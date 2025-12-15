#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a concept co-occurrence graph with year-wise citation-weighted edges
(using OpenAlex counts_by_year) from multiple CSVs organized by year folders.

Outputs:
  - nodes.csv                  : concept, doc_freq
  - edges_by_year.csv          : source, target, year, citations, contributing_papers
  - edges_series.csv           : source, target, start_year, "{v1, v2, ...}"  (requested format)
  - fetched_works_cache.jsonl  : cached OpenAlex work fields for reuse

Typical run:
  python build_concept_graph_openalex.py \
    --inputs_root "/home/donggyu/project1/outputs_canonicalized_compact" \
    --csv_name "concepts_all.canonical.compact.csv" \
    --id_col id --concept_col concept_keywords --title_col title \
    --threshold 10 --cutoff_year 2024 \
    --mailto justinyoonwk@kaist.ac.kr \
    --id_type doi --max_workers 8 --rps 10 \
    --outdir outputs_openalex_kdd_all
"""

import os
import re
import glob
import csv
import json
import time
import argparse
import itertools
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

OPENALEX_BASE = "https://api.openalex.org"

# ------------------------------
# Rate limiter
# ------------------------------
class RateLimiter:
    def __init__(self, rps: float):
        self.rps = max(0.1, float(rps))
        self.min_interval = 1.0 / self.rps
        self.lock = Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed:
                time.sleep(self.next_allowed - now)
            self.next_allowed = time.time() + self.min_interval

# ------------------------------
# Helpers
# ------------------------------
def guess_id_kind(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return "unknown"
    if s.startswith("https://doi.org/") or s.lower().startswith("doi:") or s.startswith("10."):
        return "doi"
    if s.startswith("https://openalex.org/W") or s.startswith("W"):
        return "openalex"
    if s.lower().startswith("arxiv:") or "arxiv.org/abs/" in s.lower():
        return "arxiv"
    if s.lower().startswith("pmid:"):
        return "pmid"
    return "unknown"

def norm_doi_to_https(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("https://doi.org/"):
        return s
    if s.lower().startswith("doi:"):
        s = s[4:]
    return "https://doi.org/" + s.lstrip()

def raw_doi_from_any(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("https://doi.org/"):
        return s.replace("https://doi.org/", "", 1)
    if s.lower().startswith("doi:"):
        return s[4:].lstrip()
    if s.startswith("10."):
        return s
    return ""

def norm_openalex_id(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("https://openalex.org/"):
        return s.split("/")[-1]
    if s.startswith("W"):
        return s
    return s

def split_concepts(raw: str, sep_opt: str = "auto"):
    if raw is None:
        return []
    txt = str(raw).strip()
    if not txt:
        return []
    # JSON list?
    if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("{") and txt.endswith("}")):
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    # fixed sep
    if sep_opt != "auto":
        parts = [p.strip() for p in txt.split(sep_opt)]
        return [p for p in parts if p]
    # auto seps
    seps = [",", ";", "|", "\t", "  "]
    pat = "|".join(map(re.escape, seps))
    parts = re.split(pat, txt)
    return [p.strip() for p in parts if p.strip()]

def load_rows_from_file(csv_path, id_col, concept_col, title_col=None, concept_sep="auto", casefold="lower"):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        header = [h.strip() for h in rdr.fieldnames]
        def pick(col, fallbacks):
            if col and col in header:
                return col
            for fb in fallbacks:
                if fb in header:
                    return fb
            return col
        _id_col = pick(id_col, ["id", "paper_id", "doi", "openalex_id"])
        _title_col = pick(title_col, ["title"])
        _concept_col = pick(concept_col, ["concept_keywords", "concepts"])

        for r in rdr:
            pid = (r.get(_id_col) or "").strip()
            if not pid:
                continue
            concepts = split_concepts(r.get(_concept_col, ""), sep_opt=concept_sep)
            if casefold == "lower":
                concepts = [c.strip().lower() for c in concepts if c.strip()]
            else:
                concepts = [c.strip() for c in concepts if c.strip()]
            concepts = sorted(set(concepts))  # dedup within paper
            if not concepts:
                continue
            title = (r.get(_title_col) or "").strip() if _title_col else ""
            rows.append({"paper_id": pid, "title": title, "concepts": concepts})
    return rows

def count_docfreq(rows):
    c = Counter()
    for r in rows:
        for cpt in set(r["concepts"]):
            c[cpt] += 1
    return c

def filter_concepts(rows, docfreq, threshold):
    kept = set([c for c, n in docfreq.items() if n >= threshold])
    new_rows = []
    for r in rows:
        cs = [c for c in r["concepts"] if c in kept]
        if cs:
            new_rows.append({"paper_id": r["paper_id"], "title": r["title"], "concepts": sorted(set(cs))})
    return new_rows, kept

def concept_pairs_for_paper(concepts):
    return [tuple(sorted(p)) for p in itertools.combinations(concepts, 2)]

def build_pair_to_papers(rows):
    pair2papers = defaultdict(set)
    for r in rows:
        pairs = concept_pairs_for_paper(r["concepts"])
        if not pairs:
            continue
        for p in pairs:
            pair2papers[p].add(r["paper_id"])
    return pair2papers

# ------------------------------
# OpenAlex fetching + cache
# ------------------------------
def backoff_sleep(attempt):
    time.sleep(min(60, (2 ** attempt)))

def fetch_json(session, url, rate: RateLimiter, max_retries=5, timeout=30):
    for attempt in range(max_retries):
        try:
            rate.wait()
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 403, 500, 502, 503, 504):
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                backoff_sleep(attempt)
            else:
                raise
    raise RuntimeError(f"Failed to GET after {max_retries} attempts: {url}")

def batch_fetch_works_by_ids(session, ids, id_kind, mailto, rate, select_fields=("id","doi","publication_year","counts_by_year")):
    works = []
    ids = list(ids)
    chunk = 50
    sel = ",".join(select_fields)
    for i in range(0, len(ids), chunk):
        batch = ids[i:i+chunk]
        if id_kind == "doi":
            vals = "|".join([norm_doi_to_https(x) for x in batch])
            url = f"{OPENALEX_BASE}/works?filter=doi:{vals}&per-page={len(batch)}&select={sel}"
        elif id_kind == "openalex":
            vals = "|".join([norm_openalex_id(x) for x in batch])
            url = f"{OPENALEX_BASE}/works?filter=openalex:{vals}&per-page={len(batch)}&select={sel}"
        else:
            for single in batch:
                works.extend(single_fetch_work(session, single, id_kind, mailto, rate, select_fields))
            continue
        if mailto:
            url += f"&mailto={mailto}"
        data = fetch_json(session, url, rate)
        works.extend(data.get("results", []))
    return works

def single_fetch_work(session, id_value, id_kind, mailto, rate, select_fields=("id","doi","publication_year","counts_by_year")):
    path = None
    if id_kind == "doi":
        path = f"/works/{norm_doi_to_https(id_value)}"
    elif id_kind == "openalex":
        wid = norm_openalex_id(id_value)
        path = f"/works/{wid}" if wid.startswith("W") else f"/works/{id_value}"
    elif id_kind == "arxiv":
        v = (id_value or "").strip()
        v = v.replace("https://arxiv.org/abs/", "")
        if not v.lower().startswith("arxiv:"):
            v = "arXiv:" + v
        path = f"/works/{v}"
    elif id_kind == "pmid":
        v = (id_value or "").strip()
        if not v.lower().startswith("pmid:"):
            v = "pmid:" + v
        path = f"/works/{v}"
    else:
        return []

    sel = ",".join(select_fields)  # BUG guard: keep the next line correct
    sel = ",".join(select_fields)
    url = f"{OPENALEX_BASE}{path}?select={sel}"
    if mailto:
        url += f"&mailto={mailto}"
    try:
        data = fetch_json(session, url, rate)
        return [data]
    except Exception:
        return []

def unify_work_record(w):
    """Return normalized dict and alias keys."""
    oid_full = w.get("id") or ""
    openalex_id = oid_full.split("/")[-1] if oid_full else None
    doi = w.get("doi") or None
    pub_year = w.get("publication_year")
    cby = {}
    for item in (w.get("counts_by_year") or []):
        y = item.get("year")
        c = item.get("cited_by_count")
        if isinstance(y, int) and isinstance(c, int):
            cby[y] = c

    rec = {
        "openalex_id": openalex_id,
        "doi_https": doi,
        "doi_raw": raw_doi_from_any(doi) if doi else None,
        "publication_year": pub_year,
        "counts_by_year": cby
    }

    aliases = set()
    if openalex_id:
        aliases.add(openalex_id)
        aliases.add("https://openalex.org/" + openalex_id)
    if doi:
        aliases.add(doi)  # https://doi.org/...
        rd = raw_doi_from_any(doi)
        if rd:
            aliases.add(rd)          # 10.xxxx/..
            aliases.add("doi:" + rd)
    return rec, aliases

def load_cache(cache_path):
    mapping = {}
    if not cache_path or not os.path.isfile(cache_path):
        return mapping
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                u = json.loads(line)
                aliases = set()
                if u.get("openalex_id"):
                    aliases.add(u["openalex_id"])
                    aliases.add("https://openalex.org/" + u["openalex_id"])
                if u.get("doi"):
                    doi_https = u["doi"]
                    aliases.add(doi_https)
                    rd = raw_doi_from_any(doi_https)
                    if rd:
                        aliases.add(rd)
                        aliases.add("doi:" + rd)
                rec = {
                    "openalex_id": u.get("openalex_id"),
                    "doi_https": u.get("doi") or None,
                    "doi_raw": raw_doi_from_any(u.get("doi")) if u.get("doi") else None,
                    "publication_year": u.get("publication_year"),
                    "counts_by_year": u.get("counts_by_year") or {}
                }
                for k in aliases:
                    mapping[k] = rec
            except Exception:
                continue
    return mapping

def save_cache(cache_path, cache_records):
    if not cache_path:
        return
    with open(cache_path, "w", encoding="utf-8") as f:
        for rec in cache_records:
            out = {
                "openalex_id": rec.get("openalex_id"),
                "doi": rec.get("doi_https"),
                "publication_year": rec.get("publication_year"),
                "counts_by_year": rec.get("counts_by_year"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

def collect_works_counts(paper_ids, id_type, mailto, max_workers, rps, cache_in_path=None):
    session = requests.Session()
    rate = RateLimiter(rps)

    work_info_map = load_cache(cache_in_path)

    remaining = []
    for pid in paper_ids:
        if pid in work_info_map:
            continue
        kind_guess = guess_id_kind(pid)
        if kind_guess == "doi":
            if norm_doi_to_https(pid) in work_info_map:
                continue
            rd = raw_doi_from_any(pid)
            if rd and rd in work_info_map:
                continue
        elif kind_guess == "openalex":
            wid = norm_openalex_id(pid)
            if wid in work_info_map or ("https://openalex.org/" + wid) in work_info_map:
                continue
        remaining.append(pid)

    if not remaining:
        return work_info_map, []

    buckets = defaultdict(list)
    for pid in remaining:
        k = id_type if id_type != "auto" else guess_id_kind(pid)
        buckets[k].append(pid)

    new_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for kind in ("doi", "openalex"):
            ids = buckets.get(kind, [])
            if not ids:
                continue
            def run_batch(kind=kind, ids=ids):
                ws = batch_fetch_works_by_ids(session, ids, kind, mailto, rate)
                return kind, ws
            futures.append(ex.submit(run_batch))
        for kind in buckets:
            if kind in ("doi", "openalex"):
                continue
            ids = buckets[kind]
            def run_singles(kind=kind, ids=ids):
                sel = ("id","doi","publication_year","counts_by_year")
                allw = []
                for pid in ids:
                    allw.extend(single_fetch_work(session, pid, kind, mailto, rate, select_fields=sel))
                return kind, allw
            futures.append(ex.submit(run_singles))

        for fu in as_completed(futures):
            _, works = fu.result()
            for w in works:
                rec, aliases = unify_work_record(w)
                for k in aliases:
                    work_info_map[k] = rec
                new_records.append(rec)

    return work_info_map, new_records

# ------------------------------
# Aggregation
# ------------------------------
def aggregate_edges(pair2papers, work_info_map, cutoff_year):
    """pair -> {year: total_citations}, pair -> {year: contributing_paper_count}"""
    edges_yearly = defaultdict(lambda: defaultdict(int))
    contribs = defaultdict(lambda: defaultdict(int))
    for pair, pids in pair2papers.items():
        for pid in pids:
            rec = work_info_map.get(pid)
            if not rec:
                kind = guess_id_kind(pid)
                if kind == "doi":
                    https = norm_doi_to_https(pid)
                    rec = work_info_map.get(https) or work_info_map.get(raw_doi_from_any(https)) \
                          or work_info_map.get("doi:" + raw_doi_from_any(https))
                elif kind == "openalex":
                    wid = norm_openalex_id(pid)
                    rec = work_info_map.get(wid) or work_info_map.get("https://openalex.org/" + wid)
            if not rec:
                continue
            cby = rec.get("counts_by_year", {}) or {}
            for y, c in cby.items():
                if isinstance(y, int) and y <= cutoff_year and c > 0:
                    edges_yearly[pair][y] += c
                    contribs[pair][y] += 1
    return edges_yearly, contribs

def build_edges_series(edges_yearly, cutoff_year):
    """
    For each pair, find earliest year with positive citations (<= cutoff),
    then build counts sequence from start_year..cutoff_year (fill missing years with 0).
    Returns: dict[pair] = (start_year, [c_y_start, ..., c_cutoff])
    """
    series = {}
    for pair, ydict in edges_yearly.items():
        pos_years = [y for y, c in ydict.items() if y <= cutoff_year and c > 0]
        if not pos_years:
            continue
        start_year = min(pos_years)
        years = list(range(start_year, cutoff_year + 1))
        seq = [ydict.get(y, 0) for y in years]
        series[pair] = (start_year, seq)
    return series

# ------------------------------
# I/O
# ------------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_nodes(outdir, docfreq, kept_set):
    path = os.path.join(outdir, "nodes.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["concept", "doc_freq"])
        for c in sorted(kept_set):
            wr.writerow([c, docfreq.get(c, 0)])

def save_edges_by_year(outdir, edges_yearly, contribs):
    path = os.path.join(outdir, "edges_by_year.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "target", "year", "citations", "contributing_papers"])
        for (a, b), ydict in sorted(edges_yearly.items()):
            for y, val in sorted(ydict.items()):
                wr.writerow([a, b, y, val, contribs[(a, b)].get(y, 0)])

def save_edges_series(outdir, edges_series):
    """
    CSV: source,target,start_year,counts_series
    counts_series example: "{7, 9, 21, 47}"
    """
    path = os.path.join(outdir, "edges_series.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "target", "start_year", "counts_series"])
        for (a, b), (start_y, seq) in sorted(edges_series.items()):
            series_str = "{%s}" % (", ".join(str(x) for x in seq))
            wr.writerow([a, b, start_y, series_str])

# ------------------------------
# Input collection
# ------------------------------
def collect_input_files(args):
    files = []
    if args.input_glob:
        files = sorted(glob.glob(args.input_glob))
    elif args.inputs_root:
        pattern = os.path.join(args.inputs_root, "**", args.csv_name)
        files = sorted(glob.glob(pattern, recursive=True))
    elif args.input_csv:
        files = [args.input_csv]
    else:
        raise ValueError("입력을 지정하세요: --inputs_root + --csv_name, 또는 --input_glob, 또는 --input_csv")
    if not files:
        raise ValueError("입력 CSV를 찾지 못했습니다.")
    return files

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build concept co-occurrence graph (year-wise citation weights) via OpenAlex.")
    # 입력
    ap.add_argument("--inputs_root", default=None)
    ap.add_argument("--csv_name", default="concepts_all.canonical.compact.csv")
    ap.add_argument("--input_glob", default=None)
    ap.add_argument("--input_csv", default=None)
    # 컬럼/파싱
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--concept_col", default="concept_keywords")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--concept_sep", default="auto")
    ap.add_argument("--casefold", default="lower", choices=["lower", "none"])
    # 그래프/필터
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--cutoff_year", type=int, required=True)
    # OpenAlex
    ap.add_argument("--mailto", default="")
    ap.add_argument("--id_type", default="auto", choices=["auto", "doi", "openalex", "arxiv", "pmid"])
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--rps", type=float, default=10.0)
    # 캐시/출력
    ap.add_argument("--outdir", default="outputs_openalex")
    ap.add_argument("--cache_in", default=None)
    ap.add_argument("--cache_out", default=None)
    ap.add_argument("--log_every", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.mailto and args.rps > 1.0:
        args.rps = 1.0  # polite default

    ensure_dir(args.outdir)
    if not args.cache_out:
        args.cache_out = os.path.join(args.outdir, "fetched_works_cache.jsonl")

    # 1) 입력 수집
    files = collect_input_files(args)
    logging.info(f"Found {len(files)} CSV files.")
    for p in files:
        logging.info(f" - {p}")

    # 2) 로드
    rows_all = []
    for idx, csv_path in enumerate(files, 1):
        rows = load_rows_from_file(csv_path, args.id_col, args.concept_col, args.title_col, args.concept_sep, args.casefold)
        rows_all.extend(rows)
        if idx % 20 == 0:
            logging.info(f"Loaded {idx}/{len(files)} files, cumulative rows: {len(rows_all)}")
    logging.info(f"Total rows loaded: {len(rows_all)}")

    # 3) 임계값 필터
    docfreq = count_docfreq(rows_all)
    rows_filt, kept = filter_concepts(rows_all, docfreq, args.threshold)
    logging.info(f"Kept {len(kept)} concepts (threshold={args.threshold}); rows after filter: {len(rows_filt)}")

    # 4) pair → papers
    pair2papers = build_pair_to_papers(rows_filt)
    logging.info(f"Concept pairs: {len(pair2papers)}")

    # 5) 조회 대상 ID
    paper_ids_needed = set()
    for pset in pair2papers.values():
        paper_ids_needed.update(pset)
    logging.info(f"Unique papers to fetch: {len(paper_ids_needed)}")

    # 6) OpenAlex 조회 (+ 캐시)
    work_info_map, new_records = collect_works_counts(
        paper_ids=list(paper_ids_needed),
        id_type=args.id_type,
        mailto=args.mailto,
        max_workers=args.max_workers,
        rps=args.rps,
        cache_in_path=args.cache_in
    )
    save_cache(args.cache_out, new_records)
    unique_keys = {(rec.get("openalex_id") or rec.get("doi_https")) for rec in new_records}
    unique_keys.discard(None)
    logging.info(f"Fetched/loaded metadata for ~{len(unique_keys)} new works; cache @ {args.cache_out}")

    # 7) 연도별 합산
    edges_yearly, contribs = aggregate_edges(pair2papers, work_info_map, args.cutoff_year)

    # 8) 시리즈 형태로 변환 및 저장
    edges_series = build_edges_series(edges_yearly, args.cutoff_year)
    save_edges_series(args.outdir, edges_series)

    # 9) 참조용 CSV 저장
    save_nodes(args.outdir, docfreq, kept)
    save_edges_by_year(args.outdir, edges_yearly, contribs)

    logging.info(f"Done. Outputs in: {args.outdir}")
    logging.info("Files: nodes.csv, edges_by_year.csv, edges_series.csv, fetched_works_cache.jsonl")

if __name__ == "__main__":
    main()
