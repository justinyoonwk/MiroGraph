#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, glob, argparse, json, re
from collections import defaultdict
from itertools import combinations

def split_concepts(raw: str):
    if raw is None: return []
    s = str(raw).strip()
    if not s: return []
    # JSON list 허용
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    # 구분자 자동
    seps = [",",";","|","\t","  "]
    pat = "|".join(map(re.escape, seps))
    parts = re.split(pat, s)
    return [p.strip() for p in parts if p.strip()]

def load_kept_concepts(nodes_csv):
    kept = set()
    with open(nodes_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            kept.add(r["concept"].strip())
    return kept

def pairs_from_csvs(globs, kept_concepts):
    pairs = set()
    def norm_list(lst):
        return sorted(set(x.strip().lower() for x in lst if x.strip()))
    files = []
    for g in globs:
        files.extend(glob.glob(g))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            # 컬럼 후보 유연 매칭
            header = [h.strip() for h in rdr.fieldnames]
            concept_col = "concept_keywords" if "concept_keywords" in header else ("concepts" if "concepts" in header else None)
            if not concept_col:
                continue
            for r in rdr:
                cs = norm_list(split_concepts(r.get(concept_col, "")))
                # 2019~2021 스냅샷의 어휘만 사용
                cs = [c for c in cs if c in kept_concepts]
                for a, b in combinations(cs, 2):
                    if a == b: continue
                    if a > b: a, b = b, a
                    pairs.add((a,b))
    return pairs

def load_edges_by_year(path):
    by = defaultdict(lambda: defaultdict(int))  # (a,b)->year->cit
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            a = r["source"].strip().lower()
            b = r["target"].strip().lower()
            if a > b: a, b = b, a
            y = int(r["year"])
            c = int(r["citations"])
            by[(a,b)][y] += c
    return by

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_nodes", required=True, help="outputs_openalex_kdd_2019_2021/nodes.csv")
    ap.add_argument("--train_csvs", nargs="+", required=True,
                    help="2019~2021 입력 CSV들의 경로(여러 개 글롭 허용)")
    ap.add_argument("--full_edges_by_year", required=True,
                    help="outputs_openalex_kdd_2019_2024/edges_by_year.csv")
    ap.add_argument("--sum_start", type=int, default=2022)
    ap.add_argument("--sum_end", type=int, default=2024)
    ap.add_argument("--ir", type=int, required=True, help="임계값 IR")
    ap.add_argument("--out", default="labels_newpairs_2022_2024.csv")
    args = ap.parse_args()

    kept = load_kept_concepts(args.train_nodes)
    train_pairs = pairs_from_csvs(args.train_csvs, kept)
    full_by_year = load_edges_by_year(args.full_edges_by_year)

    # full에서 나타난 모든 pair 중 kept 어휘 외 pair 제거
    full_pairs = {p for p in full_by_year.keys() if p[0] in kept and p[1] in kept}

    new_pairs = full_pairs - train_pairs  # 2019~2021엔 미연결, 2019~2024엔 연결
    print(f"Newly connected pairs (count): {len(new_pairs)}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source","target","sum_2022_2024","label","c2022","c2023","c2024"])
        for (a,b) in sorted(new_pairs):
            ymap = full_by_year.get((a,b), {})
            c22 = ymap.get(2022, 0) if args.sum_start <= 2022 <= args.sum_end else 0
            c23 = ymap.get(2023, 0) if args.sum_start <= 2023 <= args.sum_end else 0
            c24 = ymap.get(2024, 0) if args.sum_start <= 2024 <= args.sum_end else 0
            s = 0
            for y in range(args.sum_start, args.sum_end+1):
                s += ymap.get(y, 0)
            label = 1 if s >= args.ir else 0
            wr.writerow([a,b,s,label,c22,c23,c24])

if __name__ == "__main__":
    main()
