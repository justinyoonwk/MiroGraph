# concept_extractor.py
# ------------------------------------------------------------
# Bib → (Structurer→Extractor) → concepts
# - OPENAI_API_KEY: 아래 상수에 "직접" 하드코딩 (우선 사용)
# - 프롬프트: 파일 경로로 읽기 (--structurer_prompt_path / --extractor_prompt_path)
# - 입력: --data_dir (기본 현재 폴더 .) 아래의 *.bib, *.bip 일괄 처리
#   * --file_only 로 특정 파일 1개만 처리 가능
# - 엔트리(논문) 병렬: --workers / --max_inflight
# - 샤딩: --num_shards / --shard_id (엔트리 인덱스 기반 분할)
# - 출력: outputs/*.jsonl, outputs/concepts_all.csv
# ------------------------------------------------------------

import os
import time
import json
import csv
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

# ====== 🔑 여기에 네 OpenAI API Key를 직접 써 넣으세요 ======
OPENAI_BASE_URL_INLINE = None  # 필요 시 "https://api.openai.com/v1" 등으로 지정(기본 None)

# 외부 패키지 로드
def _safe_import(name, pip_hint=None):
    try:
        return __import__(name)
    except Exception as e:
        hint = f"\n[설치 필요] pip install {pip_hint or name}\n"
        raise ImportError(f"모듈 '{name}'을(를) 불러오지 못했습니다: {e}{hint}") from e

orjson = _safe_import("orjson")
requests = _safe_import("requests")
tqdm = _safe_import("tqdm").tqdm
bibtexparser = _safe_import("bibtexparser")

# 재사용 HTTP 세션 (속도/안정성 소폭 향상)
_SESSION = requests.Session()

# ============ 유틸 ============

def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_api_key(args) -> str:
    """
    하드코딩된 키 > 환경변수 > --openai_key_path 순으로 로드.
    """
    # 1) 하드코딩 키
    if OPENAI_API_KEY_INLINE and OPENAI_API_KEY_INLINE.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_INLINE.strip()
        return os.environ["OPENAI_API_KEY"]

    # 2) 환경변수
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # 3) 키 파일 경로(옵션)
    if args.openai_key_path:
        key_path = Path(args.openai_key_path).expanduser()
        if not key_path.exists():
            raise FileNotFoundError(f"[OPENAI KEY] 경로가 존재하지 않습니다: {key_path}")
        key = read_text_file(key_path).strip()
        if not key:
            raise ValueError(f"[OPENAI KEY] 파일이 비어있습니다: {key_path}")
        os.environ["OPENAI_API_KEY"] = key
        return key

    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        "코드 상단 OPENAI_API_KEY_INLINE에 직접 입력하거나, 환경변수/키파일을 사용하세요."
    )

def load_prompts(args) -> Dict[str, str]:
    if not args.structurer_prompt_path or not args.extractor_prompt_path:
        raise RuntimeError(
            "프롬프트 파일 경로가 필요합니다. "
            "--structurer_prompt_path 와 --extractor_prompt_path 를 지정하세요."
        )
    sp = Path(args.structurer_prompt_path).expanduser()
    ep = Path(args.extractor_prompt_path).expanduser()
    if not sp.exists():
        raise FileNotFoundError(f"[PROMPT] Structurer 프롬프트 파일 없음: {sp}")
    if not ep.exists():
        raise FileNotFoundError(f"[PROMPT] Extractor 프롬프트 파일 없음: {ep}")
    return {
        "STRUCTURER_PROMPT": read_text_file(sp),
        "EXTRACTOR_PROMPT": read_text_file(ep),
    }

# ============ OpenAI Chat Completions(JSON 모드) ============

def _chat_json(model: str, system_prompt: str, user_content: str, debug_label: str,
               base_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Chat Completions 호출(JSON 강제). 429/5xx 백오프 포함.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(4):
        try:
            r = _SESSION.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                content = (data["choices"][0]["message"]["content"] or "").strip()
                try:
                    return orjson.loads(content)
                except Exception:
                    last_err = RuntimeError(f"{debug_label}: non-JSON content:\n{content[:800]}")
                    time.sleep(1.5 * (attempt + 1))
                    continue
            # 에러 본문 파싱
            try:
                err = r.json().get("error", {})
                msg = err.get("message", r.text)
            except Exception:
                msg = r.text

            if r.status_code in (429, 500, 502, 503):
                last_err = RuntimeError(f"{debug_label} HTTP {r.status_code}: {msg[:400]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"{debug_label} HTTP {r.status_code}: {msg[:400]}")
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    raise last_err if last_err else RuntimeError(f"{debug_label}: unknown failure")

# ============ 파이프라인 함수 ============

def run_structurer_on_bib(entry: Dict[str, Any],
                          structurer_prompt: str,
                          model_structurer: str,
                          base_url: str,
                          headers: Dict[str, str]) -> Dict[str, Any]:
    title = (entry.get("title") or "").strip()
    abstract = (entry.get("abstract") or "").strip()
    if not title or not abstract:
        return {"status": "invalid", "reason": "missing_title_or_abstract"}

    key = entry.get("ID", "paper")
    bib_min = f"@inproceedings{{{key},\n  title = {{{title}}},\n  abstract = {{{abstract}}}\n}}"
    user_msg = "Input BibTeX Entry:\n" + bib_min

    js = _chat_json(model_structurer, structurer_prompt, user_msg, "STRUCTURER", base_url, headers)
    if "status" not in js:
        raise RuntimeError("STRUCTURER: missing 'status' in response")
    return js

def run_extractor_on_structured(structured: Dict[str, Any],
                                extractor_prompt: str,
                                model_extractor: str,
                                base_url: str,
                                headers: Dict[str, str]) -> List[str]:
    user_msg = "Structured JSON from Abstract Structurer:\n" + json.dumps(structured, ensure_ascii=False)
    js = _chat_json(model_extractor, extractor_prompt, user_msg, "EXTRACTOR", base_url, headers)

    kws = js.get("concept_keywords", [])
    if not isinstance(kws, list):
        raise RuntimeError("EXTRACTOR: 'concept_keywords' must be a list")

    cleaned, seen = [], set()
    for c in kws:
        if not isinstance(c, str):
            continue
        cc = c.strip()
        if cc and cc not in seen:
            seen.add(cc)
            cleaned.append(cc)
    if not cleaned:
        raise RuntimeError("EXTRACTOR: empty list")
    return cleaned

# ============ 메인 배치 ============

def process_batch(args):
    # 키/엔드포인트/헤더
    api_key = ensure_api_key(args)
    base_url = (
        OPENAI_BASE_URL_INLINE
        or args.base_url
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 프롬프트 로드
    prompts = load_prompts(args)
    STRUCTURER_PROMPT = prompts["STRUCTURER_PROMPT"]
    EXTRACTOR_PROMPT = prompts["EXTRACTOR_PROMPT"]

    # 경로 준비
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 입력 파일 (*.bib, *.bip)
    if args.file_only:
        bib_paths = [str(Path(args.file_only).expanduser())]
    else:
        bib_paths = sorted(glob.glob(str(data_dir / "*.bib")) + glob.glob(str(data_dir / "*.bip")))
    if not bib_paths:
        print(f"[!] 처리할 파일을 찾지 못했습니다. 경로 확인: {data_dir}/*.bib (또는 *.bip)")
        return

    print(f"[i] 대상 파일 개수: {len(bib_paths)}")
    for p in bib_paths:
        print(" -", p)

    all_results: List[Dict[str, Any]] = []
    limit = args.entry_limit if args.entry_limit and args.entry_limit > 0 else None

    for bib_path in bib_paths:
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)

        entries = bib_db.entries or []
        total_entries = len(entries)

        # entry_limit 적용
        if limit is not None:
            entries = entries[:limit]

        # ✅ 샤딩 적용 (엔트리 인덱스 기반 분할)
        if args.num_shards > 1:
            sharded = []
            for i, e in enumerate(entries):
                if (i % args.num_shards) == args.shard_id:
                    sharded.append(e)
            entries = sharded
            print(f"[i] Shard {args.shard_id}/{args.num_shards} - {len(entries)}/{total_entries} entries selected")

        rows: List[Dict[str, Any]] = []

        # ---------- 엔트리 처리 함수 ----------
        def _process_entry(entry):
            try:
                structured = run_structurer_on_bib(
                    entry,
                    structurer_prompt=STRUCTURER_PROMPT,
                    model_structurer=args.model_structurer,
                    base_url=base_url,
                    headers=headers,
                )
            except Exception:
                return None

            if structured.get("status") != "valid":
                return None

            try:
                concepts = run_extractor_on_structured(
                    structured,
                    extractor_prompt=EXTRACTOR_PROMPT,
                    model_extractor=args.model_extractor,
                    base_url=base_url,
                    headers=headers,
                )
            except Exception:
                return None

            paper_id = entry.get("ID") or (entry.get("title", "")[:50] or "paper")
            return {
                "id": paper_id,
                "title": entry.get("title", ""),
                "concept_keywords": concepts,
            }

        # ---------- 동시 요청 상한 ----------
        max_inflight = args.max_inflight or args.workers
        sem = Semaphore(max_inflight)

        def _wrapped(entry):
            with sem:  # 동시 API 요청 제한
                return _process_entry(entry)

        # ---------- 병렬/직렬 실행 ----------
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_wrapped, e) for e in entries]
                for f in tqdm(as_completed(futs), total=len(futs), desc=f"Processing {Path(bib_path).name}"):
                    rec = f.result()
                    if rec:
                        rows.append(rec)
                        all_results.append(rec)
        else:
            for entry in tqdm(entries, desc=f"Processing {Path(bib_path).name}"):
                rec = _wrapped(entry)
                if rec:
                    rows.append(rec)
                    all_results.append(rec)

        # 파일별 jsonl 저장
        out_jsonl = out_dir / f"{Path(bib_path).stem}.concepts.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(orjson.dumps(r).decode("utf-8") + "\n")
        print(f"Saved: {out_jsonl}  ({len(rows)} records)")

    # 전체 CSV 저장 (현재 실행 범위 내 전체 파일 합)
    out_csv = out_dir / "concepts_all.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "concept_keywords"])
        for r in all_results:
            w.writerow([r["id"], r["title"], "; ".join(r["concept_keywords"])])
    print(f"Saved: {out_csv}  (total {len(all_results)})")

# ============ CLI ============

def build_argparser():
    p = argparse.ArgumentParser(description="Bib → (Structurer→Extractor) → concepts")
    # 입출력
    p.add_argument("--data_dir", type=str, default=".", help="*.bib / *.bip가 있는 디렉터리 (기본: 현재 폴더)")
    p.add_argument("--out_dir", type=str, default="outputs", help="결과 저장 디렉터리")
    p.add_argument("--entry_limit", type=int, default=None, help="각 bib 파일당 처리 상한 (기본: 전체)")

    # OpenAI 설정 (하드코딩 우선, 옵션은 백업 용도)
    p.add_argument("--openai_key_path", type=str, default=None, help="OPENAI_API_KEY가 저장된 파일 경로")
    p.add_argument("--base_url", type=str, default=None, help="OpenAI 호환 API Base URL")

    # 모델
    p.add_argument("--model_structurer", type=str, default="gpt-5-mini", help="Structurer 호출 모델")
    p.add_argument("--model_extractor",  type=str, default="gpt-5-mini", help="Extractor 호출 모델")

    # 프롬프트 경로
    p.add_argument("--structurer_prompt_path", type=str, required=True, help="Structurer 프롬프트 파일 경로")
    p.add_argument("--extractor_prompt_path",  type=str, required=True, help="Extractor 프롬프트 파일 경로")

    # 단일 파일 처리 / 병렬 옵션
    p.add_argument("--file_only", type=str, default=None, help="해당 경로의 .bib/.bip 한 파일만 처리")
    p.add_argument("--workers", type=int, default=1, help="엔트리 동시 처리 스레드 수(기본 1=직렬)")
    p.add_argument("--max_inflight", type=int, default=None, help="동시 API 요청 상한(기본: workers와 동일)")

    # ✅ 샤딩 옵션
    p.add_argument("--num_shards", type=int, default=1, help="Shard count for parallel runs")
    p.add_argument("--shard_id", type=int, default=0, help="Shard id in [0..num_shards-1]")

    return p

def main():
    args = build_argparser().parse_args()
    process_batch(args)

if __name__ == "__main__":
    main()
