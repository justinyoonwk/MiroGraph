#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept canonicalization (compact form)
- 모든 하이픈('-') 제거
- 모든 공백(스페이스/탭/줄바꿈 등) 제거
- 언더스코어('_')는 공백으로 치환 후 공백 제거 → 결과적으로 제거
- 나머지 문장 부호는 그대로 두되, 말미의 .,;: 은 제거
- 소문자 변환, 논문 내 중복 제거(순서 보존)
- 스키마 유지: id,title,concept_keywords

예)
  "dual-level context-aware pre-training method"
    -> "duallevelcontextawarepretrainingmethod"
  "multimodal" / "multi-modal" / "multi modal"
    -> "multimodal"
  "cloud-edge-device"
    -> "cloudedgedevice"
"""

import os
import re
import csv
import glob
import argparse
from pathlib import Path
from typing import List, Tuple

SPLIT_RE = re.compile(r"[;,]")              # 세미콜론 우선, 콤마 보조 분할
MULTI_SPACE_RE = re.compile(r"\s+")
TRAILING_PUNCT_RE = re.compile(r"[.,;:]+\s*$")

def parse_concept_field(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in SPLIT_RE.split(str(raw)) if p.strip()]
    return parts

def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_compact(term: str) -> str:
    """
    하이픈/공백 완전 제거 + 소문자.
    1) lower + trim
    2) 언더스코어 -> 공백
    3) 말미 구두점(.,;:) 제거
    4) 하이픈('-') 제거
    5) 모든 공백 제거(스페이스/탭/개행 등)
    """
    s = (term or "").strip().lower()
    if not s:
        return ""
    s = s.replace("_", " ")
    s = TRAILING_PUNCT_RE.sub("", s).strip()
    s = s.replace("-", "")
    s = MULTI_SPACE_RE.sub("", s)  # 모든 공백류 제거
    return s

def process_file(in_csv: Path, out_csv: Path) -> Tuple[int, int, int]:
    """
    한 파일 처리: 정규화 후 저장.
    반환: (행 수, 정규화 전 개념 수, 정규화 후 개념 수)
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    before_cnt = after_cnt = 0

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "title", "concept_keywords"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"[{in_csv}] columns={reader.fieldnames} — 예상 스키마(id,title,concept_keywords)와 다릅니다.")

        for row in reader:
            raw_list = parse_concept_field(row.get("concept_keywords"))
            before_cnt += len(raw_list)

            norm_list = [normalize_compact(x) for x in raw_list]
            norm_list = [x for x in norm_list if x]           # 빈 문자열 제거
            norm_list = dedup_preserve_order(norm_list)       # 중복 제거(순서 보존)
            after_cnt += len(norm_list)

            rows.append({
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "concept_keywords": "; ".join(norm_list),
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "concept_keywords"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), before_cnt, after_cnt

def discover_and_process(in_root: Path, out_root: Path) -> None:
    """
    in_root 아래 모든 'Proceedings of KDD */concepts_all.csv' 일괄 처리.
    """
    pattern = str(in_root / "Proceedings of KDD */concepts_all.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[!] 입력 파일을 찾지 못했습니다: {pattern}")
        return

    print(f"[i] 대상 파일 수: {len(paths)}")
    total_rows = total_before = total_after = 0

    for p in paths:
        p_in = Path(p)
        rel = p_in.relative_to(in_root)
        p_out = out_root / rel.parent / "concepts_all.canonical.compact.csv"
        print(f" - {p_in}  →  {p_out}")
        rc, bf, af = process_file(p_in, p_out)
        total_rows += rc
        total_before += bf
        total_after += af

    print("\n=== 처리 요약 ===")
    print(f"- 전체 파일 수: {len(paths)}")
    print(f"- 전체 행 수(논문 수): {total_rows:,}")
    print(f"- 개념 토큰 수(정규화 전 → 후): {total_before:,} → {total_after:,}")
    if total_before > 0:
        rate = 100.0 * (total_before - total_after) / total_before
        print(f"- 중복/표면형 축소율(대략): {rate:.2f}%")

def main():
    ap = argparse.ArgumentParser(description="Concept canonicalization to compact tokens (remove hyphens and all whitespace).")
    ap.add_argument("--in_root", type=str, default=None, help="루트 디렉터리(하위에 각 연도 폴더 존재)")
    ap.add_argument("--out_root", type=str, default=None, help="출력 루트 디렉터리(미러 구조로 저장)")
    ap.add_argument("--file", type=str, default=None, help="단일 입력 CSV 경로 (concepts_all.csv)")
    ap.add_argument("--out", type=str, default=None, help="단일 출력 CSV 경로")
    args = ap.parse_args()

    if args.file:
        if not args.out:
            raise SystemExit("--out 경로를 지정하세요 (단일 파일 처리 모드).")
        in_csv = Path(args.file)
        out_csv = Path(args.out)
        rc, bf, af = process_file(in_csv, out_csv)
        print(f"[done] {in_csv} → {out_csv}  (rows={rc}, concepts {bf}->{af})")
        return

    if not args.in_root or not args.out_root:
        raise SystemExit("--in_root 와 --out_root 를 지정하거나, --file/--out 모드를 사용하세요.")

    discover_and_process(Path(args.in_root), Path(args.out_root))

if __name__ == "__main__":
    main()
