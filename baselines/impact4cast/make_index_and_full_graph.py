# baselines/impact4cast/make_index_and_full_graph.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import date
import pandas as pd
import numpy as np

def y2day(y: int) -> int:
    origin = date(1990, 1, 1)
    return (date(int(y), 12, 31) - origin).days

def read_nodes(nodes_path: str) -> pd.DataFrame:
    df = pd.read_csv(nodes_path)
    # 컬럼명 정리
    cols = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    df.columns = cols
    concept_col = "concept" if "concept" in df.columns else cols[0]
    df = df.rename(columns={concept_col: "concept"})
    df = df.drop_duplicates(subset=["concept"]).reset_index(drop=True)
    df["vid"] = df.index.astype(np.int32)
    return df[["concept", "vid"]]

def _read_csv_header_first(path: str) -> pd.DataFrame:
    # 헤더 있다고 가정하고 1차 시도
    df = pd.read_csv(path)
    cols = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    df.columns = cols
    return df

def _read_csv_no_header(path: str) -> pd.DataFrame:
    # 헤더 없다고 가정하고 2차 시도
    return pd.read_csv(
        path,
        names=["src", "dst", "year", "citations", "num_papers"],
        header=None,
    )

def read_edges_by_year(edges_path: str) -> pd.DataFrame:
    # 1) 헤더 시도
    df = _read_csv_header_first(edges_path)

    # 1-a) 기대 컬럼이 있으면 그대로 사용
    if {"src", "dst", "year"}.issubset(set(df.columns)):
        pass
    else:
        # 1-b) 기대 컬럼이 없으면 헤더가 없는 파일로 간주하고 다시 로드
        df = _read_csv_no_header(edges_path)

    # 불필요 컬럼 제거(있으면 유지)
    keep = [c for c in ["src", "dst", "year", "citations", "num_papers"] if c in df.columns]
    df = df[keep].copy()

    # 문자열 정리
    df["src"] = df["src"].astype(str).str.strip()
    df["dst"] = df["dst"].astype(str).str.strip()

    # 연도 숫자 변환(안되는 건 NaN → 제거)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    bad_year = df["year"].isna().sum()
    if bad_year:
        print(f"[warn] drop {bad_year} rows with invalid year")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df

def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    nodes_path = os.path.join(args.graph_root, "nodes.csv")
    edges_path = os.path.join(args.graph_root, "edges_by_year.csv")

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"nodes.csv not found: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"edges_by_year.csv not found: {edges_path}")

    # 1) 노드 인덱스
    nodes = read_nodes(nodes_path)
    nodes.to_parquet(os.path.join(outdir, "concept_index.parquet"), index=False)
    print(f"[index] num concepts = {len(nodes)}  → saved to {outdir}")

    vid_of = dict(zip(nodes["concept"], nodes["vid"]))

    # 2) 엣지 로드
    e = read_edges_by_year(edges_path)

    # (src, dst) 정렬 및 self-loop 제거
    src_sorted = np.minimum(e["src"].values, e["dst"].values)
    dst_sorted = np.maximum(e["src"].values, e["dst"].values)
    e = e.assign(src=src_sorted, dst=dst_sorted)
    e = e[e["src"] != e["dst"]].copy()

    # 3) 노드 사전에 없는 개념 제거
    mask = e["src"].isin(vid_of) & e["dst"].isin(vid_of)
    dropped = (~mask).sum()
    if dropped:
        print(f"[warn] drop {dropped} edge-rows whose endpoints not in nodes.csv")
    e = e.loc[mask].copy()

    # 4) 최초 연결 연도(first_year)만 사용해 full_dynamic_graph 생성
    first_year = (
        e.groupby(["src", "dst"], as_index=False)["year"]
        .min()
        .rename(columns={"year": "first_year"})
    )
    first_year["u"] = first_year["src"].map(vid_of).astype(np.int32)
    first_year["v"] = first_year["dst"].map(vid_of).astype(np.int32)
    first_year["time"] = first_year["first_year"].apply(y2day).astype(np.int32)

    full_dynamic_graph = first_year[["u", "v", "time"]].sort_values(["time", "u", "v"]).reset_index(drop=True)
    full_dynamic_graph.to_parquet(os.path.join(outdir, "full_dynamic_graph.parquet"), index=False)

    print(
        f"[graph] earliest-year edges = {len(full_dynamic_graph)} "
        f"(unique pairs). Saved → {outdir}/full_dynamic_graph.parquet"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_root", type=str, required=True, help="dir with nodes.csv & edges_by_year.csv")
    parser.add_argument("--outdir", type=str, required=True, help="output dir for parquet files")
    args = parser.parse_args()
    main(args)
