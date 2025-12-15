# baselines/impact4cast/prepare_unconnected_pairs_2021.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

def y2day(y: int) -> int:
    from datetime import date
    origin = date(1990, 1, 1)
    return (date(int(y), 12, 31) - origin).days

def read_index(index_path: str):
    idx = pd.read_parquet(index_path)
    idx = idx.rename(columns={idx.columns[0]: "concept"})
    idx["vid"] = idx["vid"].astype(int)
    return idx

def read_full_graph(path: str):
    g = pd.read_parquet(path)
    # time은 days since 1990-01-01
    return g[["u","v","time"]].copy()

def _read_csv_header_first(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    df.columns = cols
    return df

def _read_csv_no_header(path: str) -> pd.DataFrame:
    return pd.read_csv(path, names=["src","dst","year","citations","num_papers"], header=None)

def read_edges_by_year(path: str) -> pd.DataFrame:
    df = _read_csv_header_first(path)
    if not {"src","dst","year"}.issubset(set(df.columns)):
        df = _read_csv_no_header(path)
    keep = [c for c in ["src","dst","year","citations","num_papers"] if c in df.columns]
    df = df[keep].copy()
    df["src"] = df["src"].astype(str).str.strip()
    df["dst"] = df["dst"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["citations"] = pd.to_numeric(df.get("citations", 0), errors="coerce").fillna(0).astype(int)
    # (src,dst) 정렬
    s = np.minimum(df["src"].values, df["dst"].values)
    d = np.maximum(df["src"].values, df["dst"].values)
    df = df.assign(src=s, dst=d)
    return df

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    idx = read_index(args.index)
    vid_of = dict(zip(idx["concept"], idx["vid"]))
    n = len(idx)

    # 최초연도(=full_dynamic_graph) 로드
    g = read_full_graph(args.graph_parquet)
    # 일자→년도
    g["year"] = pd.to_datetime("1990-01-01") + pd.to_timedelta(g["time"], unit="D")
    g["year"] = g["year"].dt.year.astype(int)

    # 2021까지 연결된 edge 집합
    connected_by_2021 = set(map(tuple, g.loc[g["year"] <= 2021, ["u","v"]].values))

    # 2024까지 최초연도 정보
    first_year_map = {(int(u),int(v)): int(y) for u,v,y in g[["u","v","year"]].itertuples(index=False)}

    # 연도별 인용 테이블
    eby = read_edges_by_year(args.edges_csv)
    # concept → vid 매핑
    mask = eby["src"].isin(vid_of) & eby["dst"].isin(vid_of)
    eby = eby.loc[mask].copy()
    eby["u"] = eby["src"].map(vid_of).astype(int)
    eby["v"] = eby["dst"].map(vid_of).astype(int)
    # (u,v) 정렬 안전
    u = np.minimum(eby["u"].values, eby["v"].values)
    v = np.maximum(eby["u"].values, eby["v"].values)
    eby = eby.assign(u=u, v=v)

    # 2022~2024 인용 합
    eby_224 = eby[(eby["year"] >= 2022) & (eby["year"] <= 2024)]
    cit_224 = eby_224.groupby(["u","v"], as_index=False)["citations"].sum().rename(columns={"citations": "cit_sum_224"})

    # 양성: 2021까지 미연결 & 최초연도 ∈ {2022,2023,2024}
    pos_pairs = []
    for (u,v), y in first_year_map.items():
        if y >= 2022 and y <= 2024 and (u,v) not in connected_by_2021:
            pos_pairs.append((u,v))
    pos_df = pd.DataFrame(pos_pairs, columns=["u","v"])
    pos_df = pos_df.merge(cit_224, on=["u","v"], how="left").fillna({"cit_sum_224":0})
    pos_df["cit_sum_224"] = pos_df["cit_sum_224"].astype(int)

    # 음성: 2024까지도 연결되지 않은 모든 쌍
    all_pairs = np.array(list(combinations(range(n), 2)), dtype=np.int32)
    all_pairs_set = set(map(tuple, map(tuple, all_pairs)))
    connected_by_2024 = set(map(tuple, g.loc[g["year"] <= 2024, ["u","v"]].values))
    neg_set = list(all_pairs_set - connected_by_2024)
    if args.neg_sample > 0 and args.neg_sample < len(neg_set):
        rng = np.random.default_rng(42)
        neg_set = list(rng.choice(np.array(neg_set, dtype=object), size=args.neg_sample, replace=False))
    neg_df = pd.DataFrame(neg_set, columns=["u","v"])
    neg_df["cit_sum_224"] = 0

    # 저장 (impact4cast 포맷: [v1, v2, solution])
    out_dir = args.outdir
    pos_out = os.path.join(out_dir, "unconnected_2021_pair_solution_connected_2024_clean.parquet")
    neg_out = os.path.join(out_dir, "unconnected_2021_pair_solution_unconnected_2024.parquet")
    pos_df.rename(columns={"u":"v1","v":"v2","cit_sum_224":"solution"}).to_parquet(pos_out, index=False)
    neg_df.rename(columns={"u":"v1","v":"v2","cit_sum_224":"solution"}).to_parquet(neg_out, index=False)

    print(f"[pairs] positives: {len(pos_df)} → {pos_out}")
    print(f"[pairs] negatives: {len(neg_df)} → {neg_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_parquet", type=str, required=True, help="data_concept_graph/full_dynamic_graph.parquet")
    ap.add_argument("--edges_csv", type=str, required=True, help="outputs_openalex_kdd_all/edges_by_year.csv")
    ap.add_argument("--index", type=str, required=True, help="data_concept_graph/concept_index.parquet")
    ap.add_argument("--outdir", type=str, required=True, help="output dir, e.g., data_pair_solution")
    ap.add_argument("--neg_sample", type=int, default=0, help="sample size for negatives (0=all)")
    args = ap.parse_args()
    main(args)
