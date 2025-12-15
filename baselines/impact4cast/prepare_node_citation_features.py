# baselines/impact4cast/prepare_node_citation_features.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

def read_nodes(index_path: str):
    idx = pd.read_parquet(index_path)
    idx = idx.rename(columns={idx.columns[0]: "concept"})
    idx["vid"] = idx["vid"].astype(int)
    return idx

def _read_csv_header_first(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    df.columns = cols
    return df

def _read_csv_no_header(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        names=["src", "dst", "year", "citations", "num_papers"],
        header=None,
    )

def read_edges_by_year(edges_path: str) -> pd.DataFrame:
    df = _read_csv_header_first(edges_path)
    if not {"src", "dst", "year"}.issubset(set(df.columns)):
        df = _read_csv_no_header(edges_path)
    keep = [c for c in ["src", "dst", "year", "citations", "num_papers"] if c in df.columns]
    df = df[keep].copy()
    df["src"] = df["src"].astype(str).str.strip()
    df["dst"] = df["dst"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["citations"] = pd.to_numeric(df.get("citations", 0), errors="coerce").fillna(0).astype(int)
    df["num_papers"] = pd.to_numeric(df.get("num_papers", 0), errors="coerce").fillna(0).astype(int)
    # (src,dst) 정렬
    src_sorted = np.minimum(df["src"].values, df["dst"].values)
    dst_sorted = np.maximum(df["src"].values, df["dst"].values)
    df = df.assign(src=src_sorted, dst=dst_sorted)
    return df

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    nodes = read_nodes(args.index)
    vid_of = dict(zip(nodes["concept"], nodes["vid"]))

    e = read_edges_by_year(os.path.join(args.graph_root, "edges_by_year.csv"))

    # 개념별 연도 인용/논문수 집계
    cit_year = defaultdict(lambda: defaultdict(int))      # cit_year[concept][year] = citations
    pap_year = defaultdict(lambda: defaultdict(int))      # pap_year[concept][year] = num_papers

    for row in e.itertuples(index=False):
        s, d, y, c, n = row.src, row.dst, row.year, getattr(row, "citations", 0), getattr(row, "num_papers", 0)
        if s in vid_of:
            cit_year[s][y] += int(c)
            pap_year[s][y] += int(n)
        if d in vid_of:
            cit_year[d][y] += int(c)
            pap_year[d][y] += int(n)

    years = [int(y) for y in args.years]
    delta = int(args.delta)

    # 각 year 마다 concept_node_citation_data_{year}.parquet 작성
    for Y in years:
        rows = []
        for concept, vid in vid_of.items():
            # 연도별
            cY = cit_year[concept].get(Y, 0)
            pY = pap_year[concept].get(Y, 0)

            # 누적
            ctY = sum(v for y, v in cit_year[concept].items() if y <= Y)
            ptY = sum(v for y, v in pap_year[concept].items() if y <= Y)

            # 3년(=delta) 합 (경계 처리)
            y_from = Y - (delta - 1)
            window_years = [yy for yy in range(y_from, Y + 1)]
            ctDelta = sum(cit_year[concept].get(yy, 0) for yy in window_years)
            ptDelta = sum(pap_year[concept].get(yy, 0) for yy in window_years)

            # 평균(논문 수 0 보호)
            cY_m = cY / max(pY, 1)
            ctY_m = ctY / max(ptY, 1)
            ctDelta_m = ctDelta / max(ptDelta, 1)

            rows.append([vid, cY, ctY, ctDelta, ptY, cY_m, ctY_m, ctDelta_m])

        df = pd.DataFrame(rows, columns=["v1", f"c{Y}", f"ct_{Y}", f"ct_delta_{delta}", f"num_up_to_{Y}",
                                         f"c{Y}_m", f"ct_{Y}_m", f"ct_delta_{delta}_m"])
        # features_utils 기대 형태로 변환 (고정 컬럼 이름)
        out = pd.DataFrame({
            "v1": df["v1"].astype(int),
            "cY": df[f"c{Y}"].astype(np.int64),
            "ct_Y": df[f"ct_{Y}"].astype(np.int64),
            "ct_delta": df[f"ct_delta_{delta}"].astype(np.int64),
            "num": df[f"num_up_to_{Y}"].astype(np.int64),
            "cY_m": df[f"c{Y}_m"].astype(float),
            "ct_Y_m": df[f"ct_{Y}_m"].astype(float),
            "ct_delta_m": df[f"ct_delta_{delta}_m"].astype(float),
        })
        out_path = os.path.join(args.outdir, f"concept_node_citation_data_{Y}.parquet")
        out.to_parquet(out_path, index=False)
        print(f"[node-feat] saved: {out_path} (rows={len(out)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_root", type=str, required=True, help="outputs_openalex_kdd_all dir")
    ap.add_argument("--index", type=str, required=True, help="data_concept_graph/concept_index.parquet")
    ap.add_argument("--outdir", type=str, required=True, help="output dir, e.g., data_for_features")
    ap.add_argument("--years", nargs="+", required=True, help="years to prepare, e.g., 2019 2020 2021")
    ap.add_argument("--delta", type=int, default=3, help="window size for ct_delta (default=3)")
    args = ap.parse_args()
    main(args)
