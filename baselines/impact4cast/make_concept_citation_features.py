import os, argparse, collections
import pandas as pd
from collections import defaultdict

def main(args):
    idx = pd.read_parquet(os.path.join(args.index_dir, "concept_index.parquet"))
    concept2id = dict(zip(idx["concept"].astype(str), idx["vid"].astype(int)))
    N = len(concept2id)

    edges = pd.read_csv(os.path.join(args.graph_root,"edges_by_year.csv"), header=None)
    edges = edges.rename(columns={0:"a",1:"b",2:"year"})
    # 4번째 열: 해당 연도의 (pair 기준) 인용수 총합으로 가정
    cit_col = 3 if edges.shape[1] >= 4 else None
    pap_col = 4 if edges.shape[1] >= 5 else None

    # 개념 단위로 연도별 집계
    cit_by_concept_year = defaultdict(lambda: defaultdict(float))
    num_by_concept_year = defaultdict(lambda: defaultdict(float))

    for _,r in edges.iterrows():
        a,b,y = str(r["a"]), str(r["b"]), int(r["year"])
        cits = float(r[cit_col]) if cit_col is not None else 0.0
        nump = float(r[pap_col]) if pap_col is not None else 1.0  # 없으면 1로 근사
        for c in (a,b):
            if c in concept2id:
                cit_by_concept_year[c][y] += cits
                num_by_concept_year[c][y] += nump

    os.makedirs(args.outdir, exist_ok=True)
    years = [2019, 2020, 2021]  # 학습 스냅샷용
    all_years = sorted(set(edges["year"].astype(int).tolist()))

    for Y in years:
        rows = []
        for concept, vid in concept2id.items():
            # 단일 연도
            cY = cit_by_concept_year[concept].get(Y, 0.0)
            nY = num_by_concept_year[concept].get(Y, 0.0)
            cY_m = (cY / nY) if nY>0 else 0.0
            # 누적
            years_leq = [y for y in all_years if y<=Y]
            ct_Y = sum(cit_by_concept_year[concept].get(y,0.0) for y in years_leq)
            totN_Y = sum(num_by_concept_year[concept].get(y,0.0) for y in years_leq)
            ct_Y_m = (ct_Y / totN_Y) if totN_Y>0 else 0.0
            # 최근 3년
            window = [y for y in range(Y-2, Y+1)]
            ct_delta = sum(cit_by_concept_year[concept].get(y,0.0) for y in window)
            totN_delta = sum(num_by_concept_year[concept].get(y,0.0) for y in window)
            ct_delta_m = (ct_delta / totN_delta) if totN_delta>0 else 0.0
            rows.append([vid, cY, ct_Y, ct_delta, nY, cY_m, ct_Y_m, ct_delta_m])

        df = pd.DataFrame(rows, columns=["v","cY","ct_Y","ct_delta","num","cY_m","ct_Y_m","ct_delta_m"])
        outfp = os.path.join(args.outdir, f"concept_node_citation_data_{Y}.parquet")
        df.to_parquet(outfp, index=False)
        print(f"[cfeature] saved: {outfp} (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_root", default="outputs_openalex_kdd_all")
    ap.add_argument("--index_dir", default="data_concept_graph")
    ap.add_argument("--outdir", default="data_for_features")
    args = ap.parse_args()
    main(args)
