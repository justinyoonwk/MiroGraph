import os, argparse, random
import pandas as pd
from datetime import date

def main(args):
    random.seed(args.seed)

    idx = pd.read_parquet(os.path.join(args.index_dir, "concept_index.parquet"))
    concept2id = dict(zip(idx["concept"].astype(str), idx["vid"].astype(int)))
    N = len(concept2id)

    edges = pd.read_csv(os.path.join(args.graph_root,"edges_by_year.csv"), header=None)
    edges = edges.rename(columns={0:"a",1:"b",2:"year"})
    if edges.shape[1] >= 4:
        edges = edges.rename(columns={3:"cits"})
    else:
        edges["cits"] = 0.0

    # 무향 정렬 + vid 매핑
    def norm_pair(a,b):
        return tuple(sorted((a,b)))
    edges["a"] = edges["a"].astype(str)
    edges["b"] = edges["b"].astype(str)
    edges[["sa","sb"]] = pd.DataFrame(edges.apply(lambda r: norm_pair(r["a"], r["b"]), axis=1).tolist())
    # 필터: 둘 다 index에 있는 pair만
    edges = edges[edges["sa"].isin(concept2id) & edges["sb"].isin(concept2id)].copy()
    edges["u"] = edges["sa"].map(concept2id)
    edges["v"] = edges["sb"].map(concept2id)

    # pair→최초연도, pair→연도별 인용
    grp = edges.groupby(["u","v"])
    first_year = grp["year"].min().rename("first_year")
    cit_by_year = grp.apply(lambda g: dict(zip(g["year"].astype(int), g["cits"].astype(float)))).rename("year2cits")
    pairs = pd.concat([first_year, cit_by_year], axis=1).reset_index()

    base_year = 2021
    target_year = 2024

    # 2021까지 연결된 집합
    connected_leq_2021 = set(map(tuple, pairs.loc[pairs["first_year"]<=base_year, ["u","v"]].values))
    # 2024까지 연결된 집합
    connected_leq_2024 = set(map(tuple, pairs.loc[pairs["first_year"]<=target_year, ["u","v"]].values))
    # 2022~2024 사이에 새로 연결(= positive 대상)
    newly_2022_2024 = set(map(tuple, pairs.loc[(pairs["first_year"]>=2022)&(pairs["first_year"]<=2024), ["u","v"]].values))

    # Positive: 2021에는 미연결 & 2022~24 사이 첫 연결
    pos_pairs = []
    for u,v in newly_2022_2024:
        if (u,v) not in connected_leq_2021:
            y2c = pairs.loc[(pairs["u"]==u)&(pairs["v"]==v), "year2cits"].iloc[0]
            ir = sum(y2c.get(y,0.0) for y in (2022,2023,2024))
            pos_pairs.append((u,v, float(ir)))
    pos_df = pd.DataFrame(pos_pairs, columns=["v1","v2","target"])
    print(f"[pair] positives: {len(pos_df)}")

    # Negative 샘플링: 2021에도 미연결이고 2024까지도 미연결인 pair에서 랜덤 추출
    active_2019_2021 = set(edges.loc[edges["year"]<=base_year, "u"]).union(
                        set(edges.loc[edges["year"]<=base_year, "v"]))
    active_ids = sorted(active_2019_2021)
    conn_leq_2024 = connected_leq_2024  # 제외해야 함

    need = int(len(pos_df) * args.neg_multiplier) if len(pos_df)>0 else 0
    neg_pairs = set()
    trials = 0
    max_trials = need * 50 + 10000
    while len(neg_pairs) < need and trials < max_trials:
        u = random.choice(active_ids)
        v = random.choice(active_ids)
        if u==v:
            trials += 1
            continue
        a,b = (u,v) if u<v else (v,u)
        if (a,b) in conn_leq_2024:
            trials += 1
            continue
        neg_pairs.add((a,b))
        trials += 1
    neg_df = pd.DataFrame([(u,v,0.0) for (u,v) in sorted(neg_pairs)], columns=["v1","v2","target"])
    print(f"[pair] negatives(sampled): {len(neg_df)}  (trials={trials})")

    os.makedirs(args.outdir, exist_ok=True)
    pos_out = os.path.join(args.outdir, "unconnected_2021_pair_solution_connected_2024_clean.parquet")
    neg_out = os.path.join(args.outdir, "unconnected_2021_pair_solution_unconnected_2024.parquet")
    pos_df.to_parquet(pos_out, index=False)
    neg_df.to_parquet(neg_out, index=False)
    print(f"[save] {pos_out}")
    print(f"[save] {neg_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_root", default="outputs_openalex_kdd_all")
    ap.add_argument("--index_dir", default="data_concept_graph")
    ap.add_argument("--outdir", default="data_pair_solution")
    ap.add_argument("--neg_multiplier", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
