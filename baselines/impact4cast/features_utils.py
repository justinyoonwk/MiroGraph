# -*- coding: utf-8 -*-
import os
import gzip
import pickle
import time
import copy
import random
import numpy as np
import pandas as pd
import networkx as nx

from datetime import datetime
from collections import defaultdict, Counter
from scipy import sparse
from scipy.stats import rankdata

# ============================================================
# Global concept index (concept <-> integer id)
# ============================================================

_CONCEPT_TO_ID = None
_ID_TO_CONCEPT = None

def set_concept_index(nodes_csv: str):
    """
    nodes.csv (columns: concept, doc_freq) 로부터 concept -> idx 매핑을 구성.
    전역 캐시에 저장되어 이후 모든 함수에서 사용됨.
    """
    global _CONCEPT_TO_ID, _ID_TO_CONCEPT
    df = pd.read_csv(nodes_csv)
    # 안정적 인덱스를 위해 사전식 정렬
    vocab = sorted(df["concept"].astype(str).str.strip().tolist())
    _CONCEPT_TO_ID = {c: i for i, c in enumerate(vocab)}
    _ID_TO_CONCEPT = {i: c for c, i in _CONCEPT_TO_ID.items()}
    print(f"[set_concept_index] concepts: {len(_CONCEPT_TO_ID)} from {os.path.basename(nodes_csv)}")

def _ensure_index_from_edges_csv(edges_by_year_csv: str):
    """
    만약 set_concept_index가 호출되지 않았다면, edges_by_year.csv와 같은 폴더의 nodes.csv를 찾아 로드.
    """
    global _CONCEPT_TO_ID, _ID_TO_CONCEPT
    if _CONCEPT_TO_ID is not None:
        return
    folder = os.path.dirname(os.path.abspath(edges_by_year_csv))
    cand = os.path.join(folder, "nodes.csv")
    if not os.path.isfile(cand):
        raise ValueError("Concept index not set. Call set_concept_index(nodes_csv) first.")
    set_concept_index(cand)

def _N():
    if _CONCEPT_TO_ID is None:
        raise ValueError("Concept index not set. Call set_concept_index(nodes_csv) first.")
    return len(_CONCEPT_TO_ID)

def _cid(c: str) -> int:
    return _CONCEPT_TO_ID.get(str(c).strip().lower(), None)

# ============================================================
# IO helpers (cache)
# ============================================================

def _save_pickle_gz(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_pickle_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

# ============================================================
# Graph building from edges_by_year.csv
# ============================================================

def _load_edges_by_year(src):
    """
    src: str(csv path) or pandas.DataFrame with columns:
         ['source','target','year','citations'] (case-insensitive accepted)
    Returns DataFrame with standardized columns: source, target, year, citations
    """
    if isinstance(src, str):
        df = pd.read_csv(src)
        # auto columns
    elif isinstance(src, pd.DataFrame):
        df = src.copy()
    else:
        raise ValueError("full_graph must be a path to edges_by_year.csv or a DataFrame.")

    cols = {c.lower(): c for c in df.columns}
    # standardize
    s_col = cols.get("source") or cols.get("u")
    t_col = cols.get("target") or cols.get("v")
    y_col = cols.get("year")
    c_col = cols.get("citations") or cols.get("weight") or cols.get("w")

    if not all([s_col, t_col, y_col, c_col]):
        raise ValueError("edges_by_year must have columns [source,target,year,citations].")

    out = df[[s_col, t_col, y_col, c_col]].rename(
        columns={s_col: "source", t_col: "target", y_col: "year", c_col: "citations"}
    )
    # normalize concepts to lowercase/strip
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    out["target"] = out["target"].astype(str).str.strip().str.lower()
    return out

def get_adjacency_matrix(full_graph, year: int, data_file: str):
    """
    (수정버전) edges_by_year 기반으로 '연도 <= year'의 edge만 모아 이분 인접행렬 생성.
    - full_graph: edges_by_year.csv 경로 or DataFrame(source,target,year,citations)
    - year: cutoff 연도
    - data_file: 캐시 경로(.gz)
    """
    start_time = time.time()
    if os.path.exists(data_file):
        adj = _load_pickle_gz(data_file)
        print(f"{datetime.now()}: Done {year}, read adjacency_matrix; {time.time() - start_time:.2f}s")
        return adj

    # ensure index
    if isinstance(full_graph, str):
        _ensure_index_from_edges_csv(full_graph)
    else:
        if _CONCEPT_TO_ID is None:
            raise ValueError("Call set_concept_index(nodes_csv) before get_adjacency_matrix when passing a DataFrame.")

    N = _N()
    df = _load_edges_by_year(full_graph)
    df = df[df["year"] <= int(year)]
    # 존재 여부만 고려 (citations>0 인 edge만 유의미)
    df = df[df["citations"] > 0]

    # map to indices
    # (없는 개념이 있으면 스냅샷 vocab 바깥 — 학습 시점 vocab를 고정하기 위해 skip)
    sid = df["source"].map(_CONCEPT_TO_ID)
    tid = df["target"].map(_CONCEPT_TO_ID)
    mask = sid.notna() & tid.notna()
    rows = sid[mask].astype(int).to_numpy()
    cols = tid[mask].astype(int).to_numpy()

    data = np.ones(len(rows), dtype=np.uint8)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = A + A.transpose()
    A = (A > 0).astype(np.int8)

    _save_pickle_gz(A, data_file)
    print(f"Done year: {year}; num nodes: {A.shape[0]}; num edges: {A.sum()//2}; {time.time() - start_time:.2f}s")
    return A

# ============================================================
# Graph features
# ============================================================

def get_pagerank_score(adjacency_matrix: sparse.csr_matrix, data_file: str):
    print(f"{datetime.now()}: getting the pagerank score")
    if os.path.exists(data_file):
        start_time = time.time()
        pr = _load_pickle_gz(data_file)
        print(f"{datetime.now()}: Done, loading pagerank_score; {time.time() - start_time:.2f}s")
        return pr

    start_time = time.time()
    G = nx.from_scipy_sparse_array(adjacency_matrix)
    pr_dict = nx.algorithms.link_analysis.pagerank(G)
    N = adjacency_matrix.shape[0]
    pr = np.zeros(shape=(N,), dtype=np.float32)
    for i in range(N):
        pr[i] = pr_dict.get(i, 0.0)
    _save_pickle_gz(pr, data_file)
    print(f"{datetime.now()}: done pagerank_score; {time.time() - start_time:.2f}s")
    return pr

def get_node_neighbor(adjacency_matrix: sparse.csr_matrix):
    N = adjacency_matrix.shape[0]
    return [adjacency_matrix.getrow(i).indices for i in range(N)]

def get_num_neighbor(adjacency_matrix: sparse.csr_matrix):
    return np.asarray(adjacency_matrix.sum(axis=0)).ravel()

# ============================================================
# Pair features (structure-only, same 수식 유지)
# ============================================================

def get_num_shared_neighbor(node_neighbor, vertex_pairs):
    out = np.zeros(len(vertex_pairs), dtype=int)
    for k, (v1, v2) in enumerate(vertex_pairs):
        v1 = int(v1); v2 = int(v2)
        out[k] = np.intersect1d(node_neighbor[v1], node_neighbor[v2]).size
    return out

def get_pair_feature(node_neighbor, num_neighbor, vertex_list):
    num_pairs = len(vertex_list)
    num_features = 7
    feats = np.zeros((num_pairs, num_features), dtype=np.float32)

    for idx, (v1, v2) in enumerate(vertex_list):
        v1 = int(v1); v2 = int(v2)
        shared = np.intersect1d(node_neighbor[v1], node_neighbor[v2]).size
        n1 = num_neighbor[v1]; n2 = num_neighbor[v2]

        if n1 == 0 or n2 == 0:
            gem = 0.0; cos = 0.0; sps = 0.0; pre = 0.0
        else:
            gem = (shared**2) / (n1 * n2)
            cos = gem**0.5
            sps = shared / np.min([n1, n2])
            pre = n1 * n2

        sod = 0.0 if (n1 == 0 and n2 == 0) else 2 * shared / (n1 + n2)
        jac = shared / (n1 + n2 - shared) if (n1 + n2 - shared) > 0 else 0.0

        feats[idx] = [shared, gem, cos, sps, pre, sod, jac]
    return feats

# ============================================================
# Node citation features (derived from edges_by_year)
# ============================================================

def _node_cfeatures_from_edges(edges_by_year, year: int):
    """
    edges_by_year: DataFrame(source,target,year,citations)
    year: 기준 연도 y
    반환: shape (N, 8) array
      col0 dummy(id 자리), col1 c_y, col2 ct_y, col3 ct_delta(최근3년), col4 num(개수),
      col5 c_y_mean(=c_y/num), col6 ct_y_mean(=ct_y/num), col7 ct_delta_mean(=ct_delta/num)
    정의:
      - c_y(node)   = 해당 node에 인접한 edge들의 '해당 연도 citations' 합
      - ct_y(node)  = 1..y 누적(실제로는 edges가 가진 최소 연도..y)
      - ct_delta    = (y-2..y) 3개년 합(연도가 부족하면 있는 범위만)
      - num(node)   = 해당 node에 인접한 edge들의 '해당 연도 contributing edge 수'
    주의: edge 기반 근사이므로 paper 기반과 달라질 수 있음(베이스라인 목적).
    """
    N = _N()
    df = edges_by_year.copy()
    df = df[df["citations"] > 0]
    # 맵핑
    sid = df["source"].map(_CONCEPT_TO_ID)
    tid = df["target"].map(_CONCEPT_TO_ID)
    mask = sid.notna() & tid.notna()
    df = df[mask]
    df["sid"] = sid[mask].astype(int)
    df["tid"] = tid[mask].astype(int)

    # 연도별 행렬적 합산을 빠르게 하려면 pivot보다 groupby
    # c_y
    df_y = df[df["year"] == year]
    c_y_incident = np.zeros(N, dtype=np.float32)
    for col in ("sid", "tid"):
        s = df_y.groupby(col)["citations"].sum()
        c_y_incident[s.index.to_numpy()] += s.to_numpy(dtype=np.float32)

    # ct_y: <= y 누적
    df_le = df[df["year"] <= year]
    ct_y_incident = np.zeros(N, dtype=np.float32)
    for col in ("sid", "tid"):
        s = df_le.groupby(col)["citations"].sum()
        ct_y_incident[s.index.to_numpy()] += s.to_numpy(dtype=np.float32)

    # 최근 3년 (y-2..y)
    yrs = [yy for yy in (year - 2, year - 1, year) if yy >= df["year"].min()]
    df_3 = df[df["year"].isin(yrs)]
    ct_delta_incident = np.zeros(N, dtype=np.float32)
    for col in ("sid", "tid"):
        s = df_3.groupby(col)["citations"].sum()
        ct_delta_incident[s.index.to_numpy()] += s.to_numpy(dtype=np.float32)

    # num(node): 해당 연도 incident edge 수(=해당 연도에 연결된 co-occur pair 수)
    num_incident = np.zeros(N, dtype=np.float32)
    if not df_y.empty:
        for col in ("sid", "tid"):
            s = df_y.groupby(col).size()
            num_incident[s.index.to_numpy()] += s.to_numpy(dtype=np.float32)

    # 평균 안전 나눗셈
    safe_num = np.where(num_incident == 0, 1.0, num_incident)
    c_y_mean = c_y_incident / safe_num
    ct_y_mean = ct_y_incident / safe_num
    ct_delta_mean = ct_delta_incident / safe_num

    # cols: 0(dummy),1(c_y),2(ct_y),3(ct_delta),4(num),5(c_y_m),6(ct_y_m),7(ct_delta_m)
    out = np.zeros((N, 8), dtype=np.float32)
    out[:, 1] = c_y_incident
    out[:, 2] = ct_y_incident
    out[:, 3] = ct_delta_incident
    out[:, 4] = num_incident
    out[:, 5] = c_y_mean
    out[:, 6] = ct_y_mean
    out[:, 7] = ct_delta_mean
    return out

def get_all_node_cfeature(node_cfeature_list):
    """
    (원 코드 유지) 세 시점 y,y-1,y-2에 대한 node_cfeature를 받아
    차분/랭크 등을 포함한 확장 행렬을 반환.
    입력 배열 각 shape: (N, 8)  # 위 _node_cfeatures_from_edges 정의 참조
    """
    node_cfeature0, node_cfeature1, node_cfeature2 = node_cfeature_list
    all_features = []

    # 1(c_y) ~ 7(ct_delta_mean)까지 각 열에 대해 y,y-1,y-2 스택
    indices_to_process = list(range(1, node_cfeature0.shape[1]))
    for idx in indices_to_process:
        f0 = node_cfeature0[:, idx]
        f1 = node_cfeature1[:, idx]
        f2 = node_cfeature2[:, idx]
        all_features.extend([f0, f1, f2])

    # diff/Rank: index 2(ct_y), 4(num)
    for idx in [2, 4]:
        f0 = node_cfeature0[:, idx]
        f1 = node_cfeature1[:, idx]
        f2 = node_cfeature2[:, idx]
        d1 = f0 - f1
        d2 = f0 - f2
        all_features.extend([d1, d2, rankdata(d1), rankdata(d2)])

    node_cfeatures = np.vstack(all_features)
    return node_cfeatures

# ============================================================
# Node features (degree, PageRank, etc.)
# ============================================================

def get_all_node_feature(adjacency_matrix_list, year, data_folder):
    """
    adjacency_matrix_list: [A_y, A_{y-1}, A_{y-2}]
    반환: (feature_dim, N) 형태 (원 코드 호환)
    """
    A0, A1, A2 = adjacency_matrix_list
    n0 = get_num_neighbor(A0)
    n1 = get_num_neighbor(A1)
    n2 = get_num_neighbor(A2)

    d1 = n0 - n1
    d2 = n0 - n2
    r1 = rankdata(d1)
    r2 = rankdata(d2)

    pr0 = get_pagerank_score(A0, os.path.join(data_folder, f"pagerank_score_{year}.gz"))
    pr1 = get_pagerank_score(A1, os.path.join(data_folder, f"pagerank_score_{year-1}.gz"))
    pr2 = get_pagerank_score(A2, os.path.join(data_folder, f"pagerank_score_{year-2}.gz"))

    all_feats = [n0, n1, n2, d1, d2, r1, r2, pr0, pr1, pr2]
    node_features = np.vstack(all_feats)
    return node_features

# ============================================================
# Pair citation features (원 수식 유지)
# ============================================================

def get_pair_cfeature(data_cparameters, vertex_list):
    # data_cparameters: [c_y, ct_y, ct_delta, num, c_y_m, ct_y_m, ct_delta_m]
    curr_num_c, num_total_c, num_delta_c, num_cdegree, curr_num_cm, num_total_cm, num_delta_cm = data_cparameters
    num_pair_feature = 14
    out = np.zeros((len(vertex_list), num_pair_feature), dtype=np.float32)

    for idx, (v1, v2) in enumerate(vertex_list):
        v1 = int(v1); v2 = int(v2)
        feats = []
        if (num_cdegree[v1] + num_cdegree[v2]) > 0:
            feats.extend([
                (curr_num_c[v1] + curr_num_c[v2]) / (num_cdegree[v1] + num_cdegree[v2]),
                (curr_num_c[v1] * curr_num_c[v2]) / (num_cdegree[v1] + num_cdegree[v2]),
            ])
        else:
            feats.extend([0.0, 0.0])

        feats.extend([
            curr_num_cm[v1] + curr_num_cm[v2],
            num_total_cm[v1] + num_total_cm[v2],
            num_delta_c[v1] + num_delta_c[v2],
            num_delta_cm[v1] + num_delta_cm[v2],
            min(curr_num_c[v1], num_total_c[v1]),  # (원 코드 주석과 값 불일치 있어도 유지)
            max(curr_num_c[v1], num_total_c[v1]),
            min(num_total_c[v1], num_total_c[v2]),
            max(num_total_c[v1], num_total_c[v2]),
            min(num_delta_c[v1], num_delta_c[v2]),
            max(num_delta_c[v1], num_delta_c[v2]),
            min(num_cdegree[v1], num_cdegree[v2]),
            max(num_cdegree[v1], num_cdegree[v2]),
        ])
        out[idx] = feats
    return out

# ============================================================
# Normalization helpers
# ============================================================

def rescaling_col(features: np.ndarray):
    maxv = features.max(axis=0, keepdims=True)
    maxv = np.where(maxv == 0, 1, maxv)
    return features / maxv

def rescaling_row(features: np.ndarray):
    maxv = features.max(axis=1, keepdims=True)
    maxv = np.where(maxv == 0, 1, maxv)
    return features / maxv

def return_col_max(features: np.ndarray):
    maxv = features.max(axis=0, keepdims=True)
    maxv = np.where(maxv == 0, 1, maxv)
    return maxv

# ============================================================
# Feature orchestration (원 함수 시그니처 유지)
# ============================================================

def get_all_pair_features(node_cfeature_list, node_neighbor_list, num_neighbor_list, vertex_list, logs_file_name):
    node_c0, node_c1, node_c2 = node_cfeature_list
    node_neighbor0, node_neighbor1, node_neighbor2 = node_neighbor_list
    num_neighbor0, num_neighbor1, num_neighbor2 = num_neighbor_list

    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\n{datetime.now()}: start extract_features")

    t0 = time.time()
    pair_feature0 = get_pair_feature(node_neighbor0, num_neighbor0, vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_feature0, {len(pair_feature0)}; time: {time.time()-t0:.2f}")

    t1 = time.time()
    pair_feature1 = get_pair_feature(node_neighbor1, num_neighbor1, vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_feature1, {len(pair_feature1)}; time: {time.time()-t1:.2f}")

    t2 = time.time()
    pair_feature2 = get_pair_feature(node_neighbor2, num_neighbor2, vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_feature2, {len(pair_feature2)}; time: {time.time()-t2:.2f}")

    # node_cparameters: 1..7 열 추출
    def _params(arr):
        return [arr[:, i] for i in range(1, arr.shape[1])]

    t3 = time.time()
    pair_cfeature0 = get_pair_cfeature(_params(node_c0), vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_cfeature0, {len(pair_cfeature0)}; time: {time.time()-t3:.2f}")

    t4 = time.time()
    pair_cfeature1 = get_pair_cfeature(_params(node_c1), vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_cfeature1, {len(pair_cfeature1)}; time: {time.time()-t4:.2f}")

    t5 = time.time()
    pair_cfeature2 = get_pair_cfeature(_params(node_c2), vertex_list)
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish pair_cfeature2, {len(pair_cfeature2)}; time: {time.time()-t5:.2f}")

    all_pair_feature = [pair_feature0, pair_feature1, pair_feature2]
    all_pair_cfeature = [pair_cfeature0, pair_cfeature1, pair_cfeature2]
    return all_pair_feature, all_pair_cfeature

def get_all_feature(node_pair_features, vertex_list, logs_file_name):
    node_feature, node_cfeature, pair_feature, pair_cfeature = node_pair_features
    pf0, pf1, pf2 = pair_feature
    pcf0, pcf1, pcf2 = pair_cfeature

    t0 = time.time()
    norm_node_feature = rescaling_row(node_feature)
    norm_node_cfeature = rescaling_row(node_cfeature)
    norm_pf0 = rescaling_col(pf0)
    norm_pf1 = rescaling_col(pf1)
    norm_pf2 = rescaling_col(pf2)
    norm_pcf0 = rescaling_col(pcf0)
    norm_pcf1 = rescaling_col(pcf1)
    norm_pcf2 = rescaling_col(pcf2)
    print(f"Finish rescaling; time: {time.time()-t0:.2f}")
    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write(f"\nFinish rescaling; time: {time.time()-t0:.2f}")

    # 원 코드 포맷 유지
    num_features = (
        2 * len(norm_node_feature) +
        2 * len(norm_node_cfeature) +
        3 * pf0.shape[1] +
        3 * pcf0.shape[1]
    )
    store = np.zeros((len(vertex_list), num_features), dtype=np.float32)

    for i, (v1, v2) in enumerate(vertex_list):
        v1 = int(v1); v2 = int(v2)
        vals = []
        for k in range(len(norm_node_feature)):
            vals.append(norm_node_feature[k][v1])
            vals.append(norm_node_feature[k][v2])
        for k in range(len(norm_node_cfeature)):
            vals.append(norm_node_cfeature[k][v1])
            vals.append(norm_node_cfeature[k][v2])
        for k in range(pf0.shape[1]):
            vals.extend([norm_pf0[i, k], norm_pf1[i, k], norm_pf2[i, k]])
        for k in range(pcf0.shape[1]):
            vals.extend([norm_pcf0[i, k], norm_pcf1[i, k], norm_pcf2[i, k]])
        store[i] = vals

        if i % 100_000 == 0:
            with open(logs_file_name + "_logs.txt", "a") as f:
                f.write(f"\nprogress: {i}/{len(vertex_list)}")

    with open(logs_file_name + "_logs.txt", "a") as f:
        f.write("\nFinish store_features")
    return store

# ============================================================
# Convenience: build node_cfeature triple directly from edges_by_year.csv
# ============================================================

def build_node_cfeature_triplet(edges_by_year, year: int):
    """
    edges_by_year: path or DataFrame
    year: 기준 y  (y, y-1, y-2 반환)
    """
    if isinstance(edges_by_year, str):
        _ensure_index_from_edges_csv(edges_by_year)
        df = _load_edges_by_year(edges_by_year)
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(edges_by_year)), ".cache_cfeat")
    else:
        if _CONCEPT_TO_ID is None:
            raise ValueError("Call set_concept_index(nodes_csv) first.")
        df = _load_edges_by_year(edges_by_year)
        cache_dir = ".cache_cfeat"

    os.makedirs(cache_dir, exist_ok=True)

    outs = []
    for y in [year, year - 1, year - 2]:
        cache_path = os.path.join(cache_dir, f"node_cfeature_{y}.pkl.gz")
        if os.path.exists(cache_path):
            arr = _load_pickle_gz(cache_path)
        else:
            arr = _node_cfeatures_from_edges(df, y)
            _save_pickle_gz(arr, cache_path)
        outs.append(arr)
    return outs  # [cfeat_y, cfeat_y-1, cfeat_y-2]
