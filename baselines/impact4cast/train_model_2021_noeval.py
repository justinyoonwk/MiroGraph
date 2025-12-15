# -*- coding: utf-8 -*-
import os

# ===== 경로 상수: threshold=2로 생성한 그래프 산출물 =====
EDGES_ROOT = "outputs_openalex_kdd_2019_2024_t2"
EDGES_CSV = os.path.join(EDGES_ROOT, "edges_by_year.csv")
NODES_CSV = os.path.join(EDGES_ROOT, "nodes.csv")

import pickle
import gzip
import copy
import torch
from torch import nn
import torch.nn.functional as F
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import rankdata
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, date
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

from general_utils import *
from preprocess_utils import *
from features_utils import *
from train_model_utils import *

"""
목표: 2019–2021 스냅샷(= 2021, 2020, 2019)을 사용해 2021 시점에서 2022–2024 IR 예측.
설정: year_start=2021, years_delta=3 → 타깃 연도 2024.
차이점: 외부 평가(eval 세트) 없이 학습/내부 검증(Train/Test)까지만 수행.
입력:
  - data_concept_graph/full_dynamic_graph.parquet (로그용)
  - data_for_features/adjacency_matrix_{2019,2020,2021}.gz   <-- edges_by_year.csv로 생성
  - data_for_features/concept_node_citation_data_{2019,2020,2021}.parquet
  - data_pair_solution/unconnected_2021_pair_solution_connected_2024_clean.parquet
  - data_pair_solution/unconnected_2021_pair_solution_unconnected_2024.parquet
출력:
  - 2021_train/t0_c2_* 하위 폴더(로그, 네트워크, 손실곡선, ROC 등 — eval ROC는 없음)
"""

rn_time = random.random() * 30
time.sleep(rn_time)

if __name__ == '__main__':

    split_type = 0
    out_norm = False
    num_class = 2
    day_origin = date(1990, 1, 1)

    vertex_degree_cutoff = 1
    min_edges = 1
    years_delta = 3
    year_start = 2021

    graph_parameter = [year_start, years_delta, vertex_degree_cutoff, min_edges]

    # ===== 결과 폴더 생성 =====
    save_folders, log_folder = make_folders(year_start, split_type, num_class, "train")
    log_run = os.path.join(log_folder, f"train_model_{year_start + years_delta}_run_1")
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\n\nstart: {datetime.now()}\n")

    # ===== 콘셉트 인덱스 고정 (threshold=2 vocab) =====
    set_concept_index(NODES_CSV)
    with open(log_run + "_logs.txt", "a") as f:
        f.write(f"\n[set_concept_index] from {NODES_CSV}")

    # ===== 전체 동적 그래프 (로그용) =====
    start_time = time.time()
    data_folder = "data_concept_graph"
    graph_file = os.path.join(data_folder, "full_dynamic_graph.parquet")
    if os.path.exists(graph_file):
        full_dynamic_graph = pd.read_parquet(graph_file)
        with open(log_run + "_logs.txt", "a") as myfile:
            myfile.write(f"\n{datetime.now()}: Done, read full_dynamic_graph: {len(full_dynamic_graph)}; elapsed_time: {time.time() - start_time}")
    else:
        full_dynamic_graph = pd.DataFrame()
        with open(log_run + "_logs.txt", "a") as myfile:
            myfile.write(f"\n{datetime.now()}: [warn] full_dynamic_graph.parquet not found (logging only)")

    # ===== 피처 준비(인접행렬/이웃/노드피처) — adjacency는 edges_by_year.csv 기반 =====
    feature_folder = "data_for_features"
    start_time = time.time()
    adj_mat_sparse = []
    node_neighbor_list = []
    num_neighbor_list = []
    for yy in [year_start, year_start - 1, year_start - 2]:  # 2021, 2020, 2019
        data_file = os.path.join(feature_folder, f"adjacency_matrix_{yy}.gz")
        # EDGES_CSV를 소스로 사용하여 cutoff<=yy의 스냅샷을 만든다
        adj_mat = get_adjacency_matrix(EDGES_CSV, yy, data_file)
        adj_mat_sparse.append(adj_mat)

        curr_node_neighbor = get_node_neighbor(adj_mat)
        node_neighbor_list.append(curr_node_neighbor)

        curr_num_neighbor = np.array(adj_mat.sum(axis=0)).flatten()
        num_neighbor_list.append(curr_num_neighbor)

    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")

    start_time = time.time()
    vertex_features = get_all_node_feature(adj_mat_sparse, year_start, feature_folder)
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, vertex_features; elapsed_time: {time.time() - start_time}")

    # ===== 인용 기반 노드 피처 (이미 t=2019,2020,2021로 생성됨) =====
    start_time = time.time()
    vc_feature_list = []
    for yy in [year_start, year_start - 1, year_start - 2]:
        data_file = os.path.join(feature_folder, f"concept_node_citation_data_{yy}.parquet")
        vc_df = pd.read_parquet(data_file)
        vc_feature = vc_df.values
        vc_feature_list.append(vc_feature)

    vertex_cfeatures = get_all_node_cfeature(vc_feature_list)
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, vertex_cfeatures; elapsed_time: {time.time() - start_time}")

    pair_cf_parameter = [vc_feature_list, node_neighbor_list, num_neighbor_list, vertex_features, vertex_cfeatures]

    # ===== 라벨(학습/테스트용) =====
    train_data_folder = 'data_pair_solution'
    train_pair_file1 = os.path.join(train_data_folder, f"unconnected_{year_start}_pair_solution_connected_{year_start + years_delta}_clean.parquet")
    train_pair_file2 = os.path.join(train_data_folder, f"unconnected_{year_start}_pair_solution_unconnected_{year_start + years_delta}.parquet")

    time_start = time.time()
    train_pair_data_yes = pd.read_parquet(train_pair_file1)
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, read unconnected_{year_start}_pair_solution_connected_{year_start + years_delta}: {len(train_pair_data_yes)}; elapsed_time: {time.time() - time_start}")

    time_start = time.time()
    train_pair_data_no = pd.read_parquet(train_pair_file2)
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, read unconnected_{year_start}_pair_solution_unconnected_{year_start + years_delta}: {len(train_pair_data_no)}; elapsed_time: {time.time() - time_start}")

    time_start = time.time()
    full_train_data = np.concatenate((train_pair_data_yes.values, train_pair_data_no.values), axis=0)
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, combine all: {len(full_train_data)}; elapsed_time: {time.time() - time_start}")

    # 메모리 정리
    full_dynamic_graph = pd.DataFrame()
    train_pair_data_yes = pd.DataFrame()
    train_pair_data_no = pd.DataFrame()

    # ===== IR 스윕 (고정 목록) =====
    for ir in [5, 10, 20, 30, 40]:
        IR_num = [ir]
        IR_Str = format_IR(IR_num, split_type)

        logs_file_name = os.path.join(log_folder, f"train_model_{year_start + years_delta}_" + IR_Str)
        if not os.path.exists(logs_file_name + "_logs.txt"):
            open(logs_file_name + "_logs.txt", 'a').close()

            batch_size = 1000
            lr_enc = 3e-5
            rnd_seed = 42
            hyper_parameter = [batch_size, lr_enc, rnd_seed]
            graph_parameter = [year_start, years_delta, vertex_degree_cutoff, min_edges]
            user_parameter = [num_class, IR_num, split_type, out_norm]

            # pos/neg 존재 확인 — 없으면 스킵
            neg = full_train_data[full_train_data[:, 2] < IR_num[0]]
            pos = full_train_data[full_train_data[:, 2] >= IR_num[0]]
            if len(neg) == 0 or len(pos) == 0:
                with open(logs_file_name + "_logs.txt", "a") as f:
                    f.write(f"\n[skip] IR={IR_num[0]} -> neg={len(neg)}, pos={len(pos)}")
                print(f"[skip] IR={IR_num[0]} -> neg={len(neg)}, pos={len(pos)}")
                continue

            impact_classfication(
                full_train_data,
                [],     # noeval: eval 없음
                [],     # noeval: eval 없음
                pair_cf_parameter,
                hyper_parameter,
                graph_parameter,
                user_parameter,
                save_folders,
                logs_file_name
            )

            # 반복 간 랜덤 대기(동시 실행 시 시드·I/O 충돌 방지용)
            rn_time = random.random() * 30
            time.sleep(rn_time)
        else:
            # 이미 해당 IR 결과가 있으면 스킵
            continue

    # 모든 IR 완료 후 종료 로그
    with open(log_run + "_logs.txt", "a") as myfile:
        myfile.write(f"\nfinish: {datetime.now()}\n\n")
