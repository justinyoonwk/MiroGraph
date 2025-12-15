import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import rankdata
import networkx as nx
import pandas as pd
from collections import defaultdict,Counter
from datetime import datetime, date
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from features_utils import *
 
# =========================
# PATCHED for our dataset
# - binary(0/1) labels are supported end-to-end
# =========================

def _is_binary_labels(arr: np.ndarray) -> bool:
    """Return True if third column contains only {0,1} (binary labels)."""
    v = np.unique(arr)
    return set(v).issubset({0,1})

def prepare_split_datasets(full_train_data, user_parameter, logs_file_name):
    """
    Split the whole dataset (unconnected pairs) into two classes (neg/pos) and
    enforce class balance at 1:1 *before* train/test split.

    full_train_data: np.ndarray with shape (N, 3) -> [:,2] = future citations
    user_parameter: [num_class, IR_num, split_type, out_norm]
    """
    num_class, IR_num, split_type, out_norm = user_parameter
    assert split_type == 0, "This balancing helper assumes binary setting (split_type=0)."

    t0 = time.time()
    neg = full_train_data[full_train_data[:, 2] < IR_num[0]]
    pos = full_train_data[full_train_data[:, 2] >= IR_num[0]]

    n_neg, n_pos = len(neg), len(pos)
    if n_neg == 0 or n_pos == 0:
        # 한쪽이 비어 있으면 학습이 불가능하므로 그대로 반환 (상위 루프에서 이 IR을 skip 권장)
        with open(logs_file_name+"_logs.txt", "a") as f:
            f.write(f"\n[warn] IR={IR_num[0]} -> neg={n_neg}, pos={n_pos} (one side empty) -> skip recommended")
        print(f"[warn] IR={IR_num[0]} -> neg={n_neg}, pos={n_pos} (one side empty) -> skip recommended")
        return [neg, pos]

    # 목표: pos 개수에 맞춰 1:1
    target = n_pos
    if n_neg >= target:
        sel = np.random.choice(n_neg, size=target, replace=False)
        neg_bal = neg[sel]
        pos_bal = pos  # 그대로
    else:
        # neg가 더 적으면 oversample해서 맞춤 (중복 허용)
        need = target - n_neg
        dup = np.random.choice(n_neg, size=need, replace=True)
        neg_bal = np.concatenate([neg, neg[dup]], axis=0)
        pos_bal = pos

    # 셔플
    np.random.shuffle(neg_bal)
    np.random.shuffle(pos_bal)

    with open(logs_file_name+"_logs.txt", "a") as f:
        f.write(f"\n[balance] IR={IR_num[0]}  before: neg={n_neg}, pos={n_pos}  -> after: neg={len(neg_bal)}, pos={len(pos_bal)}; {time.time()-t0:.2f}s")
    print(f"[balance] IR={IR_num[0]}  before: neg={n_neg}, pos={n_pos}  -> after: neg={len(neg_bal)}, pos={len(pos_bal)}; {time.time()-t0:.2f}s")

    return [neg_bal, pos_bal]



def shuffle_split_datasets(data_subsets, train_valid_test_size):
    """
    Split the dataset into training and testing sets

    data_subsets: [negatives, positives], each as np.ndarray
    train_valid_test_size=[train, test, (unused)]
    """
    dataset_train = []
    dataset_test = []
    for subset in data_subsets:
        if len(subset) == 0:
            dataset_train.append(subset)
            dataset_test.append(subset)
            continue
        np.random.shuffle(subset)
        idx_train = int(len(subset) * train_valid_test_size[0])
        train_set = subset[:idx_train]
        test_set = subset[idx_train:]
        dataset_train.append(train_set)
        dataset_test.append(test_set)
    return dataset_train, dataset_test


def get_pair_solution_datasets(data_subsets, hyper_parameter, user_parameter, logs_file_name):
    """
    Prepare the unconnected pairs and their corresponding solutions (third column).
    Output:
      - train_edge_pair: (N, 2) int
      - train_edge_solution: (N,) float or int
    """    
    num_class, IR_num, split_type, out_norm = user_parameter # not used here
    batch_size, lr_enc, rnd_seed=hyper_parameter
    start_time=time.time()   

    for idx, subset in enumerate(data_subsets):
        if len(subset) == 0:
            continue
        min_num_row=min(batch_size, len(subset))
        num_new_samples = len(subset) - min_num_row
        if num_new_samples<0:
            upsamples=np.abs(num_new_samples)
            new_samples_idx = np.random.choice(subset.shape[0], size=upsamples, replace=True)
            data_subsets[idx]= np.concatenate([subset, subset[new_samples_idx]], axis=0)
            
    train_data_for_checking = np.concatenate(data_subsets, axis=0) if all(len(s)>0 for s in data_subsets) else np.concatenate(data_subsets, axis=0)
    train_edge_pair=train_data_for_checking[:,:2]
    train_edge_solution=train_data_for_checking[:,2]
    
    print(f"\nDone, data_for_checking: {len(train_data_for_checking)}; elapsed_time: {time.time() - start_time}") 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, data_for_checking: {len(train_data_for_checking)}; elapsed_time: {time.time() - start_time}")
        
    return train_edge_pair, train_edge_solution


def prepare_train_data(data_pair, data_solution, all_features, user_parameter, logs_file_name):
    """
    Make 141-dim features and split them into neg/pos by label or IR.
    """
    num_class, IR_num, split_type, out_norm = user_parameter
    data_feature=get_all_feature(all_features, data_pair, logs_file_name) # (N, 141)
    
    data_input = []
    start_time=time.time()

    # Binary 라벨(0/1) 자동 인식 → 그대로 split
    if set(np.unique(data_solution)).issubset({0,1}):
        data_input.append(data_feature[data_solution == 0])
        data_input.append(data_feature[data_solution == 1])
    else:
        if split_type==0:
            data_input.append(data_feature[data_solution < IR_num[0]])
            data_input.append(data_feature[data_solution >= IR_num[0]])        
        else:
            for i in range(len(IR_num) - 1):
                data_input.append(data_feature[(data_solution >= IR_num[i][0]) & (data_solution <=IR_num[i][1])])
            data_input.append(data_feature[data_solution >= IR_num[-1]])
        
    print(f"\n   finish split_data_features: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n   finish split_data_features: {time.time()-start_time}")    
    
    return data_feature, data_input
 

######### classify_solution ###############
def classify_solution(data_solution, user_parameter):
    """
    Map solution → class labels.
    - If solution already in {0,1}, return as-is.
    - Else keep legacy IR-based binarization.
    """     
    num_class, IR_num, split_type, out_norm = user_parameter

    # --- NEW: binary passthrough ---
    if set(np.unique(data_solution)).issubset({0,1}):
        return data_solution.astype(int)

    # --- legacy behavior ---
    solution_arr = np.zeros_like(data_solution)
    if split_type==0: ## only binary
        solution_arr[data_solution < IR_num[0]] = 0
        solution_arr[data_solution >= IR_num[0]] = 1
    else: # conditional case
        for i in range(len(IR_num) - 1):
            solution_arr[(data_solution >= IR_num[i][0]) & (data_solution <=IR_num[i][1])] = i
        solution_arr[data_solution >= IR_num[-1]] = len(IR_num) - 1
    return solution_arr
