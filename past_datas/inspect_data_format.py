import pandas as pd
import gzip
import pickle
import numpy as np
import scipy.sparse
import os

# 경로 설정 (사용자 환경 기준)
base_dir = "/home/donggyu/project1"
concept_path = os.path.join(base_dir, "data_concept_graph/concept_index.parquet")
adj_path = os.path.join(base_dir, "data_for_features/adjacency_matrix_2021.gz")

print("=== 1. Concept Index Inspection ===")
try:
    if os.path.exists(concept_path):
        df = pd.read_parquet(concept_path)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("First 3 rows:")
        print(df.head(3))
    else:
        print(f"File not found: {concept_path}")
except Exception as e:
    print(f"Error reading parquet: {e}")

print("\n=== 2. Adjacency Matrix Inspection ===")
try:
    if os.path.exists(adj_path):
        with gzip.open(adj_path, 'rb') as f:
            print(f"Attempting to load {adj_path}...")
            
            # 1. Pickle 로딩 시도 (가장 일반적)
            try:
                data = pickle.load(f)
                print(f"Successfully loaded with pickle.")
                print(f"Data Type: {type(data)}")
                
                if hasattr(data, 'shape'):
                    print(f"Shape: {data.shape}")
                
                # 만약 Scipy Sparse Matrix라면 0이 아닌 요소 개수 확인
                if scipy.sparse.issparse(data):
                    print(f"Non-zero elements: {data.nnz}")
                    
            except Exception as e_pickle:
                print(f"Pickle load failed: {e_pickle}")
                # 2. Numpy 로딩 시도 (혹시 .npy가 압축된 경우)
                f.seek(0)
                try:
                    data = np.load(f)
                    print(f"Successfully loaded with numpy.")
                    print(f"Data Type: {type(data)}")
                    print(f"Shape: {data.shape}")
                except Exception as e_numpy:
                    print(f"Numpy load failed: {e_numpy}")
                    print("Unknown format inside gzip.")
    else:
        print(f"File not found: {adj_path}")
except Exception as e:
    print(f"Error inspecting adjacency matrix: {e}")