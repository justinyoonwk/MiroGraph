import pandas as pd
import os

# 경로 설정 (2021년 데이터 기준)
base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, "data_for_features", "concept_node_citation_data_2021.parquet")

print(f"=== Inspecting: {parquet_path} ===")

if os.path.exists(parquet_path):
    try:
        df = pd.read_parquet(parquet_path)
        
        print(f"\n1. Shape: {df.shape}")
        
        print("\n2. Columns & Data Types:")
        print(df.dtypes)
        
        print("\n3. First 5 Rows (Sample):")
        # 컬럼이 잘리지 않게 출력 설정
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.head(5))
        
        # 특정 컬럼(예: 시계열이 리스트나 딕셔너리인지 확인) 샘플링
        # 컬럼 이름에 'history', 'year', 'series' 등이 들어가는지 확인
        print("\n4. Detailed Check on Potential Series Columns:")
        potential_series_cols = [c for c in df.columns if any(x in c.lower() for x in ['year', 'hist', 'count', 'cit'])]
        for col in potential_series_cols:
            print(f" - Column '{col}' sample value: {df[col].iloc[0]}")

    except Exception as e:
        print(f"Error reading parquet: {e}")
else:
    print("File not found. Please check the directory structure.")