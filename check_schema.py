import os
import pandas as pd

# 파일 경로 설정 (2021년 데이터 기준)
base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, "data_for_features", "concept_node_citation_data_2021.parquet")

print(f"📂 Checking file: {parquet_path}")

if os.path.exists(parquet_path):
    try:
        df = pd.read_parquet(parquet_path)
        print("\n✅ File Loaded Successfully!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print("\n📊 First 3 Rows:")
        print(df.head(3))
        
        # 데이터 타입 확인
        print("\nℹ️ Data Types:")
        print(df.dtypes)
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
else:
    print("❌ File not found. Please check the path.")