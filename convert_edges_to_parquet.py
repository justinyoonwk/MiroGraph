import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert edges_by_year.csv to concept_pair_citation_data_{year}.parquet")
    parser.add_argument("--year", type=int, default=2021, help="Cutoff year for cumulative sum")
    parser.add_argument("--base_dir", type=str, default=".", help="Project base directory")
    args = parser.parse_args()

    # 경로 설정 (사용자 환경에 맞게 수정 가능)
    # 1. 입력 CSV 파일 경로 (build_concept_graph_openalex.py의 출력 폴더 확인 필요)
    # 예: outputs_openalex_kdd_all 폴더 안에 있다고 가정
    input_csv_path = os.path.join(args.base_dir, "outputs_openalex_kdd_all", "edges_by_year.csv")
    
    # 2. Concept Index 경로 (이름 -> ID 매핑용)
    index_path = os.path.join(args.base_dir, "data_concept_graph", "concept_index.parquet")
    
    # 3. 출력 Parquet 저장 경로
    output_dir = os.path.join(args.base_dir, "data_for_features")
    output_path = os.path.join(output_dir, f"concept_pair_citation_data_{args.year}.parquet")

    # 파일 존재 확인
    if not os.path.exists(input_csv_path):
        print(f"❌ Input CSV not found: {input_csv_path}")
        return
    if not os.path.exists(index_path):
        print(f"❌ Concept Index not found: {index_path}")
        return

    print(f"🚀 Processing for Year {args.year}...")

    # 1. Load Concept Index (Name -> ID)
    print("   Loading concept index...")
    df_idx = pd.read_parquet(index_path)
    # concept(str) -> vid(int) 매핑 생성
    name_to_id = dict(zip(df_idx['concept'], df_idx['vid']))

    # 2. Load Edges Data
    print("   Loading edges_by_year.csv...")
    df_edges = pd.read_csv(input_csv_path)
    
    # 3. Filter by Year
    # 지정한 연도(2021) 이하의 데이터만 남김
    df_filtered = df_edges[df_edges['year'] <= args.year]
    
    # 4. Aggregate Citations (Sum)
    # source, target 별로 citations 합계 계산
    print("   Aggregating citations...")
    df_agg = df_filtered.groupby(['source', 'target'])['citations'].sum().reset_index()
    
    # 5. Map Names to IDs
    # source/target 문자열을 vid 정수로 변환
    print("   Mapping names to IDs...")
    df_agg['v1'] = df_agg['source'].map(name_to_id)
    df_agg['v2'] = df_agg['target'].map(name_to_id)
    
    # 매핑되지 않은(Index에 없는) 데이터 제거
    df_final = df_agg.dropna(subset=['v1', 'v2'])
    
    # ID를 정수형으로 변환
    df_final['v1'] = df_final['v1'].astype(int)
    df_final['v2'] = df_final['v2'].astype(int)
    
    # 컬럼 이름 변경 (citations -> ct_Y) 및 정렬
    df_final = df_final.rename(columns={'citations': 'ct_Y'})
    df_final = df_final[['v1', 'v2', 'ct_Y']]
    
    # 6. Save to Parquet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"   Saving to {output_path}...")
    df_final.to_parquet(output_path, index=False)
    
    print(f"✅ Done! Created {len(df_final)} pair records.")

if __name__ == "__main__":
    main()