import os
import argparse
import pandas as pd

def load_resources(base_dir):
    """Load concept index and solution dataset."""
    # 1. Concept Index (ID -> Name)
    idx_path = os.path.join(base_dir, "data_concept_graph", "concept_index.parquet")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Concept index not found: {idx_path}")
    
    print("Loading concept index...")
    df_idx = pd.read_parquet(idx_path)
    id_to_concept = dict(zip(df_idx['vid'], df_idx['concept']))
    
    # 2. Solution Dataset
    sol_path = os.path.join(base_dir, "data_pair_solution", "unconnected_2021_pair_solution_connected_2024_clean.parquet")
    if not os.path.exists(sol_path):
        raise FileNotFoundError(f"Solution dataset not found: {sol_path}")
    
    print(f"Loading dataset: {os.path.basename(sol_path)}...")
    df_sol = pd.read_parquet(sol_path)
    
    return df_sol, id_to_concept

def find_pairs(df, id_to_concept, ir, limit=20):
    """Filter pairs where future citation count >= IR."""
    
    # 컬럼 이름 확인 ('solution'이 인용 수/엣지 가중치라고 가정)
    target_col = 'solution' 
    if target_col not in df.columns:
        print(f"⚠️ Column '{target_col}' not found. Available columns: {list(df.columns)}")
        # 만약 컬럼명이 다르다면 여기서 수정 필요 (예: 'weight', 'citations' 등)
        return

    # Filtering
    # IR 이상인 행만 추출
    filtered_df = df[df[target_col] >= ir]
    
    total_found = len(filtered_df)
    print(f"\n🔍 Found {total_found} pairs with Future Citations >= {ir}")
    
    if total_found == 0:
        return

    # Sort by citation count (descending)
    filtered_df = filtered_df.sort_values(by=target_col, ascending=False)
    
    print(f"\n🏆 Top {limit} High-Impact Pairs (IR >= {ir}):")
    print("-" * 60)
    print(f"{'Concept U':<35} | {'Concept V':<35} | {'Citations (2024)'}")
    print("-" * 60)

    count = 0
    for _, row in filtered_df.iterrows():
        if count >= limit: break
        
        # 컬럼명 처리 (v1, v2 혹은 source, target)
        if 'v1' in df.columns:
            u_raw, v_raw = row['v1'], row['v2']
        else:
            u_raw, v_raw = row['source'], row['target']
            
        cit = row[target_col]
        
        # ID -> Name 변환
        u_name = id_to_concept.get(int(u_raw), str(u_raw))
        v_name = id_to_concept.get(int(v_raw), str(v_raw))
        
        print(f"{u_name:<35} | {v_name:<35} | {int(cit)}")
        count += 1
    print("-" * 60)

    # (옵션) 결과 저장
    save_path = "high_impact_candidates.csv"
    # filtered_df.head(100).to_csv(save_path, index=False)
    # print(f"Creating '{save_path}' with top 100 pairs... (Uncomment in code to enable)")

def main():
    parser = argparse.ArgumentParser(description="Find High-Impact Ground Truth Pairs")
    parser.add_argument("--ir", type=int, default=10, help="Minimum citation count threshold (Ground Truth)")
    parser.add_argument("--limit", type=int, default=20, help="Number of pairs to display")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        df, id_map = load_resources(base_dir)
        find_pairs(df, id_map, args.ir, args.limit)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()