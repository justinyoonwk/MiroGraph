import os
import pandas as pd
import argparse

def load_trend_data(base_dir, year):
    """
    Load concept index and citation statistics for the given year.
    """
    # 1. Load Concept Index (Map: Name -> ID)
    idx_path = os.path.join(base_dir, "data_concept_graph", "concept_index.parquet")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Concept index not found: {idx_path}")
    
    df_idx = pd.read_parquet(idx_path)
    # concept string -> vertex id (int)
    concept_to_id = dict(zip(df_idx['concept'], df_idx['vid']))
    
    # 2. Load Citation Data (Map: ID -> Stats)
    cit_path = os.path.join(base_dir, "data_for_features", f"concept_node_citation_data_{year}.parquet")
    if not os.path.exists(cit_path):
        raise FileNotFoundError(f"Citation data not found: {cit_path}")
    
    df_cit = pd.read_parquet(cit_path)
    # Set index to v1 (vertex id) for fast lookup
    df_cit = df_cit.set_index('v1')
    
    return concept_to_id, df_cit

def get_concept_stats(df_cit, vid):
    """
    Retrieve stats for a specific vertex ID. Returns a dict or None.
    """
    if vid not in df_cit.index:
        return None
    
    row = df_cit.loc[vid]
    return {
        "cY": int(row['cY']),          # Current Year Citations
        "ct_Y": int(row['ct_Y']),      # Total Accumulated Citations
        "ct_delta": int(row['ct_delta']), # Recent Delta
        "num": int(row['num'])         # Paper Count
    }

def format_trend_context(u_name, v_name, u_stats, v_stats, year, ir):
    """
    Generate the input prompt for the Trend Forecaster Agent.
    Includes explicit IR comparison logic.
    """
    lines = []
    lines.append(f"Task: Forecast trend synergy for ({u_name}, {v_name}) to reach {ir} citations.")
    lines.append(f"Target Threshold (IR): {ir}")
    lines.append(f"Data at Snapshot Year {year}:")
    
    for name, stats in [(u_name, u_stats), (v_name, v_stats)]:
        lines.append(f"[Concept: {name}]")
        if stats:
            lines.append(f"  - citations_current_year: {stats['cY']}")
            lines.append(f"  - citations_total: {stats['ct_Y']}")
            lines.append(f"  - recent_momentum (3-yr delta): {stats['ct_delta']}")
            lines.append(f"  - paper_count: {stats['num']}")
            
            # Derived Metrics for context
            growth_ratio = (stats['cY'] / stats['ct_Y']) if stats['ct_Y'] > 0 else 0
            lines.append(f"  - growth_ratio (current/total): {growth_ratio:.2%}")
            
            # IR Comparison Hint
            ratio_ir = stats['cY'] / ir if ir > 0 else 0
            lines.append(f"  - [Analysis] Current yearly citations are {ratio_ir:.1f}x the Target IR.")
        else:
            lines.append("  - Data not available (New or rare concept)")
            
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate trend features for LLM Agent.")
    parser.add_argument("--u", type=str, required=True, help="Source concept")
    parser.add_argument("--v", type=str, required=True, help="Target concept")
    parser.add_argument("--year", type=int, default=2021, help="Snapshot year")
    parser.add_argument("--ir", type=int, default=10, help="Impact Range")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        concept_to_id, df_cit = load_trend_data(base_dir, args.year)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Lookup IDs
    u_id = concept_to_id.get(args.u)
    v_id = concept_to_id.get(args.v)

    if u_id is None:
        print(f"Error: Concept '{args.u}' not found in index.")
        return
    if v_id is None:
        print(f"Error: Concept '{args.v}' not found in index.")
        return

    # Get Stats
    u_stats = get_concept_stats(df_cit, u_id)
    v_stats = get_concept_stats(df_cit, v_id)

    # Format Output
    llm_input = format_trend_context(args.u, args.v, u_stats, v_stats, args.year, args.ir)
    
    print("\n" + "="*50)
    print(" [Generated Input for Trend Forecaster Agent]")
    print("="*50)
    print(llm_input)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()