import os
import gzip
import pickle
import pandas as pd
import networkx as nx
import argparse
from collections import deque

def load_graph_data(base_dir: str, year: int = 2021) -> nx.Graph:
    """
    Load Binary Structure and apply Edge Weights from Pair Citation Data.
    """
    print(f"Loading graph structure and edge weights for year {year}...")
    
    # 1. Load Concept Index (ID -> Name)
    idx_path = os.path.join(base_dir, "data_concept_graph", "concept_index.parquet")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Concept index not found: {idx_path}")
    
    df_idx = pd.read_parquet(idx_path)
    id_to_concept = dict(zip(df_idx['vid'], df_idx['concept']))
    print(f" - Concept Index loaded: {len(id_to_concept)} nodes.")

    # 2. Load Binary Adjacency Matrix (Structure)
    adj_path = os.path.join(base_dir, "data_for_features", f"adjacency_matrix_{year}.gz")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")

    with gzip.open(adj_path, 'rb') as f:
        adj_matrix = pickle.load(f)
    
    # Create Graph (Nodes are IDs initially)
    try:
        G = nx.from_scipy_sparse_array(adj_matrix)
    except AttributeError:
        G = nx.from_scipy_sparse_matrix(adj_matrix)

    # 3. Load PAIR Citation Data (Edge Weights)
    pair_path = os.path.join(base_dir, "data_for_features", f"concept_pair_citation_data_{year}.parquet")
    
    if os.path.exists(pair_path):
        print(f" - Loading Edge Weights from: {os.path.basename(pair_path)}")
        df_pair = pd.read_parquet(pair_path)
        
        # 컬럼 확인 (v1, v2, ct_Y) - convert_edges_to_parquet.py 결과물 기준
        if {'v1', 'v2', 'ct_Y'}.issubset(df_pair.columns):
            # 엣지 가중치 딕셔너리 생성
            edge_weights = {}
            for row in df_pair.itertuples():
                u, v = int(row.v1), int(row.v2)
                # Undirected Graph이므로 순서 통일 (작은값, 큰값)
                if u > v: u, v = v, u
                edge_weights[(u, v)] = int(row.ct_Y)
            
            # 그래프 엣지에 가중치 할당
            update_count = 0
            for u, v in G.edges():
                k1, k2 = (u, v) if u < v else (v, u)
                weight = edge_weights.get((k1, k2), 0) # 없으면 0 (또는 1)
                
                # 가중치 저장
                G[u][v]['citation_weight'] = weight
                
                if weight > 0:
                    update_count += 1
            
            print(f" - Assigned weights to {update_count} edges.")
        else:
            print(f"⚠️ Column mismatch. Expected v1, v2, ct_Y. Found: {list(df_pair.columns)}")
    else:
        print(f"⚠️ Pair citation file not found at: {pair_path}")

    # 4. Relabel Nodes to Names
    G = nx.relabel_nodes(G, id_to_concept)
    
    return G, id_to_concept

def bfs_path_explorer(G: nx.Graph, source: str, target: str, max_k: int = 3, max_p: int = 10):
    if source not in G: return None, None, f"Source '{source}' not found."
    if target not in G: return None, None, f"Target '{target}' not found."

    queue = deque([(source, [source])])
    found_paths = []
    visited_paths = set() 
    max_nodes = max_k + 1

    while queue and len(found_paths) < max_p:
        curr_node, curr_path = queue.popleft()
        
        if len(curr_path) > max_nodes: continue
        if curr_node == target:
            found_paths.append(curr_path)
            continue

        for neighbor in G.neighbors(curr_node):
            if neighbor not in curr_path:
                new_path = curr_path + [neighbor]
                if tuple(new_path) not in visited_paths:
                    visited_paths.add(tuple(new_path))
                    queue.append((neighbor, new_path))
    
    if not found_paths: return [], None, "No paths found."

    subgraph_nodes = set()
    for path in found_paths: subgraph_nodes.update(path)
    G_sub = G.subgraph(subgraph_nodes).copy()
    
    return found_paths, G_sub, "Success"

def format_for_llm(source, target, paths, G_sub, ir):
    """
    Format: Node -[Edge Weight]-> Node
    """
    lines = []
    lines.append(f"Task: Analyze structural potential for link ({source}, {target}) to reach {ir} citations.")
    lines.append(f"Target Threshold (IR): {ir}")
    lines.append(f"\n1. Path Analysis (Top {len(paths)} shortest paths):")
    lines.append("   Format: Concept -[Citation Count]-> Concept ...")

    for i, p in enumerate(paths):
        path_str_parts = []
        total_weight = 0
        edge_cnt = 0
        
        for j in range(len(p) - 1):
            u, v = p[j], p[j+1]
            
            # 엣지 가중치 가져오기 (기본값 0)
            # 주의: G_sub는 Subgraph이므로 원본 G의 속성을 유지함
            weight = G_sub[u][v].get('citation_weight', 0)
            
            path_str_parts.append(f"{u} -[{weight}]->")
            total_weight += weight
            edge_cnt += 1
        
        # 마지막 노드
        path_str_parts.append(p[-1])
        
        path_str = " ".join(path_str_parts)
        avg_weight = total_weight / max(1, edge_cnt)
        
        status = "Strong" if avg_weight >= ir else "Weak"
        
        lines.append(f"  - Path {i+1}: {path_str}")
        lines.append(f"    (Avg Edge Citations: {avg_weight:.1f} vs IR {ir} -> {status})")

    lines.append(f"\n2. Subgraph Statistics:")
    lines.append(f"  - Nodes: {G_sub.number_of_nodes()}")
    lines.append(f"  - Edges: {G_sub.number_of_edges()}")
    
    # Bridge Info (Degree based)
    degrees = dict(G_sub.degree())
    bridges = sorted([n for n in G_sub.nodes() if n not in (source, target)], key=lambda n: degrees[n], reverse=True)[:5]
    
    if bridges:
        lines.append(f"  - Key Bridges (Degree): {', '.join([f'{b}({degrees[b]})' for b in bridges])}")
    else:
        lines.append("  - Key Bridges: None")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=str, required=True)
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--ir", type=int, default=10)
    parser.add_argument("--year", type=int, default=2021)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        G, _ = load_graph_data(base_dir, args.year)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\nSearching paths: '{args.u}' <--> '{args.v}'")
    paths, G_sub, msg = bfs_path_explorer(G, args.u, args.v)
    
    if paths:
        print("\n" + "="*50)
        print(format_for_llm(args.u, args.v, paths, G_sub, args.ir))
        print("="*50 + "\n")
    else:
        print(f"Result: {msg}")

if __name__ == "__main__":
    main()