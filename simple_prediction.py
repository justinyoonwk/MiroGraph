#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple LLM-based prediction: Predict impact using only concept pair and LLM
Supports both single pair and batch processing with parallel execution
"""

import os
import json
import argparse
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import extract_path

load_dotenv()

# =========================================================
# Path helpers
# =========================================================
def infer_dataset_setting_from_path(test_file: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Robustly infer (dataset, setting) from either relative or absolute paths.
    Expected layout: .../dataset/<DATASET>/<SETTING>/test_data_10.csv
    """
    p = os.path.abspath(test_file).replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if "dataset" in parts:
        i = parts.index("dataset")
        if i + 2 < len(parts):
            dataset = parts[i + 1]
            setting = parts[i + 2]
            return dataset, setting
    # fallback: if already relative like "SIGIR/inductive/test_data_10.csv"
    if len(parts) >= 3 and parts[-3] in ("inductive", "transductive"):
        return parts[-4] if len(parts) >= 4 else None, parts[-3]
    return None, None

# =========================================================
# Graph loading and subgraph extraction functions
# =========================================================
def load_concept_graph_csv(csv_path: str) -> Tuple[nx.Graph, int]:
    """
    concept_graph.csv format:
      concept1, concept2, year, citation_count

    Build undirected graph:
      edge attr: citations_by_year = {year:int}
    Return (G, snapshot_year=max year)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept_graph.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"concept1", "concept2", "year", "citation_count"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}")

    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)

    snapshot_year = int(df["year"].max())

    # aggregate in case duplicates
    # key: (c1,c2,year) with undirected canonical pair
    agg = {}
    for row in df.itertuples(index=False):
        c1, c2, y, cc = row.concept1, row.concept2, int(row.year), int(row.citation_count)
        a, b = (c1, c2) if str(c1) <= str(c2) else (c2, c1)
        k = (a, b, y)
        agg[k] = agg.get(k, 0) + cc

    # build edge_map: (a,b)->{year:count}
    edge_map: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (a, b, y), cc in agg.items():
        edge_map.setdefault((a, b), {})
        edge_map[(a, b)][y] = edge_map[(a, b)].get(y, 0) + cc

    G = nx.Graph()
    for (a, b), year_dict in edge_map.items():
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b, citations_by_year=dict(year_dict))

    return G, snapshot_year


def get_edge_year_dict(G: nx.Graph, a: str, b: str) -> Dict[int, int]:
    if not G.has_edge(a, b):
        return {}
    data = G.get_edge_data(a, b) or {}
    yd = data.get("citations_by_year", {})
    return dict(yd) if isinstance(yd, dict) else {}


def recent_3yr_weight(year_dict: Dict[int, int], snapshot_year: int) -> int:
    """Calculate total citation weight (all years, not just recent 3 years)"""
    if not year_dict:
        return 0
    # Sum all citations across all years
    return sum(int(count) for count in year_dict.values())


def build_enclosing_subgraph_from_paths(G: nx.Graph, paths: List[List[str]]) -> nx.Graph:
    nodes = set()
    for p in paths:
        nodes.update(p)
    if not nodes:
        return nx.Graph()
    return G.subgraph(nodes).copy()


def mask_subgraph_edge_list(
    G_sub: nx.Graph, u: str, v: str, snapshot_year: int
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """
    Returns subgraph with actual concept names (no masking)
    Returns:
      concept_to_id (empty, not used), id_to_concept (empty, not used), json_payload={"query_pair":[...],"edge_list":[[conceptA,conceptB,weight],...]}
    """
    nodes = list(G_sub.nodes())
    if u not in nodes or v not in nodes:
        # still allow empty/fallback
        nodes = [u, v]

    edge_list = []
    # stable order
    edges = sorted(G_sub.edges(), key=lambda e: (str(e[0]), str(e[1])))
    for a, b in edges:
        ya = get_edge_year_dict(G_sub, a, b)
        # Use year-by-year citation count dictionary instead of total weight
        citations_by_year = {int(year): int(count) for year, count in ya.items()}
        edge_list.append([a, b, citations_by_year])  # Use actual concept names with year-by-year citations

    payload = {
        "query_pair": [u, v],  # Use actual concept names
        "edge_list": edge_list,
    }
    return {}, {}, payload  # Return empty dicts for compatibility


def extract_enclosing_subgraph(
    G: nx.Graph, u: str, v: str, snapshot_year: int, max_k: int = 4, max_p: int = 8
) -> Optional[Dict[str, Any]]:
    """Extract enclosing subgraph for concept pair (u, v)"""
    try:
        # Find induced paths
        paths, _G_sub_unused, msg = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
        if paths is None:
            paths = []
        paths = list(paths)

        # Build enclosing subgraph from paths
        G_sub = build_enclosing_subgraph_from_paths(G, paths)
        
        # Get subgraph payload (without masking, using actual concept names)
        _, _, subgraph_payload = mask_subgraph_edge_list(G_sub, u, v, snapshot_year)
        
        return {
            "paths": paths,
            "subgraph": subgraph_payload
        }
    except Exception as e:
        print(f"[WARNING] Failed to extract subgraph for ({u}, {v}): {e}")
        return None


def load_simple_prompt(ir: int) -> str:
    """Load simple prompt for direct prediction"""
    prompt = f"""You are a scientific impact prediction expert.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

[PREDICTION RULES]
You must predict "Yes" or "No" based on:
- Semantic relationship: Are these concepts related? Could they be combined in research?
- Research trends: Are these areas active? Is there growing interest?
- Potential impact: If combined, would this research likely receive significant citations?

[OUTPUT]
Return a JSON object with:
{{
  "reasoning": "<your reasoning about why this pair will or won't have high impact>",
  "prediction": "Yes" or "No"
}}

The "prediction" field MUST be exactly one of: "Yes", "No".
Do NOT include additional text outside the JSON object.
"""
    return prompt

def predict_pair(
    client: OpenAI, 
    concept1: str, 
    concept2: str, 
    ir: int, 
    model: str = "gpt-4o-mini",
    subgraph_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Predict impact for a concept pair using simple LLM call"""
    prompt = load_simple_prompt(ir)
    
    user_input = f"""Concept Pair:
- Concept X: {concept1}
- Concept Y: {concept2}"""
    
    # Add subgraph information if available
    if subgraph_info and subgraph_info.get("subgraph"):
        subgraph_payload = subgraph_info["subgraph"]
        user_input += f"""

[ENCLOSING SUBGRAPH]
The following is the structural context around this concept pair in the concept graph:
{json.dumps(subgraph_payload, ensure_ascii=False, indent=2)}

This subgraph shows the connectivity patterns and relationships between concepts in the graph. Use this structural information to inform your prediction."""
    
    user_input += f"""

Please predict whether this concept pair will have high impact (co-occur within 3 years with cumulative citations ≥ {ir})."""
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    try:
        # gpt-5-mini 등 일부 모델은 temperature=0.0을 지원하지 않을 수 있음
        temp = 1.0 if "gpt-5" in model else 0.0
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Convert prediction to int (1 for "Yes", 0 for "No")
        prediction_str = result.get("prediction", "").strip().lower()
        if prediction_str == "yes":
            prediction_int = 1
        elif prediction_str == "no":
            prediction_int = 0
        else:
            # Try to infer from reasoning if prediction field is missing
            reasoning = result.get("reasoning", "").lower()
            if "yes" in reasoning or "high impact" in reasoning or "will co-occur" in reasoning:
                prediction_int = 1
            else:
                prediction_int = 0
        
        return {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "reasoning": result.get("reasoning", ""),
            "prediction": result.get("prediction", ""),
            "prediction_int": prediction_int
        }
    except Exception as e:
        print(f"Error predicting pair ({concept1}, {concept2}): {e}")
        return {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "reasoning": f"Error: {str(e)}",
            "prediction": "No",
            "prediction_int": 0,
            "error": str(e)
        }

def process_single_pair(args_tuple: Tuple) -> Tuple[int, Dict[str, Any]]:
    """Process a single pair (for parallel execution)"""
    idx, row, ir, model, client, G, snapshot_year, max_k, max_p, use_subgraph = args_tuple
    
    concept1 = row['concept1']
    concept2 = row['concept2']
    ground_truth = int(row['label'])
    total_citations = int(row.get('total_citations', 0))
    
    # Extract enclosing subgraph (optional)
    subgraph_info = None
    if use_subgraph and G is not None:
        subgraph_info = extract_enclosing_subgraph(G, concept1, concept2, snapshot_year, max_k, max_p)
    
    try:
        result = predict_pair(client, concept1, concept2, ir, model, subgraph_info)
        
        # Add ground truth info
        result['ground_truth_label'] = ground_truth
        result['total_citations'] = total_citations
        result['pair'] = {
            'concept1': concept1,
            'concept2': concept2
        }
        
        return (idx, result)
    except Exception as e:
        print(f"[ERROR] Failed to process pair {idx} ({concept1} <-> {concept2}): {str(e)}")
        return (idx, {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "ground_truth_label": ground_truth,
            "total_citations": total_citations,
            "pair": {
                'concept1': concept1,
                'concept2': concept2
            },
            "reasoning": f"Error: {str(e)}",
            "prediction": "No",
            "prediction_int": 0,
            "error": str(e)
        })

def run_batch_prediction(
    client: OpenAI,
    test_file: str,
    ir: int,
    model: str = "gpt-4o-mini",
    api_key_name: str = "1",
    output_file: str = None,
    batch_size: int = 100,
    max_k: int = 4,
    max_p: int = 8,
    use_subgraph: bool = True
):
    """Run simple prediction on all test pairs with parallel processing"""
    
    # Load test data
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} test pairs from {test_file}")
    print()
    
    # Infer dataset/setting robustly from absolute/relative paths
    dataset, setting = infer_dataset_setting_from_path(test_file)
    
    # Load concept graph
    G = None
    snapshot_year = None
    if dataset and setting and use_subgraph:
        # Get the directory containing the test_file
        test_file_dir = os.path.dirname(os.path.abspath(test_file))
        # The test_file should be in dataset/setting/ directory
        # So we can directly use test_file_dir
        if setting == "inductive":
            # inductive: use concept_graph_test.csv
            graph_path = os.path.join(test_file_dir, "concept_graph_test.csv")
        else:
            # transductive: use concept_graph.csv
            graph_path = os.path.join(test_file_dir, "concept_graph.csv")
        
        if os.path.exists(graph_path):
            print(f"Loading concept graph from: {graph_path}")
            G, snapshot_year = load_concept_graph_csv(graph_path)
            print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, snapshot_year={snapshot_year}")
        else:
            print(f"[WARNING] Concept graph not found: {graph_path}, proceeding without subgraph information")
    else:
        if not use_subgraph:
            print("[INFO] Subgraph extraction disabled (--disable_subgraph)")
        else:
            print(f"[WARNING] Could not extract dataset/setting from test_file path, proceeding without subgraph information")
    
    # Set max_workers
    max_workers = batch_size
    
    if output_file is None:
        dataset_name, setting_name = infer_dataset_setting_from_path(test_file)
        if dataset_name and setting_name:
            output_file = f"results/{dataset_name}/{setting_name}/simple_prediction_ir{ir}_{model.replace('/', '_')}.jsonl"
        else:
            output_file = f"results/simple_prediction_ir{ir}_{model.replace('/', '_')}.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Resume logic: check existing results
    processed_pairs = set()
    all_results = []
    y_true = []
    y_pred = []

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('type') == 'result':
                        p1 = data.get('concept1')
                        p2 = data.get('concept2')
                        processed_pairs.add(tuple(sorted((str(p1), str(p2)))))
                        all_results.append(data)
                        y_true.append(data['ground_truth_label'])
                        y_pred.append(data['prediction_int'])
                except:
                    continue
        if processed_pairs:
            print(f"Found {len(processed_pairs)} existing results in {output_file}. Resuming...")

    # Initialize file with metadata if new
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            metadata = {
                'type': 'metadata',
                'ir': ir,
                'model': model,
                'test_file': test_file,
                'use_subgraph': use_subgraph,
                'max_k': max_k,
                'max_p': max_p
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

    print(f"Processing {len(df)} pairs with IR={ir}")
    print(f"Concurrent workers: {max_workers}")
    if use_subgraph and G is not None:
        print(f"Subgraph extraction: max_k={max_k}, max_p={max_p}")
    print()
    
    # Prepare arguments for all pairs (excluding processed ones)
    args_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        c1, c2 = str(row['concept1']), str(row['concept2'])
        if tuple(sorted((c1, c2))) in processed_pairs:
            continue
        args_list.append((i, row, ir, model, client, G, snapshot_year, max_k, max_p, use_subgraph))

    if not args_list:
        print("All pairs already processed.")
    else:
        # Process in parallel using as_completed for real-time saving
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_pair, args): args[0] for args in args_list}
            
            with tqdm(total=len(args_list), desc="Predicting") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        _, result = future.result()
                        # Add metadata type
                        result['type'] = 'result'
                        
                        # Save to file immediately
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        
                        all_results.append(result)
                        y_true.append(result['ground_truth_label'])
                        y_pred.append(result['prediction_int'])
                    except Exception as e:
                        print(f"\n[ERROR] Future failed: {e}")
                    pbar.update(1)
    
    # Calculate metrics
    if not y_true:
        print("No results to calculate metrics.")
        return [], 0, 0, 0, 0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print()
    print("=" * 80)
    print("Cumulative Results (including resumed)")
    print("=" * 80)
    print(f"Total pairs: {len(all_results)}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {cm[1][1]}, TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}")
    print()
    
    # Final metrics record
    final_metrics = {
        'type': 'metrics',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'tp': int(cm[1][1]),
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0])
        }
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_metrics, ensure_ascii=False) + '\n')
    
    print(f"[Finished] All results saved to: {output_file}")
    
    return all_results, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Simple LLM-based impact prediction")
    parser.add_argument("--concept1", type=str, default=None, help="First concept (for single prediction)")
    parser.add_argument("--concept2", type=str, default=None, help="Second concept (for single prediction)")
    parser.add_argument("--test_file", type=str, default=None, help="Test data CSV file (for batch prediction)")
    parser.add_argument("--ir", type=int, required=True, help="IR threshold")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api_key_name", type=str, default="1", help="API key name (OPENAI_API_KEY_1, OPENAI_API_KEY_2, etc.)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (direct input)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for parallel processing")
    parser.add_argument("--max_k", type=int, default=4, help="Maximum path length for subgraph extraction (default: 4)")
    parser.add_argument("--max_p", type=int, default=8, help="Maximum number of paths for subgraph extraction (default: 8)")
    parser.add_argument("--disable_subgraph", action="store_true", help="Disable enclosing subgraph extraction and remove it from the prompt")
    
    args = parser.parse_args()
    
    # Get API key (and create client once)
    if args.api_key:
        api_key = args.api_key
    else:
        api_key_name = f"OPENAI_API_KEY_{args.api_key_name}"
        api_key = os.getenv(api_key_name)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"API key not found. Set {api_key_name} or OPENAI_API_KEY in .env file")

    client = OpenAI(api_key=api_key)
    
    # Single prediction mode
    if args.concept1 and args.concept2:
        print(f"Predicting impact for: {args.concept1} <-> {args.concept2}")
        print(f"IR threshold: {args.ir}")
        print()
        
        result = predict_pair(client, args.concept1, args.concept2, args.ir, args.model)
        
        # Print result
        print("=" * 80)
        print("Prediction Result")
        print("=" * 80)
        print(f"Concept Pair: {result['concept1']} <-> {result['concept2']}")
        print(f"IR Threshold: {result['ir']}")
        print(f"Prediction: {result['prediction']} ({result['prediction_int']})")
        print()
        print("Reasoning:")
        print(result['reasoning'])
        print()
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {args.output}")
        
        return result
    
    # Batch prediction mode
    elif args.test_file:
        run_batch_prediction(
            client=client,
            test_file=args.test_file,
            ir=args.ir,
            model=args.model,
            api_key_name=args.api_key_name,
            output_file=args.output,
            batch_size=args.batch_size,
            max_k=args.max_k,
            max_p=args.max_p,
            use_subgraph=(not args.disable_subgraph)
        )
    
    else:
        parser.error("Either --concept1 and --concept2 (for single prediction) or --test_file (for batch prediction) must be provided")

if __name__ == "__main__":
    main()
