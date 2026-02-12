import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import extract_path

load_dotenv()


# ---------- Graph utilities ----------
def load_concept_graph_csv(csv_path: str) -> Tuple[nx.Graph, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept_graph.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"concept1", "concept2", "year", "citation_count"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}")
    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)
    snapshot_year = int(df["year"].max())
    agg: Dict[Tuple[str, str, int], int] = {}
    for row in df.itertuples(index=False):
        c1, c2, y, cc = row.concept1, row.concept2, int(row.year), int(row.citation_count)
        a, b = (c1, c2) if str(c1) <= str(c2) else (c2, c1)
        k = (a, b, y)
        agg[k] = agg.get(k, 0) + cc
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


def edge_year_dict(G: nx.Graph, u: str, v: str) -> Dict[int, int]:
    if not G.has_edge(u, v):
        return {}
    data = G.get_edge_data(u, v) or {}
    yd = data.get("citations_by_year", {})
    return dict(yd) if isinstance(yd, dict) else {}


def enclosing_subgraph(G: nx.Graph, u: str, v: str, snapshot_year: int, max_k: int = 4, max_p: int = 8) -> Dict[str, Any]:
    paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    paths = paths or []
    nodes = set()
    for p in paths:
        nodes.update(p)
    G_sub = G.subgraph(nodes).copy() if nodes else nx.Graph()
    edge_list = []
    for a, b in sorted(G_sub.edges()):
        yd = edge_year_dict(G_sub, a, b)
        citations_by_year = {int(y): int(c) for y, c in yd.items()}
        edge_list.append([a, b, citations_by_year])
    return {
        "query_pair": [u, v],
        "edge_list": edge_list,
    }


# ---------- Prompts ----------
def structure_prompt(ir: int) -> str:
    return (
        f"""You are the Structure Agent. Focus ONLY on structural evidence in the subgraph (paths, bridges, hubs, degree, connectivity). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output concise JSON (<=1024 tokens) with reasoning and score.

[TASK]
Given a pair of scientific concepts (X, Y), estimate how likely they are to co-occur in papers within the next 3 years with cumulative citations reaching at least {ir}.

[OUTPUT]
Return JSON only:
{{
  "reasoning": "<brief structural reasoning>",
  "score": <float between 0 and 1 with exactly two decimal places>
}}
Do NOT include any text outside the JSON object.
"""
    )


def semantic_prompt(ir: int) -> str:
    return (
        f"""You are the Semantic Agent. Focus ONLY on topical/semantic compatibility (meaning overlap, shared context). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output concise JSON (<=1024 tokens) with reasoning and score.

[TASK]
Given a pair of scientific concepts (X, Y), estimate how likely they are to co-occur in papers within the next 3 years with cumulative citations reaching at least {ir}.

[OUTPUT]
Return JSON only:
{{
  "reasoning": "<brief semantic reasoning>",
  "score": <float between 0 and 1 with exactly two decimal places>
}}
Do NOT include any text outside the JSON object.
"""
    )


def trend_prompt(ir: int) -> str:
    return (
        f"""You are the Trend Agent. Focus ONLY on temporal signals in the subgraph (year-by-year citations, recency, growth/decline). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output concise JSON (<=1024 tokens) with reasoning and score.

[TASK]
Given a pair of scientific concepts (X, Y), estimate how likely they are to co-occur in papers within the next 3 years with cumulative citations reaching at least {ir}.

[OUTPUT]
Return JSON only:
{{
  "reasoning": "<brief temporal reasoning>",
  "score": <float between 0 and 1 with exactly two decimal places>
}}
Do NOT include any text outside the JSON object.
"""
    )


# ---------- LLM calls ----------
def run_agent(client: OpenAI, model: str, temp: float, sys_prompt: str, query_block: str) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query_block},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=temp, response_format={"type": "json_object"})
    content = resp.choices[0].message.content
    data = json.loads(content)
    score_val = float(data.get("score", 0))
    reasoning = data.get("reasoning", "")
    return {"score": score_val, "reasoning": reasoning}


def call_multi_agent(client: OpenAI, model: str, ir: int, concept1: str, concept2: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
    query_block = (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )
    temp = 1.0 if "gpt-5" in model.lower() else 0.0
    struct_res = run_agent(client, model, temp, structure_prompt(ir), query_block)
    sem_res = run_agent(client, model, temp, semantic_prompt(ir), query_block)
    trend_res = run_agent(client, model, temp, trend_prompt(ir), query_block)
    return {
        "structure": struct_res,
        "semantic": sem_res,
        "trend": trend_res,
        "concept1": concept1,
        "concept2": concept2,
    }


# ---------- Cache helpers ----------
def save_scores_cache(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_scores_cache(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# ---------- Scoring ----------
def generate_agent_scores(
    client: OpenAI,
    model: str,
    ir: int,
    rows: List[Dict[str, Any]],
    subgraph_fn,
    max_workers: int,
    keep_label: bool = True,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    def task(idx_row: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, row = idx_row
        subg = subgraph_fn(row["concept1"], row["concept2"])
        res = call_multi_agent(client, model, ir, row["concept1"], row["concept2"], subg)
        if keep_label:
            res["ground_truth_label"] = int(row.get("label", 0))
        res["index"] = idx
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, (idx, row)) for idx, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent scoring"):
            outputs.append(fut.result())
    return outputs


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent scoring only (no meta inference).")
    p.add_argument("--train_csv", required=True, help="Training CSV (concept1, concept2, label).")
    p.add_argument("--test_csv", required=True, help="Test CSV.")
    p.add_argument("--concept_graph", required=True, help="Path to concept_graph.csv for subgraph extraction.")
    p.add_argument("--ir", type=int, required=True, help="IR threshold.")
    p.add_argument("--model", default="gpt-5-mini", help="OpenAI model.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--max_workers", type=int, default=100, help="ThreadPoolExecutor workers for LLM calls.")
    p.add_argument("--scores_cache_train", type=str, default="results/metallm/train_scores.jsonl", help="Cache path for train agent scores.")
    p.add_argument("--scores_cache_test", type=str, default="results/metallm/test_scores.jsonl", help="Cache path for test agent scores.")
    p.add_argument("--reuse_train_cache", action="store_true", help="Reuse train scores cache.")
    p.add_argument("--reuse_test_cache", action="store_true", help="Reuse test scores cache.")
    p.add_argument("--keep_test_labels", action="store_true", help="If set, attach label for test rows when available (for metrics only).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for any shuffling.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    client = OpenAI(api_key=api_key)

    # Graph for subgraph extraction
    G, snapshot_year = load_concept_graph_csv(args.concept_graph)

    def subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G, a, b, snapshot_year, max_k=4, max_p=8)

    np.random.seed(args.seed)

    # Train scores
    if args.reuse_train_cache and os.path.exists(args.scores_cache_train):
        print(f"[CACHE] Reusing train scores from {args.scores_cache_train}")
        train_records = load_scores_cache(args.scores_cache_train)
    else:
        train_df = pd.read_csv(args.train_csv)
        train_rows = train_df.to_dict(orient="records")
        train_records = generate_agent_scores(client, args.model, args.ir, train_rows, subgraph_fn, args.max_workers, keep_label=True)
        if args.scores_cache_train:
            save_scores_cache(args.scores_cache_train, train_records)
            print(f"[CACHE] Saved train scores to {args.scores_cache_train}")

    # Test scores
    if args.reuse_test_cache and os.path.exists(args.scores_cache_test):
        print(f"[CACHE] Reusing test scores from {args.scores_cache_test}")
        test_records = load_scores_cache(args.scores_cache_test)
    else:
        test_df = pd.read_csv(args.test_csv)
        test_rows = test_df.to_dict(orient="records")
        test_records = generate_agent_scores(
            client,
            args.model,
            args.ir,
            test_rows,
            subgraph_fn,
            args.max_workers,
            keep_label=args.keep_test_labels,
        )
        if args.scores_cache_test:
            save_scores_cache(args.scores_cache_test, test_records)
            print(f"[CACHE] Saved test scores to {args.scores_cache_test}")

    print("[DONE] scores computed and cached.")


if __name__ == "__main__":
    main()
