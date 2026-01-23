#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-agent scoring + SVM decision boundary.

Flow:
1) Three agents (Structure, Semantic, Trend) each score the pair + subgraph (simple_prediction-style prompt with score field).
2) Collect train scores -> fit SVM (RBF) on (s_struct, s_sem, s_trend).
3) Collect test scores -> use SVM to predict Yes/No.
Caching:
- Agent scores can be cached to JSONL; reuse to avoid repeated LLM calls.
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
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
    agg = {}
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
def base_simple_prompt(ir: int) -> str:
    return f"""You are a scientific impact evaluator expert. 

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
  "prediction": "Yes" or "No",
  "score": <float 0-1 with exactly two decimal places>
}}

The "prediction" field MUST be exactly one of: "Yes", "No".
Do NOT include additional text outside the JSON object.
score=0.00 very unlikely; score=1.00 very likely with citations ≥ {ir}.
"""


def structure_prompt(ir: int) -> str:
    return (
        f"""You are the Structure Agent. Provide structural reasoning and a score based on subgraph connectivity. JSON only.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

Focus only on structural patterns in the subgraph (connectivity, paths, hubs, bridges, degree). Output JSON ONLY:
{{
  "reasoning": "<structural reasoning>",
  "prediction": "Yes" or "No",
  "score": <float 0-1 with exactly two decimal places>
}}
score=0.00 very unlikely; score=1.00 very likely with citations ≥ {ir}. Do NOT include text outside JSON.
"""
    )


def semantic_prompt(ir: int) -> str:
    return (
        f"""You are the Semantic Agent. Provide semantic reasoning and a score based on topical/meaningful compatibility. JSON only.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

Focus only on topical/semantic compatibility (concept meanings, overlaps, co-occurring nodes/edges). Output JSON ONLY:
{{
  "reasoning": "<semantic reasoning>",
  "prediction": "Yes" or "No",
  "score": <float 0-1 with exactly two decimal places>
}}
score=0.00 very unlikely; score=1.00 very likely with citations ≥ {ir}. Do NOT include text outside JSON.
"""
    )


def trend_prompt(ir: int) -> str:
    return (
        f"""You are the Trend Agent. Provide temporal reasoning and a score based on year-by-year citations. JSON only.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

Focus only on temporal signals (year-by-year citations, recency, growth/decline) from the subgraph. Output JSON ONLY:
{{
  "reasoning": "<temporal reasoning>",
  "prediction": "Yes" or "No",
  "score": <float 0-1 with exactly two decimal places>
}}
score=0.00 very unlikely; score=1.00 very likely with citations ≥ {ir}. Do NOT include text outside JSON.
"""
    )


def build_query_block(concept1: str, concept2: str, subgraph: Dict[str, Any]) -> str:
    return (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )


def run_agent(client: OpenAI, model: str, temp: float, sys_prompt: str, query_block: str) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query_block},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=temp, response_format={"type": "json_object"})
    content = resp.choices[0].message.content
    data = json.loads(content)
    # normalize fields
    score_val = float(data.get("score", 0))
    pred_str = str(data.get("prediction", "")).strip().lower()
    if pred_str == "yes":
        pred_int = 1
    elif pred_str == "no":
        pred_int = 0
    else:
        pred_int = 1 if score_val >= 0.5 else 0
    return {
        "reasoning": data.get("reasoning", ""),
        "prediction": data.get("prediction", ""),
        "prediction_int": pred_int,
        "score": round(score_val, 2),
    }


def call_multi_agent(client: OpenAI, model: str, ir: int, concept1: str, concept2: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
    query_block = build_query_block(concept1, concept2, subgraph)
    temp = 1.0 if "gpt-5" in model.lower() else 0.0
    struct_res = run_agent(client, model, temp, structure_prompt(ir), query_block)
    sem_res = run_agent(client, model, temp, semantic_prompt(ir), query_block)
    trend_res = run_agent(client, model, temp, trend_prompt(ir), query_block)
    # return aggregated agent scores
    return {
        "structure": struct_res,
        "semantic": sem_res,
        "trend": trend_res,
        "concept1": concept1,
        "concept2": concept2,
    }


# ---------- Caching ----------
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
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    def task(idx_row: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, row = idx_row
        subg = subgraph_fn(row["concept1"], row["concept2"])
        res = call_multi_agent(client, model, ir, row["concept1"], row["concept2"], subg)
        res["ground_truth_label"] = int(row.get("label", 0))
        res["index"] = idx
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, (idx, row)) for idx, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent scoring"):
            outputs.append(fut.result())
    return outputs


def flatten_scores(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for r in records:
        s = r["structure"]["score"]
        sem = r["semantic"]["score"]
        tr = r["trend"]["score"]
        X.append([s, sem, tr])
        y.append(int(r.get("ground_truth_label", 0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)


# ---------- CLI / Main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent scoring + SVM decision boundary.")
    p.add_argument("--train_csv", required=True, help="Training CSV (concept1, concept2, label).")
    p.add_argument("--test_csv", required=True, help="Test CSV.")
    p.add_argument("--concept_graph", required=True, help="Path to concept_graph.csv for subgraph extraction.")
    p.add_argument("--ir", type=int, required=True, help="IR threshold.")
    p.add_argument("--model", default="gpt-5-mini", help="OpenAI model.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--max_workers", type=int, default=100, help="ThreadPoolExecutor workers for LLM calls.")
    p.add_argument("--scores_cache_train", type=str, default="results/svm/train_scores.jsonl", help="Cache path for train agent scores.")
    p.add_argument("--scores_cache_test", type=str, default="results/svm/test_scores.jsonl", help="Cache path for test agent scores.")
    p.add_argument("--reuse_train_cache", action="store_true", help="Reuse train scores cache.")
    p.add_argument("--reuse_test_cache", action="store_true", help="Reuse test scores cache.")
    p.add_argument("--svm_model_path", type=str, default="results/svm/svm_model.pkl", help="Path to save/load SVM model.")
    p.add_argument("--svm_C", type=float, default=10.0, help="SVM C.")
    p.add_argument("--svm_gamma", type=float, default=0.5, help="SVM gamma.")
    p.add_argument("--output", type=str, default="results/svm/test_predictions.jsonl", help="Output JSONL for test predictions.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    client = OpenAI(api_key=api_key)

    # Load graph for subgraph extraction
    G, snapshot_year = load_concept_graph_csv(args.concept_graph)

    def subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G, a, b, snapshot_year, max_k=4, max_p=8)

    # Train scores
    if args.reuse_train_cache and os.path.exists(args.scores_cache_train):
        print(f"[CACHE] Reusing train scores from {args.scores_cache_train}")
        train_records = load_scores_cache(args.scores_cache_train)
    else:
        train_df = pd.read_csv(args.train_csv)
        train_rows = train_df.to_dict(orient="records")
        train_records = generate_agent_scores(client, args.model, args.ir, train_rows, subgraph_fn, args.max_workers)
        if args.scores_cache_train:
            save_scores_cache(args.scores_cache_train, train_records)
            print(f"[CACHE] Saved train scores to {args.scores_cache_train}")

    X_train, y_train = flatten_scores(train_records)

    # Fit SVM
    clf = SVC(C=args.svm_C, gamma=args.svm_gamma, kernel="rbf", probability=False)
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(args.svm_model_path), exist_ok=True)
    with open(args.svm_model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[SVM] Trained and saved to {args.svm_model_path}")

    # Test scores
    if args.reuse_test_cache and os.path.exists(args.scores_cache_test):
        print(f"[CACHE] Reusing test scores from {args.scores_cache_test}")
        test_records = load_scores_cache(args.scores_cache_test)
    else:
        test_df = pd.read_csv(args.test_csv)
        test_rows = test_df.to_dict(orient="records")
        test_records = generate_agent_scores(client, args.model, args.ir, test_rows, subgraph_fn, args.max_workers)
        if args.scores_cache_test:
            save_scores_cache(args.scores_cache_test, test_records)
            print(f"[CACHE] Saved test scores to {args.scores_cache_test}")

    X_test, y_test = flatten_scores(test_records)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_neg = f1_score(y_test, y_pred, zero_division=0, pos_label=0)
    macro_f1 = (f1 + f1_neg) / 2
    print("[METRICS]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    # Save predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, yp in zip(test_records, y_pred):
            out = {
                "concept1": r.get("concept1"),
                "concept2": r.get("concept2"),
                "structure": r["structure"],
                "semantic": r["semantic"],
                "trend": r["trend"],
                "svm_pred": int(yp),
                "svm_pred_label": "Yes" if yp == 1 else "No",
                "ground_truth_label": int(r.get("ground_truth_label", 0)),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[DONE] saved predictions to {args.output}")


if __name__ == "__main__":
    main()
