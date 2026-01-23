#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured prediction (multi-agent, no clustering, no demos):
- Three agents (Structure, Semantic, Trend) analyze the subgraph.
- Impact Evaluator agent integrates their outputs and returns prediction + score.
- Use train (or provided val) scores/labels to search best threshold.
- Apply threshold to test scores for final Yes/No.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import extract_path

load_dotenv()


# ========== Graph utilities ==========
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


def graph_features(G: nx.Graph, u: str, v: str, snapshot_year: int) -> Dict[str, Any]:
    return {}


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


# ========== Prompting (multi-agent, simple_prediction-style with score) ==========
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
  "prediction": "Yes" or "No"
}}

The "prediction" field MUST be exactly one of: "Yes", "No".
Do NOT include additional text outside the JSON object.
Additionally include a field "score" as a float in [0,1] with exactly two decimal places (0.00 very unlikely, 1.00 very likely with citations ≥ {ir}).
You will also receive summaries from Structure/Semantic/Trend agents; treat them as references only."""


def structure_prompt(ir: int) -> str:
    return (
        base_simple_prompt(ir)
        + "\n\nYou are the Structure Agent. Provide only factual observations about structural patterns in the subgraph (connectivity, paths, hubs, bridges, degree)."
        + " Do NOT give opinions, predictions, or sentiment; avoid Yes/No/likelihood language."
    )


def semantic_prompt(ir: int) -> str:
    return (
        base_simple_prompt(ir)
        + "\n\nYou are the Semantic Agent. Provide only factual observations about topical/semantic aspects of the concepts and subgraph (names, overlaps, co-occurring nodes/edges)."
        + " Do NOT give opinions, predictions, or sentiment; avoid Yes/No/likelihood language."
    )


def trend_prompt(ir: int) -> str:
    return (
        base_simple_prompt(ir)
        + "\n\nYou are the Trend Agent. Provide only factual observations about temporal signals (year-by-year citations, recency, growth/decline) from the subgraph."
        + " Do NOT give opinions, predictions, or sentiment; avoid Yes/No/likelihood language."
    )


def evaluator_prompt(ir: int) -> str:
    return f"""{base_simple_prompt(ir)}

You are a scientific impact evaluator expert. You receive the Concept Pair + Subgraph and, in addition, summaries from Structure/Semantic/Trend agents. Treat agent outputs as references only.

Return JSON ONLY:
{{
  "reasoning": "<your reasoning about why this pair will or won't have high impact>",
  "prediction": "Yes" or "No",
  "score": <float 0-1 with exactly two decimal places>
}}
- Keep reasoning under 120 words.
- Do NOT include text outside the JSON.
"""


def build_query_block(concept1: str, concept2: str, subgraph: Dict[str, Any]) -> str:
    return (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )


def call_llm(client: OpenAI, model: str, ir: int, concept1: str, concept2: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
    query_block = build_query_block(concept1, concept2, subgraph)
    temp = 1.0 if "gpt-5" in model.lower() else 0.0

    def run_agent(sys_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query_block},
        ]
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=temp)
        return resp.choices[0].message.content or ""

    structure_out = run_agent(structure_prompt(ir))
    semantic_out = run_agent(semantic_prompt(ir))
    trend_out = run_agent(trend_prompt(ir))

    eval_msgs = [
        {"role": "system", "content": evaluator_prompt(ir)},
        {
            "role": "user",
            "content": f"""{query_block}

[AGENT OUTPUTS]
Structure Agent: {structure_out}
Semantic Agent: {semantic_out}
Trend Agent: {trend_out}
""",
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=eval_msgs,
        temperature=temp,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    result = json.loads(content)
    pred_str = str(result.get("prediction", "")).strip().lower()
    score_val = float(result.get("score", 0))
    if pred_str == "yes":
        pred_int = 1
    elif pred_str == "no":
        pred_int = 0
    else:
        pred_int = 1 if score_val >= 0.5 else 0
    return {
        "concept1": concept1,
        "concept2": concept2,
        "reasoning": result.get("reasoning", ""),
        "prediction": result.get("prediction", ""),
        "prediction_int": pred_int,
        "score": round(score_val, 2),
    }


# ========== Scoring / Threshold ==========
def save_candidates_cache(path: str, candidates: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cand in candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")


def load_candidates_cache(path: str) -> List[Dict[str, Any]]:
    cands = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cands.append(json.loads(line))
            except Exception:
                continue
    return cands


def generate_scores(
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
        res = call_llm(client, model, ir, row["concept1"], row["concept2"], subg)
        res["ground_truth_label"] = int(row.get("label", 0))
        res["total_citations"] = int(row.get("total_citations", 0))
        res["index"] = idx
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, (idx, row)) for idx, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM scoring"):
            outputs.append(fut.result())
    return outputs


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    y_true = [int(r.get("ground_truth_label", 0)) for r in results]
    y_pred = [int(r.get("prediction_int", 0)) for r in results]
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1_pos = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, (tn + fp))
    f1_neg = 0 if (prec_neg + rec_neg) == 0 else 2 * prec_neg * rec_neg / (prec_neg + rec_neg)
    macro_f1 = (f1_pos + f1_neg) / 2
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(results: List[Dict[str, Any]], step: float = 0.01, metric: str = "accuracy") -> Tuple[float, Dict[str, float]]:
    y_true = [int(r.get("ground_truth_label", 0)) for r in results]
    scores = [float(r.get("score", 0)) for r in results]
    best_t = 0.5
    best_val = -1.0
    best_metrics = {}
    t = 0.0
    while t <= 1.0001:
        preds = [1 if s >= t else 0 for s in scores]
        tmp = compute_metrics([{"ground_truth_label": gt, "prediction_int": p} for gt, p in zip(y_true, preds)])
        val = tmp["accuracy"] if metric == "accuracy" else tmp["macro_f1"]
        if val > best_val:
            best_val = val
            best_t = t
            best_metrics = tmp
        t += step
    return best_t, best_metrics


# ========== CLI ==========
def parse_args():
    p = argparse.ArgumentParser(description="Structured graph-aware prediction with thresholded scores.")
    p.add_argument("--train_csv", required=True, help="Training CSV (concept1, concept2, label[, total_citations]) for scoring.")
    p.add_argument("--test_csv", required=True, help="Test CSV.")
    p.add_argument("--val_csv_for_threshold", help="Optional CSV to compute best threshold on scores (concept1, concept2, label).")
    p.add_argument("--ir", type=int, required=True, help="IR threshold.")
    p.add_argument("--concept_graph", required=True, help="Path to concept_graph.csv (used for subgraph extraction).")
    p.add_argument("--model", default="gpt-5-mini", help="OpenAI model.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--max_workers", type=int, default=100, help="ThreadPoolExecutor workers.")
    p.add_argument("--candidates_cache", type=str, default=None, help="Cache path for train scores (JSONL).")
    p.add_argument("--reuse_candidates_cache", action="store_true", help="Reuse existing candidates cache.")
    p.add_argument("--apply_threshold", type=float, default=None, help="Force a fixed threshold on scores at test time.")
    p.add_argument("--threshold_grid_step", type=float, default=0.01, help="Grid step for threshold search.")
    p.add_argument("--output", default="results/structured/test_predictions.jsonl", help="Output JSONL.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    client = OpenAI(api_key=api_key)

    # Load data
    train_df = pd.read_csv(args.train_csv)
    train_rows = train_df.to_dict(orient="records")
    if args.val_csv_for_threshold:
        val_df_thr = pd.read_csv(args.val_csv_for_threshold)
        val_rows_thr = val_df_thr.to_dict(orient="records")
    else:
        val_rows_thr = None
    test_df = pd.read_csv(args.test_csv)
    test_rows = test_df.to_dict(orient="records")

    # Graph for subgraph extraction
    G, snapshot_year = load_concept_graph_csv(args.concept_graph)

    def subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G, a, b, snapshot_year, max_k=4, max_p=8)

    # Train scoring (cacheable)
    if args.reuse_candidates_cache and args.candidates_cache and os.path.exists(args.candidates_cache):
        print(f"[CACHE] Reusing train scores from {args.candidates_cache}")
        train_scored = load_candidates_cache(args.candidates_cache)
    else:
        train_scored = generate_scores(client, args.model, args.ir, train_rows, subgraph_fn, args.max_workers)
        if args.candidates_cache:
            save_candidates_cache(args.candidates_cache, train_scored)
            print(f"[CACHE] Saved train scores to {args.candidates_cache}")

    # Threshold search
    best_threshold = args.apply_threshold if args.apply_threshold is not None else None
    if best_threshold is None:
        if val_rows_thr is not None:
            if os.path.abspath(args.val_csv_for_threshold) == os.path.abspath(args.train_csv):
                thr_src = train_scored  # reuse train scores, no extra LLM calls
            else:
                val_scored = generate_scores(client, args.model, args.ir, val_rows_thr, subgraph_fn, args.max_workers)
                thr_src = val_scored
        else:
            thr_src = train_scored
        best_threshold, thr_metrics = find_best_threshold(thr_src, step=args.threshold_grid_step, metric="accuracy")
        print(f"[THRESHOLD] best={best_threshold:.3f} (acc={thr_metrics['accuracy']:.4f}, macro_f1={thr_metrics['macro_f1']:.4f})")

    # Test scoring
    test_scored = generate_scores(client, args.model, args.ir, test_rows, subgraph_fn, args.max_workers)
    for r in test_scored:
        s = float(r.get("score", 0))
        r["prediction_int"] = 1 if s >= best_threshold else 0
        r["prediction"] = "Yes" if r["prediction_int"] == 1 else "No"
    metrics = compute_metrics(test_scored)
    print("[METRICS]")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in test_scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] saved to {args.output}")


if __name__ == "__main__":
    main()
