#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic retrieval → cluster-level best 1-shot demo selection → test-time 1-shot prompting.

Pipeline steps:
1) Load validation clustering output (from cap_constrained_clustering.py).
2) Use train assignments in that JSON to create candidate demos (LLM predictions on train pairs).
3) Evaluate candidates on their validation cluster with 1-shot prompting; pick the best demo per cluster.
4) For test pairs, embed and route to the nearest validation cluster (centroid cosine), then predict with the cluster's best demo.

Assumptions:
- Clustering JSON includes `train_assignments` with train rows and cluster ids (produced by cap_constrained_clustering.py).
- Validation clustering was built with the same embedding model/sep_token you supply here.
- LLM outputs JSON with fields {reasoning, prediction} where prediction is "Yes"/"No".
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from cap_constrained_clustering import build_texts, embed_texts

load_dotenv()

# =============================
# Prompt helpers
# =============================
def base_prompt(ir: int) -> str:
    return f"""You are a scientific impact prediction expert.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

[PREDICTION RULES]
Predict "Yes" or "No" based on:
- Semantic relationship: Are these concepts related? Could they be combined in research?
- Research trends: Are these areas active? Is there growing interest?
- Potential impact: If combined, would this research likely receive significant citations?

[OUTPUT]
Return a JSON object with:
{{
  "reasoning": "<why this pair will/won't be high impact>",
  "prediction": "Yes" or "No"
}}
The "prediction" field MUST be exactly one of: "Yes", "No".
Do NOT include additional text outside the JSON object.
"""


def build_messages(ir: int, concept1: str, concept2: str, demo: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    system = {"role": "system", "content": base_prompt(ir)}
    user_parts = []
    if demo:
        demo_txt = json.dumps(
            {
                "concept1": demo["concept1"],
                "concept2": demo["concept2"],
                "ground_truth_label": demo.get("ground_truth_label"),
                "prediction": demo.get("prediction"),
                "reasoning": demo.get("reasoning"),
            },
            ensure_ascii=False,
            indent=2,
        )
        user_parts.append("[DEMONSTRATION]\nUse the following example as 1-shot guidance:\n" + demo_txt)
        user_parts.append(
            "You must follow the exact same reasoning procedure and structure as in the demonstration before giving the final Yes/No JSON."
        )
    user_parts.append(f"[QUERY PAIR]\nConcept X: {concept1}\nConcept Y: {concept2}\nPlease predict Yes/No with JSON output.")
    return [system, {"role": "user", "content": "\n\n".join(user_parts)}]


def call_llm(client: OpenAI, model: str, ir: int, concept1: str, concept2: str, demo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    messages = build_messages(ir, concept1, concept2, demo)
    temperature = 1.0 if "gpt-5" in model.lower() else 0.0
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    result = json.loads(content)
    pred_str = str(result.get("prediction", "")).strip().lower()
    if pred_str == "yes":
        pred_int = 1
    elif pred_str == "no":
        pred_int = 0
    else:
        reasoning = (result.get("reasoning") or "").lower()
        pred_int = 1 if "yes" in reasoning or "high impact" in reasoning else 0
    return {
        "concept1": concept1,
        "concept2": concept2,
        "reasoning": result.get("reasoning", ""),
        "prediction": result.get("prediction", ""),
        "prediction_int": pred_int,
    }


# =============================
# Clustering payload helpers
# =============================
def load_clustering(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidates_cache(path: str, cluster_candidates: Dict[int, List[Dict[str, Any]]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid, lst in cluster_candidates.items():
            for cand in lst:
                f.write(json.dumps({"cluster_id": cid, "candidate": cand}, ensure_ascii=False) + "\n")


def load_candidates_cache(path: str) -> Dict[int, List[Dict[str, Any]]]:
    cluster_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cid = data.get("cluster_id")
            cand = data.get("candidate")
            if cid is None or cand is None:
                continue
            cluster_candidates[int(cid)].append(cand)
    return cluster_candidates


def extract_clusters(clustering: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, List[int]], Dict[int, List[int]], List[Dict[str, Any]]]:
    clusters = clustering["clusters"]
    centroids = np.stack([np.array(c["centroid"], dtype=float) for c in clusters], axis=0)
    val_cluster_id = clustering["val_cluster_id"]
    val_rows = clustering["validation_rows"]

    val_map: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(val_cluster_id):
        val_map[cid].append(idx)

    train_map: Dict[int, List[int]] = defaultdict(list)
    if "train_assignments" in clustering and clustering["train_assignments"]:
        train_assign = clustering["train_assignments"]
        cluster_ids = train_assign["cluster_id"]
        for idx, cid in enumerate(cluster_ids):
            train_map[cid].append(idx)

    return centroids, val_map, train_map, val_rows


# =============================
# Candidate demo generation & evaluation
# =============================
def generate_candidate_demos(
    client: OpenAI,
    model: str,
    ir: int,
    train_rows: List[Dict[str, Any]],
    cluster_ids: List[int],
    max_workers: int,
) -> Dict[int, List[Dict[str, Any]]]:
    cluster_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def task(idx_row: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, row = idx_row
        cid = cluster_ids[idx]
        res = call_llm(client, model, ir, row["concept1"], row["concept2"], demo=None)
        res["ground_truth_label"] = int(row.get("label", 0))
        res["train_index"] = idx
        return cid, res

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, (idx, row)) for idx, row in enumerate(train_rows)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Candidate demos (train)"):
                cid, res = fut.result()
                cluster_candidates[cid].append(res)
    else:
        for idx, row in enumerate(tqdm(train_rows, desc="Candidate demos (train)")):
            cid, res = task((idx, row))
            cluster_candidates[cid].append(res)
    return cluster_candidates


def evaluate_candidates_on_validation(
    client: OpenAI,
    model: str,
    ir: int,
    candidates: Dict[int, List[Dict[str, Any]]],
    val_rows: List[Dict[str, Any]],
    val_map: Dict[int, List[int]],
    max_workers: int,
    max_val_per_cluster: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    best_demo_per_cluster: Dict[int, Dict[str, Any]] = {}
    for cid, val_indices in tqdm(list(val_map.items()), desc="Cluster eval (val)"):
        cand_list = candidates.get(cid, [])
        if not cand_list or not val_indices:
            continue
        if max_val_per_cluster:
            val_indices = val_indices[:max_val_per_cluster]
        # Evaluate each candidate on this cluster's validation rows
        scores = []
        def eval_cand(cand: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
            correct = 0
            for vid in val_indices:
                vrow = val_rows[vid]
                pred = call_llm(client, model, ir, vrow["concept1"], vrow["concept2"], demo=cand)
                gt = int(vrow.get("label", 0))
                correct += 1 if pred["prediction_int"] == gt else 0
            acc = correct / max(1, len(val_indices))
            return acc, cand

        # Parallel over candidates for this cluster
        if cand_list:
            if max_workers and max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(eval_cand, cand) for cand in cand_list]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Candidates for cluster {cid}", leave=False):
                        scores.append(fut.result())
            else:
                for cand in tqdm(cand_list, desc=f"Candidates for cluster {cid}", leave=False):
                    scores.append(eval_cand(cand))
        scores.sort(key=lambda x: x[0], reverse=True)
        best_demo_per_cluster[cid] = {
            "best_demo": scores[0][1],
            "accuracy": scores[0][0],
            "num_validation": len(val_indices),
        }
    return best_demo_per_cluster


# =============================
# Test-time inference
# =============================
def assign_pairs_to_clusters(
    pair_emb: np.ndarray,
    centroids: np.ndarray,
    allowed_cids: List[int],
) -> List[int]:
    sims = np.dot(pair_emb, centroids.T)
    cluster_ids = []
    for row in sims:
        # restrict to allowed cluster ids
        best_cid = allowed_cids[0]
        best_sim = -1e9
        for cid in allowed_cids:
            if row[cid] > best_sim:
                best_sim = row[cid]
                best_cid = cid
        cluster_ids.append(best_cid)
    return cluster_ids


def predict_test(
    client: OpenAI,
    model: str,
    ir: int,
    test_rows: List[Dict[str, Any]],
    test_cluster_ids: List[int],
    best_demos: Dict[int, Dict[str, Any]],
    max_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    results = []

    def task(row: Dict[str, Any], cid: int) -> Dict[str, Any]:
        demo_entry = best_demos.get(cid, {})
        demo = demo_entry.get("best_demo")
        pred = call_llm(client, model, ir, row["concept1"], row["concept2"], demo=demo)
        pred["cluster_id"] = cid
        pred["ground_truth_label"] = int(row.get("label", 0))
        pred["total_citations"] = int(row.get("total_citations", 0))
        return pred

    pairs = list(zip(test_rows, test_cluster_ids))
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, row, cid) for row, cid in pairs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Test inference"):
                results.append(fut.result())
    else:
        for row, cid in tqdm(pairs, desc="Test inference"):
            results.append(task(row, cid))

    cluster_usage: Dict[int, int] = {}
    for r in results:
        cid = r["cluster_id"]
        cluster_usage[cid] = cluster_usage.get(cid, 0) + 1

    return results, cluster_usage


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, macro F1 from results containing prediction_int and ground_truth_label."""
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
    # For macro F1, compute also for negative class
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


# =============================
# CLI
# =============================
def parse_args():
    p = argparse.ArgumentParser(description="Cluster-level 1-shot demo selection and test inference.")
    p.add_argument("--clustering_json", required=True, help="Output JSON from cap_constrained_clustering.py (must include train_assignments).")
    p.add_argument("--test_csv", required=True, help="Test CSV (concept1, concept2[, label, total_citations]).")
    p.add_argument("--ir", type=int, required=True, help="IR threshold.")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for LLM predictions.")
    p.add_argument("--api_key", default=None, help="OpenAI API key (overrides env).")
    p.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2", help="SBERT model used for clustering (must match).")
    p.add_argument("--sep_token", default="[SEP]", help="Separator used for SBERT texts.")
    p.add_argument("--batch_size", type=int, default=100, help="Embedding batch size.")
    p.add_argument("--output", default="results/demo_pipeline/test_predictions.jsonl", help="Output JSONL for test predictions.")
    p.add_argument("--max_workers", type=int, default=8, help="ThreadPoolExecutor workers for LLM calls.")
    p.add_argument("--max_candidates_per_cluster", type=int, default=50, help="Limit number of train candidates per cluster (highest centroid similarity).")
    p.add_argument("--max_val_per_cluster", type=int, default=None, help="Optionally subsample validation rows per cluster for faster scoring.")
    p.add_argument("--candidates_cache", type=str, default=None, help="Path to cache candidate demos (JSONL).")
    p.add_argument("--reuse_candidates_cache", action="store_true", help="Reuse existing candidates cache instead of regenerating.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found (set --api_key or OPENAI_API_KEY).")
    client = OpenAI(api_key=api_key)

    clustering = load_clustering(args.clustering_json)
    centroids, val_map, train_map, val_rows = extract_clusters(clustering)

    if "train_assignments" not in clustering or not clustering["train_assignments"]:
        raise ValueError("clustering_json must include train_assignments to build candidate demos.")
    train_assign = clustering["train_assignments"]
    train_rows = train_assign["rows"]
    train_cluster_ids = train_assign["cluster_id"]
    train_sims = train_assign.get("similarity", [0.0] * len(train_rows))

    # If reuse cache, skip filtering/generation and load existing candidates
    if args.reuse_candidates_cache and args.candidates_cache and os.path.exists(args.candidates_cache):
        print(f"[CACHE] Reusing candidate demos from {args.candidates_cache}")
        candidates = load_candidates_cache(args.candidates_cache)
    else:
        # Limit candidates per cluster by highest similarity to centroid
        if args.max_candidates_per_cluster:
            cluster_to_indices = defaultdict(list)
            for idx, cid in enumerate(train_cluster_ids):
                sim = float(train_sims[idx]) if idx < len(train_sims) else 0.0
                cluster_to_indices[cid].append((sim, idx))
            selected_indices = []
            for cid, pairs in cluster_to_indices.items():
                pairs.sort(key=lambda x: x[0], reverse=True)
                keep = pairs[: args.max_candidates_per_cluster]
                selected_indices.extend([idx for _, idx in keep])
            selected_indices = sorted(selected_indices)
            train_rows = [train_rows[i] for i in selected_indices]
            train_cluster_ids = [train_cluster_ids[i] for i in selected_indices]
            train_sims = [train_sims[i] for i in selected_indices]
            print(f"[FILTER] limited train candidates per cluster to top {args.max_candidates_per_cluster}; total candidates: {len(train_rows)}")

        candidates = generate_candidate_demos(client, args.model, args.ir, train_rows, train_cluster_ids, args.max_workers)
        if args.candidates_cache:
            save_candidates_cache(args.candidates_cache, candidates)
            print(f"[CACHE] Saved candidate demos to {args.candidates_cache}")

    # Step 1: candidate demos
    candidates = generate_candidate_demos(client, args.model, args.ir, train_rows, train_cluster_ids, args.max_workers)

    # Step 2: evaluate on validation
    best_demos = evaluate_candidates_on_validation(
        client, args.model, args.ir, candidates, val_rows, val_map, args.max_workers, args.max_val_per_cluster
    )

    # Step 3: test embedding and cluster routing (only clusters with at least one candidate)
    allowed_cids = [cid for cid in best_demos.keys()]
    if not allowed_cids:
        raise ValueError("No clusters have candidate demos; cannot proceed to test inference.")
    test_df = pd.read_csv(args.test_csv)
    test_texts = build_texts(test_df, sep=args.sep_token)
    test_emb = embed_texts(args.embedding_model, test_texts, batch_size=args.batch_size, normalize=True)
    test_cluster_ids = assign_pairs_to_clusters(test_emb, centroids, allowed_cids)

    # Step 4: test inference with best demo
    test_rows = test_df.to_dict(orient="records")
    test_results, cluster_usage = predict_test(client, args.model, args.ir, test_rows, test_cluster_ids, best_demos, args.max_workers)

    # Metrics
    metrics = compute_metrics(test_results)
    print("[METRICS]")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print("[CLUSTER USAGE]")
    for cid, count in sorted(cluster_usage.items()):
        print(f"Cluster {cid}: {count} test pairs")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in test_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Saved test predictions to {args.output}")


if __name__ == "__main__":
    main()
