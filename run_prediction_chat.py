#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

import extract_path

# Load environment variables
load_dotenv()

# =========================================================
# Helpers
# =========================================================
def parse_llm_json(text: str) -> Dict[str, Any]:
    if not text: return {}
    t = text.strip()
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1:
        try: return json.loads(t[l:r+1])
        except: pass
    return {"reasoning": t}

def parse_final_verdict_from_chat(text: str) -> Tuple[str, str]:
    """
    Parse Turn-3 JSON output from the chat analysis agent.
    Returns: (final_verdict, integrated_reasoning_text)
    """
    parsed = parse_llm_json(text or "")
    verdict = (parsed.get("final_verdict") if isinstance(parsed, dict) else None)
    reasoning = ""
    if isinstance(parsed, dict):
        reasoning = (
            parsed.get("integrated_reasoning")
            or parsed.get("impact_reasoning")
            or parsed.get("analysis")
            or parsed.get("reasoning")
            or (text or "")
        )
    else:
        reasoning = text or ""

    if not verdict:
        v_lower = (text or "").lower()
        verdict = "Yes" if "yes" in v_lower else "No"
    if verdict not in ("Yes", "No"):
        verdict = "Yes" if str(verdict).strip().lower() == "yes" else "No"
    return verdict, (reasoning or "").strip()

def compute_binary_metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# =========================================================
# Subgraph Payload Generators
# =========================================================
def get_masked_structural_payload(G_sub: nx.Graph, u: str, v: str) -> Dict[str, Any]:
    """Turn 1: Masked IDs + total citation weights only"""
    nodes = sorted(list(G_sub.nodes()))
    mapping = {u: "node_1", v: "node_2"}
    idx = 3
    for n in nodes:
        if n not in mapping:
            mapping[n] = f"node_{idx}"
            idx += 1

    edge_list = []
    for a, b in sorted(G_sub.edges()):
        yd = G_sub.get_edge_data(a, b).get("citations_by_year", {})
        weight = sum(int(v) for v in yd.values())
        edge_list.append([mapping[a], mapping[b], int(weight)])
    return {"query_pair": ["node_1", "node_2"], "edge_list": edge_list}

def get_unmasked_semantic_payload(G_sub: nx.Graph, u: str, v: str) -> Dict[str, Any]:
    """Turn 2: Actual names, no weights"""
    edge_list = [[a, b] for a, b in sorted(G_sub.edges())]
    return {"query_pair": [u, v], "edge_list": edge_list}

def get_temporal_trend_payload(G_sub: nx.Graph, u: str, v: str) -> Dict[str, Any]:
    """Turn 3: Actual names + year-by-year trends"""
    edge_list = []
    for a, b in sorted(G_sub.edges()):
        yd = G_sub.get_edge_data(a, b).get("citations_by_year", {})
        sorted_yd = {int(k): int(v) for k, v in sorted(yd.items())}
        edge_list.append([a, b, sorted_yd])
    return {"query_pair": [u, v], "edge_list": edge_list}

# =========================================================
# Core Chat Pipeline
# =========================================================
def call_chat_agent(client, model, messages, max_tokens=None, retries=3):
    is_gpt5 = "gpt-5" in model.lower() or "o3" in model.lower()
    temp = 1.0 if is_gpt5 else 0.2
    if max_tokens is None:
        actual_max_tokens = 16000 if is_gpt5 else 2048
    else:
        actual_max_tokens = max_tokens

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
        model=model,
                messages=messages, 
                temperature=temp, 
                max_completion_tokens=actual_max_tokens
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            else:
                finish_reason = resp.choices[0].finish_reason
                print(f"[WARNING] Empty content (Attempt {attempt+1}/{retries}). Reason: {finish_reason}, Model: {model}")
                if attempt < retries - 1: time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[ERROR] API Call (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(2 * (attempt + 1))
    return ""

def run_one_pair_chat(client, model, prompts, G, snapshot_year, u, v, ir, max_k, max_p):
    # 1) 서브그래프 추출
    paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    nodes = set()
    for p in (paths or []): nodes.update(p)
    if not nodes: return None
    G_sub = G.subgraph(nodes).copy()

    # 2) 메시지 초기화
    messages = [{"role": "system", "content": prompts["chat_analysis_unified"]}]
    
    # [Turn 1: Structure]
    p1 = get_masked_structural_payload(G_sub, u, v)
    inp1 = f"[TURN 1: STRUCTURE ANALYSIS]\nIR={ir}\nSNAPSHOT_YEAR={snapshot_year}\n[MASKED SUBGRAPH]\n{json.dumps(p1, ensure_ascii=False, indent=2)}"
    out1 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp1}])
    if out1:
        messages.append({"role": "user", "content": inp1})
        messages.append({"role": "assistant", "content": out1})

    # [Turn 2: Semantic]
    p2 = get_unmasked_semantic_payload(G_sub, u, v)
    inp2 = (
        "[TURN 2: SEMANTIC ANALYSIS]\n"
        "Now concept names are revealed. Mapping Turn 1 concepts to these names, provide semantic analysis.\n"
        "[UNMASKED SUBGRAPH]\n"
        f"{json.dumps(p2, ensure_ascii=False, indent=2)}"
    )
    out2 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp2}])
    if out2:
        messages.append({"role": "user", "content": inp2})
        messages.append({"role": "assistant", "content": out2})

    # [Turn 3: Trend & Integration]
    p3 = get_temporal_trend_payload(G_sub, u, v)
    inp3 = (
        "[TURN 3: TREND ANALYSIS & FINAL INTEGRATION]\n"
        "Now incorporate temporal data (year-by-year citations). Provide a final, refined integrated analysis.\n"
        "[TEMPORAL SUBGRAPH]\n"
        f"{json.dumps(p3, ensure_ascii=False, indent=2)}"
    )
    out3 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp3}])
    if out3:
        messages.append({"role": "user", "content": inp3})
        messages.append({"role": "assistant", "content": out3})

    final_analysis = (out3 or out2 or out1 or "").strip()
    return {"final_analysis": final_analysis, "history": messages, "subgraph": p3}

def run_impact_evaluator(client, model, prompts, u, v, snapshot_year, ir, final_analysis, subgraph_data=None, demonstrations=None):
    demos_block = ""
    if demonstrations:
        d_texts = []
        for i, d in enumerate(demonstrations, 1):
            pair = d.get("pair", {})
            d_texts.append(f"Example {i}:\n  Pair: {pair.get('concept1')} <-> {pair.get('concept2')}\n  Ground Truth: {d.get('ground_truth_label')}\n  Final Analysis: {d.get('final_impact_output', {}).get('impact_reasoning')[:1000]}...")
        demos_block = "\n\n[FEW-SHOT EXAMPLES]\n" + "\n\n".join(d_texts)

    subgraph_block = ""
    if subgraph_data:
        subgraph_block = f"\n\n[ORIGINAL SUBGRAPH DATA]\n{json.dumps(subgraph_data, ensure_ascii=False, indent=2)}"

    inp = f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n[INTEGRATED_ANALYSIS_OUTPUT]\n{final_analysis}{subgraph_block}{demos_block}"
    
    is_gpt5 = "gpt-5" in model.lower() or "o3" in model.lower()
    temp = 1.0 if is_gpt5 else 0.2
    actual_max_tokens = 16000 if is_gpt5 else 2048
    
    try:
        resp = client.chat.completions.create(
        model=model,
            messages=[{"role": "system", "content": prompts["impact_evaluator"]}, {"role": "user", "content": inp}],
            temperature=temp, 
            max_completion_tokens=actual_max_tokens
        )
        raw = resp.choices[0].message.content or ""
        parsed = parse_llm_json(raw)
        verdict = parsed.get("final_verdict")
        if not verdict:
            v_lower = raw.lower()
            verdict = "Yes" if "yes" in v_lower else "No"
        return {"impact_reasoning": parsed.get("impact_reasoning", raw), "final_verdict": verdict}
    except Exception as e:
        print(f"[ERROR] Evaluator Call: {e}")
        return {"impact_reasoning": f"Error: {e}", "final_verdict": "No"}

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, choices=[0, 1], required=True)
    parser.add_argument("--setting", type=str, choices=["transductive", "inductive"], required=True)
    parser.add_argument("--ir", type=int, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key_name", type=str, default="4")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_demos", type=int, default=0)
    parser.add_argument("--n_pairs", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv(f"OPENAI_API_KEY_{args.api_key_name}") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "dataset" / "kdd" / args.setting
    graph_path = dataset_dir / ("concept_graph_test.csv" if args.setting == "inductive" and args.mode == 1 else "concept_graph.csv")
    pair_path = dataset_dir / (f"train_data_{args.ir}.csv" if args.mode == 0 else f"test_data_{args.ir}.csv")

    def load_p(fn):
        with open(base_dir / "prompts" / fn, "r") as f: return f.read().replace("{IR}", str(args.ir))
    prompts = {
        "chat_analysis_unified": load_p("chat_analysis_unified.txt"),
        "impact_evaluator": load_p("impact_evaluator.txt"),
    }

    df_pairs = pd.read_csv(pair_path)
    pos, neg = df_pairs[df_pairs["label"]==1], df_pairs[df_pairs["label"]==0]
    n = min(len(pos), len(neg))
    if args.n_pairs is not None:
        n = min(n, args.n_pairs // 2) if args.n_pairs >= 2 else 1
    sampled = pd.concat([pos.sample(n, random_state=args.seed), neg.sample(n, random_state=args.seed)]).sample(frac=1, random_state=args.seed)

    df_g = pd.read_csv(graph_path)
    G = nx.Graph()
    for row in df_g.itertuples():
        u, v, y, c = str(row.concept1), str(row.concept2), int(row.year), int(row.citation_count)
        if not G.has_edge(u, v): G.add_edge(u, v, citations_by_year={})
        G[u][v]["citations_by_year"][y] = G[u][v]["citations_by_year"].get(y, 0) + c
    snapshot_year = df_g["year"].max()

    out_path = base_dir / "results" / "kdd" / args.setting / f"mode{args.mode}_ir{args.ir}_run_prediction_impactEval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_pairs = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("type") == "result": processed_pairs.add(tuple(sorted(d["pair"])))
                except: continue
        print(f"[INFO] Resuming: {len(processed_pairs)} pairs already processed.")
    else:
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(json.dumps({
                "type": "metadata",
                "model": args.model,
                "setting": args.setting,
                "ir": args.ir,
                "pipeline": "chat_3turn_analysis_then_impact_evaluator"
            }, ensure_ascii=False) + "\n")

    def process_pair_wrapper(row):
        u, v, gt = str(row.concept1), str(row.concept2), int(row.label)
        if tuple(sorted([u, v])) in processed_pairs: return "SKIP"
        try:
            res = run_one_pair_chat(client, args.model, prompts, G, snapshot_year, u, v, args.ir, 4, 8)
            if not res or not res.get("final_analysis"):
                # No path found or API failure
                sub_data = get_temporal_trend_payload(G.subgraph([u, v]), u, v) if u in G and v in G else None
                impact = run_impact_evaluator(
                    client, args.model, prompts,
                    u, v, snapshot_year, args.ir,
                    "No relevant subgraph found for analysis.",
                    subgraph_data=sub_data
                )
                return {
                    "type": "result",
                    "model": args.model,
                    "pair": [u, v],
                    "gt": gt,
                    "pred": 1 if impact["final_verdict"] == "Yes" else 0,
                    "analysis": "NO_SUBGRAPH",
                    "impact": impact
                }

            impact = run_impact_evaluator(
                client, args.model, prompts,
                u, v, snapshot_year, args.ir,
                res["final_analysis"],
                subgraph_data=res.get("subgraph")
            )
            return {
                "type": "result",
                "model": args.model,
                "pair": [u, v],
                "gt": gt,
                "pred": 1 if impact["final_verdict"] == "Yes" else 0,
                "analysis": res["final_analysis"],
                "impact": impact
            }
        except Exception as e:
            print(f"[ERROR] Pair {u}-{v}: {str(e)}")
            return None

    print(f"Starting Prediction Mode {args.mode} with {args.model} (Batch: {args.batch_size})...")
    all_y_true, all_y_pred = [], []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("type") == "result":
                        all_y_true.append(d["gt"]); all_y_pred.append(d["pred"])
                except: continue

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = [executor.submit(process_pair_wrapper, row) for _, row in sampled.iterrows()]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            r = f.result()
            if r and r != "SKIP":
                with open(out_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                all_y_true.append(r["gt"]); all_y_pred.append(r["pred"])

    if all_y_true:
        m = compute_binary_metrics(all_y_true, all_y_pred)
        print(f"\nFinal Results: Acc={m['accuracy']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}")
        with open(out_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps({"type": "metrics", **m}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
