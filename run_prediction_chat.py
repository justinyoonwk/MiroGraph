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
import httpx
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

def parse_p_yes(text: str) -> Optional[float]:
    """
    Best-effort parse of p_yes from a JSON-only response.
    Accepts keys: p_yes, yes_probability, prob_yes, score.
    """
    parsed = parse_llm_json(text or "")
    val = None
    if isinstance(parsed, dict):
        for k in ("p_yes", "yes_probability", "prob_yes", "score"):
            if k in parsed:
                val = parsed.get(k)
                break
    try:
        if val is None:
            return None
        p = float(val)
        # clamp defensively
        if p < 0: p = 0.0
        if p > 1: p = 1.0
        return p
    except Exception:
        return None

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

def majority_vote(preds: List[int]) -> int:
    """Return 1 if >=2 of 3 predict 1, else 0."""
    return 1 if sum(int(x) for x in preds) >= 2 else 0

def find_best_threshold_for_accuracy(y_true: List[int], p_yes: List[float]) -> Tuple[float, float]:
    """
    Finds threshold t in [0,1] maximizing accuracy for rule: pred = 1 if p_yes >= t else 0.
    Uses candidate thresholds at 0, 1, and midpoints between unique sorted probabilities.
    Tie-break: choose threshold closest to 0.5 (more neutral).
    Returns (best_t, best_acc).
    """
    if not y_true or not p_yes or len(y_true) != len(p_yes):
        return 0.5, 0.0
    probs = sorted(set(float(x) for x in p_yes))
    candidates = [0.0]
    for a, b in zip(probs, probs[1:]):
        candidates.append((a + b) / 2.0)
    candidates.append(1.0)

    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        pred = [1 if p >= t else 0 for p in p_yes]
        acc = sum(1 for a, b in zip(y_true, pred) if a == b) / max(1, len(y_true))
        if acc > best_acc:
            best_t, best_acc = t, acc
        elif acc == best_acc:
            if abs(t - 0.5) < abs(best_t - 0.5):
                best_t = t
    return float(best_t), float(best_acc)

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

    # 2) 메시지 초기화 (single agent multi-turn)
    messages = [{"role": "system", "content": prompts["chat_analysis_unified"]}]

    # [Turn 1: Structure] (FULL subgraph is provided ONLY here)
    payload = get_temporal_trend_payload(G_sub, u, v)
    inp1 = (
        "[TURN 1: STRUCTURE ANALYSIS]\n"
        f"IR={ir}\nSNAPSHOT_YEAR={snapshot_year}\n"
        "[FULL SUBGRAPH WITH YEAR-BY-YEAR CITATIONS]\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    out1 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp1}])
    if out1:
        messages.append({"role": "user", "content": inp1})
        messages.append({"role": "assistant", "content": out1})
    p_yes_1 = parse_p_yes(out1 or "")

    # [Turn 2: Semantic] (NO additional subgraph data is provided)
    inp2 = (
        "[TURN 2: SEMANTIC ANALYSIS]\n"
        "Using the concepts/edges you already saw in Turn 1 and the conversation history, provide semantic analysis.\n"
        "Do NOT ask for more graph data."
    )
    out2 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp2}])
    if out2:
        messages.append({"role": "user", "content": inp2})
        messages.append({"role": "assistant", "content": out2})
    p_yes_2 = parse_p_yes(out2 or "")

    # [Turn 3: Trend & Integration] (NO additional subgraph data is provided)
    inp3 = (
        "[TURN 3: TREND ANALYSIS & FINAL INTEGRATION]\n"
        "Using the year-by-year citations you already saw in Turn 1 and the conversation history, "
        "provide trend analysis and a final, refined integrated analysis.\n"
        "Do NOT ask for more graph data."
    )
    out3 = call_chat_agent(client, model, messages + [{"role": "user", "content": inp3}])
    if out3:
        messages.append({"role": "user", "content": inp3})
        messages.append({"role": "assistant", "content": out3})
    p_yes_3 = parse_p_yes(out3 or "")

    final_analysis = (out3 or out2 or out1 or "").strip()
    final_p_yes = p_yes_3 if p_yes_3 is not None else (p_yes_2 if p_yes_2 is not None else (p_yes_1 if p_yes_1 is not None else None))
    return {
        "final_analysis": final_analysis,
        "turn1_output": (out1 or "").strip(),
        "turn2_output": (out2 or "").strip(),
        "turn3_output": (out3 or "").strip(),
        "p_yes_turn1": p_yes_1,
        "p_yes_turn2": p_yes_2,
        "p_yes_turn3": p_yes_3,
        "p_yes": final_p_yes,
        "history": messages,
        "subgraph_payload": payload,
    }

def run_three_agents_scores(client, model, prompts, u: str, v: str, snapshot_year: int, ir: int, enclosing_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run 3 independent agents sequentially. Each agent receives the same enclosing subgraph and outputs JSON with a score."""
    target_block = f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n"
    subgraph_block = f"[ENCLOSING_SUBGRAPH]\n{json.dumps(enclosing_payload, ensure_ascii=False, indent=2)}"
    user_content = f"{target_block}\n{subgraph_block}"

    def call_one(system_prompt: str) -> str:
        return call_chat_agent(
            client,
            model,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        )

    out_s = call_one(prompts["agent_structure_score"])
    out_m = call_one(prompts["agent_semantic_score"])
    out_t = call_one(prompts["agent_trend_score"])

    s = parse_p_yes(out_s or "")
    m = parse_p_yes(out_m or "")
    t = parse_p_yes(out_t or "")

    return {
        "outputs": {"structure": out_s, "semantic": out_m, "trend": out_t},
        "scores": {"structure": s, "semantic": m, "trend": t},
    }

def run_impact_evaluator(
    client,
    model,
    prompts,
    u,
    v,
    snapshot_year,
    ir,
    integrated_analysis_reference: str,
    enclosing_subgraph_payload: Dict[str, Any],
    demonstrations=None,
):
    demos_block = ""
    if demonstrations:
        d_texts = []
        for i, d in enumerate(demonstrations, 1):
            pair = d.get("pair", {})
            d_texts.append(f"Example {i}:\n  Pair: {pair.get('concept1')} <-> {pair.get('concept2')}\n  Ground Truth: {d.get('ground_truth_label')}\n  Final Analysis: {d.get('final_impact_output', {}).get('impact_reasoning')[:1000]}...")
        demos_block = "\n\n[FEW-SHOT EXAMPLES]\n" + "\n\n".join(d_texts)

    enclosing_block = f"\n\n[ENCLOSING_SUBGRAPH]\n{json.dumps(enclosing_subgraph_payload or {}, ensure_ascii=False, indent=2)}"

    inp = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[INTEGRATED_ANALYSIS_REFERENCE]\n{integrated_analysis_reference}\n"
        f"{enclosing_block}"
        f"{demos_block}"
    )
    
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
        p_yes = parse_p_yes(raw)
        reasoning = raw
        if isinstance(parsed, dict):
            reasoning = parsed.get("impact_reasoning", raw)
        if p_yes is None:
            # fallback: if model accidentally emits verdict, map to 0/1
            v = (parsed.get("final_verdict") if isinstance(parsed, dict) else None)
            if not v:
                v_lower = raw.lower()
                v = "Yes" if "yes" in v_lower else "No"
            p_yes = 1.0 if str(v).strip().lower() == "yes" else 0.0
        return {"impact_reasoning": reasoning, "p_yes": float(p_yes)}
    except Exception as e:
        print(f"[ERROR] Evaluator Call: {e}")
        return {"impact_reasoning": f"Error: {e}", "p_yes": 0.0}

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
    parser.add_argument("--threshold", type=float, default=None, help="If set, skip train calibration and use this threshold for p_yes.")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip 3-turn chat analysis; run impact evaluator using ENCLOSING_SUBGRAPH only (empty integrated analysis).",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default="",
        help="Optional string appended to output filenames to avoid collisions (e.g., 'testonly').",
    )
    parser.add_argument(
        "--debug_save_turns",
        action="store_true",
        help="Save turn1/turn2/turn3 raw outputs into the result JSONL for debugging prompt adherence.",
    )
    parser.add_argument(
        "--use_impact_evaluator",
        action="store_true",
        help="If set, use the separate impact evaluator model call instead of using chat agent scores as p_yes.",
    )
    parser.add_argument(
        "--agents_majority_vote",
        action="store_true",
        help="Use 3 independent agents (structure/semantic/trend). Calibrate per-agent thresholds on train and decide by 2/3 majority vote on test.",
    )
    parser.add_argument(
        "--agent_thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds for structure,semantic,trend (e.g., '0.30,0.45,0.40'). If set, skip per-agent calibration.",
    )
    args = parser.parse_args()

    api_key = os.getenv(f"OPENAI_API_KEY_{args.api_key_name}") or os.getenv("OPENAI_API_KEY")
    # With high parallelism (e.g., batch_size=100), default HTTP client limits can stall the first responses.
    # Increase connection limits and set timeouts to avoid indefinite hangs.
    http_client = httpx.Client(
        timeout=httpx.Timeout(connect=60.0, read=120.0, write=120.0, pool=60.0),
        limits=httpx.Limits(
            max_connections=max(50, int(args.batch_size) * 2),
            max_keepalive_connections=max(20, int(args.batch_size)),
        ),
    )
    client = OpenAI(api_key=api_key, http_client=http_client)

    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "dataset" / "kdd" / args.setting
    graph_path = dataset_dir / ("concept_graph_test.csv" if args.setting == "inductive" and args.mode == 1 else "concept_graph.csv")
    pair_path = dataset_dir / (f"train_data_{args.ir}.csv" if args.mode == 0 else f"test_data_{args.ir}.csv")

    def load_p(fn):
        with open(base_dir / "prompts" / fn, "r") as f: return f.read().replace("{IR}", str(args.ir))
    prompts = {
        # single agent multi-turn (3 turns) that outputs JSON with score each turn
        "chat_analysis_unified": load_p("chat_analysis_unified.txt"),
        # 3 independent agents (each outputs JSON with score)
        "agent_structure_score": load_p("agent_structure_score.txt"),
        "agent_semantic_score": load_p("agent_semantic_score.txt"),
        "agent_trend_score": load_p("agent_trend_score.txt"),
        "impact_evaluator": load_p("impact_evaluator_probability.txt"),
    }

    # Load both train/test pairs (train used for threshold calibration)
    train_path = dataset_dir / f"train_data_{args.ir}.csv"
    test_path = dataset_dir / f"test_data_{args.ir}.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
        if args.n_pairs is None:
            return df
        n = min(int(args.n_pairs), len(df))
        return df.sample(n=n, random_state=args.seed)

    df_g = pd.read_csv(graph_path)
    G = nx.Graph()
    for row in df_g.itertuples():
        u, v, y, c = str(row.concept1), str(row.concept2), int(row.year), int(row.citation_count)
        if not G.has_edge(u, v): G.add_edge(u, v, citations_by_year={})
        G[u][v]["citations_by_year"][y] = G[u][v]["citations_by_year"].get(y, 0) + c
    snapshot_year = df_g["year"].max()

    out_dir = base_dir / "results" / "kdd" / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = "_evalonly" if args.eval_only else ""
    tag = f"_{args.out_tag}" if args.out_tag else ""
    train_out = out_dir / f"calib_ir{args.ir}_run_prediction_chat{variant}{tag}_prob_{args.model}_key{args.api_key_name}_seed{args.seed}.jsonl"
    test_out = out_dir / f"mode1_ir{args.ir}_run_prediction_chat{variant}{tag}_prob_{args.model}_key{args.api_key_name}_seed{args.seed}.jsonl"

    def load_existing_results(out_path: Path):
        processed = set()
        y_true, p_list = [], []
        agent_scores = {"structure": [], "semantic": [], "trend": []}
        has_agents = False
        if not out_path.exists():
            return processed, y_true, p_list, None
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "result":
                    processed.add(tuple(sorted(d["pair"])))
                    if "gt" in d and "p_yes" in d:
                        y_true.append(int(d["gt"]))
                        try:
                            p_list.append(float(d["p_yes"]))
                        except Exception:
                            pass
                    if isinstance(d.get("agent_scores"), dict):
                        has_agents = True
                        for k in ("structure", "semantic", "trend"):
                            try:
                                agent_scores[k].append(float(d["agent_scores"].get(k)))
                            except Exception:
                                agent_scores[k].append(float("nan"))
        return processed, y_true, p_list, (agent_scores if has_agents else None)

    def ensure_metadata(out_path: Path, split: str):
        if out_path.exists():
            return
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(json.dumps({
                "type": "metadata",
                "model": args.model,
                "setting": args.setting,
                "ir": args.ir,
                "split": split,
                "pipeline": (
                    "impact_evaluator_probability_enclosing_only"
                    if args.eval_only
                    else ("agents_majority_vote_3scores" if args.agents_majority_vote else "chat_3turn_analysis_then_impact_evaluator_probability")
                ),
                "out_tag": args.out_tag,
            }, ensure_ascii=False) + "\n")

    def process_pair_wrapper(row):
        u, v, gt = str(row.concept1), str(row.concept2), int(row.label)
        try:
            if args.eval_only:
                # Build enclosing subgraph payload only (same payload format as simple_prediction.py)
                paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=4, max_p=8)
                nodes = set()
                for p in (paths or []):
                    nodes.update(p)
                if nodes:
                    G_sub = G.subgraph(nodes).copy()
                    enclosing_payload = get_temporal_trend_payload(G_sub, u, v)
                else:
                    sub_g = G.subgraph([u, v]).copy() if (u in G and v in G) else nx.Graph()
                    if sub_g.number_of_nodes() == 0:
                        sub_g.add_node(u)
                        sub_g.add_node(v)
                    enclosing_payload = get_temporal_trend_payload(sub_g, u, v)

                impact = run_impact_evaluator(
                    client, args.model, prompts,
                    u, v, snapshot_year, args.ir,
                    integrated_analysis_reference="",
                    enclosing_subgraph_payload=enclosing_payload
                )
                return {
                    "type": "result",
                    "model": args.model,
                    "pair": [u, v],
                    "gt": gt,
                    "p_yes": impact["p_yes"],
                    "analysis": "EVAL_ONLY_ENCLOSING_SUBGRAPH",
                    "impact": impact
                }

            if args.agents_majority_vote:
                # Build enclosing subgraph payload (same payload format as simple_prediction.py)
                paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=4, max_p=8)
                nodes = set()
                for p in (paths or []):
                    nodes.update(p)
                if nodes:
                    G_sub = G.subgraph(nodes).copy()
                    enclosing_payload = get_temporal_trend_payload(G_sub, u, v)
                else:
                    sub_g = G.subgraph([u, v]).copy() if (u in G and v in G) else nx.Graph()
                    if sub_g.number_of_nodes() == 0:
                        sub_g.add_node(u)
                        sub_g.add_node(v)
                    enclosing_payload = get_temporal_trend_payload(sub_g, u, v)

                agents = run_three_agents_scores(client, args.model, prompts, u, v, int(snapshot_year), args.ir, enclosing_payload)
                s = float(agents["scores"].get("structure") or 0.0)
                m = float(agents["scores"].get("semantic") or 0.0)
                t = float(agents["scores"].get("trend") or 0.0)

                return {
                    "type": "result",
                    "model": args.model,
                    "pair": [u, v],
                    "gt": gt,
                    # store something compatible (not used for majority vote decision)
                    "p_yes": float(t),
                    "agent_scores": {"structure": s, "semantic": m, "trend": t},
                    "analysis": "AGENTS_3_SCORES",
                    "impact": {
                        "structure": {"raw": agents["outputs"]["structure"], "p_yes": s},
                        "semantic": {"raw": agents["outputs"]["semantic"], "p_yes": m},
                        "trend": {"raw": agents["outputs"]["trend"], "p_yes": t},
                    },
                }

            res = run_one_pair_chat(client, args.model, prompts, G, snapshot_year, u, v, args.ir, 4, 8)
            if not res or not res.get("final_analysis"):
                # No path found or API failure
                sub_g = G.subgraph([u, v]).copy() if (u in G and v in G) else nx.Graph()
                if sub_g.number_of_nodes() == 0:
                    sub_g.add_node(u)
                    sub_g.add_node(v)
                enclosing_payload = get_temporal_trend_payload(sub_g, u, v)
                if args.use_impact_evaluator:
                    impact = run_impact_evaluator(
                        client, args.model, prompts,
                        u, v, snapshot_year, args.ir,
                        "No relevant subgraph found for analysis.",
                        enclosing_subgraph_payload=enclosing_payload
                    )
                    p_yes = impact["p_yes"]
                else:
                    # minimal 1-turn scoring using the chat agent prompt
                    inp = (
                        "[TURN 1: STRUCTURE ANALYSIS]\n"
                        f"IR={args.ir}\nSNAPSHOT_YEAR={snapshot_year}\n"
                        "[FULL SUBGRAPH WITH YEAR-BY-YEAR CITATIONS]\n"
                        f"{json.dumps(enclosing_payload, ensure_ascii=False, indent=2)}"
                    )
                    out = call_chat_agent(
                        client,
                        args.model,
                        [{"role": "system", "content": prompts["chat_analysis_unified"]}, {"role": "user", "content": inp}],
                    )
                    p_yes = parse_p_yes(out) or 0.0
                    impact = {"impact_reasoning": out, "p_yes": float(p_yes)}
                return {
                    "type": "result",
                    "model": args.model,
                    "pair": [u, v],
                    "gt": gt,
                    "p_yes": float(p_yes),
                    "analysis": "NO_SUBGRAPH",
                    "impact": impact
                }

            if args.use_impact_evaluator:
                impact = run_impact_evaluator(
                    client, args.model, prompts,
                    u, v, snapshot_year, args.ir,
                    (res.get("turn3_output") or res.get("final_analysis") or ""),
                    enclosing_subgraph_payload=res.get("subgraph_payload") or {}
                )
                p_yes = float(impact["p_yes"])
            else:
                # use chat agent stage-3 score as final p_yes
                p_yes = res.get("p_yes")
                if p_yes is None:
                    p_yes = 0.0
                impact = {
                    "impact_reasoning": (res.get("turn3_output") or res.get("final_analysis") or ""),
                    "p_yes": float(p_yes),
                    "stage_p_yes": {
                        "turn1": res.get("p_yes_turn1"),
                        "turn2": res.get("p_yes_turn2"),
                        "turn3": res.get("p_yes_turn3"),
                    },
                }
            out_obj = {
                "type": "result",
                "model": args.model,
                "pair": [u, v],
                "gt": gt,
                "p_yes": float(p_yes),
                "analysis": res["final_analysis"],
                "impact": impact
            }
            if args.debug_save_turns:
                out_obj["turn1_output"] = res.get("turn1_output", "")
                out_obj["turn2_output"] = res.get("turn2_output", "")
                out_obj["turn3_output"] = res.get("turn3_output", "")
            return out_obj
        except Exception as e:
            print(f"[ERROR] Pair {u}-{v}: {str(e)}")
            return None

    def run_split(df: pd.DataFrame, out_path: Path, split_name: str):
        ensure_metadata(out_path, split=split_name)
        processed_pairs, y_true_existing, p_existing, agent_existing = load_existing_results(out_path)
        if processed_pairs:
            print(f"[INFO] Resuming {split_name}: {len(processed_pairs)} pairs already processed.")
        df_use = maybe_sample(df).sample(frac=1, random_state=args.seed)

        y_true, p_yes_list = list(y_true_existing), list(p_existing)
        agent_scores = agent_existing if agent_existing is not None else {"structure": [], "semantic": [], "trend": []}

        def wrapper_with_skip(row):
            u, v = str(row.concept1), str(row.concept2)
            if tuple(sorted([u, v])) in processed_pairs:
                return "SKIP"
            return process_pair_wrapper(row)

        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = [executor.submit(wrapper_with_skip, row) for _, row in df_use.iterrows()]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
                r = f.result()
                if r and r != "SKIP":
                    with open(out_path, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    y_true.append(int(r["gt"]))
                    p_yes_list.append(float(r["p_yes"]))
                    if args.agents_majority_vote and isinstance(r.get("agent_scores"), dict):
                        for k in ("structure", "semantic", "trend"):
                            agent_scores[k].append(float(r["agent_scores"].get(k, 0.0)))
        return y_true, p_yes_list, (agent_scores if args.agents_majority_vote else None)

    print(f"Starting run_prediction_chat (probability) with {args.model} (Batch: {args.batch_size})...")

    # 1) Train calibration (unless threshold(s) explicitly provided)
    agent_thresholds = None
    if args.agents_majority_vote:
        if args.agent_thresholds:
            parts = [p.strip() for p in args.agent_thresholds.split(",")]
            if len(parts) != 3:
                raise ValueError("--agent_thresholds must be 3 comma-separated floats: structure,semantic,trend")
            agent_thresholds = {"structure": float(parts[0]), "semantic": float(parts[1]), "trend": float(parts[2])}
            print(f"[CALIB] Using provided agent thresholds: {agent_thresholds}")
        else:
            y_train, _p_train, agent_train = run_split(df_train, train_out, "train")
            assert agent_train is not None, "agent_train scores missing"
            agent_thresholds = {}
            agent_best_acc = {}
            for k in ("structure", "semantic", "trend"):
                t_k, acc_k = find_best_threshold_for_accuracy(y_train, agent_train[k])
                agent_thresholds[k] = float(t_k)
                agent_best_acc[k] = float(acc_k)
            print(f"[CALIB] Best per-agent thresholds on train (max-acc): {agent_thresholds} (n={len(y_train)})")
            with open(train_out, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps({"type": "agent_thresholds", "thresholds": agent_thresholds, "best_accuracy": agent_best_acc}, ensure_ascii=False) + "\n")
        # (note) args.threshold is not used in this scheme, but keep a default for compatibility
        threshold = float(args.threshold) if args.threshold is not None else 0.5
        if args.threshold is not None:
            print(f"[CALIB] (note) --threshold={threshold:.4f} is ignored for majority-vote decisions.")
    else:
        if args.threshold is None:
            y_train, p_train, _ = run_split(df_train, train_out, "train")
            best_t, best_acc = find_best_threshold_for_accuracy(y_train, p_train)
            threshold = best_t
            print(f"[CALIB] Best threshold on train (max-acc): t={threshold:.4f} (acc={best_acc:.4f}, n={len(y_train)})")
            with open(train_out, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps({"type": "threshold", "best_threshold": threshold, "best_accuracy": best_acc}, ensure_ascii=False) + "\n")
        else:
            threshold = float(args.threshold)
            print(f"[CALIB] Using provided threshold: t={threshold:.4f}")

    # If mode==0, stop after calibration
    if args.mode == 0:
        return

    # 2) Test evaluation
    y_test, p_test, agent_test = run_split(df_test, test_out, "test")
    if args.agents_majority_vote:
        assert agent_test is not None and agent_thresholds is not None
        y_pred = []
        for s, m_, t in zip(agent_test["structure"], agent_test["semantic"], agent_test["trend"]):
            votes = [
                1 if float(s) >= agent_thresholds["structure"] else 0,
                1 if float(m_) >= agent_thresholds["semantic"] else 0,
                1 if float(t) >= agent_thresholds["trend"] else 0,
            ]
            y_pred.append(majority_vote(votes))
        m = compute_binary_metrics(y_test, y_pred)
        print(f"\n[Test] MajorityVote(2/3) thresholds={agent_thresholds} => Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f} (n={len(y_test)})")
        with open(test_out, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps({"type": "metrics", "scheme": "agents_majority_vote", "agent_thresholds": agent_thresholds, **m}, ensure_ascii=False) + "\n")
    else:
        y_pred = [1 if p >= threshold else 0 for p in p_test]
        m = compute_binary_metrics(y_test, y_pred)
        print(f"\n[Test] Threshold t={threshold:.4f} => Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f} (n={len(y_test)})")
        with open(test_out, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps({"type": "metrics", "threshold": threshold, **m}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
