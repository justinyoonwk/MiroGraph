#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-agent pipeline with weight calibration via Skeptic (train/demo generation) and weighted Impact Evaluator (test).

Mode 0 (train): generate demonstrations from training data by iterating weights until Impact Evaluator matches GT (max 3 tries).
Mode 1 (test): run 3 agents, then weighted Impact Evaluator using averaged weights from Mode 0 + few-shot demos (min-citation GT=1).
"""

import os
import json
import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

import extract_path

load_dotenv()


# =========================================================
# JSON helpers
# =========================================================
def parse_llm_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {"raw": t}
    except Exception:
        pass
    l, r = t.find("{"), t.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            obj = json.loads(t[l : r + 1])
            return obj if isinstance(obj, dict) else {"raw": t}
        except Exception:
            pass
    return {"raw": t}


def verdict_to_int(v: Any) -> int:
    s = (str(v or "")).strip().lower()
    return 1 if s == "yes" else 0


def normalize_weights(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    w1 = max(0.0, float(w1))
    w2 = max(0.0, float(w2))
    w3 = max(0.0, float(w3))
    s = w1 + w2 + w3
    if s <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w1 / s, w2 / s, w3 / s)


# =========================================================
# Metrics
# =========================================================
def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


# =========================================================
# Prompt loader / OpenAI caller
# =========================================================
def load_prompt(prompts_dir: Path, filename: str, ir: int) -> str:
    p = prompts_dir / filename
    txt = p.read_text(encoding="utf-8")
    return txt.replace("{IR}", str(ir))


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    max_tokens: Optional[int] = None,
    retries: int = 3,
) -> str:
    is_gpt5 = ("gpt-5" in model.lower()) or ("o3" in model.lower())
    temperature = 1.0 if is_gpt5 else 0.2
    actual_max_tokens = max_tokens if max_tokens is not None else (16000 if is_gpt5 else 2048)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=actual_max_tokens,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content.strip()
        except Exception as e:
            if attempt == retries - 1:
                return f""
            time.sleep(2 * (attempt + 1))
    return ""


# =========================================================
# Graph / subgraph payload
# =========================================================
def load_concept_graph_csv(csv_path: Path) -> Tuple[nx.Graph, int]:
    df = pd.read_csv(csv_path)
    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)
    snapshot_year = int(df["year"].max())

    G = nx.Graph()
    for row in df.itertuples(index=False):
        u, v, y, c = str(row.concept1), str(row.concept2), int(row.year), int(row.citation_count)
        if not G.has_edge(u, v):
            G.add_edge(u, v, citations_by_year={})
        G[u][v]["citations_by_year"][y] = G[u][v]["citations_by_year"].get(y, 0) + c
    return G, snapshot_year


def get_enclosing_subgraph_payload(
    G: nx.Graph, u: str, v: str, max_k: int = 4, max_p: int = 8
) -> Optional[Dict[str, Any]]:
    paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    nodes: set = set()
    for p in (paths or []):
        nodes.update(p)
    if not nodes:
        return None
    G_sub = G.subgraph(nodes).copy()

    edge_list = []
    for a, b in sorted(G_sub.edges(), key=lambda e: (str(e[0]), str(e[1]))):
        yd = (G_sub.get_edge_data(a, b) or {}).get("citations_by_year", {}) or {}
        yd_sorted = {int(k): int(v) for k, v in sorted(yd.items())}
        edge_list.append([str(a), str(b), yd_sorted])
    return {"query_pair": [str(u), str(v)], "edge_list": edge_list}


# =========================================================
# Agents
# =========================================================
def run_three_agents(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    subgraph_payload: Dict[str, Any],
) -> Tuple[str, str, str]:
    payload_json = json.dumps(subgraph_payload, ensure_ascii=False, indent=2)
    common = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[ENCLOSING SUBGRAPH]\n{payload_json}"
    )
    structure = call_llm(client, model, prompts["structure"], common)
    semantic = call_llm(client, model, prompts["semantic"], common)
    trend = call_llm(client, model, prompts["trend"], common)
    return structure, semantic, trend


def format_fewshot_examples(demos: List[Dict[str, Any]]) -> str:
    if not demos:
        return ""
    blocks = []
    for i, d in enumerate(demos, 1):
        pair = d.get("pair", {})
        w = d.get("weights_used", {})
        impact = d.get("impact_output", {})
        struct_ex = (d.get("structure_analysis", "")[:600]).replace("\n", " ")
        sem_ex = (d.get("semantic_analysis", "")[:600]).replace("\n", " ")
        trend_ex = (d.get("trend_analysis", "")[:600]).replace("\n", " ")
        impact_ex = (impact.get("impact_reasoning", "")[:800]).replace("\n", " ")
        blocks.append(
            "\n".join(
                [
                    f"Example {i}:",
                    f"  Pair: {pair.get('concept1','')} <-> {pair.get('concept2','')}",
                    f"  Ground Truth: {d.get('ground_truth_label')}",
                    f"  Weights used: w_structure={w.get('w_structure')}, w_semantic={w.get('w_semantic')}, w_trend={w.get('w_trend')}",
                    "  Structure analysis (excerpt):",
                    "    " + struct_ex,
                    "  Semantic analysis (excerpt):",
                    "    " + sem_ex,
                    "  Trend analysis (excerpt):",
                    "    " + trend_ex,
                    f"  Final Verdict: {impact.get('final_verdict')}",
                    "  Impact reasoning (excerpt):",
                    "    " + impact_ex,
                ]
            )
        )
    return "\n\n[FEW-SHOT EXAMPLES]\n" + "\n\n".join(blocks)


def run_impact_evaluator_weighted(
    client: OpenAI,
    model: str,
    prompt: str,
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    w_structure: float,
    w_semantic: float,
    w_trend: float,
    structure_analysis: str,
    semantic_analysis: str,
    trend_analysis: str,
    demonstrations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    demos_block = format_fewshot_examples(demonstrations or [])
    user_input = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[WEIGHTS]\n"
        f"w_structure={w_structure}\n"
        f"w_semantic={w_semantic}\n"
        f"w_trend={w_trend}\n\n"
        f"[STRUCTURE_ANALYSIS]\n{structure_analysis}\n\n"
        f"[SEMANTIC_ANALYSIS]\n{semantic_analysis}\n\n"
        f"[TREND_ANALYSIS]\n{trend_analysis}"
        f"{demos_block}"
    )
    raw = call_llm(client, model, prompt, user_input)
    parsed = parse_llm_json(raw)
    verdict = parsed.get("final_verdict")
    if verdict is None:
        verdict = "Yes" if "yes" in raw.lower() else "No"
    return {
        "impact_reasoning": parsed.get("impact_reasoning", parsed.get("raw", raw)),
        "final_verdict": "Yes" if str(verdict).strip().lower() == "yes" else "No",
        "raw": raw,
    }


def run_skeptic_weight_selector(
    client: OpenAI,
    model: str,
    prompt: str,
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    structure_analysis: str,
    semantic_analysis: str,
    trend_analysis: str,
    impact_output: Dict[str, Any],
    ground_truth: int,
) -> Tuple[float, float, float, Dict[str, Any]]:
    user_input = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[STRUCTURE_ANALYSIS]\n{structure_analysis}\n\n"
        f"[SEMANTIC_ANALYSIS]\n{semantic_analysis}\n\n"
        f"[TREND_ANALYSIS]\n{trend_analysis}\n\n"
        f"[IMPACT_EVALUATOR_OUTPUT]\n{json.dumps(impact_output, ensure_ascii=False, indent=2)}\n\n"
        f"[GROUND_TRUTH]\n{ground_truth}\n"
    )
    raw = call_llm(client, model, prompt, user_input)
    parsed = parse_llm_json(raw)
    w1 = parsed.get("w_structure", 1.0)
    w2 = parsed.get("w_semantic", 1.0)
    w3 = parsed.get("w_trend", 1.0)
    try:
        w1f, w2f, w3f = float(w1), float(w2), float(w3)
    except Exception:
        w1f, w2f, w3f = 1.0, 1.0, 1.0
    return w1f, w2f, w3f, {"raw": raw, "parsed": parsed}


# =========================================================
# Demo selection (same as run_prediction_old.py behavior)
# =========================================================
def select_min_citation_gt1_demos(
    demonstrations: List[Dict[str, Any]],
    ir: int,
    num_demos: int,
    seed: int,
    salt: int,
) -> List[Dict[str, Any]]:
    if not demonstrations or num_demos <= 0:
        return []
    rng = random.Random(seed + salt * 101)
    gt1_demos = [
        d
        for d in demonstrations
        if int(d.get("ground_truth_label", -1)) == 1
        and int(d.get("final_prediction_int", -1)) == 1
    ]
    demo_with_citations: List[Tuple[Dict[str, Any], int]] = []
    for d in gt1_demos:
        tc = d.get("total_citations")
        if tc is not None and int(tc) >= ir:
            demo_with_citations.append((d, int(tc)))
    if not demo_with_citations:
        return []

    citation_groups: Dict[int, List[Dict[str, Any]]] = {}
    for demo, citations in demo_with_citations:
        citation_groups.setdefault(citations, []).append(demo)

    selected: List[Dict[str, Any]] = []
    for citations in sorted(citation_groups.keys()):
        if len(selected) >= num_demos:
            break
        remaining = num_demos - len(selected)
        available = citation_groups[citations]
        k = min(remaining, len(available))
        selected.extend(rng.sample(available, k))
    return selected


def load_demonstrations_and_avg_weights(mode0_out: Path) -> Tuple[List[Dict[str, Any]], Tuple[float, float, float]]:
    demos: List[Dict[str, Any]] = []
    weights: Optional[Tuple[float, float, float]] = None
    if not mode0_out.exists():
        return demos, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    with mode0_out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "demonstration":
                demos.append(d)
            elif d.get("type") == "weights_summary":
                w = d.get("avg_weights", {})
                try:
                    weights = normalize_weights(w.get("w_structure", 1 / 3), w.get("w_semantic", 1 / 3), w.get("w_trend", 1 / 3))
                except Exception:
                    weights = None
    if weights is None:
        # compute from demos
        ws = []
        for d in demos:
            if int(d.get("final_prediction_int", -1)) == int(d.get("ground_truth_label", -2)):
                w = d.get("weights_used", {})
                try:
                    ws.append((float(w.get("w_structure")), float(w.get("w_semantic")), float(w.get("w_trend"))))
                except Exception:
                    continue
        if ws:
            w1 = sum(x[0] for x in ws) / len(ws)
            w2 = sum(x[1] for x in ws) / len(ws)
            w3 = sum(x[2] for x in ws) / len(ws)
            weights = normalize_weights(w1, w2, w3)
        else:
            weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return demos, weights


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, choices=[0, 1], required=True)
    parser.add_argument("--dataset", type=str, default="kdd")
    parser.add_argument("--setting", type=str, choices=["transductive", "inductive"], required=True)
    parser.add_argument("--ir", type=int, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--api_key_name", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_pairs", type=int, default=None, help="Optional cap for debugging (balanced for mode1; first N for mode0)")
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=4)
    parser.add_argument("--max_p", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=3, help="Max skeptic reweight iterations per training pair")
    parser.add_argument("--mode0_out", type=str, default=None)
    parser.add_argument("--mode1_out", type=str, default=None)
    args = parser.parse_args()

    api_key = os.getenv(f"OPENAI_API_KEY_{args.api_key_name}") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    base_dir = Path(__file__).parent
    prompts_dir = base_dir / "prompts"

    prompts = {
        "structure": load_prompt(prompts_dir, "structure_analysis.txt", args.ir),
        "semantic": load_prompt(prompts_dir, "semantic_agent_enclosing.txt", args.ir),
        "trend": load_prompt(prompts_dir, "trend_agent_enclosing.txt", args.ir),
        "impact_weighted": load_prompt(prompts_dir, "impact_evaluator_weighted.txt", args.ir),
        "skeptic_weight": load_prompt(prompts_dir, "skeptic_weight_selector.txt", args.ir),
    }

    dataset_dir = base_dir / "dataset" / args.dataset / args.setting
    graph_path = dataset_dir / ("concept_graph_test.csv" if args.setting == "inductive" and args.mode == 1 else "concept_graph.csv")
    pair_path = dataset_dir / (f"train_data_{args.ir}.csv" if args.mode == 0 else f"test_data_{args.ir}.csv")

    G, snapshot_year = load_concept_graph_csv(graph_path)
    df_pairs = pd.read_csv(pair_path)

    # Output paths
    if args.mode0_out:
        mode0_out = Path(args.mode0_out)
    else:
        mode0_out = base_dir / "demonstrations" / args.dataset / f"mode0_ir{args.ir}_multiagent_weighted.jsonl"
    mode0_out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode1_out:
        mode1_out = Path(args.mode1_out)
    else:
        mode1_out = base_dir / "results" / args.dataset / args.setting / f"mode1_ir{args.ir}_multiagent_weighted.jsonl"
    mode1_out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == 0:
        # =========================================================
        # Mode 0: demonstration generation + weight calibration
        # =========================================================
        processed = set()
        if mode0_out.exists():
            with mode0_out.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get("type") == "demonstration":
                            p = d.get("pair", {})
                            processed.add(tuple(sorted((str(p.get("concept1")), str(p.get("concept2"))))))
                    except Exception:
                        continue
        else:
            with mode0_out.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "metadata",
                            "dataset": args.dataset,
                            "setting": args.setting,
                            "ir": args.ir,
                            "model": args.model,
                            "snapshot_year": snapshot_year,
                            "pipeline": "train_demo_multiagent_with_skeptic_weight_calibration",
                            "max_iters": args.max_iters,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        rows = list(df_pairs.itertuples(index=False))
        if args.n_pairs is not None:
            rows = rows[: max(1, args.n_pairs)]

        def process_train_row(row):
            u, v = str(row.concept1), str(row.concept2)
            gt = int(row.label)
            tc = int(row.total_citations) if hasattr(row, "total_citations") else None
            if tuple(sorted((u, v))) in processed:
                return "SKIP"

            sub = get_enclosing_subgraph_payload(G, u, v, max_k=args.max_k, max_p=args.max_p)
            if not sub:
                # No subgraph -> cannot analyze; keep as non-demo
                return {
                    "type": "demonstration",
                    "pair": {"concept1": u, "concept2": v},
                    "ground_truth_label": gt,
                    "total_citations": tc,
                    "final_prediction_int": 0,
                    "weights_used": {"w_structure": 1 / 3, "w_semantic": 1 / 3, "w_trend": 1 / 3},
                    "structure_analysis": "NO_SUBGRAPH",
                    "semantic_analysis": "NO_SUBGRAPH",
                    "trend_analysis": "NO_SUBGRAPH",
                    "impact_output": {"impact_reasoning": "No enclosing subgraph.", "final_verdict": "No"},
                    "success": int(gt == 0),
                    "attempts": 0,
                }

            structure_a, semantic_a, trend_a = run_three_agents(
                client, args.model, prompts, u, v, snapshot_year, args.ir, sub
            )

            w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            last_impact = None
            last_pred = 0
            last_skeptic = None
            for attempt in range(1, args.max_iters + 1):
                impact = run_impact_evaluator_weighted(
                    client=client,
                    model=args.model,
                    prompt=prompts["impact_weighted"],
                    u=u,
                    v=v,
                    snapshot_year=snapshot_year,
                    ir=args.ir,
                    w_structure=w[0],
                    w_semantic=w[1],
                    w_trend=w[2],
                    structure_analysis=structure_a,
                    semantic_analysis=semantic_a,
                    trend_analysis=trend_a,
                    demonstrations=None,
                )
                pred = verdict_to_int(impact.get("final_verdict"))
                last_impact, last_pred = impact, pred
                if pred == gt:
                    return {
                        "type": "demonstration",
                        "pair": {"concept1": u, "concept2": v},
                        "ground_truth_label": gt,
                        "total_citations": tc,
                        "final_prediction_int": pred,
                        "weights_used": {"w_structure": w[0], "w_semantic": w[1], "w_trend": w[2]},
                        "structure_analysis": structure_a,
                        "semantic_analysis": semantic_a,
                        "trend_analysis": trend_a,
                        "impact_output": {"impact_reasoning": impact.get("impact_reasoning", ""), "final_verdict": impact.get("final_verdict")},
                        "skeptic_trace": last_skeptic,
                        "success": 1,
                        "attempts": attempt,
                    }
                if attempt < args.max_iters:
                    sw1, sw2, sw3, skeptic_meta = run_skeptic_weight_selector(
                        client=client,
                        model=args.model,
                        prompt=prompts["skeptic_weight"],
                        u=u,
                        v=v,
                        snapshot_year=snapshot_year,
                        ir=args.ir,
                        structure_analysis=structure_a,
                        semantic_analysis=semantic_a,
                        trend_analysis=trend_a,
                        impact_output=impact,
                        ground_truth=gt,
                    )
                    w = normalize_weights(sw1, sw2, sw3)
                    last_skeptic = {
                        "attempt": attempt,
                        "suggested_raw": {"w_structure": sw1, "w_semantic": sw2, "w_trend": sw3},
                        "normalized": {"w_structure": w[0], "w_semantic": w[1], "w_trend": w[2]},
                        "llm": skeptic_meta,
                    }

            # failed to match within max_iters
            return {
                "type": "demonstration",
                "pair": {"concept1": u, "concept2": v},
                "ground_truth_label": gt,
                "total_citations": tc,
                "final_prediction_int": last_pred,
                "weights_used": {"w_structure": w[0], "w_semantic": w[1], "w_trend": w[2]},
                "structure_analysis": structure_a,
                "semantic_analysis": semantic_a,
                "trend_analysis": trend_a,
                "impact_output": {"impact_reasoning": (last_impact or {}).get("impact_reasoning", ""), "final_verdict": (last_impact or {}).get("final_verdict", "No")},
                "skeptic_trace": last_skeptic,
                "success": 0,
                "attempts": args.max_iters,
            }

        max_workers = args.batch_size
        successes = 0
        ws: List[Tuple[float, float, float]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_train_row, r) for r in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Mode0 Training"):
                rec = fut.result()
                if not rec or rec == "SKIP":
                    continue
                with mode0_out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if int(rec.get("success", 0)) == 1:
                    successes += 1
                    wrec = rec.get("weights_used", {})
                    try:
                        ws.append((float(wrec["w_structure"]), float(wrec["w_semantic"]), float(wrec["w_trend"])))
                    except Exception:
                        pass

        if ws:
            avg = normalize_weights(
                sum(x[0] for x in ws) / len(ws),
                sum(x[1] for x in ws) / len(ws),
                sum(x[2] for x in ws) / len(ws),
            )
        else:
            avg = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        summary = {
            "type": "weights_summary",
            "n_total": len(rows),
            "n_success": successes,
            "avg_weights": {"w_structure": avg[0], "w_semantic": avg[1], "w_trend": avg[2]},
        }
        with mode0_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(
            f"[Mode0 Done] n_total={len(rows)} n_success={successes} "
            f"avg_weights=({avg[0]:.4f},{avg[1]:.4f},{avg[2]:.4f})"
        )
        return

    # =========================================================
    # Mode 1: test with averaged weights + few-shot demos
    # =========================================================
    demonstrations, avg_w = load_demonstrations_and_avg_weights(mode0_out)
    print(f"[INFO] Loaded demonstrations={len(demonstrations)} avg_weights=({avg_w[0]:.4f},{avg_w[1]:.4f},{avg_w[2]:.4f}) from {mode0_out}")

    processed = set()
    y_true: List[int] = []
    y_pred: List[int] = []
    if mode1_out.exists():
        with mode1_out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("type") == "result":
                        p = d.get("pair", {})
                        processed.add(tuple(sorted((str(p.get("concept1")), str(p.get("concept2"))))))
                        y_true.append(int(d.get("ground_truth_label", 0)))
                        y_pred.append(int(d.get("final_prediction_int", 0)))
                except Exception:
                    continue
    else:
        with mode1_out.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "dataset": args.dataset,
                        "setting": args.setting,
                        "ir": args.ir,
                        "model": args.model,
                        "snapshot_year": snapshot_year,
                        "pipeline": "test_multiagent_weighted_impact_evaluator",
                        "avg_weights": {"w_structure": avg_w[0], "w_semantic": avg_w[1], "w_trend": avg_w[2]},
                        "num_demos": args.num_demos,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # balanced sampling for test (like previous scripts)
    pos = df_pairs[df_pairs["label"] == 1]
    neg = df_pairs[df_pairs["label"] == 0]
    n = min(len(pos), len(neg))
    if args.n_pairs is not None:
        n = min(n, max(1, args.n_pairs // 2))
    sampled = pd.concat(
        [pos.sample(n, random_state=args.seed), neg.sample(n, random_state=args.seed)]
    ).sample(frac=1, random_state=args.seed)

    def process_test_row(i_row):
        i, row = i_row
        u, v = str(row.concept1), str(row.concept2)
        gt = int(row.label)
        if tuple(sorted((u, v))) in processed:
            return "SKIP"

        sub = get_enclosing_subgraph_payload(G, u, v, max_k=args.max_k, max_p=args.max_p)
        if not sub:
            impact = run_impact_evaluator_weighted(
                client=client,
                model=args.model,
                prompt=prompts["impact_weighted"],
                u=u,
                v=v,
                snapshot_year=snapshot_year,
                ir=args.ir,
                w_structure=avg_w[0],
                w_semantic=avg_w[1],
                w_trend=avg_w[2],
                structure_analysis="NO_SUBGRAPH",
                semantic_analysis="NO_SUBGRAPH",
                trend_analysis="NO_SUBGRAPH",
                demonstrations=None,
            )
            pred = verdict_to_int(impact.get("final_verdict"))
            return {
                "type": "result",
                "pair": {"concept1": u, "concept2": v},
                "ground_truth_label": gt,
                "final_prediction_int": pred,
                "weights_used": {"w_structure": avg_w[0], "w_semantic": avg_w[1], "w_trend": avg_w[2]},
                "structure_analysis": "NO_SUBGRAPH",
                "semantic_analysis": "NO_SUBGRAPH",
                "trend_analysis": "NO_SUBGRAPH",
                "impact_output": {"impact_reasoning": impact.get("impact_reasoning", ""), "final_verdict": impact.get("final_verdict")},
                "used_demonstrations": [],
            }

        structure_a, semantic_a, trend_a = run_three_agents(
            client, args.model, prompts, u, v, snapshot_year, args.ir, sub
        )
        selected_demos = (
            select_min_citation_gt1_demos(demonstrations, args.ir, args.num_demos, args.seed, i)
            if args.num_demos > 0
            else []
        )
        impact = run_impact_evaluator_weighted(
            client=client,
            model=args.model,
            prompt=prompts["impact_weighted"],
            u=u,
            v=v,
            snapshot_year=snapshot_year,
            ir=args.ir,
            w_structure=avg_w[0],
            w_semantic=avg_w[1],
            w_trend=avg_w[2],
            structure_analysis=structure_a,
            semantic_analysis=semantic_a,
            trend_analysis=trend_a,
            demonstrations=selected_demos if selected_demos else None,
        )
        pred = verdict_to_int(impact.get("final_verdict"))
        used_pairs = [
            {"concept1": d.get("pair", {}).get("concept1"), "concept2": d.get("pair", {}).get("concept2"), "total_citations": d.get("total_citations")}
            for d in (selected_demos or [])
        ]
        return {
            "type": "result",
            "pair": {"concept1": u, "concept2": v},
            "ground_truth_label": gt,
            "final_prediction_int": pred,
            "weights_used": {"w_structure": avg_w[0], "w_semantic": avg_w[1], "w_trend": avg_w[2]},
            "structure_analysis": structure_a,
            "semantic_analysis": semantic_a,
            "trend_analysis": trend_a,
            "impact_output": {"impact_reasoning": impact.get("impact_reasoning", ""), "final_verdict": impact.get("final_verdict")},
            "used_demonstrations": used_pairs,
        }

    max_workers = args.batch_size
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_test_row, x) for x in list(enumerate(sampled.itertuples(index=False), 1))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Mode1 Testing"):
            rec = fut.result()
            if not rec or rec == "SKIP":
                continue
            with mode1_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            y_true.append(int(rec["ground_truth_label"]))
            y_pred.append(int(rec["final_prediction_int"]))

    if y_true:
        m = compute_binary_metrics(y_true, y_pred)
        with mode1_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "metrics", **m}, ensure_ascii=False) + "\n")
        print(
            f"[Mode1 Done] n={len(y_true)} "
            f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
