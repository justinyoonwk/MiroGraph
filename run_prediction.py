#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
from openai import OpenAI

import extract_path

# =========================================================
# Robust JSON parsing
# =========================================================
def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # try substring from first "{" to last "}"
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        sub = t[l : r + 1]
        try:
            obj = json.loads(sub)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def parse_llm_json(output_text: str, fallback_key: str = "reasoning") -> Dict[str, Any]:
    obj = _extract_json_object(output_text or "")
    if obj is not None:
        return obj
    return {fallback_key: (output_text or "").strip()}


# =========================================================
# Metrics
# =========================================================
def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
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
# Prompt loader
# =========================================================
def load_prompt(prompts_dir: str, filename: str, ir: int) -> str:
    path = os.path.join(prompts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return txt.replace("{IR}", str(ir))


# =========================================================
# OpenAI call (JSON mode)
# =========================================================
def call_agent_json(
    client: OpenAI,
    model: str,
    agent_name: str,
    agent_prompt: str,
    user_input: str,
    max_tokens: int = 1600,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    system_guard = "Return ONLY a single valid JSON object. No markdown. No extra text."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_guard},
            {"role": "system", "content": f"[AGENT]\n{agent_name}\n\n[PROMPT]\n{agent_prompt}"},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    return parse_llm_json(raw)


# =========================================================
# Data loading: concept_graph.csv -> networkx graph
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


def load_pairs_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"pair file not found: {path}")
    df = pd.read_csv(path)
    required = {"concept1", "concept2", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{path} must have columns {sorted(required)}")
    df["label"] = df["label"].astype(int)
    return df


def balanced_sample_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    half = n_pairs // 2
    if half == 0:
        raise ValueError("n_pairs must be >= 2 to enforce pos:neg=1:1 sampling")

    if len(pos) < half or len(neg) < half:
        raise ValueError(f"Not enough pos/neg to sample {half} each: pos={len(pos)}, neg={len(neg)}")

    pos_idx = rng.sample(list(pos.index), half)
    neg_idx = rng.sample(list(neg.index), half)

    out = pd.concat([pos.loc[pos_idx], neg.loc[neg_idx]], axis=0)
    # shuffle
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


# =========================================================
# Evidence formatting (paths + local 1-hop)
# =========================================================
def format_year_dict(d: Dict[int, int]) -> str:
    if not d:
        return "{}"
    items = sorted((int(k), int(v)) for k, v in d.items())
    return "{" + ", ".join(f"{y}:{v}" for y, v in items) + "}"


def get_edge_year_dict(G: nx.Graph, a: str, b: str) -> Dict[int, int]:
    if not G.has_edge(a, b):
        return {}
    data = G.get_edge_data(a, b) or {}
    yd = data.get("citations_by_year", {})
    return dict(yd) if isinstance(yd, dict) else {}


def serialize_global_paths_node_only(paths: List[List[str]]) -> str:
    if not paths:
        return "NONE"
    lines = []
    for i, p in enumerate(paths, 1):
        lines.append(f"Path {i}: " + " -> ".join(p))
    return "\n".join(lines)


def serialize_global_paths_with_citations(G: nx.Graph, paths: List[List[str]]) -> str:
    if not paths:
        return "NONE"
    lines = []
    for i, p in enumerate(paths, 1):
        parts = [p[0]]
        for j in range(len(p) - 1):
            a, b = p[j], p[j + 1]
            yd = get_edge_year_dict(G, a, b)
            parts.append(f"-{format_year_dict(yd)}-> {b}")
        lines.append(f"Path {i}: " + " ".join(parts))
    return "\n".join(lines)


def sample_one_hop_neighbors(G: nx.Graph, node: str, m: int, seed: int) -> List[str]:
    if node not in G:
        return []
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return []
    rng = random.Random(seed)
    if len(nbrs) <= m:
        return sorted(nbrs, key=lambda x: str(x))
    picked = rng.sample(nbrs, m)
    return sorted(picked, key=lambda x: str(x))


def serialize_local_neighbors_node_only(nbrs: List[str]) -> str:
    if not nbrs:
        return "NONE"
    return "\n".join([f"- {x}" for x in nbrs])


def serialize_local_neighbors_with_citations(G: nx.Graph, center: str, nbrs: List[str]) -> str:
    if not nbrs:
        return "NONE"
    lines = []
    for w in nbrs:
        yd = get_edge_year_dict(G, center, w)
        lines.append(f"- ({w}, {format_year_dict(yd)})")
    return "\n".join(lines)


# =========================================================
# Structure analysis: masking/unmasking
# =========================================================
def recent_3yr_weight(year_dict: Dict[int, int], snapshot_year: int) -> int:
    if not year_dict:
        return 0
    yrs = [snapshot_year - 2, snapshot_year - 1, snapshot_year]
    return sum(int(year_dict.get(y, 0)) for y in yrs)


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
    Node mapping:
      u -> node_1
      v -> node_2
      others -> node_3, node_4, ...
    Returns:
      concept_to_id, id_to_concept, json_payload={"query_pair":[...],"edge_list":[[idA,idB,weight],...]}
    """
    nodes = list(G_sub.nodes())
    if u not in nodes or v not in nodes:
        # still allow empty/fallback
        nodes = [u, v]

    others = sorted([x for x in nodes if x not in {u, v}], key=lambda x: str(x))

    concept_to_id: Dict[str, str] = {u: "node_1", v: "node_2"}
    id_to_concept: Dict[str, str] = {"node_1": u, "node_2": v}

    idx = 3
    for n in others:
        nid = f"node_{idx}"
        concept_to_id[n] = nid
        id_to_concept[nid] = n
        idx += 1

    edge_list = []
    # stable order
    edges = sorted(G_sub.edges(), key=lambda e: (str(e[0]), str(e[1])))
    for a, b in edges:
        ya = get_edge_year_dict(G_sub, a, b)
        w = recent_3yr_weight(ya, snapshot_year)
        edge_list.append([concept_to_id[a], concept_to_id[b], int(w)])

    payload = {
        "query_pair": ["node_1", "node_2"],
        "edge_list": edge_list,
    }
    return concept_to_id, id_to_concept, payload


def unmask_text(text: str, id_to_concept: Dict[str, str]) -> str:
    if not text:
        return text
    # replace node_10 before node_1 to avoid partial overlaps
    keys = sorted(id_to_concept.keys(), key=lambda k: -len(k))
    out = text
    for k in keys:
        concept = id_to_concept[k]
        out = re.sub(rf"\b{re.escape(k)}\b", concept, out)
    return out


def unmask_json(obj: Dict[str, Any], id_to_concept: Dict[str, str]) -> Dict[str, Any]:
    # deep map only over strings
    def _walk(x):
        if isinstance(x, dict):
            return {kk: _walk(vv) for kk, vv in x.items()}
        if isinstance(x, list):
            return [_walk(vv) for vv in x]
        if isinstance(x, str):
            return unmask_text(x, id_to_concept)
        return x

    return _walk(obj)


# =========================================================
# Verdict parsing
# =========================================================
def verdict_to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1", "high", "high-impact", "high_impact"}:
        return 1
    if s in {"no", "n", "false", "0", "low", "non-impact", "non_impact"}:
        return 0
    # common: "YES/NO"
    if "yes" in s:
        return 1
    if "no" in s:
        return 0
    return None


# =========================================================
# Full per-pair pipeline (no terminal agent logs)
# =========================================================
def run_one_pair_once(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    G: nx.Graph,
    snapshot_year: int,
    u: str,
    v: str,
    ir: int,
    max_k_global: int,
    max_p_global: int,
    m_local: int,
    seed: int,
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # 1) induced paths
    paths, _G_sub_unused, msg = extract_path.enumerate_induced_paths(G, u, v, L=max_k_global, max_p=max_p_global)
    if paths is None:
        paths = []
    paths = list(paths)

    # 2) global inputs
    global_node_only = serialize_global_paths_node_only(paths)
    global_with_cit = serialize_global_paths_with_citations(G, paths)

    # 3) local 1-hop sampled neighbors
    # seed split for reproducibility
    u_nbrs = sample_one_hop_neighbors(G, u, m_local, seed + 17)
    v_nbrs = sample_one_hop_neighbors(G, v, m_local, seed + 29)

    u_local_node_only = serialize_local_neighbors_node_only(u_nbrs)
    v_local_node_only = serialize_local_neighbors_node_only(v_nbrs)

    u_local_with_cit = serialize_local_neighbors_with_citations(G, u, u_nbrs)
    v_local_with_cit = serialize_local_neighbors_with_citations(G, v, v_nbrs)

    # 4) structure analysis (masked enclosing subgraph)
    G_sub = build_enclosing_subgraph_from_paths(G, paths)
    _, id_to_concept, masked_payload = mask_subgraph_edge_list(G_sub, u, v, snapshot_year)

    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    # Structure Analysis (masked)
    inp_struct = json.dumps(masked_payload, ensure_ascii=False)
    out_struct_masked = call_agent_json(
        client=client,
        model=model,
        agent_name="Structure Analysis Agent",
        agent_prompt=prompts["structure_analysis"],
        user_input=inp_struct,
        max_tokens=2200,
        temperature=0.2,
    )
    out_struct_unmasked = unmask_json(out_struct_masked, id_to_concept)

    # 5) Global Semantic
    inp_gs = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS]\n{global_node_only}"
        f"{skeptic_block}"
    )
    out_gs = call_agent_json(client, model, "Global Semantic Agent", prompts["global_semantic"], inp_gs)

    # 6) Global Trend
    inp_gt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS_WITH_CITATIONS]\n{global_with_cit}"
        f"{skeptic_block}"
    )
    out_gt = call_agent_json(client, model, "Global Trend Agent", prompts["global_trend"], inp_gt)

    # 7) Local Semantic
    inp_ls = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS_NODE_ONLY]\n{u_local_node_only}\n\n"
        f"[Y_1HOP_NEIGHBORS_NODE_ONLY]\n{v_local_node_only}"
        f"{skeptic_block}"
    )
    out_ls = call_agent_json(client, model, "Local Semantic Agent", prompts["local_semantic"], inp_ls)

    # 8) Local Trend
    inp_lt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS_WITH_CITATIONS]\n{u_local_with_cit}\n\n"
        f"[Y_1HOP_NEIGHBORS_WITH_CITATIONS]\n{v_local_with_cit}"
        f"{skeptic_block}"
    )
    out_lt = call_agent_json(client, model, "Local Trend Agent", prompts["local_trend"], inp_lt)

    # 9) Integrators
    inp_sa = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GLOBAL_SEMANTIC_OUTPUT]\n{json.dumps(out_gs, ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_SEMANTIC_OUTPUT]\n{json.dumps(out_ls, ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_sa = call_agent_json(client, model, "Semantic Analysis Agent", prompts["semantic_analysis"], inp_sa)

    inp_ta = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GLOBAL_TREND_OUTPUT]\n{json.dumps(out_gt, ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_TREND_OUTPUT]\n{json.dumps(out_lt, ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_ta = call_agent_json(client, model, "Trend Analysis Agent", prompts["trend_analysis"], inp_ta)

    return {
        "snapshot_year": snapshot_year,
        "u": u,
        "v": v,
        "ir": ir,
        "paths_msg": msg,
        "global_paths_node_only": global_node_only,
        "global_paths_with_citations": global_with_cit,
        "local_u_neighbors_node_only": u_local_node_only,
        "local_v_neighbors_node_only": v_local_node_only,
        "local_u_neighbors_with_citations": u_local_with_cit,
        "local_v_neighbors_with_citations": v_local_with_cit,
        "structure_analysis_masked_input": masked_payload,
        "structure_analysis_masked_output": out_struct_masked,
        "structure_analysis_unmasked_output": out_struct_unmasked,
        "first_stage": {
            "global_semantic": out_gs,
            "global_trend": out_gt,
            "local_semantic": out_ls,
            "local_trend": out_lt,
        },
        "integrators": {
            "semantic_analysis": out_sa,
            "trend_analysis": out_ta,
        },
    }


def run_impact_evaluator_once(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    structure_unmasked: Dict[str, Any],
    semantic_analysis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    inp_ie = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT]\n{json.dumps(structure_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TREND_ANALYSIS_OUTPUT]\n{json.dumps(trend_analysis, ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_ie = call_agent_json(client, model, "Impact Evaluator", prompts["impact_evaluator"], inp_ie, max_tokens=1700)
    return out_ie


def run_skeptic_with_gt(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    structure_unmasked: Dict[str, Any],
    semantic_analysis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    impact_draft: Dict[str, Any],
    gt_label: int,
) -> Dict[str, Any]:
    """
    mode 0 training: Skeptic MUST know GT.
    """
    inp_sk = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GROUND_TRUTH_LABEL]\n{gt_label}\n"
        f"(1 means YES/high-impact, 0 means NO/non-impact)\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT]\n{json.dumps(structure_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TREND_ANALYSIS_OUTPUT]\n{json.dumps(trend_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[IMPACT_EVALUATOR_DRAFT]\n{json.dumps(impact_draft, ensure_ascii=False, indent=2)}\n"
    )
    out_sk = call_agent_json(client, model, "Skeptic Agent (GT-known)", prompts["skeptic"], inp_sk, max_tokens=1200)
    return out_sk


# =========================================================
# Mode-0 loop (mandatory skeptic; up to 3 rebuttals)
# =========================================================
def mode0_run_pair_with_rebuttals(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    G: nx.Graph,
    snapshot_year: int,
    u: str,
    v: str,
    ir: int,
    gt_label: int,
    max_k_global: int,
    max_p_global: int,
    m_local: int,
    seed: int,
    max_rebuttals: int = 3,
) -> Dict[str, Any]:
    # 0) build evidence + analyses once (cheaper; aligns with your earlier training intent)
    bundle = run_one_pair_once(
        client=client,
        model=model,
        prompts=prompts,
        G=G,
        snapshot_year=snapshot_year,
        u=u,
        v=v,
        ir=ir,
        max_k_global=max_k_global,
        max_p_global=max_p_global,
        m_local=m_local,
        seed=seed,
        skeptic_feedback=None,
    )

    structure_unmasked = bundle["structure_analysis_unmasked_output"]
    semantic_analysis = bundle["integrators"]["semantic_analysis"]
    trend_analysis = bundle["integrators"]["trend_analysis"]

    impact_attempts: List[Dict[str, Any]] = []
    skeptic_feedbacks: List[Dict[str, Any]] = []

    feedback: Optional[Dict[str, Any]] = None

    # initial + up to max_rebuttals
    for t in range(max_rebuttals + 1):
        impact_out = run_impact_evaluator_once(
            client=client,
            model=model,
            prompts=prompts,
            u=u,
            v=v,
            snapshot_year=snapshot_year,
            ir=ir,
            structure_unmasked=structure_unmasked,
            semantic_analysis=semantic_analysis,
            trend_analysis=trend_analysis,
            skeptic_feedback=feedback,
        )

        pred_int = verdict_to_int(impact_out.get("final_verdict", None))
        matched = (pred_int is not None) and (pred_int == int(gt_label))

        impact_attempts.append(
            {
                "attempt_index": t,
                "impact_output": impact_out,
                "pred_int": pred_int,
                "gt_label": int(gt_label),
                "matched_gt": bool(matched),
            }
        )

        if matched:
            # success
            break

        if t == max_rebuttals:
            # stop after max rebuttals exhausted
            break

        # Skeptic critique with GT (mandatory in mode0)
        sk = run_skeptic_with_gt(
            client=client,
            model=model,
            prompts=prompts,
            u=u,
            v=v,
            snapshot_year=snapshot_year,
            ir=ir,
            structure_unmasked=structure_unmasked,
            semantic_analysis=semantic_analysis,
            trend_analysis=trend_analysis,
            impact_draft=impact_out,
            gt_label=int(gt_label),
        )
        skeptic_feedbacks.append(sk)
        feedback = sk  # feed into next impact run

    final_impact = impact_attempts[-1]["impact_output"]
    final_pred_int = verdict_to_int(final_impact.get("final_verdict", None))

    out = {
        "setting": "mode0",
        "snapshot_year": snapshot_year,
        "ir": ir,
        "pair": {"concept1": u, "concept2": v},
        "ground_truth_label": int(gt_label),
        "final_prediction_int": final_pred_int,
        "final_impact_output": final_impact,
        "impact_attempts": impact_attempts,       # includes reasoning + GT
        "skeptic_feedbacks": skeptic_feedbacks,   # critiques (GT-known)
        "bundle": bundle,                         # includes semantic/trend/structure analyses
    }
    return out


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=int, choices=[0, 1], required=True)
    parser.add_argument("--setting", type=str, choices=["transductive", "inductive"], required=True)
    parser.add_argument("--ir", type=int, required=True)

    # canonical args
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_k_global", type=int, default=5)
    parser.add_argument("--max_p_global", type=int, default=10)
    parser.add_argument("--m_local", type=int, default=30)
    parser.add_argument("--model", type=str, default="gpt-4o")

    # logging/outputs
    parser.add_argument("--mode0_out", type=str, default=None)

    # aliases (for your previous command style)
    parser.add_argument("--num_pairs", type=int, default=None)          # alias -> n_pairs
    parser.add_argument("--show_agent_logs", action="store_true")       # alias -> ignored (mode0 prints minimal anyway)
    parser.add_argument("--fewshot_out", type=str, default=None)        # alias -> mode0_out

    args = parser.parse_args()

    # alias resolve
    if args.num_pairs is not None:
        args.n_pairs = args.num_pairs
    if args.fewshot_out is not None:
        args.mode0_out = args.fewshot_out

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY 환경변수가 비어 있습니다. 예) export OPENAI_API_KEY="YOUR_KEY"')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # dataset paths
    kdd_dir = os.path.join(base_dir, "kdd", args.setting)
    graph_path = os.path.join(kdd_dir, "concept_graph.csv")

    if args.setting == "inductive" and args.mode == 1:
        # inductive test uses concept_graph_test.csv
        graph_path = os.path.join(kdd_dir, "concept_graph_test.csv")

    # pair file for sampling
    if args.mode == 0:
        pair_path = os.path.join(kdd_dir, f"train_data_{args.ir}.csv")
    else:
        pair_path = os.path.join(kdd_dir, f"test_data_{args.ir}.csv")

    # prompts
    prompts_dir = os.path.join(base_dir, "prompts")
    prompts = {
        "global_semantic": load_prompt(prompts_dir, "global_semantic.txt", args.ir),
        "global_trend": load_prompt(prompts_dir, "global_trend.txt", args.ir),
        "local_semantic": load_prompt(prompts_dir, "local_semantic.txt", args.ir),
        "local_trend": load_prompt(prompts_dir, "local_trend.txt", args.ir),
        "semantic_analysis": load_prompt(prompts_dir, "semantic_analysis.txt", args.ir),
        "trend_analysis": load_prompt(prompts_dir, "trend_analysis.txt", args.ir),
        "structure_analysis": load_prompt(prompts_dir, "structure_analysis.txt", args.ir),
        "impact_evaluator": load_prompt(prompts_dir, "impact_evaluator.txt", args.ir),
        "skeptic": load_prompt(prompts_dir, "skeptic.txt", args.ir),
    }

    # load graph + snapshot year
    G, snapshot_year = load_concept_graph_csv(graph_path)

    # load pairs + balanced sampling
    df_pairs = load_pairs_file(pair_path)
    sampled = balanced_sample_pairs(df_pairs, args.n_pairs, args.seed)

    client = OpenAI()

    # run
    y_true: List[int] = []
    y_pred: List[int] = []
    records: List[Dict[str, Any]] = []

    # mode 0: mandatory skeptic loop; minimal terminal printing
    if args.mode == 0:
        for i, row in enumerate(sampled.itertuples(index=False), 1):
            u, v, gt = row.concept1, row.concept2, int(row.label)

            rec = mode0_run_pair_with_rebuttals(
                client=client,
                model=args.model,
                prompts=prompts,
                G=G,
                snapshot_year=snapshot_year,
                u=u,
                v=v,
                ir=args.ir,
                gt_label=gt,
                max_k_global=args.max_k_global,
                max_p_global=args.max_p_global,
                m_local=args.m_local,
                seed=args.seed + i * 101,
                max_rebuttals=3,  # fixed per requirement
            )

            pred_int = rec.get("final_prediction_int", None)
            if pred_int is None:
                pred_int = 0  # conservative fallback
            y_true.append(gt)
            y_pred.append(int(pred_int))
            records.append(rec)

            # per-pair minimal print
            print(f"[{i}/{len(sampled)}] final_verdict={pred_int} | GT={gt}")

        # save jsonl if requested
        if args.mode0_out:
            os.makedirs(os.path.dirname(args.mode0_out) or ".", exist_ok=True)
            with open(args.mode0_out, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # (optional but useful) summary metrics
        metrics = compute_binary_metrics(y_true, y_pred)
        print("============================================================")
        print(f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
        print("============================================================")
        if args.mode0_out:
            print(f"[Saved] mode0 examples -> {args.mode0_out}")
        return

    # mode 1: (placeholder) keep behavior consistent with your earlier request,
    # but printing is not the focus of this patch.
    # For now: run single-pass Impact Evaluator (no GT-known skeptic loop).
    for i, row in enumerate(sampled.itertuples(index=False), 1):
        u, v, gt = row.concept1, row.concept2, int(row.label)

        bundle = run_one_pair_once(
            client=client,
            model=args.model,
            prompts=prompts,
            G=G,
            snapshot_year=snapshot_year,
            u=u,
            v=v,
            ir=args.ir,
            max_k_global=args.max_k_global,
            max_p_global=args.max_p_global,
            m_local=args.m_local,
            seed=args.seed + i * 101,
            skeptic_feedback=None,
        )

        impact = run_impact_evaluator_once(
            client=client,
            model=args.model,
            prompts=prompts,
            u=u,
            v=v,
            snapshot_year=snapshot_year,
            ir=args.ir,
            structure_unmasked=bundle["structure_analysis_unmasked_output"],
            semantic_analysis=bundle["integrators"]["semantic_analysis"],
            trend_analysis=bundle["integrators"]["trend_analysis"],
            skeptic_feedback=None,
        )

        pred_int = verdict_to_int(impact.get("final_verdict", None))
        if pred_int is None:
            pred_int = 0

        y_true.append(gt)
        y_pred.append(int(pred_int))

        print(f"[{i}/{len(sampled)}] final_verdict={pred_int} | GT={gt}")

    metrics = compute_binary_metrics(y_true, y_pred)
    print("============================================================")
    print(f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
    print("============================================================")


if __name__ == "__main__":
    main()
