#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from openai import OpenAI
import networkx as nx

import extract_path
import generate_path_features as graph_module


# =========================================================
# JSON parsing (robust)
# =========================================================
def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_llm_json(output_text: str, fallback_key: str = "reasoning") -> Dict[str, Any]:
    obj = _try_json_loads(output_text)
    if obj is not None:
        return obj
    return {fallback_key: (output_text or "").strip()}


# =========================================================
# Pretty logging
# =========================================================
def print_block(title: str, content: str):
    print(f"\n[{title}]")
    print("─" * 80)
    print(content)


def print_agent_io(agent_name: str, agent_input: str, agent_output: Dict[str, Any]):
    print(f"\n==================== {agent_name} ====================")
    print_block("INPUT", agent_input)
    print_block("OUTPUT(JSON)", json.dumps(agent_output, ensure_ascii=False, indent=2))
    print("=======================================================")


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
# Graph evidence extraction (YOUR intended definitions)
# =========================================================
def extract_induced_paths_only(
    G: nx.Graph, u: str, v: str, max_k: int, max_p: int
) -> Tuple[List[List[str]], str]:
    """
    Use extract_path.enumerate_induced_paths.
    Return only the induced paths list (as-is), plus message.
    """
    paths, _G_sub_unused, msg = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    if paths is None:
        paths = []
    return paths, msg


def collect_edges_on_paths(G: nx.Graph, paths: List[List[str]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Global edges MUST be only consecutive edges that appear in induced paths.
    No extra edges in enclosing subgraph.
    """
    edge_keys: Set[Tuple[str, str]] = set()
    edges: List[Tuple[str, str, Dict[str, Any]]] = []

    for p in paths:
        if len(p) < 2:
            continue
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            key = (a, b) if str(a) <= str(b) else (b, a)
            if key in edge_keys:
                continue
            edge_keys.add(key)

            if G.has_edge(a, b):
                data = dict(G.get_edge_data(a, b) or {})
            else:
                data = {}
            edges.append((a, b, data))

    edges.sort(key=lambda e: (str(e[0]), str(e[1])))
    return edges


def one_hop_neighbor_edges(G: nx.Graph, node: str) -> Tuple[List[str], List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Local evidence MUST be only:
    - 1-hop neighbors of node
    - edges (node, neighbor) only
    No neighbor-neighbor edges, no 2-hop.
    """
    if node not in G:
        return [], []
    nbrs = sorted(list(G.neighbors(node)), key=lambda x: str(x))
    edges = []
    for w in nbrs:
        data = dict(G.get_edge_data(node, w) or {})
        edges.append((node, w, data))
    return nbrs, edges


# =========================================================
# Weight extraction for structure edge list
# =========================================================
def get_edge_weight_last3(data: Dict[str, Any], snapshot_year: int) -> int:
    """
    Try to extract the 'recent 3 years citation sum' weight.
    Priority:
      1) citation_weight (assumed already recent-3yr sum)
      2) weight (fallback)
      3) dict-like yearly citations (try to sum snapshot_year-2..snapshot_year)
    Return int >= 0.
    """
    if not data:
        return 0

    if "citation_weight" in data:
        try:
            return int(data.get("citation_weight") or 0)
        except Exception:
            return 0

    if "weight" in data:
        try:
            return int(data.get("weight") or 0)
        except Exception:
            return 0

    for k in ["citations_by_year", "yearly_citations", "temporal_citations"]:
        if k in data and isinstance(data[k], dict):
            s = 0
            for y in [snapshot_year - 2, snapshot_year - 1, snapshot_year]:
                if y in data[k]:
                    try:
                        s += int(data[k][y] or 0)
                    except Exception:
                        pass
                elif str(y) in data[k]:
                    try:
                        s += int(data[k][str(y)] or 0)
                    except Exception:
                        pass
            return int(s)

    return 0


# =========================================================
# Serialization (raw evidence; no precomputed stats)
# =========================================================
def _edge_attr_to_str(data: Dict[str, Any]) -> str:
    if not data:
        return ""
    if "citation_weight" in data:
        return f"citation_weight={data.get('citation_weight')}"
    if "weight" in data:
        return f"weight={data.get('weight')}"
    for k in ["citations_by_year", "yearly_citations", "temporal_citations"]:
        if k in data and isinstance(data[k], dict):
            years = sorted(list(data[k].keys()))
            preview = years[:12]
            more = "" if len(years) <= 12 else f",...(+{len(years)-12})"
            return f"{k}_years=[{','.join(map(str, preview))}{more}]"
    return ""


def serialize_paths(paths: List[List[str]]) -> str:
    if not paths:
        return "NONE"
    lines = []
    for i, p in enumerate(paths, 1):
        lines.append(f"- Path {i}: " + " -> ".join(p))
    return "\n".join(lines)


def serialize_edge_tuples(edges: List[Tuple[str, str, Dict[str, Any]]]) -> str:
    if not edges:
        return "NONE"
    lines = []
    for a, b, d in edges:
        attr = _edge_attr_to_str(d)
        if attr:
            lines.append(f"- {a} -- {b} | {attr}")
        else:
            lines.append(f"- {a} -- {b}")
    return "\n".join(lines)


def serialize_neighbors(nbrs: List[str]) -> str:
    if not nbrs:
        return "NONE"
    return "\n".join([f"- {x}" for x in nbrs])


# =========================================================
# Masking / Unmasking modules
# =========================================================
def build_node_mask_maps(nodes: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    concept -> node_k, node_k -> concept
    Nodes are sorted for determinism.
    """
    nodes_sorted = sorted(list(set(nodes)), key=lambda x: str(x))
    concept_to_mask: Dict[str, str] = {}
    mask_to_concept: Dict[str, str] = {}
    for i, n in enumerate(nodes_sorted, 1):
        masked = f"node_{i}"
        concept_to_mask[str(n)] = masked
        mask_to_concept[masked] = str(n)
    return concept_to_mask, mask_to_concept


def make_masked_edge_list(
    edges: List[Tuple[str, str, Dict[str, Any]]],
    concept_to_mask: Dict[str, str],
    snapshot_year: int,
) -> List[List[Union[str, int]]]:
    """
    Build [NODE_ID_A, NODE_ID_B, WEIGHT_VALUE] list.
    NODE_ID_* are strings like node_1, node_2.
    """
    out: List[List[Union[str, int]]] = []
    for a, b, d in edges:
        ma = concept_to_mask[str(a)]
        mb = concept_to_mask[str(b)]
        w = get_edge_weight_last3(d, snapshot_year=snapshot_year)
        out.append([ma, mb, int(w)])
    # stable order
    out.sort(key=lambda x: (x[0], x[1], int(x[2])))
    return out


def unmask_string(s: str, mask_to_concept: Dict[str, str]) -> str:
    """
    Replace occurrences of node_k in a string with concept names.
    Uses word-boundary style replacement to reduce accidental partial matches.
    """
    if not s:
        return s
    # Replace longer ids first (node_10 before node_1)
    keys = sorted(mask_to_concept.keys(), key=lambda x: (-len(x), x))
    for k in keys:
        v = mask_to_concept[k]
        # word boundary-ish: ensure we match exact token node_#
        s = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, s)
    return s


def unmask_json(obj: Any, mask_to_concept: Dict[str, str]) -> Any:
    """
    Recursively unmask node ids inside JSON-like structures.
    - If string equals a node id -> replace
    - If string contains node ids in sentences -> replace via regex
    """
    if isinstance(obj, dict):
        return {k: unmask_json(v, mask_to_concept) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unmask_json(x, mask_to_concept) for x in obj]
    if isinstance(obj, str):
        if obj in mask_to_concept:
            return mask_to_concept[obj]
        return unmask_string(obj, mask_to_concept)
    return obj


# =========================================================
# OpenAI call (JSON mode)
# =========================================================
def call_agent_json(
    client: OpenAI,
    model: str,
    agent_name: str,
    agent_prompt: str,
    user_input: str,
    max_tokens: int = 1400,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    system_guard = "Return ONLY a single valid JSON object. No markdown. No extra text."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_guard},
            {"role": "system", "content": f"[AGENT_NAME]\n{agent_name}\n\n[AGENT_INSTRUCTIONS]\n{agent_prompt}"},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw = resp.choices[0].message.content or ""
    return parse_llm_json(raw)


# =========================================================
# Agent stages
# =========================================================
def run_structure_analysis_masked(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    year: int,
    ir: int,
    enclosing_edges: List[Tuple[str, str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Masking Module -> Structure Analysis -> (return both masked and unmasked outputs)
    Node IDs are node_1, node_2, ...
    """
    # nodes that appear in edges + ensure u,v included
    nodes: List[str] = [u, v]
    for a, b, _d in enclosing_edges:
        nodes.append(str(a))
        nodes.append(str(b))

    concept_to_mask, mask_to_concept = build_node_mask_maps(nodes)

    masked_u = concept_to_mask[str(u)]
    masked_v = concept_to_mask[str(v)]
    masked_edge_list = make_masked_edge_list(enclosing_edges, concept_to_mask, snapshot_year=year)

    inp_struct = (
        f"[TARGET]\n"
        f"X_ID={masked_u}\nY_ID={masked_v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[EDGE_LIST_FORMAT]\nEach edge is [NODE_ID_A, NODE_ID_B, WEIGHT_VALUE].\n"
        f"NODE_ID_* are opaque identifiers like node_1, node_2.\n"
        f"WEIGHT_VALUE is the sum of citations over the last 3 years.\n\n"
        f"[EDGE_LIST]\n{json.dumps(masked_edge_list, ensure_ascii=False)}"
    )

    out_struct_masked = call_agent_json(
        client=client,
        model=model,
        agent_name="Structure Analysis Agent (Masked)",
        agent_prompt=prompts["structure_analysis"],
        user_input=inp_struct,
    )
    print_agent_io("Structure Analysis Agent (Masked)", inp_struct, out_struct_masked)

    # Unmasking Module (deterministic)
    out_struct_unmasked = unmask_json(out_struct_masked, mask_to_concept)

    # Log unmasked (for debugging / readability)
    print_agent_io(
        "Unmasking Module (Structure Output)",
        f"[id_to_concept size]={len(mask_to_concept)} (hidden mapping applied)",
        out_struct_unmasked if isinstance(out_struct_unmasked, dict) else {"unmasked": out_struct_unmasked},
    )

    return {
        "masked_input": {
            "query_pair_ids": [masked_u, masked_v],
            "edge_list": masked_edge_list,
        },
        "masked_output": out_struct_masked,
        "unmasked_output": out_struct_unmasked,
        "mask_to_concept": mask_to_concept,  # keep for traceability (do not feed to structure agent)
    }


def run_first_stage_agents(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    year: int,
    ir: int,
    induced_paths: List[List[str]],
    global_path_edges: List[Tuple[str, str, Dict[str, Any]]],
    u_nbrs: List[str],
    u_edges: List[Tuple[str, str, Dict[str, Any]]],
    v_nbrs: List[str],
    v_edges: List[Tuple[str, str, Dict[str, Any]]],
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths_str = serialize_paths(induced_paths)
    global_edges_str = serialize_edge_tuples(global_path_edges)

    u_nbrs_str = serialize_neighbors(u_nbrs)
    v_nbrs_str = serialize_neighbors(v_nbrs)
    u_edges_str = serialize_edge_tuples(u_edges)
    v_edges_str = serialize_edge_tuples(v_edges)

    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    # Global Semantic
    inp_gs = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS]\n{paths_str}\n\n"
        f"[GLOBAL_EDGES_ON_PATHS_ONLY]\n{global_edges_str}"
        f"{skeptic_block}"
    )
    out_gs = call_agent_json(client, model, "Global Semantic Agent", prompts["global_semantic"], inp_gs)
    print_agent_io("Global Semantic Agent", inp_gs, out_gs)

    # Global Trend
    inp_gt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS]\n{paths_str}\n\n"
        f"[GLOBAL_EDGES_ON_PATHS_ONLY]\n{global_edges_str}"
        f"{skeptic_block}"
    )
    out_gt = call_agent_json(client, model, "Global Trend Agent", prompts["global_trend"], inp_gt)
    print_agent_io("Global Trend Agent", inp_gt, out_gt)

    # Local Semantic (1-hop only)
    inp_ls = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS]\n{u_nbrs_str}\n\n"
        f"[X_1HOP_EDGES_ONLY]\n{u_edges_str}\n\n"
        f"[Y_1HOP_NEIGHBORS]\n{v_nbrs_str}\n\n"
        f"[Y_1HOP_EDGES_ONLY]\n{v_edges_str}"
        f"{skeptic_block}"
    )
    out_ls = call_agent_json(client, model, "Local Semantic Agent", prompts["local_semantic"], inp_ls)
    print_agent_io("Local Semantic Agent", inp_ls, out_ls)

    # Local Trend (1-hop only)
    inp_lt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS]\n{u_nbrs_str}\n\n"
        f"[X_1HOP_EDGES_ONLY]\n{u_edges_str}\n\n"
        f"[Y_1HOP_NEIGHBORS]\n{v_nbrs_str}\n\n"
        f"[Y_1HOP_EDGES_ONLY]\n{v_edges_str}"
        f"{skeptic_block}"
    )
    out_lt = call_agent_json(client, model, "Local Trend Agent", prompts["local_trend"], inp_lt)
    print_agent_io("Local Trend Agent", inp_lt, out_lt)

    return {
        "global_semantic": out_gs,
        "global_trend": out_gt,
        "local_semantic": out_ls,
        "local_trend": out_lt,
    }


def run_integrators(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    year: int,
    ir: int,
    first_stage: Dict[str, Any],
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    inp_sa = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_SEMANTIC_OUTPUT]\n{json.dumps(first_stage['global_semantic'], ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_SEMANTIC_OUTPUT]\n{json.dumps(first_stage['local_semantic'], ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_sa = call_agent_json(client, model, "Semantic Analysis Agent", prompts["semantic_analysis"], inp_sa)
    print_agent_io("Semantic Analysis Agent", inp_sa, out_sa)

    inp_ta = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_TREND_OUTPUT]\n{json.dumps(first_stage['global_trend'], ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_TREND_OUTPUT]\n{json.dumps(first_stage['local_trend'], ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_ta = call_agent_json(client, model, "Trend Analysis Agent", prompts["trend_analysis"], inp_ta)
    print_agent_io("Trend Analysis Agent", inp_ta, out_ta)

    return {"semantic_analysis": out_sa, "trend_analysis": out_ta}


def run_impact_evaluator(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    year: int,
    ir: int,
    structure_analysis_unmasked: Any,
    semantic_analysis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    inp_ie = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT_UNMASKED]\n{json.dumps(structure_analysis_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TREND_ANALYSIS_OUTPUT]\n{json.dumps(trend_analysis, ensure_ascii=False, indent=2)}"
        f"{skeptic_block}"
    )
    out_ie = call_agent_json(client, model, "Impact Evaluator", prompts["impact_evaluator"], inp_ie)
    print_agent_io("Impact Evaluator", inp_ie, out_ie)
    return out_ie


def run_skeptic(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    year: int,
    ir: int,
    structure_analysis_unmasked: Any,
    semantic_analysis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    impact_decision: Dict[str, Any],
) -> Dict[str, Any]:
    inp_sk = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT_UNMASKED]\n{json.dumps(structure_analysis_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TREND_ANALYSIS_OUTPUT]\n{json.dumps(trend_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[IMPACT_EVALUATOR_DRAFT]\n{json.dumps(impact_decision, ensure_ascii=False, indent=2)}\n"
    )
    out_sk = call_agent_json(client, model, "Skeptic Agent", prompts["skeptic"], inp_sk)
    print_agent_io("Skeptic Agent", inp_sk, out_sk)
    return out_sk


# =========================================================
# Round runner
# =========================================================
def run_round(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    G_full: nx.Graph,
    u: str,
    v: str,
    year: int,
    ir: int,
    max_k_global: int,
    max_p_global: int,
    skeptic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    induced_paths, msg = extract_induced_paths_only(G_full, u, v, max_k=max_k_global, max_p=max_p_global)
    global_path_edges = collect_edges_on_paths(G_full, induced_paths)

    # (NEW) Structure analysis uses enclosing subgraph edge list.
    # Here we pass "global_path_edges" as the enclosing edge set (edges on induced paths only),
    # consistent with your current evidence definition.
    structure_bundle = run_structure_analysis_masked(
        client=client,
        model=model,
        prompts=prompts,
        u=u,
        v=v,
        year=year,
        ir=ir,
        enclosing_edges=global_path_edges,
    )

    u_nbrs, u_edges = one_hop_neighbor_edges(G_full, u)
    v_nbrs, v_edges = one_hop_neighbor_edges(G_full, v)

    first_stage = run_first_stage_agents(
        client=client,
        model=model,
        prompts=prompts,
        u=u,
        v=v,
        year=year,
        ir=ir,
        induced_paths=induced_paths,
        global_path_edges=global_path_edges,
        u_nbrs=u_nbrs,
        u_edges=u_edges,
        v_nbrs=v_nbrs,
        v_edges=v_edges,
        skeptic_feedback=skeptic_feedback,
    )

    integrators = run_integrators(
        client=client,
        model=model,
        prompts=prompts,
        u=u,
        v=v,
        year=year,
        ir=ir,
        first_stage=first_stage,
        skeptic_feedback=skeptic_feedback,
    )

    impact = run_impact_evaluator(
        client=client,
        model=model,
        prompts=prompts,
        u=u,
        v=v,
        year=year,
        ir=ir,
        structure_analysis_unmasked=structure_bundle["unmasked_output"],
        semantic_analysis=integrators["semantic_analysis"],
        trend_analysis=integrators["trend_analysis"],
        skeptic_feedback=skeptic_feedback,
    )

    return {
        "induced_paths_msg": msg,
        "induced_paths": induced_paths,
        "global_path_edges": global_path_edges,
        "structure": structure_bundle,
        "local_u_neighbors": u_nbrs,
        "local_v_neighbors": v_nbrs,
        "first_stage": first_stage,
        "integrators": integrators,
        "impact": impact,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=str, required=True)
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--ir", type=int, required=True)
    parser.add_argument("--refine", action="store_true")

    # induced path search constraints
    parser.add_argument("--max_k_global", type=int, default=5)
    parser.add_argument("--max_p_global", type=int, default=10)

    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY 환경변수가 비어 있습니다. 예) export OPENAI_API_KEY="YOUR_KEY"')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, "prompts")

    prompts = {
        "global_semantic": load_prompt(prompts_dir, "global_semantic.txt", args.ir),
        "global_trend": load_prompt(prompts_dir, "global_trend.txt", args.ir),
        "local_semantic": load_prompt(prompts_dir, "local_semantic.txt", args.ir),
        "local_trend": load_prompt(prompts_dir, "local_trend.txt", args.ir),
        "semantic_analysis": load_prompt(prompts_dir, "semantic_analysis.txt", args.ir),
        "trend_analysis": load_prompt(prompts_dir, "trend_analysis.txt", args.ir),
        # NEW
        "structure_analysis": load_prompt(prompts_dir, "structure_analysis.txt", args.ir),
        "impact_evaluator": load_prompt(prompts_dir, "impact_evaluator.txt", args.ir),
        "skeptic": load_prompt(prompts_dir, "skeptic.txt", args.ir),
    }

    print(f"\n[1] Loading graph data for year={args.year} ...")
    G_full, _ = graph_module.load_graph_data(base_dir, args.year)
    print(f"    Graph loaded: |V|={G_full.number_of_nodes()}, |E|={G_full.number_of_edges()}")

    if args.u not in G_full:
        raise ValueError(f"Node not found in graph: u='{args.u}'")
    if args.v not in G_full:
        raise ValueError(f"Node not found in graph: v='{args.v}'")

    client = OpenAI()

    print("\n[2] Round-0: running agents ...")
    round0 = run_round(
        client=client,
        model=args.model,
        prompts=prompts,
        G_full=G_full,
        u=args.u,
        v=args.v,
        year=args.year,
        ir=args.ir,
        max_k_global=args.max_k_global,
        max_p_global=args.max_p_global,
        skeptic_feedback=None,
    )

    final_bundle = {"round0": round0, "skeptic": None, "round1": None, "final": round0["impact"]}

    if args.refine:
        print("\n[3] Skeptic review (Round-0 -> feedback) ...")
        skeptic0 = run_skeptic(
            client=client,
            model=args.model,
            prompts=prompts,
            u=args.u,
            v=args.v,
            year=args.year,
            ir=args.ir,
            structure_analysis_unmasked=round0["structure"]["unmasked_output"],
            semantic_analysis=round0["integrators"]["semantic_analysis"],
            trend_analysis=round0["integrators"]["trend_analysis"],
            impact_decision=round0["impact"],
        )
        final_bundle["skeptic"] = skeptic0

        print("\n[4] Round-1: re-running agents with Skeptic feedback ...")
        round1 = run_round(
            client=client,
            model=args.model,
            prompts=prompts,
            G_full=G_full,
            u=args.u,
            v=args.v,
            year=args.year,
            ir=args.ir,
            max_k_global=args.max_k_global,
            max_p_global=args.max_p_global,
            skeptic_feedback=skeptic0,
        )
        final_bundle["round1"] = round1
        final_bundle["final"] = round1["impact"]

    final = final_bundle["final"]

    final_verdict = final.get("Final Verdict", None)

    print("\n============================================================")
    print("FINAL REPORT")
    print("============================================================")
    print(f"- Final verdict: {final_verdict}")
    if "decision_rationale" in final:
        print("- Rationale:")
        print(final["decision_rationale"])
    else:
        for k in ["rationale", "reasoning", "final_commentary", "final_comment"]:
            if k in final:
                print(f"- {k}:")
                print(final[k])
                break
    print("============================================================")


if __name__ == "__main__":
    main()
