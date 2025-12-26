#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
import random
import glob
from datetime import datetime
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
# Pretty logging (print ALL I/O)
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
# Prompt loader (+ fallback)
# =========================================================
def load_prompt(prompts_dir: str, filename: str, ir: int) -> str:
    path = os.path.join(prompts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return txt.replace("{IR}", str(ir))


def load_prompt_with_fallback(prompts_dir: str, primary: str, fallback: str, ir: int) -> str:
    primary_path = os.path.join(prompts_dir, primary)
    if os.path.exists(primary_path):
        return load_prompt(prompts_dir, primary, ir)
    return load_prompt(prompts_dir, fallback, ir)


# =========================================================
# Graph evidence extraction
# =========================================================
def extract_induced_paths_only(
    G: nx.Graph, u: str, v: str, max_k: int, max_p: int
) -> Tuple[List[List[str]], str]:
    """
    Use extract_path.enumerate_induced_paths.
    Return only induced paths list, plus message.
    """
    paths, _G_sub_unused, msg = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    if paths is None:
        paths = []
    return paths, msg


def collect_edges_on_paths(G: nx.Graph, paths: List[List[str]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Global edges = only consecutive edges that appear in induced paths.
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


def one_hop_neighbor_edges(
    G: nx.Graph,
    node: str,
    max_neighbors: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Local evidence:
      - 1-hop neighbors of node
      - edges (node, neighbor) only
    If neighbors > max_neighbors, randomly sample max_neighbors (reproducible).
    """
    if node not in G:
        return [], []

    nbrs_all = list(G.neighbors(node))
    if len(nbrs_all) > max_neighbors:
        rnd = random.Random(seed)
        nbrs = rnd.sample(nbrs_all, max_neighbors)
    else:
        nbrs = nbrs_all

    nbrs = sorted(nbrs, key=lambda x: str(x))

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
    Extract 'recent 3 years citation sum' weight.
    Priority:
      1) citation_weight (assumed already recent-3yr sum)
      2) weight (fallback)
      3) dict-like yearly citations (sum snapshot_year-2..snapshot_year)
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
# Serialization (for non-structure agents)
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
    [NODE_ID_A, NODE_ID_B, WEIGHT_VALUE]
    """
    out: List[List[Union[str, int]]] = []
    for a, b, d in edges:
        ma = concept_to_mask[str(a)]
        mb = concept_to_mask[str(b)]
        w = get_edge_weight_last3(d, snapshot_year=snapshot_year)
        out.append([ma, mb, int(w)])
    out.sort(key=lambda x: (x[0], x[1], int(x[2])))
    return out


def unmask_string(s: str, mask_to_concept: Dict[str, str]) -> str:
    """
    Replace node ids in flexible formats:
      node_10, Node 10, node 10, NODE_10, etc.
    """
    if not s:
        return s

    def _replace_node_token(m: re.Match) -> str:
        num = m.group(1)
        key = f"node_{num}"
        return mask_to_concept.get(key, m.group(0))

    s = re.sub(r"(?i)\bnode[_\s]?(\d+)\b", _replace_node_token, s)

    # extra safety: exact node_k tokens
    keys = sorted(mask_to_concept.keys(), key=lambda x: (-len(x), x))
    for k in keys:
        v = mask_to_concept[k]
        s = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, s)

    return s


def unmask_json(obj: Any, mask_to_concept: Dict[str, str]) -> Any:
    """
    Recursively unmask both dict keys and values.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_k = unmask_string(k, mask_to_concept) if isinstance(k, str) else k
            new_obj[new_k] = unmask_json(v, mask_to_concept)
        return new_obj
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
# Structure Analysis stage (masked input: query_pair + edge_list JSON)
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
    Masking -> Structure Analysis -> Unmasking.
    Structure agent receives ONLY:
      { "query_pair": ["node_x", "node_y"], "edge_list": [[node_a,node_b,w], ...] }
    """
    nodes: List[str] = [u, v]
    for a, b, _d in enclosing_edges:
        nodes.append(str(a))
        nodes.append(str(b))

    concept_to_mask, mask_to_concept = build_node_mask_maps(nodes)

    masked_u = concept_to_mask[str(u)]
    masked_v = concept_to_mask[str(v)]
    masked_edge_list = make_masked_edge_list(enclosing_edges, concept_to_mask, snapshot_year=year)

    masked_payload = {
        "query_pair": [masked_u, masked_v],
        "edge_list": masked_edge_list,
    }

    inp_struct = json.dumps(masked_payload, ensure_ascii=False, indent=2)

    out_struct_masked = call_agent_json(
        client=client,
        model=model,
        agent_name="Structure Analysis Agent (Masked)",
        agent_prompt=prompts["structure_analysis"],
        user_input=inp_struct,
    )
    print_agent_io("Structure Analysis Agent (Masked)", inp_struct, out_struct_masked)

    out_struct_unmasked = unmask_json(out_struct_masked, mask_to_concept)

    print_agent_io(
        "Unmasking Module (Structure Output)",
        f"[id_to_concept size]={len(mask_to_concept)} (hidden mapping applied)",
        out_struct_unmasked if isinstance(out_struct_unmasked, dict) else {"unmasked": out_struct_unmasked},
    )

    return {
        "masked_input": masked_payload,
        "masked_output": out_struct_masked,
        "unmasked_output": out_struct_unmasked,
        "mask_to_concept": mask_to_concept,  # traceability only
    }


# =========================================================
# First stage agents + integrators (trend -> temporal naming)
# =========================================================
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
) -> Dict[str, Any]:
    paths_str = serialize_paths(induced_paths)
    global_edges_str = serialize_edge_tuples(global_path_edges)

    u_nbrs_str = serialize_neighbors(u_nbrs)
    v_nbrs_str = serialize_neighbors(v_nbrs)
    u_edges_str = serialize_edge_tuples(u_edges)
    v_edges_str = serialize_edge_tuples(v_edges)

    # Global Semantic
    inp_gs = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS]\n{paths_str}\n\n"
        f"[GLOBAL_EDGES_ON_PATHS_ONLY]\n{global_edges_str}"
    )
    out_gs = call_agent_json(client, model, "Global Semantic Agent", prompts["global_semantic"], inp_gs)
    print_agent_io("Global Semantic Agent", inp_gs, out_gs)

    # Global Temporal (was Global Trend)
    inp_gt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_INDUCED_PATHS]\n{paths_str}\n\n"
        f"[GLOBAL_EDGES_ON_PATHS_ONLY]\n{global_edges_str}"
    )
    out_gt = call_agent_json(client, model, "Global Temporal Agent", prompts["global_temporal"], inp_gt)
    print_agent_io("Global Temporal Agent", inp_gt, out_gt)

    # Local Semantic
    inp_ls = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS]\n{u_nbrs_str}\n\n"
        f"[X_1HOP_EDGES_ONLY]\n{u_edges_str}\n\n"
        f"[Y_1HOP_NEIGHBORS]\n{v_nbrs_str}\n\n"
        f"[Y_1HOP_EDGES_ONLY]\n{v_edges_str}"
    )
    out_ls = call_agent_json(client, model, "Local Semantic Agent", prompts["local_semantic"], inp_ls)
    print_agent_io("Local Semantic Agent", inp_ls, out_ls)

    # Local Temporal (was Local Trend)
    inp_lt = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[X_1HOP_NEIGHBORS]\n{u_nbrs_str}\n\n"
        f"[X_1HOP_EDGES_ONLY]\n{u_edges_str}\n\n"
        f"[Y_1HOP_NEIGHBORS]\n{v_nbrs_str}\n\n"
        f"[Y_1HOP_EDGES_ONLY]\n{v_edges_str}"
    )
    out_lt = call_agent_json(client, model, "Local Temporal Agent", prompts["local_temporal"], inp_lt)
    print_agent_io("Local Temporal Agent", inp_lt, out_lt)

    return {
        "global_semantic": out_gs,
        "global_temporal": out_gt,
        "local_semantic": out_ls,
        "local_temporal": out_lt,
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
) -> Dict[str, Any]:
    inp_sa = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_SEMANTIC_OUTPUT]\n{json.dumps(first_stage['global_semantic'], ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_SEMANTIC_OUTPUT]\n{json.dumps(first_stage['local_semantic'], ensure_ascii=False, indent=2)}"
    )
    out_sa = call_agent_json(client, model, "Semantic Analysis Agent", prompts["semantic_analysis"], inp_sa)
    print_agent_io("Semantic Analysis Agent", inp_sa, out_sa)

    inp_ta = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GLOBAL_TEMPORAL_OUTPUT]\n{json.dumps(first_stage['global_temporal'], ensure_ascii=False, indent=2)}\n\n"
        f"[LOCAL_TEMPORAL_OUTPUT]\n{json.dumps(first_stage['local_temporal'], ensure_ascii=False, indent=2)}"
    )
    out_ta = call_agent_json(client, model, "Temporal Analysis Agent", prompts["temporal_analysis"], inp_ta)
    print_agent_io("Temporal Analysis Agent", inp_ta, out_ta)

    return {"semantic_analysis": out_sa, "temporal_analysis": out_ta}


# =========================================================
# Impact Evaluator + Skeptic
# =========================================================
def run_impact_evaluator(
    client: OpenAI,
    model: str,
    impact_prompt: str,
    u: str,
    v: str,
    year: int,
    ir: int,
    structure_analysis_unmasked: Any,
    semantic_analysis: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    skeptic_feedback: Optional[Dict[str, Any]] = None,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + json.dumps(skeptic_feedback, ensure_ascii=False, indent=2)

    fewshot_block = ""
    if few_shot_examples is not None:
        fewshot_block = "\n\n[FEW_SHOT_EXAMPLES]\n" + json.dumps(few_shot_examples, ensure_ascii=False, indent=2)

    inp_ie = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT_UNMASKED]\n{json.dumps(structure_analysis_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TEMPORAL_ANALYSIS_OUTPUT]\n{json.dumps(temporal_analysis, ensure_ascii=False, indent=2)}"
        f"{fewshot_block}"
        f"{skeptic_block}"
    )

    out_ie = call_agent_json(client, model, "Impact Evaluator", impact_prompt, inp_ie)
    print_agent_io("Impact Evaluator", inp_ie, out_ie)
    return out_ie


def run_skeptic(
    client: OpenAI,
    model: str,
    skeptic_prompt: str,
    u: str,
    v: str,
    year: int,
    ir: int,
    structure_analysis_unmasked: Any,
    semantic_analysis: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    impact_decision: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    inp_sk = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={year}\nIR={ir}\n\n"
        f"[GROUND_TRUTH]\n{json.dumps(ground_truth, ensure_ascii=False, indent=2)}\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT_UNMASKED]\n{json.dumps(structure_analysis_unmasked, ensure_ascii=False, indent=2)}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{json.dumps(semantic_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[TEMPORAL_ANALYSIS_OUTPUT]\n{json.dumps(temporal_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"[IMPACT_EVALUATOR_DRAFT]\n{json.dumps(impact_decision, ensure_ascii=False, indent=2)}\n"
    )
    out_sk = call_agent_json(client, model, "Skeptic Agent", skeptic_prompt, inp_sk)
    print_agent_io("Skeptic Agent", inp_sk, out_sk)
    return out_sk


# =========================================================
# Ground Truth computation
# =========================================================
def compute_ground_truth(
    G: nx.Graph,
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
) -> Dict[str, Any]:
    """
    Ground truth based on future citations in years (snapshot_year+1..snapshot_year+3).
    connected = True if edge exists at all (in your dataset representation).
    future_citation_sum defaults to 0 if yearly dict is unavailable.
    """
    if not G.has_edge(u, v):
        return {"connected": False, "future_citation_sum": 0, "impact_ge_ir": False}

    data = dict(G.get_edge_data(u, v) or {})
    future_sum = 0

    found_yearly = False
    for k in ["citations_by_year", "yearly_citations", "temporal_citations"]:
        if k in data and isinstance(data[k], dict):
            found_yearly = True
            for y in [snapshot_year + 1, snapshot_year + 2, snapshot_year + 3]:
                if y in data[k]:
                    try:
                        future_sum += int(data[k][y] or 0)
                    except Exception:
                        pass
                elif str(y) in data[k]:
                    try:
                        future_sum += int(data[k][str(y)] or 0)
                    except Exception:
                        pass
            break

    if not found_yearly:
        future_sum = 0

    return {"connected": True, "future_citation_sum": int(future_sum), "impact_ge_ir": int(future_sum) >= ir}


# =========================================================
# Verdict parsing helpers (robust)
# =========================================================
def extract_final_verdict_value(impact_json: Dict[str, Any]) -> Any:
    keys = [
        "Final Verdict",
        "final_verdict",
        "final_prediction",
        "final_prediction_label",
        "prediction",
        "verdict",
        "answer",
    ]
    for k in keys:
        if k in impact_json:
            return impact_json[k]
    return None


def verdict_to_bool(verdict_val: Any) -> Optional[bool]:
    if verdict_val is None:
        return None
    if isinstance(verdict_val, bool):
        return verdict_val
    if isinstance(verdict_val, (int, float)):
        # convention: 1=True, 0=False
        if int(verdict_val) == 1:
            return True
        if int(verdict_val) == 0:
            return False
        return None
    if isinstance(verdict_val, str):
        s = verdict_val.strip().lower()
        # common labels
        if s in ["true", "yes", "y", "positive", "pos", "1", "impact", "high", "high-impact", "high_impact"]:
            return True
        if s in ["false", "no", "n", "negative", "neg", "0", "non-impact", "low", "low-impact", "low_impact"]:
            return False
        # sometimes "impact >= ir" etc
        if "true" in s or "yes" in s:
            return True
        if "false" in s or "no" in s:
            return False
    return None


# =========================================================
# Save / Load few-shot cases
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_case_filename(year: int, ir: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = random.randint(1000, 9999)
    return f"case_year{year}_ir{ir}_{ts}_{rnd}.json"


def save_case(save_dir: str, year: int, ir: int, payload: Dict[str, Any]) -> str:
    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, make_case_filename(year, ir))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def load_cases(cases_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(cases_dir):
        return []
    files = sorted(glob.glob(os.path.join(cases_dir, "*.json")))
    cases: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # 최소 키 검증
            if not isinstance(obj, dict):
                continue
            if "IR" not in obj or "GT" not in obj:
                continue
            if "structure_analysis" not in obj or "semantic_analysis" not in obj or "temporal_analysis" not in obj:
                continue
            if "impact_evaluator_output" not in obj:
                continue
            cases.append(obj)
        except Exception:
            continue
    return cases


def sample_two_cases_by_ir(cases: List[Dict[str, Any]], target_ir: int, seed: int = 7) -> List[Dict[str, Any]]:
    if len(cases) <= 2:
        return cases[:2]

    diffs = []
    for c in cases:
        try:
            cir = int(c.get("IR", 0))
        except Exception:
            cir = 0
        diffs.append((abs(cir - target_ir), c))

    diffs.sort(key=lambda x: x[0])
    # "같거나 비슷한"을 반영: IR 차이가 작은 상위 일부에서 랜덤 샘플
    top_k = min(20, len(diffs))
    pool = [x[1] for x in diffs[:top_k]]

    rnd = random.Random(seed)
    return rnd.sample(pool, 2)


# =========================================================
# Compute analyses once (structure + semantic + temporal)
# =========================================================
def compute_all_analyses_once(
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
) -> Dict[str, Any]:
    induced_paths, msg = extract_induced_paths_only(G_full, u, v, max_k=max_k_global, max_p=max_p_global)
    global_path_edges = collect_edges_on_paths(G_full, induced_paths)

    print("\n==================== Structure Analysis Stage ====================")
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

    # Local evidence (1-hop, sample <= 10)
    u_nbrs, u_edges = one_hop_neighbor_edges(G_full, u, max_neighbors=10, seed=42)
    v_nbrs, v_edges = one_hop_neighbor_edges(G_full, v, max_neighbors=10, seed=43)

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
    )

    return {
        "induced_paths_msg": msg,
        "induced_paths": induced_paths,
        "global_path_edges": global_path_edges,
        "structure_bundle": structure_bundle,
        "first_stage": first_stage,
        "integrators": integrators,
    }


# =========================================================
# Mode 0: Skeptic <-> Impact 반복 (최대 3회), 최종 결과만 저장
# =========================================================
def run_mode0_self_refine_and_save(
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
    save_dir: str,
) -> Dict[str, Any]:
    analyses = compute_all_analyses_once(
        client=client,
        model=model,
        prompts=prompts,
        G_full=G_full,
        u=u,
        v=v,
        year=year,
        ir=ir,
        max_k_global=max_k_global,
        max_p_global=max_p_global,
    )

    gt = compute_ground_truth(G_full, u, v, snapshot_year=year, ir=ir)

    structure_unmasked = analyses["structure_bundle"]["unmasked_output"]
    semantic_analysis = analyses["integrators"]["semantic_analysis"]
    temporal_analysis = analyses["integrators"]["temporal_analysis"]

    skeptic_feedback: Optional[Dict[str, Any]] = None
    final_impact: Optional[Dict[str, Any]] = None
    attempts_used = 0

    for attempt in range(1, 4):
        attempts_used = attempt
        print(f"\n[Mode=0] Attempt {attempt}/3: Impact Evaluator ...")

        impact = run_impact_evaluator(
            client=client,
            model=model,
            impact_prompt=prompts["impact_evaluator"],
            u=u,
            v=v,
            year=year,
            ir=ir,
            structure_analysis_unmasked=structure_unmasked,
            semantic_analysis=semantic_analysis,
            temporal_analysis=temporal_analysis,
            skeptic_feedback=skeptic_feedback,
            few_shot_examples=None,
        )
        final_impact = impact

        pred_val = extract_final_verdict_value(impact)
        pred_bool = verdict_to_bool(pred_val)
        gt_bool = bool(gt.get("impact_ge_ir", False))

        if pred_bool is not None and pred_bool == gt_bool:
            print(f"\n[Mode=0] Matched GT at attempt {attempt}. Stopping.")
            break

        if attempt < 3:
            print(f"\n[Mode=0] Not matched (or verdict unparseable). Running Skeptic ...")
            skeptic_feedback = run_skeptic(
                client=client,
                model=model,
                skeptic_prompt=prompts["skeptic"],
                u=u,
                v=v,
                year=year,
                ir=ir,
                structure_analysis_unmasked=structure_unmasked,
                semantic_analysis=semantic_analysis,
                temporal_analysis=temporal_analysis,
                impact_decision=impact,
                ground_truth=gt,
            )

    # 저장: 최종 impact + structure/semantic/temporal + IR + GT만
    payload = {
        "IR": ir,
        "GT": gt,
        "structure_analysis": structure_unmasked,
        "semantic_analysis": semantic_analysis,
        "temporal_analysis": temporal_analysis,
        "impact_evaluator_output": final_impact,
    }
    out_path = save_case(save_dir, year=year, ir=ir, payload=payload)
    print(f"\n[Mode=0] Saved case to: {out_path}")

    return {
        "saved_path": out_path,
        "ground_truth": gt,
        "final_impact": final_impact,
        "attempts_used": attempts_used,
    }


# =========================================================
# Mode 1: Few-shot Impact Evaluator (저장된 케이스 2개 샘플)
# =========================================================
def run_mode1_few_shot_predict(
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
    cases_dir: str,
    fewshot_seed: int = 7,
) -> Dict[str, Any]:
    cases = load_cases(cases_dir)
    if len(cases) < 2:
        raise RuntimeError(
            f"Few-shot requires at least 2 saved cases in cases_dir='{cases_dir}'. "
            f"Current loaded cases: {len(cases)}"
        )

    few_shot_examples = sample_two_cases_by_ir(cases, target_ir=ir, seed=fewshot_seed)

    analyses = compute_all_analyses_once(
        client=client,
        model=model,
        prompts=prompts,
        G_full=G_full,
        u=u,
        v=v,
        year=year,
        ir=ir,
        max_k_global=max_k_global,
        max_p_global=max_p_global,
    )

    gt = compute_ground_truth(G_full, u, v, snapshot_year=year, ir=ir)

    structure_unmasked = analyses["structure_bundle"]["unmasked_output"]
    semantic_analysis = analyses["integrators"]["semantic_analysis"]
    temporal_analysis = analyses["integrators"]["temporal_analysis"]

    print("\n[Mode=1] Impact Evaluator (Few-shot) ...")
    impact = run_impact_evaluator(
        client=client,
        model=model,
        impact_prompt=prompts["impact_evaluator_few_shot"],
        u=u,
        v=v,
        year=year,
        ir=ir,
        structure_analysis_unmasked=structure_unmasked,
        semantic_analysis=semantic_analysis,
        temporal_analysis=temporal_analysis,
        skeptic_feedback=None,
        few_shot_examples=few_shot_examples,
    )

    return {"ground_truth": gt, "impact": impact}


# =========================================================
# Final report
# =========================================================
def print_final_report(final: Dict[str, Any], ir: int, gt: Dict[str, Any], title: str = "FINAL REPORT"):
    final_verdict = extract_final_verdict_value(final)

    print("\n============================================================")
    print(title)
    print("============================================================")
    print(f"- Model verdict: {final_verdict}")
    print(f"- IR: {ir}")

    print("\n[Ground Truth]")
    print(f"- Connected in future window (t+1..t+3): {gt.get('connected')}")
    print(f"- Future citation sum (t+1..t+3): {gt.get('future_citation_sum')}")
    print(f"- Impact ≥ IR: {gt.get('impact_ge_ir')}")

    if "decision_rationale" in final:
        print("\n[Rationale]")
        print(final["decision_rationale"])
    else:
        for k in ["rationale", "reasoning", "final_commentary", "final_comment", "Analytical Reasoning"]:
            if k in final:
                print(f"\n[{k}]")
                val = final[k]
                print(val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, indent=2))
                break
        else:
            print("\n[Raw Final JSON]")
            print(json.dumps(final, ensure_ascii=False, indent=2))

    print("============================================================")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, choices=[0, 1], required=True, help="0: self-refine+save, 1: few-shot predict")
    parser.add_argument("--u", type=str, required=True)
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--ir", type=int, required=True)

    parser.add_argument("--max_k_global", type=int, default=5)
    parser.add_argument("--max_p_global", type=int, default=10)

    parser.add_argument("--model", type=str, default="gpt-4o")

    # mode=0: save folder, mode=1: cases folder
    parser.add_argument("--save_dir", type=str, default="fewshot_cases", help="(mode=0) folder to save cases")
    parser.add_argument("--cases_dir", type=str, default="fewshot_cases", help="(mode=1) folder to load cases")
    parser.add_argument("--fewshot_seed", type=int, default=7)

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY 환경변수가 비어 있습니다. 예) export OPENAI_API_KEY="YOUR_KEY"')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(base_dir, "prompts")

    # --- Prompts (temporal naming + fallback) ---
    prompts: Dict[str, str] = {
        "global_semantic": load_prompt(prompts_dir, "global_semantic.txt", args.ir),
        "local_semantic": load_prompt(prompts_dir, "local_semantic.txt", args.ir),
        "semantic_analysis": load_prompt(prompts_dir, "semantic_analysis.txt", args.ir),

        # temporal prompts: try *_temporal.txt else fallback to *_trend.txt
        "global_temporal": load_prompt_with_fallback(prompts_dir, "global_temporal.txt", "global_trend.txt", args.ir),
        "local_temporal": load_prompt_with_fallback(prompts_dir, "local_temporal.txt", "local_trend.txt", args.ir),
        "temporal_analysis": load_prompt_with_fallback(prompts_dir, "temporal_analysis.txt", "trend_analysis.txt", args.ir),

        # structure prompt (your updated file)
        "structure_analysis": load_prompt(prompts_dir, "structure_analysis.txt", args.ir),

        # impact evaluator prompts
        "impact_evaluator": load_prompt(prompts_dir, "impact_evaluator.txt", args.ir),
        "impact_evaluator_few_shot": load_prompt(prompts_dir, "impact_evaluator_few_shot.txt", args.ir),

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

    if args.mode == 0:
        print("\n[2] Mode=0: Self-refine (<=3) then save final case ...")
        result = run_mode0_self_refine_and_save(
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
            save_dir=args.save_dir,
        )

        final = result["final_impact"] or {}
        gt = result["ground_truth"]
        print_final_report(final, ir=args.ir, gt=gt, title="FINAL REPORT (MODE=0)")

    else:
        print("\n[2] Mode=1: Few-shot prediction using saved cases ...")
        result = run_mode1_few_shot_predict(
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
            cases_dir=args.cases_dir,
            fewshot_seed=args.fewshot_seed,
        )

        final = result["impact"]
        gt = result["ground_truth"]
        print_final_report(final, ir=args.ir, gt=gt, title="FINAL REPORT (MODE=1)")



if __name__ == "__main__":
    main()
