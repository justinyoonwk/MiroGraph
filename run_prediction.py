#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

import extract_path

# Load environment variables from .env file
load_dotenv()

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
    max_tokens: int = 2048,
    temperature: float = 0.2,
    use_python_execution: bool = False,
) -> str:
    # GPT 5.1의 메시지 역할:
    # - system: 모델의 전반적인 행동과 목적 정의 (최고 우선순위)
    # - developer: 출력 형식, 도구 사용 등 세부 조정 (중간 우선순위)
    # - user: 사용자 입력 (최저 우선순위)
    
    # JSON 제한 제거: agent가 자유롭게 텍스트를 출력할 수 있도록 함
    
    # 모델 버전에 따라 적절한 role 사용
    if "gpt-5" in model.lower() or "o3" in model.lower():
        # GPT 5.x 또는 O3 모델: developer role 사용
        messages = [
            {"role": "system", "content": f"[AGENT]\n{agent_name}\n\n[PROMPT]\n{agent_prompt}"},
            {"role": "user", "content": user_input},
        ]
    else:
        # 기존 모델: system role 사용
        messages = [
            {"role": "system", "content": f"[AGENT]\n{agent_name}\n\n[PROMPT]\n{agent_prompt}"},
            {"role": "user", "content": user_input},
        ]

    try:
        # Python 코드 실행을 위한 function calling 도구 정의
        tools = None
        if use_python_execution:
            tools = [{
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Execute Python code and return the output. Use this to run Python scripts for data analysis, graph analysis, or any computational tasks. The code should be complete and executable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The complete Python code to execute. Must be valid Python code that can be run independently."
                            }
                        },
                        "required": ["code"]
                    }
                }
            }]
        
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # response_format 제거: JSON 제한 없음
            temperature=temperature if not use_python_execution else 0.0,  # Python 실행 시 temperature=0.0
            max_completion_tokens=max_tokens,
            tools=tools,
        )
        
        message = resp.choices[0].message
        raw = message.content or ""
        finish_reason = resp.choices[0].finish_reason
        
        # tool_calls가 있는 경우: Python 코드 실행 후 결과를 포함한 최종 응답을 받기
        if tools and hasattr(message, 'tool_calls') and message.tool_calls:
            # finish_reason이 "tool_calls"이고 raw가 비어있는 것은 정상 (tool_calls만 있고 content가 없는 것은 정상)
            # 이 경우는 경고를 출력하지 않음
            # tool_calls를 messages에 추가
            messages.append(message)
            
            # Python 코드 실행 및 결과 수집
            tool_outputs = []
            for tool_call in message.tool_calls:
                tool_id = None
                python_code = None
                
                # tool_call 구조 확인
                if isinstance(tool_call, dict):
                    tool_id = tool_call.get("id", "")
                    function = tool_call.get("function", {})
                    if function:
                        function_name = function.get("name", "")
                        arguments = function.get("arguments", "")
                        if function_name == "execute_python_code" and arguments:
                            try:
                                import json
                                args_dict = json.loads(arguments)
                                python_code = args_dict.get("code", "")
                            except:
                                python_code = arguments
                else:
                    # tool_call 객체인 경우
                    tool_id = getattr(tool_call, 'id', '')
                    if hasattr(tool_call, 'function'):
                        func = tool_call.function
                        func_name = getattr(func, 'name', '')
                        if func_name == "execute_python_code":
                            if hasattr(func, 'arguments'):
                                try:
                                    import json
                                    args_dict = json.loads(func.arguments)
                                    python_code = args_dict.get("code", "")
                                except:
                                    python_code = func.arguments
                
                # Python 코드 실행
                import subprocess
                import tempfile
                import os
                output = ""
                try:
                    if python_code:
                        # 임시 파일에 Python 코드 작성
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
                            f.write(python_code)
                            temp_file = f.name
                        
                        try:
                            # Python 코드 실행 (보안을 위해 제한된 환경에서 실행)
                            result = subprocess.run(
                                ["python3", temp_file],
                                capture_output=True,
                                text=True,
                                timeout=30,  # 30초 타임아웃
                                cwd="/tmp"  # 임시 디렉토리에서 실행
                            )
                            output = result.stdout + result.stderr
                            if result.returncode != 0:
                                output = f"[Error code: {result.returncode}]\n{output}"
                        finally:
                            # 임시 파일 삭제
                            try:
                                os.unlink(temp_file)
                            except:
                                pass
                    else:
                        output = "No Python code found in tool_call"
                        print(f"[WARNING] {agent_name}: tool_call에 Python 코드가 없음")
                    
                    if tool_id:
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": output
                        })
                except Exception as e:
                    print(f"[ERROR] {agent_name}: Python code execution failed - {str(e)}")
                    if tool_id:
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": f"Error executing Python code: {str(e)}"
                        })
            
            # tool_outputs를 messages에 추가
            for tool_output in tool_outputs:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_output["tool_call_id"],
                    "content": tool_output["output"]
                })
            
            # 최종 응답 받기 (재귀 호출 없음 - 비용 절감)
            follow_up_resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # Python 실행 후 분석은 deterministic하게
                max_completion_tokens=max_tokens,
                tools=tools,
            )
            follow_up_message = follow_up_resp.choices[0].message
            raw = follow_up_message.content or ""
            finish_reason = follow_up_resp.choices[0].finish_reason
            
            # tool_calls 처리 후 최종 응답이 비어있는 경우 경고
            if not raw or not raw.strip():
                print(f"[WARNING] {agent_name}: Empty response from API after tool_calls")
                print(f"  Finish reason: {finish_reason}")
                return ""
        
        # 빈 응답 체크 (tool_calls가 없는 경우)
        # tool_calls가 있는 경우는 이미 위에서 처리했으므로 여기서는 tool_calls가 없는 경우만 체크
        if not raw or not raw.strip():
            # finish_reason이 "tool_calls"인 경우는 tool_calls 처리 블록에서 처리했으므로 여기서는 발생하지 않음
            print(f"[WARNING] {agent_name}: Empty response from API")
            print(f"  Finish reason: {finish_reason if 'finish_reason' in locals() else 'unknown'}")
            if tools:
                print(f"  Tools were provided")
            return ""
        
        # Raw text를 그대로 반환 (JSON 파싱 제거)
        return raw.strip()
            
    except Exception as e:
        print(f"[ERROR] {agent_name}: API call failed - {str(e)}")
        return ""


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


def extract_verdict_from_text(text: str) -> Optional[str]:
    """Raw text에서 verdict를 추출 (Yes/No)"""
    if not text:
        return None
    text_lower = text.lower()
    # "final_verdict" 키워드 주변에서 찾기
    if '"final_verdict"' in text_lower or "'final_verdict'" in text_lower:
        # JSON 형식인 경우
        import re
        pattern = r'["\']final_verdict["\']\s*:\s*["\'](yes|no)["\']'
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).capitalize()
    # 텍스트에서 직접 찾기
    if "final verdict" in text_lower or "verdict:" in text_lower:
        if "yes" in text_lower and text_lower.index("yes") < text_lower.index("no") if "no" in text_lower else True:
            return "Yes"
        if "no" in text_lower:
            return "No"
    # 마지막으로 전체 텍스트에서 yes/no 찾기
    if "yes" in text_lower:
        return "Yes"
    if "no" in text_lower:
        return "No"
    return None


def parse_impact_output(raw_text: str) -> Dict[str, Any]:
    """Impact Evaluator의 raw text 출력을 파싱하여 dict 형태로 변환"""
    if not raw_text:
        return {"impact_reasoning": "", "final_verdict": "No"}
    
    # JSON 파싱 시도
    parsed = parse_llm_json(raw_text)
    if isinstance(parsed, dict):
        verdict = parsed.get("final_verdict", None)
        reasoning = parsed.get("impact_reasoning", raw_text)
        if not verdict:
            verdict = extract_verdict_from_text(raw_text)
        return {
            "impact_reasoning": reasoning if reasoning else raw_text,
            "final_verdict": verdict or "No"
        }
    
    # JSON 파싱 실패 시 텍스트에서 추출
    verdict = extract_verdict_from_text(raw_text)
    return {
        "impact_reasoning": raw_text,
        "final_verdict": verdict or "No"
    }


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
    skeptic_feedback: Optional[str] = None,
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
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + skeptic_feedback

    # Structure Analysis (masked)
    inp_struct = json.dumps(masked_payload, ensure_ascii=False)
    out_struct_masked = call_agent_json(
        client=client,
        model=model,
        agent_name="Structure Analysis Agent",
        agent_prompt=prompts["structure_analysis"],
        user_input=inp_struct,
        max_tokens=2048,  # max_completion_tokens로 전달됨
        temperature=0.2,
        use_python_execution=True,  # Python 코드 실행 기능 활성화
    )
    # Raw text를 그대로 사용 (unmask는 텍스트에서 node_id를 concept 이름으로 변환)
    out_struct_unmasked = unmask_text(out_struct_masked, id_to_concept)

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
        f"[GLOBAL_SEMANTIC_OUTPUT]\n{out_gs}\n\n"
        f"[LOCAL_SEMANTIC_OUTPUT]\n{out_ls}"
        f"{skeptic_block}"
    )
    out_sa = call_agent_json(client, model, "Semantic Analysis Agent", prompts["semantic_analysis"], inp_sa)

    inp_ta = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GLOBAL_TREND_OUTPUT]\n{out_gt}\n\n"
        f"[LOCAL_TREND_OUTPUT]\n{out_lt}"
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
    structure_unmasked: str,
    semantic_analysis: str,
    trend_analysis: str,
    skeptic_feedback: Optional[str] = None,
    demonstrations: Optional[List[Dict[str, Any]]] = None,
    no_structure: bool = False,
    no_semantic: bool = False,
    no_trend: bool = False,
) -> Dict[str, Any]:
    skeptic_block = ""
    if skeptic_feedback:
        skeptic_block = "\n\n[SKEPTIC_FEEDBACK]\n" + skeptic_feedback
    
    demonstrations_block = ""
    if demonstrations:
        demos_text = []
        for i, demo in enumerate(demonstrations, 1):
            demo_pair = demo.get("pair", {})
            demo_gt = demo.get("ground_truth_label", -1)
            demo_reasoning = demo.get("final_impact_output", {}).get("impact_reasoning", "")
            demo_verdict = demo.get("final_impact_output", {}).get("final_verdict", "")
            demos_text.append(
                f"Example {i}:\n"
                f"  Pair: {demo_pair.get('concept1', '')} <-> {demo_pair.get('concept2', '')}\n"
                f"  Ground Truth: {demo_gt} ({'High Impact (IR≥threshold)' if demo_gt == 1 else 'Low Impact (IR<threshold)'})\n"
                f"  Reasoning: {demo_reasoning}\n"
                f"  Verdict: {demo_verdict}"
            )
        demonstrations_block = "\n\n[FEW-SHOT EXAMPLES]\n" + "\n\n".join(demos_text)

    # Build input with optional ablation
    parts = [f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n"]
    
    if not no_structure:
        parts.append(f"[STRUCTURE_ANALYSIS_OUTPUT]\n{structure_unmasked}\n\n")
    
    if not no_semantic:
        parts.append(f"[SEMANTIC_ANALYSIS_OUTPUT]\n{semantic_analysis}\n\n")
    
    if not no_trend:
        parts.append(f"[TREND_ANALYSIS_OUTPUT]\n{trend_analysis}")
    
    parts.append(skeptic_block)
    parts.append(demonstrations_block)
    
    inp_ie = "".join(parts)
    out_ie_raw = call_agent_json(client, model, "Impact Evaluator", prompts["impact_evaluator"], inp_ie, max_tokens=2048)  # max_completion_tokens로 전달됨
    
    # Impact Evaluator의 출력을 JSON 형식으로 파싱하여 dict로 반환
    return parse_impact_output(out_ie_raw)


def run_impact_evaluator_batch(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    batch_items: List[Dict[str, Any]],
    max_workers: int = 5,
    no_structure: bool = False,
    no_semantic: bool = False,
    no_trend: bool = False,
) -> List[Dict[str, Any]]:
    """
    배치 단위로 Impact Evaluator를 병렬 실행합니다.
    
    Args:
        client: OpenAI client
        model: 모델 이름
        prompts: 프롬프트 딕셔너리
        batch_items: 배치 아이템 리스트, 각 아이템은 다음 키를 포함:
            - u, v: 개념 쌍
            - snapshot_year: 스냅샷 연도
            - ir: IR 값
            - structure_unmasked: 구조 분석 결과
            - semantic_analysis: 의미 분석 결과
            - trend_analysis: 트렌드 분석 결과
            - skeptic_feedback: (optional) skeptic 피드백
            - demonstrations: (optional) few-shot examples
        max_workers: 병렬 처리 최대 워커 수
        no_structure: Structure analysis 제거 여부
        no_semantic: Semantic analysis 제거 여부
        no_trend: Trend analysis 제거 여부
    
    Returns:
        각 아이템에 대한 Impact Evaluator 출력 리스트 (입력 순서 유지)
    """
    def process_item(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """단일 아이템 처리 (인덱스와 함께 반환하여 순서 유지)"""
        idx = item["_batch_idx"]
        u = item["u"]
        v = item["v"]
        snapshot_year = item["snapshot_year"]
        ir = item["ir"]
        structure_unmasked = item["structure_unmasked"]
        semantic_analysis = item["semantic_analysis"]
        trend_analysis = item["trend_analysis"]
        skeptic_feedback = item.get("skeptic_feedback")
        demonstrations = item.get("demonstrations")
        
        result = run_impact_evaluator_once(
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
            skeptic_feedback=skeptic_feedback,
            demonstrations=demonstrations,
            no_structure=no_structure,
            no_semantic=no_semantic,
            no_trend=no_trend,
        )
        return (idx, result)
    
    # 인덱스 추가
    indexed_items = []
    for idx, item in enumerate(batch_items):
        item_copy = item.copy()
        item_copy["_batch_idx"] = idx
        indexed_items.append(item_copy)
    
    # 병렬 처리
    results_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in indexed_items}
        
        for future in as_completed(future_to_item):
            try:
                idx, result = future.result()
                results_dict[idx] = result
            except Exception as e:
                item = future_to_item[future]
                print(f"[ERROR] Failed to process {item['u']} <-> {item['v']}: {str(e)}")
                # 에러 발생 시 빈 결과 반환
                idx = item["_batch_idx"]
                results_dict[idx] = {"impact_reasoning": f"[ERROR] {str(e)}", "final_verdict": "No"}
    
    # 인덱스 순서대로 결과 반환
    results = [results_dict[i] for i in range(len(batch_items))]
    return results


def run_skeptic_with_gt(
    client: OpenAI,
    model: str,
    prompts: Dict[str, str],
    u: str,
    v: str,
    snapshot_year: int,
    ir: int,
    structure_unmasked: str,
    semantic_analysis: str,
    trend_analysis: str,
    impact_draft: str,
    gt_label: int,
) -> str:
    """
    mode 0 training: Skeptic MUST know GT.
    """
    inp_sk = (
        f"[TARGET]\nX={u}\nY={v}\nSNAPSHOT_YEAR={snapshot_year}\nIR={ir}\n\n"
        f"[GROUND_TRUTH_LABEL]\n{gt_label}\n"
        f"(1 means YES/high-impact, 0 means NO/non-impact)\n\n"
        f"[STRUCTURE_ANALYSIS_OUTPUT]\n{structure_unmasked}\n\n"
        f"[SEMANTIC_ANALYSIS_OUTPUT]\n{semantic_analysis}\n\n"
        f"[TREND_ANALYSIS_OUTPUT]\n{trend_analysis}\n\n"
        f"[IMPACT_EVALUATOR_DRAFT]\n{impact_draft}\n"
    )
    out_sk = call_agent_json(client, model, "Skeptic Agent (GT-known)", prompts["skeptic"], inp_sk, max_tokens=2048)  # max_completion_tokens로 전달됨
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
    no_structure: bool = False,
    no_semantic: bool = False,
    no_trend: bool = False,
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
    skeptic_feedbacks: List[str] = []

    feedback: Optional[str] = None

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
            no_structure=no_structure,
            no_semantic=no_semantic,
            no_trend=no_trend,
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
            impact_draft=json.dumps(impact_out, ensure_ascii=False, indent=2),  # JSON 형식으로 변환하여 skeptic에 전달
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
    parser.add_argument("--dataset", type=str, default="kdd", help="Dataset name (default: kdd)")

    # canonical args
    parser.add_argument("--n_pairs", type=int, default=None, help="Number of pairs to sample (omit to use all pairs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_k_global", type=int, default=5)
    parser.add_argument("--max_p_global", type=int, default=10)
    parser.add_argument("--m_local", type=int, default=30)
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing pairs (also sets max_workers)")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (if None, uses batch_size)")

    # ablation study options
    parser.add_argument("--no_structure", action="store_true", help="Remove structure analysis from impact evaluation")
    parser.add_argument("--no_semantic", action="store_true", help="Remove semantic analysis from impact evaluation")
    parser.add_argument("--no_trend", action="store_true", help="Remove trend analysis from impact evaluation")

    # API key selection
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (direct input, overrides --api_key_name)")
    parser.add_argument("--api_key_name", type=str, default=None, help="OpenAI API key name from .env (e.g., '1' for OPENAI_API_KEY_1, '2' for OPENAI_API_KEY_2, or 'default' for OPENAI_API_KEY)")

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

    # Load .env file if exists (already loaded at top, but ensure it's loaded from base_dir)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # API key 선택 로직
    api_key = None
    if args.api_key:
        # 직접 키 입력
        api_key = args.api_key
        print(f"[INFO] Using API key from --api_key argument")
    elif args.api_key_name:
        # .env에서 키 이름으로 선택
        if args.api_key_name == "default":
            env_key_name = "OPENAI_API_KEY"
        else:
            env_key_name = f"OPENAI_API_KEY_{args.api_key_name}"
        api_key = os.getenv(env_key_name)
        if not api_key:
            raise RuntimeError(f'{env_key_name}가 .env 파일에 설정되지 않았습니다. .env 파일에 {env_key_name}="YOUR_KEY"를 추가하세요.')
        print(f"[INFO] Using API key from .env: {env_key_name}")
    else:
        # 기본값: OPENAI_API_KEY
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY="YOUR_KEY"를 추가하거나 --api_key 또는 --api_key_name을 사용하세요.')
        print(f"[INFO] Using default API key from OPENAI_API_KEY")

    # dataset paths
    dataset_dir = os.path.join(base_dir, args.dataset, args.setting)
    graph_path = os.path.join(dataset_dir, "concept_graph.csv")

    if args.setting == "inductive" and args.mode == 1:
        # inductive test uses concept_graph_test.csv
        graph_path = os.path.join(dataset_dir, "concept_graph_test.csv")

    # pair file for sampling
    if args.mode == 0:
        pair_path = os.path.join(dataset_dir, f"train_data_{args.ir}.csv")
    else:
        pair_path = os.path.join(dataset_dir, f"test_data_{args.ir}.csv")

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

    # load pairs + sampling
    df_pairs = load_pairs_file(pair_path)
    if args.n_pairs is None:
        # Use all pairs
        sampled = df_pairs.copy()
        print(f"[INFO] Using all {len(sampled)} pairs (no sampling)")
    else:
        # Balanced sampling
        sampled = balanced_sample_pairs(df_pairs, args.n_pairs, args.seed)
        print(f"[INFO] Sampled {len(sampled)} pairs (balanced)")

    client = OpenAI(api_key=api_key)

    # run
    y_true: List[int] = []
    y_pred: List[int] = []
    records: List[Dict[str, Any]] = []

    # mode 0: mandatory skeptic loop with parallel processing
    if args.mode == 0:
        def process_pair(args_tuple):
            """단일 페어 처리 함수 (병렬 처리용)"""
            i, row, pair_idx = args_tuple
            u, v, gt = row.concept1, row.concept2, int(row.label)
            
            try:
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
                    no_structure=args.no_structure,
                    no_semantic=args.no_semantic,
                    no_trend=args.no_trend,
                )
                
                pred_int = rec.get("final_prediction_int", None)
                if pred_int is None:
                    pred_int = 0  # conservative fallback
                
                return (pair_idx, {
                    "rec": rec,
                    "pred_int": pred_int,
                    "gt": gt,
                    "u": u,
                    "v": v,
                    "index": i,
                })
            except Exception as e:
                print(f"[ERROR] Failed to process pair {i} ({u} <-> {v}): {str(e)}")
                return (pair_idx, {
                    "rec": None,
                    "pred_int": 0,
                    "gt": gt,
                    "u": u,
                    "v": v,
                    "index": i,
                    "error": str(e),
                })
        
        # 출력 경로 결정 (저장 및 재시작을 위해 먼저 결정)
        # Ablation study 정보 출력
        ablation_info = []
        if args.no_structure:
            ablation_info.append("no_structure")
        if args.no_semantic:
            ablation_info.append("no_semantic")
        if args.no_trend:
            ablation_info.append("no_trend")
        
        if ablation_info:
            print(f"[INFO] Ablation study: {' + '.join(ablation_info)}")
        else:
            print(f"[INFO] Full analysis (structure + semantic + trend)")
        
        if args.mode0_out:
            output_path = args.mode0_out
        else:
            demonstrations_dir = os.path.join(base_dir, "demonstrations")
            dataset_demo_dir = os.path.join(demonstrations_dir, args.dataset)
            os.makedirs(dataset_demo_dir, exist_ok=True)
            
            # Ablation study에 따른 파일명 변경
            ablation_suffix = ""
            if args.no_structure:
                ablation_suffix += "_no_struct"
            if args.no_semantic:
                ablation_suffix += "_no_sem"
            if args.no_trend:
                ablation_suffix += "_no_trend"
            
            # n_pairs가 None이 아니면 파일명에 포함
            n_pairs_suffix = f"_n{args.n_pairs}" if args.n_pairs is not None else ""
            
            output_path = os.path.join(dataset_demo_dir, f"mode0_ir{args.ir}{ablation_suffix}{n_pairs_suffix}.jsonl")
        
        # 기존 결과 불러오기 (재시작 지원)
        processed_indices = set()
        existing_records = []
        existing_y_true = []
        existing_y_pred = []
        if os.path.exists(output_path):
            print(f"[INFO] 기존 결과 파일 발견: {output_path}")
            print(f"[INFO] 기존 결과를 불러와서 재시작합니다...")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            # pair 정보로 인덱스 추출 (bundle에서 u, v로 매칭)
                            if "bundle" in rec and "u" in rec["bundle"] and "v" in rec["bundle"]:
                                u = rec["bundle"]["u"]
                                v = rec["bundle"]["v"]
                                # sampled에서 해당 페어 찾기
                                for idx, row in sampled.iterrows():
                                    if row.concept1 == u and row.concept2 == v:
                                        processed_indices.add(idx)
                                        existing_records.append(rec)
                                        # ground truth와 prediction도 저장
                                        if "ground_truth_label" in rec:
                                            existing_y_true.append(rec["ground_truth_label"])
                                            if "final_prediction_int" in rec:
                                                existing_y_pred.append(rec["final_prediction_int"])
                                            else:
                                                existing_y_pred.append(0)
                                        break
                        except json.JSONDecodeError:
                            continue
            print(f"[INFO] 이미 처리된 페어: {len(processed_indices)}개")
        
        # 배치 단위로 병렬 처리
        # batch_size: 한 번에 처리할 페어 수 (실제 동시 처리 수)
        # max_workers: 배치 사이즈와 동일하게 설정 (명시적으로 지정된 경우만 사용)
        total_pairs = len(sampled)
        batch_size = args.batch_size
        max_workers = args.max_workers if args.max_workers is not None else batch_size
        num_batches = (total_pairs + batch_size - 1) // batch_size
        
        print(f"[INFO] Processing {total_pairs} pairs in {num_batches} batches")
        print(f"[INFO] Batch size: {batch_size} (동시 처리 수: {max_workers})")
        print(f"[INFO] Output path: {output_path}")
        
        # 기존 결과를 records에 추가
        records = existing_records.copy()
        y_true = existing_y_true.copy()
        y_pred = existing_y_pred.copy()
        all_results_dict = {}
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_pairs)
            batch_sampled = sampled.iloc[batch_start:batch_end]
            
            # 이미 처리된 페어 필터링
            batch_sampled_filtered = batch_sampled[~batch_sampled.index.isin(processed_indices)]
            
            if len(batch_sampled_filtered) == 0:
                print(f"\n[Batch {batch_idx + 1}/{num_batches}] 모든 페어가 이미 처리됨. 건너뜁니다.")
                continue
            
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing pairs {batch_start + 1}-{batch_end} (동시 처리: {max_workers}개, 새로 처리: {len(batch_sampled_filtered)}개)...")
            
            # 배치 내 페어 인자 준비 (필터링된 것만)
            pair_args = []
            for local_idx, (idx, row) in enumerate(batch_sampled_filtered.iterrows()):
                # 원본 인덱스 유지
                original_idx = sampled.index.get_loc(idx) + 1  # 1-based index
                pair_args.append((original_idx, row, batch_start + local_idx))
            
            batch_results_dict = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {executor.submit(process_pair, args): args for args in pair_args}
                
                # 진행상황 표시
                with tqdm(total=len(pair_args), desc=f"  Batch {batch_idx + 1}", leave=False) as pbar:
                    for future in as_completed(future_to_pair):
                        try:
                            pair_idx, result = future.result()
                            batch_results_dict[pair_idx] = result
                            pbar.update(1)
                        except Exception as e:
                            args = future_to_pair[future]
                            print(f"\n[ERROR] Unexpected error processing pair: {str(e)}")
                            batch_results_dict[args[2]] = {
                                "rec": None,
                                "pred_int": 0,
                                "gt": args[1].label,
                                "u": args[1].concept1,
                                "v": args[1].concept2,
                                "index": args[0],
                                "error": str(e),
                            }
                            pbar.update(1)
            
            all_results_dict.update(batch_results_dict)
            
            # 배치 완료 후 결과 요약 출력
            batch_records = []
            batch_y_true = []
            batch_y_pred = []
            for idx in sorted(batch_results_dict.keys()):
                result = batch_results_dict[idx]
                if result["rec"] is not None:
                    batch_records.append(result["rec"])
                batch_y_true.append(result["gt"])
                batch_y_pred.append(result["pred_int"])
            
            # 배치 결과 요약
            batch_metrics = compute_binary_metrics(batch_y_true, batch_y_pred)
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Completed: {batch_start + 1}-{batch_end}")
            print(f"  Accuracy={batch_metrics['accuracy']:.4f} | Precision={batch_metrics['precision']:.4f} | Recall={batch_metrics['recall']:.4f} | F1={batch_metrics['f1']:.4f}")
            
            # 배치 내 각 페어 결과 출력
            print(f"  Results:")
            for idx in sorted(batch_results_dict.keys()):
                result = batch_results_dict[idx]
                status = "✓" if result["pred_int"] == result["gt"] else "✗"
                print(f"    [{result['index']:4d}] {status} {result['u']} <-> {result['v']}: pred={result['pred_int']}, GT={result['gt']}")
            
            # 전체 records에 추가
            records.extend(batch_records)
            y_true.extend(batch_y_true)
            y_pred.extend(batch_y_pred)
            
            # 배치 완료 시마다 저장 (append 모드)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as f:  # append 모드
                for rec in batch_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()  # 즉시 디스크에 쓰기
            
            # 처리된 인덱스 업데이트
            for idx in batch_sampled_filtered.index:
                processed_indices.add(idx)
            
            print(f"  [Saved] 배치 결과 저장됨: {len(batch_records)}개 레코드 추가 (총 {len(records)}개)")

        # (optional but useful) summary metrics
        metrics = compute_binary_metrics(y_true, y_pred)
        print("============================================================")
        print(f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
        print("============================================================")
        print(f"[Saved] mode0 examples -> {output_path}")
        return

    # mode 1: batch processing for impact evaluation
    # Ablation study 정보 출력
    ablation_info = []
    if args.no_structure:
        ablation_info.append("no_structure")
    if args.no_semantic:
        ablation_info.append("no_semantic")
    if args.no_trend:
        ablation_info.append("no_trend")
    
    if ablation_info:
        print(f"[INFO] Ablation study: {' + '.join(ablation_info)}")
    else:
        print(f"[INFO] Full analysis (structure + semantic + trend)")
    
    # Step 0: Load demonstrations for few-shot learning
    demonstrations = []
    demonstrations_path = None
    if args.mode0_out:
        # 사용자가 지정한 demonstrations 경로가 있으면 사용
        demonstrations_path = args.mode0_out
    else:
        # 자동 경로: demonstrations/{dataset}/mode0_ir{ir}{ablation_suffix}.jsonl (transductive/inductive 공통)
        # Mode 0에서 저장할 때와 동일한 ablation_suffix 사용
        demonstrations_dir = os.path.join(base_dir, "demonstrations")
        dataset_demo_dir = os.path.join(demonstrations_dir, args.dataset)
        
        # Ablation study에 따른 파일명 변경 (Mode 0과 동일한 로직)
        ablation_suffix = ""
        if args.no_structure:
            ablation_suffix += "_no_struct"
        if args.no_semantic:
            ablation_suffix += "_no_sem"
        if args.no_trend:
            ablation_suffix += "_no_trend"
        
        demonstrations_path = os.path.join(dataset_demo_dir, f"mode0_ir{args.ir}{ablation_suffix}.jsonl")
    
    # demonstrations 로드
    if os.path.exists(demonstrations_path):
        print(f"[INFO] Loading demonstrations from: {demonstrations_path}")
        with open(demonstrations_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        demo = json.loads(line)
                        # 필요한 필드가 있는지 확인
                        if "ground_truth_label" in demo and "final_impact_output" in demo:
                            demonstrations.append(demo)
                    except json.JSONDecodeError:
                        continue
        print(f"[INFO] Loaded {len(demonstrations)} demonstrations")
    else:
        print(f"[WARNING] Demonstrations file not found: {demonstrations_path}")
        print(f"[WARNING] Continuing without few-shot examples...")
    
    # 출력 경로 결정 (저장을 위해 먼저 결정)
    output_dir = os.path.join(base_dir, "results")
    dataset_result_dir = os.path.join(output_dir, args.dataset)
    setting_result_dir = os.path.join(dataset_result_dir, args.setting)
    os.makedirs(setting_result_dir, exist_ok=True)
    
    # Ablation study에 따른 파일명 변경
    ablation_suffix = ""
    if args.no_structure:
        ablation_suffix += "_no_struct"
    if args.no_semantic:
        ablation_suffix += "_no_sem"
    if args.no_trend:
        ablation_suffix += "_no_trend"
    
    output_path = os.path.join(setting_result_dir, f"mode1_ir{args.ir}{ablation_suffix}.jsonl")
    
    # 기존 결과 파일이 있으면 삭제 (새로 시작)
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"[INFO] 기존 결과 파일 삭제: {output_path}")
    
    # 메타데이터 저장 (파일 첫 줄)
    metadata = {
        "setting": args.setting,
        "snapshot_year": snapshot_year,
        "ir": args.ir,
        "dataset": args.dataset,
        "model": args.model,
        "total_pairs": len(sampled),
        "ablation": {
            "no_structure": args.no_structure,
            "no_semantic": args.no_semantic,
            "no_trend": args.no_trend,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
    
    # 배치 단위로 병렬 처리 (Mode 0과 동일한 구조)
    # batch_size: 한 번에 처리할 페어 수 (실제 동시 처리 수)
    # max_workers: 배치 사이즈와 동일하게 설정 (명시적으로 지정된 경우만 사용)
    total_pairs = len(sampled)
    batch_size = args.batch_size
    max_workers = args.max_workers if args.max_workers is not None else batch_size
    num_batches = (total_pairs + batch_size - 1) // batch_size
    
    print(f"[INFO] Processing {total_pairs} pairs in {num_batches} batches")
    print(f"[INFO] Batch size: {batch_size} (동시 처리 수: {max_workers})")
    print(f"[INFO] Output path: {output_path}")
    
    all_impacts = []
    all_pairs_info = []
    
    def process_pair_mode1(pair_arg):
        """Mode 1에서 단일 페어 처리"""
        i, row = pair_arg
        u, v, gt = row.concept1, row.concept2, int(row.label)
        
        try:
            # Step 1: Bundle 수집
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
            
            # Step 2: Demonstrations 선택 (prediction이 ground truth를 맞춘 경우만)
            selected_demos = []
            if demonstrations:
                # GT=0인 케이스 (IR 이하) 중 prediction이 맞춘 것만
                gt0_demos = [
                    d for d in demonstrations 
                    if d.get("ground_truth_label") == 0 
                    and d.get("final_prediction_int") == 0
                ]
                if gt0_demos:
                    selected_demos.append(random.choice(gt0_demos))
                
                # GT=1인 케이스 (IR 이상) 중 prediction이 맞춘 것만
                gt1_demos = [
                    d for d in demonstrations 
                    if d.get("ground_truth_label") == 1 
                    and d.get("final_prediction_int") == 1
                ]
                if gt1_demos:
                    selected_demos.append(random.choice(gt1_demos))
            
            # Step 3: Impact evaluation
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
                demonstrations=selected_demos if selected_demos else None,
                no_structure=args.no_structure,
                no_semantic=args.no_semantic,
                no_trend=args.no_trend,
            )
            
            pred_int = verdict_to_int(impact.get("final_verdict", None))
            if pred_int is None:
                pred_int = 0
            
            return (i, {
                "impact": impact,  # JSON 파싱된 dict 형태
                "pred_int": pred_int,
                "gt": gt,
                "u": u,
                "v": v,
                "index": i,
                "bundle": bundle,  # 중간 analysis 결과들
                "selected_demos": selected_demos,  # 사용한 demonstrations
            })
        except Exception as e:
            print(f"[ERROR] Failed to process pair {i} ({u} <-> {v}): {str(e)}")
            return (i, {
                "impact": None,
                "pred_int": 0,
                "gt": gt,
                "u": u,
                "v": v,
                "index": i,
                "error": str(e),
            })
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_pairs)
        batch_sampled = sampled.iloc[batch_start:batch_end]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing pairs {batch_start + 1}-{batch_end} (동시 처리: {max_workers}개)...")
        
        # 배치 내 페어 인자 준비
        pair_args = []
        for local_idx, (idx, row) in enumerate(batch_sampled.iterrows()):
            original_idx = sampled.index.get_loc(idx) + 1  # 1-based index
            pair_args.append((original_idx, row))
        
        batch_results_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(process_pair_mode1, args): args for args in pair_args}
            
            # 진행상황 표시
            with tqdm(total=len(pair_args), desc=f"  Batch {batch_idx + 1}", leave=False) as pbar:
                for future in as_completed(future_to_pair):
                    try:
                        pair_idx, result = future.result()
                        batch_results_dict[pair_idx] = result
                        pbar.update(1)
                    except Exception as e:
                        pair_arg = future_to_pair[future]
                        print(f"\n[ERROR] Unexpected error processing pair: {str(e)}")
                        batch_results_dict[pair_arg[0]] = {
                            "impact": None,
                            "pred_int": 0,
                            "gt": pair_arg[1].label,
                            "u": pair_arg[1].concept1,
                            "v": pair_arg[1].concept2,
                            "index": pair_arg[0],
                            "error": str(e),
                        }
                        pbar.update(1)
        
        # 배치 완료 후 결과 요약 출력 및 저장 준비
        batch_y_true = []
        batch_y_pred = []
        batch_records = []
        for idx in sorted(batch_results_dict.keys()):
            result = batch_results_dict[idx]
            batch_y_true.append(result["gt"])
            batch_y_pred.append(result["pred_int"])
            all_impacts.append(result["impact"])
            all_pairs_info.append({
                "index": result["index"],
                "u": result["u"],
                "v": result["v"],
                "gt": result["gt"],
            })
            
            # 저장용 레코드 생성 (메타데이터는 제외, pair 정보만 저장)
            if result.get("impact") is not None:
                rec = {
                    "type": "result",
                    "pair": {"concept1": result["u"], "concept2": result["v"]},
                    "ground_truth_label": result["gt"],
                    "final_prediction_int": result["pred_int"],
                    "final_impact_output": result["impact"],
                    "bundle": result.get("bundle"),  # 중간 analysis 결과들
                    "used_demonstrations": result.get("selected_demos", []),  # 사용한 demonstrations
                }
                batch_records.append(rec)
            elif "error" in result:
                # 에러가 발생한 경우도 저장
                rec = {
                    "type": "result",
                    "pair": {"concept1": result["u"], "concept2": result["v"]},
                    "ground_truth_label": result["gt"],
                    "final_prediction_int": result["pred_int"],
                    "error": result["error"],
                }
                batch_records.append(rec)
        
        # 배치 결과 요약
        batch_metrics = compute_binary_metrics(batch_y_true, batch_y_pred)
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Completed: {batch_start + 1}-{batch_end}")
        print(f"  Accuracy={batch_metrics['accuracy']:.4f} | Precision={batch_metrics['precision']:.4f} | Recall={batch_metrics['recall']:.4f} | F1={batch_metrics['f1']:.4f}")
        
        # 배치 내 각 페어 결과 출력
        print(f"  Results:")
        for idx in sorted(batch_results_dict.keys()):
            result = batch_results_dict[idx]
            status = "✓" if result["pred_int"] == result["gt"] else "✗"
            print(f"    [{result['index']:4d}] {status} {result['u']} <-> {result['v']}: pred={result['pred_int']}, GT={result['gt']}")
        
        # 전체 결과에 추가
        y_true.extend(batch_y_true)
        y_pred.extend(batch_y_pred)
        
        # 배치 완료 시마다 저장 (append 모드)
        if batch_records:
            with open(output_path, "a", encoding="utf-8") as f:  # append 모드
                for rec in batch_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()  # 즉시 디스크에 쓰기
            
            print(f"  [Saved] 배치 결과 저장됨: {len(batch_records)}개 레코드 추가")

    # 최종 결과 출력 및 저장
    metrics = compute_binary_metrics(y_true, y_pred)
    print("============================================================")
    print(f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}")
    print("============================================================")
    
    # 최종 메트릭 저장 (파일 끝에 추가)
    final_metrics = {
        "type": "metrics",
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "total_pairs": len(y_true),
        "correct_predictions": sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(final_metrics, ensure_ascii=False) + "\n")
    
    print(f"[Saved] mode1 test results -> {output_path}")


if __name__ == "__main__":
    main()
