#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

load_dotenv()


# ------------ Graph utils (from metallm_prediction simplified) ------------
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
    agg: Dict[Tuple[str, str, int], int] = {}
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
    try:
        import extract_path
        paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    except Exception:
        paths = []
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


# ------------ Prompts ------------
def structure_prompt_binary(ir: int) -> str:
    return f"""You are the Structure Agent. Focus ONLY on structural evidence (connectivity, paths, hubs, bridges, degree). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output JSON ONLY with a single field.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and reach cumulative citations ≥ {ir} (IR).

[OUTPUT]
Return JSON only:
{{
  "prediction": "Yes" or "No"
}}
Do NOT include reasoning or other fields."""


def semantic_prompt_binary(ir: int) -> str:
    return f"""You are the Semantic Agent. Focus ONLY on topical/semantic compatibility (meaning overlap, shared context). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output JSON ONLY with a single field.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and reach cumulative citations ≥ {ir} (IR).

[OUTPUT]
Return JSON only:
{{
  "prediction": "Yes" or "No"
}}
Do NOT include reasoning or other fields."""


def trend_prompt_binary(ir: int) -> str:
    return f"""You are the Trend Agent. Focus ONLY on temporal signals (year-by-year citations, recency, growth/decline). The query pair is currently unconnected; do not penalize the lack of a direct edge. Output JSON ONLY with a single field.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and reach cumulative citations ≥ {ir} (IR).

[OUTPUT]
Return JSON only:
{{
  "prediction": "Yes" or "No"
}}
Do NOT include reasoning or other fields."""


# ------------ Agent calls ------------
def run_agent_binary(client: OpenAI, model: str, sys_prompt: str, query_block: str) -> str:
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query_block},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"})
    data = json.loads(resp.choices[0].message.content)
    pred = str(data.get("prediction", "")).strip().lower()
    return "Yes" if pred == "yes" else "No"


def call_agents_binary(client: OpenAI, model: str, ir: int, concept1: str, concept2: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
    query_block = (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )
    struct = run_agent_binary(client, model, structure_prompt_binary(ir), query_block)
    sem = run_agent_binary(client, model, semantic_prompt_binary(ir), query_block)
    trend = run_agent_binary(client, model, trend_prompt_binary(ir), query_block)
    return {
        "structure": {"prediction": struct},
        "semantic": {"prediction": sem},
        "trend": {"prediction": trend},
        "concept1": concept1,
        "concept2": concept2,
    }


# ------------ Code generation helpers (reuse 3-dim binary features) ------------
def clean_code_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("python", "", 1).strip()
    return t


def meta_code_prompt(num_candidates: int, dim: int) -> str:
    return f"""You are an expert Python ML engineer.
Generate {num_candidates} diverse Python implementations of fit_and_predict(train_scores, train_labels, test_scores).
Each implementation must define fit_and_predict(train_scores, train_labels, test_scores) returning a list of 0/1 predictions for each row in test_scores.
Input shapes:
- train_scores: numpy array shape (N,{dim}) with 0/1 features from agent Yes/No outputs
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{dim}) matching train_scores
Rules:
- You may use any Python library commonly available (numpy, sklearn, scipy). Prefer strong predictive performance.
- Encourage diversity: include at least one strong RBF SVM/hinge/soft-margin variant, one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
- Return JSON ONLY:
{{
  "codes": [
    {{"id": 1, "code": "<python code for fit_and_predict>"}}
    ...
  ]
}}
Do NOT wrap code in markdown fences. Provide exactly {num_candidates} entries in the codes list."""


def single_code_prompt(dim: int) -> str:
    return f"""You are an expert Python ML engineer.
Write a single Python implementation of fit_and_predict(train_scores, train_labels, test_scores).
Input shapes:
- train_scores: numpy array shape (N,{dim}) with 0/1 features from agent Yes/No outputs
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{dim}) matching train_scores
Rules:
- You may use any Python library commonly available (numpy, sklearn, scipy). Prefer strong predictive performance.
- Encourage diversity: include at least one strong RBF SVM/hinge/soft-margin variant, one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
Return ONLY executable Python code (no JSON, no markdown)."""


def generate_model_codes(client: OpenAI, num_candidates: int, dim: int) -> List[str]:
    system = "You write runnable Python code. Output only code."
    user = meta_code_prompt(num_candidates, dim)
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
    )
    content = resp.choices[0].message.content or ""
    codes: List[str] = []
    try:
        payload = json.loads(content)
        for entry in payload.get("codes", []):
            if isinstance(entry, dict) and "code" in entry:
                code_text = clean_code_text(entry["code"])
                if code_text:
                    codes.append(code_text)
    except Exception:
        pass
    if not codes:
        parts = content.split("\n\n")
        for p in parts:
            p = clean_code_text(p)
            if "def fit_and_predict" in p:
                codes.append(p)
    return codes


def generate_single_code(client: OpenAI, dim: int) -> str:
    system = "You write runnable Python code. Output only code."
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": single_code_prompt(dim)}],
        temperature=0.6,
    )
    return clean_code_text(resp.choices[0].message.content or "")


def run_generated_code(code: str, X_train, y_train, X_test):
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)
    if "fit_and_predict" not in ns:
        raise RuntimeError("Generated code missing fit_and_predict")
    return ns["fit_and_predict"](X_train, y_train, X_test)


def split_train_val(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    if len(val_idx) == 0:
        val_idx = idx[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ------------ Pipeline ------------
def parse_args():
    p = argparse.ArgumentParser(description="No-score ablation: Yes/No agents + meta-code.")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--concept_graph", required=True)
    p.add_argument("--ir", type=int, required=True)
    p.add_argument("--model", default="gpt-5-mini", help="Agent model for Yes/No.")
    p.add_argument("--api_key", default=None)
    p.add_argument("--max_workers", type=int, default=64)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Predictions JSONL.")
    p.add_argument("--codes_out", default=None)
    p.add_argument("--best_code_out", default=None)
    p.add_argument("--best_code_in", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    client = OpenAI(api_key=api_key)

    # Load graph
    G, snapshot_year = load_concept_graph_csv(args.concept_graph)
    def subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G, a, b, snapshot_year, max_k=4, max_p=8)

    # Load train/test CSV
    train_rows = pd.read_csv(args.train_csv).to_dict(orient="records")
    test_rows = pd.read_csv(args.test_csv).to_dict(orient="records")

    # Agent scoring Yes/No
    def score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outs = []
        def task(idx_row):
            idx, row = idx_row
            subg = subgraph_fn(row["concept1"], row["concept2"])
            res = call_agents_binary(client, args.model, args.ir, row["concept1"], row["concept2"], subg)
            res["ground_truth_label"] = int(row.get("label", 0))
            res["index"] = idx
            return res
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(task, (i, r)) for i, r in enumerate(rows)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent scoring (Yes/No)"):
                outs.append(fut.result())
        return outs

    train_records = score_rows(train_rows)
    test_records = score_rows(test_rows)

    # Build features (3 binary)
    def recs_to_xy(records):
        X, y = [], []
        for r in records:
            def to_int(agent):
                return 1 if str(agent.get("prediction", "")).strip().lower() == "yes" else 0
            X.append([to_int(r["structure"]), to_int(r["semantic"]), to_int(r["trend"])])
            y.append(int(r.get("ground_truth_label", 0)))
        return np.array(X, dtype=float), np.array(y, dtype=int)

    X_train, y_train = recs_to_xy(train_records)
    X_test, y_test = recs_to_xy(test_records)

    # Train/val split
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, args.train_ratio, args.val_ratio, args.seed)
    print(f"[INFO] Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    best_code = None
    code_records: List[Dict[str, Any]] = []

    if args.best_code_in:
        if not os.path.exists(args.best_code_in):
            raise FileNotFoundError(f"best_code_in not found: {args.best_code_in}")
        best_code = open(args.best_code_in, "r", encoding="utf-8").read()
        print(f"[INFO] Loaded best code from {args.best_code_in}")
    else:
        best_acc = -1.0
        codes = generate_model_codes(client, args.num_candidates, dim=3)
        if not codes:
            print("[WARN] Bulk generation empty; fallback to single-code generation.")
            for idx in tqdm(range(args.num_candidates), desc="Code candidates (fallback)"):
                code = generate_single_code(client, dim=3)
                if not code.strip():
                    continue
                try:
                    val_preds = run_generated_code(code, X_tr, y_tr, X_val)
                    val_preds = np.array(val_preds).astype(int)
                    acc_val = accuracy_score(y_val, val_preds)
                except Exception:
                    continue
                code_records.append({"index": idx, "val_accuracy": float(acc_val), "code": code})
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_code = code
        for idx, code in enumerate(codes):
            try:
                val_preds = run_generated_code(code, X_tr, y_tr, X_val)
                val_preds = np.array(val_preds).astype(int)
                acc_val = accuracy_score(y_val, val_preds)
            except Exception:
                continue
            code_records.append({"index": idx, "val_accuracy": float(acc_val), "code": code})
            if acc_val > best_acc:
                best_acc = acc_val
                best_code = code
        if best_code is None:
            for idx in range(args.num_candidates):
                code = generate_single_code(client, dim=3)
                if not code.strip():
                    continue
                try:
                    val_preds = run_generated_code(code, X_tr, y_tr, X_val)
                    val_preds = np.array(val_preds).astype(int)
                    acc_val = accuracy_score(y_val, val_preds)
                except Exception:
                    continue
                code_records.append({"index": idx + len(code_records), "val_accuracy": float(acc_val), "code": code})
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_code = code
            if best_code is None:
                raise RuntimeError("No valid code candidates produced a score.")

    if args.codes_out and code_records:
        os.makedirs(os.path.dirname(args.codes_out), exist_ok=True)
        with open(args.codes_out, "w", encoding="utf-8") as f:
            for rec in code_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[CODES] saved to {args.codes_out}")
    if args.best_code_out and best_code:
        os.makedirs(os.path.dirname(args.best_code_out), exist_ok=True)
        with open(args.best_code_out, "w", encoding="utf-8") as f:
            f.write(best_code)
        print(f"[BEST] saved to {args.best_code_out}")

    # Final train on full train, predict test
    preds = run_generated_code(best_code, X_train, y_train, X_test)
    preds = np.array(preds).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f1_neg = f1_score(y_test, preds, zero_division=0, pos_label=0)
    macro_f1 = (f1 + f1_neg) / 2
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    print("[METRICS]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Save predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, yhat in zip(test_records, preds):
            out = {
                "concept1": r.get("concept1"),
                "concept2": r.get("concept2"),
                "meta_code_pred": int(yhat),
                "meta_code_pred_label": "Yes" if int(yhat) == 1 else "No",
                "ground_truth_label": int(r.get("ground_truth_label", 0)),
                "structure": r.get("structure"),
                "semantic": r.get("semantic"),
                "trend": r.get("trend"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[DONE] saved to {args.output}")


if __name__ == "__main__":
    main()
