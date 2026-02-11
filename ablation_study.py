#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

load_dotenv()


def load_scores(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def extract_xy(records: List[Dict[str, Any]], mode: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in records:
        if mode == "no_structure":
            feats = [
                float(r["semantic"]["score"]),
                float(r["trend"]["score"]),
            ]
        elif mode == "no_semantic":
            feats = [
                float(r["structure"]["score"]),
                float(r["trend"]["score"]),
            ]
        elif mode == "no_trend":
            feats = [
                float(r["structure"]["score"]),
                float(r["semantic"]["score"]),
            ]
        else:
            feats = [
                float(r["structure"]["score"]),
                float(r["semantic"]["score"]),
                float(r["trend"]["score"]),
            ]
        X.append(feats)
        y.append(int(r.get("ground_truth_label", 0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def meta_code_prompt(num_candidates: int, dim: int, mode: str) -> str:
    feat_desc = f"(N,{dim}) with floats in [0,1]"
    return f"""You are an expert Python ML engineer.
Generate {num_candidates} diverse Python implementations of fit_and_predict(train_scores, train_labels, test_scores).
Each implementation must define fit_and_predict(train_scores, train_labels, test_scores) returning a list of 0/1 predictions for each row in test_scores.
Input shapes:
- train_scores: numpy array shape {feat_desc}
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{dim}) matching train_scores
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Encourage diversity: include at least one margin-based classifier (explicitly provide one strong RBF SVM/hinge/soft-margin style variant with tuned hyperparameters), one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
- Return JSON ONLY:
{{
  "codes": [
    {{"id": 1, "code": "<python code for fit_and_predict>"}}
    ...
  ]
}}
Do NOT wrap code in markdown fences. Provide exactly {num_candidates} entries in the codes list."""


def clean_code_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("python", "", 1).strip()
    return t


def single_code_prompt(dim: int, mode: str) -> str:
    feat_desc = f"(N,{dim}) with floats in [0,1]"
    return f"""You are an expert Python ML engineer.
Write a single Python implementation of fit_and_predict(train_scores, train_labels, test_scores).
Input shapes:
- train_scores: numpy array shape {feat_desc}
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{dim}) matching train_scores
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Encourage diversity: include at least one margin-based classifier (explicitly provide one strong RBF SVM/hinge/soft-margin style variant with tuned hyperparameters), one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
Return ONLY executable Python code (no JSON, no markdown)."""


def generate_single_code(api_key: str, dim: int, mode: str) -> str:
    client = OpenAI(api_key=api_key)
    system = "You write runnable Python code. Output only code."
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": single_code_prompt(dim, mode)},
        ],
        temperature=0.6,
    )
    content = resp.choices[0].message.content or ""
    return clean_code_text(content)


def generate_model_codes(api_key: str, num_candidates: int, dim: int, mode: str) -> List[str]:
    client = OpenAI(api_key=api_key)
    system = "You write runnable Python code. Output only code."
    user = meta_code_prompt(num_candidates, dim, mode)
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )
    content = resp.choices[0].message.content or ""
    codes: List[str] = []
    try:
        payload = json.loads(content)
        codes_raw = payload.get("codes", [])
        for entry in codes_raw:
            if not isinstance(entry, dict):
                continue
            code_text = entry.get("code") or ""
            code_text = clean_code_text(code_text)
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


def run_generated_code(code: str, X_train, y_train, X_test):
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)
    if "fit_and_predict" not in ns:
        raise RuntimeError("Generated code missing fit_and_predict")
    preds = ns["fit_and_predict"](X_train, y_train, X_test)
    return preds


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
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    return X_train, y_train, X_val, y_val


def classify_no_designer(client: OpenAI, model: str, ir: int, scores: List[float]) -> str:
    sys = "You are a scientific impact evaluator. Output JSON only."
    user = f"""Given three agent scores in [0,1]:
- structure_score: {scores[0]:.4f}
- semantic_score: {scores[1]:.4f}
- trend_score: {scores[2]:.4f}
IR threshold = {ir}.
Predict whether the pair will reach high-impact (Yes/No)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=1.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    pred = str(data.get("prediction", "")).strip().lower()
    return "Yes" if pred == "yes" else "No"


def parse_args():
    p = argparse.ArgumentParser(description="Ablation meta-code prediction.")
    p.add_argument("--mode", choices=["no_structure", "no_semantic", "no_trend", "no_designer"], help="Single mode run.")
    p.add_argument("--modes", type=str, help="Comma-separated modes to run sequentially (e.g., no_structure,no_semantic,no_trend).")
    p.add_argument("--train_scores_cache", required=True, help="Path to train scores JSONL.")
    p.add_argument("--test_scores_cache", required=True, help="Path to test scores JSONL.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--output", required=True, help="Output JSONL (or base name when running multiple modes).")
    p.add_argument("--codes_out", default=None, help="Optional JSONL to save code candidates (when applicable).")
    p.add_argument("--best_code_out", default=None, help="Optional path to save the best code (plain text).")
    p.add_argument("--best_code_in", default=None, help="Optional path to load an existing best code and skip generation.")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio from train cache.")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Val split ratio from train cache.")
    p.add_argument("--num_candidates", type=int, default=10, help="Number of code candidates to generate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--ir", type=int, default=10, help="IR threshold (used in no_designer prompt).")
    p.add_argument("--model_direct", default="gpt-5-mini", help="Model for no_designer direct classification.")
    p.add_argument("--codes_out_prefix", default=None, help="If running multiple modes, prefix for candidate JSONL outputs.")
    p.add_argument("--best_code_out_prefix", default=None, help="If running multiple modes, prefix for best code outputs.")
    p.add_argument("--runs", type=int, default=1, help="Number of runs (default 1).")
    p.add_argument("--summary_out", default=None, help="Optional summary JSON output.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    train_records = load_scores(args.train_scores_cache)
    test_records = load_scores(args.test_scores_cache)

    # Resolve modes list (runs/summary_out are ignored in this simplified version)
    if args.modes:
        modes_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        if not args.mode:
            raise ValueError("Either --mode or --modes must be provided.")
        modes_list = [args.mode]
    valid_modes = {"no_structure", "no_semantic", "no_trend", "no_designer"}
    for m in modes_list:
        if m not in valid_modes:
            raise ValueError(f"Invalid mode: {m}")

    # Helper to build per-mode outputs
    def with_suffix(base: str, mode: str) -> str:
        if base.endswith(".jsonl"):
            base_no = base[: -len(".jsonl")]
            return f"{base_no}_{mode}.jsonl"
        return f"{base}_{mode}"

    outputs = {}
    for m in modes_list:
        outputs[m] = {
            "output": args.output if len(modes_list) == 1 else with_suffix(args.output, m),
            "codes_out": None,
            "best_out": None,
        }
        if args.codes_out:
            outputs[m]["codes_out"] = args.codes_out if len(modes_list) == 1 else with_suffix(args.codes_out, m)
        if args.best_code_out:
            outputs[m]["best_out"] = args.best_code_out if len(modes_list) == 1 else with_suffix(args.best_code_out, m)
        if args.codes_out_prefix:
            outputs[m]["codes_out"] = f"{args.codes_out_prefix}_{m}.jsonl"
        if args.best_code_out_prefix:
            outputs[m]["best_out"] = f"{args.best_code_out_prefix}_{m}.py"

    client_direct = None

    for mode in modes_list:
        # Feature dimension by mode
        dim = 2 if mode in ("no_structure", "no_semantic", "no_trend") else 3

        X_train, y_train = extract_xy(train_records, mode)
        X_test, y_test = extract_xy(test_records, mode)

        if mode == "no_designer":
            if client_direct is None:
                client_direct = OpenAI(api_key=api_key)
            preds = []
            for r in tqdm(test_records, desc=f"{mode} direct classify"):
                scores = [
                    float(r["structure"]["score"]),
                    float(r["semantic"]["score"]),
                    float(r["trend"]["score"]),
                ]
                pred = classify_no_designer(client_direct, args.model_direct, args.ir, scores)
                preds.append(1 if pred == "Yes" else 0)
            preds = np.array(preds, dtype=int)
        else:
            # Split train/val
            X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, args.train_ratio, args.val_ratio, args.seed)
            print(f"[INFO] Mode={mode} | dim={dim} | Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            best_code = None
            code_records: List[Dict[str, Any]] = []

            if args.best_code_in:
                if not os.path.exists(args.best_code_in):
                    raise FileNotFoundError(f"best_code_in not found: {args.best_code_in}")
                best_code = open(args.best_code_in, "r", encoding="utf-8").read()
                print(f"[INFO] Loaded best code from {args.best_code_in}")
                try:
                    val_preds = run_generated_code(best_code, X_tr, y_tr, X_val)
                    val_preds = np.array(val_preds).astype(int)
                    acc_val = accuracy_score(y_val, val_preds)
                    print(f"[INFO] Loaded code val accuracy: {acc_val:.4f}")
                except Exception:
                    print("[WARN] Loaded code failed on val; proceeding to test anyway.")
            else:
                print(f"[INFO] Generating {args.num_candidates} code candidates with a single gpt-5.2 call ...")
                best_acc = -1.0
                codes = generate_model_codes(api_key, args.num_candidates, dim, mode)
                if not codes:
                    print("[WARN] Bulk generation returned no parsable codes; falling back to per-code generation.")
                    for idx in tqdm(range(args.num_candidates), desc="Code candidates (fallback)"):
                        code = generate_single_code(api_key, dim, mode)
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
                        code = generate_single_code(api_key, dim, mode)
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

            co_path = outputs[mode]["codes_out"]
            bo_path = outputs[mode]["best_out"]
            if co_path and mode != "no_designer":
                os.makedirs(os.path.dirname(co_path), exist_ok=True)
                with open(co_path, "w", encoding="utf-8") as f:
                    for rec in code_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[CODES] saved all candidates to {co_path}")

            if bo_path and mode != "no_designer" and best_code:
                os.makedirs(os.path.dirname(bo_path), exist_ok=True)
                with open(bo_path, "w", encoding="utf-8") as f:
                    f.write(best_code)
                print(f"[BEST] saved best code to {bo_path}")

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
        print(f"[METRICS][{mode}]")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Macro F1:  {macro_f1:.4f}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        # Save predictions
        out_path = outputs[mode]["output"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r, yhat in zip(test_records, preds):
                out = {
                    "concept1": r.get("concept1"),
                    "concept2": r.get("concept2"),
                    "meta_code_pred": int(yhat),
                    "meta_code_pred_label": "Yes" if int(yhat) == 1 else "No",
                    "ground_truth_label": int(r.get("ground_truth_label", 0)),
                }
                out["structure"] = r.get("structure")
                out["semantic"] = r.get("semantic")
                out["trend"] = r.get("trend")
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[DONE][{mode}] saved to {out_path}")


if __name__ == "__main__":
    main()
