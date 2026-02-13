import argparse
import json
import os
from typing import Any, Dict, List
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


def extract_xy(records: List[Dict[str, Any]]):
    X, y = [], []
    for r in records:
        X.append([
            float(r["structure"]["score"]),
            float(r["semantic"]["score"]),
            float(r["trend"]["score"]),
        ])
        y.append(int(r.get("ground_truth_label", 0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def meta_code_prompt(num_candidates: int) -> str:
    return f"""You are an expert Python ML engineer.
Generate {num_candidates} diverse Python implementations of fit_and_predict(train_scores, train_labels, test_scores).
Each implementation must define fit_and_predict(train_scores, train_labels, test_scores) returning a list of 0/1 predictions for each row in test_scores.
Input shapes:
- train_scores: numpy array shape (N,3) with floats in [0,1]
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,3) with floats in [0,1]
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Encourage diversity: include at least one margin-based classifier (explicitly provide one strong RBF SVM/hinge/soft-margin style variant with tuned hyperparameters), one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
- GaussianNB is NOT allowed (overly simple baseline)
- Return JSON ONLY:
{{
  "codes": [
    {{"id": 1, "code": "<python code for fit_and_predict>"}}
    ...
  ]
}}
Do NOT wrap code in markdown fences. Provide exactly {num_candidates} entries in the codes list."""


def clean_code_text(text: str) -> str:
    """Strip markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("python", "", 1).strip()
    return t


def single_code_prompt() -> str:
    return """You are an expert Python ML engineer.
Write a single Python implementation of fit_and_predict(train_scores, train_labels, test_scores).
Input shapes:
- train_scores: numpy array shape (N,3) with floats in [0,1]
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,3) with floats in [0,1]
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Encourage diversity: include at least one margin-based classifier (explicitly provide one strong RBF SVM/hinge/soft-margin style variant with tuned hyperparameters), one neighbor/distance-weighted method, one tree/ensemble-style approach, one linear baseline, and one nonlinear feature mapping
Return ONLY executable Python code (no JSON, no markdown)."""


def generate_single_code(api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    system = "You write runnable Python code. Output only code."
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": single_code_prompt()},
        ],
        temperature=0.6,
    )
    content = resp.choices[0].message.content or ""
    return clean_code_text(content)


def generate_model_codes(api_key: str, num_candidates: int) -> List[str]:
    client = OpenAI(api_key=api_key)
    system = "You write runnable Python code. Output only code."
    user = meta_code_prompt(num_candidates)
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
    # First try JSON bundle
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
        codes_raw = []
    # If JSON bundle failed, fall back to splitting by markers or newlines
    if not codes:
        # Try to split heuristically on separators
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


def format_examples_for_prompt(X: np.ndarray, y: np.ndarray, num_examples: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(num_examples, n)
    idx = rng.choice(n, size=k, replace=False)
    lines = []
    for i in idx:
        lines.append(f"({X[i,0]:.2f}, {X[i,1]:.2f}, {X[i,2]:.2f}) -> {int(y[i])}")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Meta LLM code generation for score-based prediction.")
    p.add_argument("--train_scores_cache", required=True, help="Path to train scores JSONL.")
    p.add_argument("--test_scores_cache", required=True, help="Path to test scores JSONL.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--output", default="results/metallm/meta_code_test_predictions.jsonl", help="Output JSONL.")
    p.add_argument("--codes_out", default=None, help="Optional JSONL file to save all generated code candidates with val accuracy.")
    p.add_argument("--best_code_out", default=None, help="Optional path to save the best code (plain text).")
    p.add_argument("--best_code_in", default=None, help="Optional path to load an existing best code and skip generation.")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio from train cache.")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Val split ratio from train cache.")
    p.add_argument("--num_candidates", type=int, default=10, help="Number of code candidates to generate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    # Load caches
    train_records = load_scores(args.train_scores_cache)
    test_records = load_scores(args.test_scores_cache)
    X_train, y_train = extract_xy(train_records)
    X_test, y_test = extract_xy(test_records)

    # Split train/val
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, args.train_ratio, args.val_ratio, args.seed)

    print(f"[INFO] Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    best_code = None
    code_records = []

    if args.best_code_in:
        if not os.path.exists(args.best_code_in):
            raise FileNotFoundError(f"best_code_in not found: {args.best_code_in}")
        best_code = open(args.best_code_in, "r", encoding="utf-8").read()
        print(f"[INFO] Loaded best code from {args.best_code_in}")
        # Optional: evaluate on val for logging
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
        codes = generate_model_codes(api_key, args.num_candidates)
        if not codes:
            print("[WARN] Bulk generation returned no parsable codes; falling back to per-code generation.")
            for idx in tqdm(range(args.num_candidates), desc="Code candidates (fallback)"):
                code = generate_single_code(api_key)
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
        # Evaluate codes from bulk (if any)
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
            # Final safeguard: try extra single-code attempts
            for idx in range(args.num_candidates):
                code = generate_single_code(api_key)
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

    if args.codes_out:
        os.makedirs(os.path.dirname(args.codes_out), exist_ok=True)
        with open(args.codes_out, "w", encoding="utf-8") as f:
            for rec in code_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[CODES] saved all candidates to {args.codes_out}")

    if args.best_code_out and best_code:
        os.makedirs(os.path.dirname(args.best_code_out), exist_ok=True)
        with open(args.best_code_out, "w", encoding="utf-8") as f:
            f.write(best_code)
        print(f"[BEST] saved best code to {args.best_code_out}")

    # Retrain best code on full train and predict test
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
        # Fallback if a class is missing
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
                "structure_score": r["structure"]["score"],
                "semantic_score": r["semantic"]["score"],
                "trend_score": r["trend"]["score"],
                "meta_code_pred": int(yhat),
                "meta_code_pred_label": "Yes" if int(yhat) == 1 else "No",
                "ground_truth_label": int(r.get("ground_truth_label", 0)),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[DONE] saved to {args.output}")


if __name__ == "__main__":
    main()
