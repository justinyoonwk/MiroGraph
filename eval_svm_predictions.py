#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate SVM predictions JSONL and print metrics.

Input file format (from svm_prediction.py):
{ "svm_pred": 0/1, "ground_truth_label": 0/1, ... }
"""

import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SVM predictions JSONL.")
    p.add_argument("--pred_file", required=True, help="Path to test_predictions.jsonl from svm_prediction.py.")
    return p.parse_args()


def load_preds(path):
    y_true, y_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "svm_pred" in obj and "ground_truth_label" in obj:
                y_pred.append(int(obj["svm_pred"]))
                y_true.append(int(obj["ground_truth_label"]))
    return y_true, y_pred


def main():
    args = parse_args()
    y_true, y_pred = load_preds(args.pred_file)
    if not y_true:
        print("No valid predictions found.")
        return
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, zero_division=0)
    f1_neg = f1_score(y_true, y_pred, zero_division=0, pos_label=0)
    macro_f1 = (f1_pos + f1_neg) / 2
    print("[METRICS]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1_pos:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")


if __name__ == "__main__":
    main()
