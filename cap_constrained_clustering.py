#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cap-constrained agglomerative clustering for validation SBERT embeddings.

Implements the design: cosine (dot-product on L2-normalized vectors),
hard cap on cluster size (default 50), stop when the best merge falls
below sim_threshold. Produces validation cluster assignments and
optional train→cluster matching for downstream demo selection.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers not found. Install with `pip install sentence-transformers`."
    ) from e


# =============================
# Embedding helpers
# =============================
def build_texts(df: pd.DataFrame, sep: str = "[SEP]") -> List[str]:
    if not {"concept1", "concept2"}.issubset(df.columns):
        raise ValueError("Input CSV must have columns: concept1, concept2")
    return [f"{a} {sep} {b}" for a, b in zip(df["concept1"], df["concept2"])]


def embed_texts(
    model_name: str,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    if not normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
    return emb


# =============================
# Clustering core
# =============================
@dataclass
class Cluster:
    cid: int
    members: List[int]
    sum_vec: np.ndarray
    centroid: np.ndarray
    size: int
    version: int = 0
    active: bool = True
    meta: Dict = field(default_factory=dict)


def cap_constrained_agglomerative(
    X: np.ndarray,
    cap: int = 50,
    sim_threshold: float = 0.65,
) -> Tuple[List[int], List[Cluster]]:
    """
    X: (N, d) L2-normalized embeddings for validation set.
    Returns:
      val_cluster_id: list of length N with cluster ids (0..K-1)
      clusters: list of active Cluster objects (centroid/members/size filled)
    """
    import heapq

    N, d = X.shape
    clusters: Dict[int, Cluster] = {}
    heap = []

    for i in range(N):
        clusters[i] = Cluster(cid=i, members=[i], sum_vec=X[i].copy(), centroid=X[i].copy(), size=1)

    # Initialize heap with all pairs (cap satisfied because size=1)
    for i in range(N):
        xi = X[i]
        sims = np.dot(X[i + 1 :], xi)
        for j_offset, sim in enumerate(sims, start=i + 1):
            heapq.heappush(heap, (-float(sim), i, j_offset, clusters[i].version, clusters[j_offset].version))

    next_cid = N
    active_ids = set(clusters.keys())

    while heap:
        neg_sim, a, b, ver_a, ver_b = heapq.heappop(heap)
        if a not in active_ids or b not in active_ids:
            continue
        ca, cb = clusters[a], clusters[b]
        if ver_a != ca.version or ver_b != cb.version:
            continue
        if ca.size + cb.size > cap:
            continue
        sim = -neg_sim
        if sim < sim_threshold:
            break

        # Merge
        new_sum = ca.sum_vec + cb.sum_vec
        new_centroid = new_sum / (np.linalg.norm(new_sum) + 1e-12)
        new_members = ca.members + cb.members
        new_cluster = Cluster(
            cid=next_cid,
            members=new_members,
            sum_vec=new_sum,
            centroid=new_centroid,
            size=ca.size + cb.size,
        )
        clusters[next_cid] = new_cluster
        active_ids.add(next_cid)

        # Deactivate old clusters
        ca.active = False
        cb.active = False
        ca.version += 1
        cb.version += 1
        active_ids.discard(a)
        active_ids.discard(b)

        # Push similarities between new cluster and all other active clusters
        for cid in list(active_ids):
            if cid == next_cid:
                continue
            c_other = clusters[cid]
            if not c_other.active:
                continue
            if new_cluster.size + c_other.size > cap:
                continue
            sim_new = float(np.dot(new_cluster.centroid, c_other.centroid))
            heapq.heappush(heap, (-sim_new, next_cid, cid, new_cluster.version, c_other.version))

        next_cid += 1

    # Collect active clusters and reindex 0..K-1
    active_clusters = [c for c in clusters.values() if c.active]
    active_clusters.sort(key=lambda c: min(c.members))
    cid_map = {c.cid: idx for idx, c in enumerate(active_clusters)}
    val_cluster_id = [-1] * N
    for c in active_clusters:
        new_id = cid_map[c.cid]
        for m in c.members:
            val_cluster_id[m] = new_id
        c.cid = new_id  # overwrite with compact id

    return val_cluster_id, active_clusters


# =============================
# Train→cluster assignment
# =============================
def assign_to_clusters(
    train_emb: np.ndarray,
    clusters: List[Cluster],
) -> Tuple[List[int], List[float]]:
    """
    Assign each train embedding to the nearest cluster centroid.
    Returns (cluster_ids, similarities).
    """
    if not clusters:
        raise ValueError("No clusters provided for assignment.")
    centroids = np.stack([c.centroid for c in clusters], axis=0)
    sims = np.dot(train_emb, centroids.T)  # (N_train, K)
    cluster_ids = np.argmax(sims, axis=1).tolist()
    best_sims = sims[np.arange(sims.shape[0]), cluster_ids].tolist()
    return cluster_ids, best_sims


# =============================
# I/O
# =============================
def save_clusters(
    path: str,
    val_cluster_id: List[int],
    clusters: List[Cluster],
    val_df: pd.DataFrame,
    train_assignments: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "meta": {
            "cap": max(c.size for c in clusters) if clusters else 0,
            "num_clusters": len(clusters),
            "num_validation": len(val_cluster_id),
        },
        "val_cluster_id": val_cluster_id,
        "clusters": [
            {
                "cluster_id": c.cid,
                "size": c.size,
                "members": c.members,
                "centroid": c.centroid.tolist(),
            }
            for c in clusters
        ],
        "validation_rows": val_df.to_dict(orient="records"),
    }
    if train_assignments:
        payload["train_assignments"] = train_assignments
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =============================
# CLI
# =============================
def parse_args():
    p = argparse.ArgumentParser(description="Cap-constrained agglomerative clustering on validation SBERT embeddings.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--val_csv", help="CSV with validation pairs (concept1, concept2[, label ...]).")
    group.add_argument(
        "--trainval_csv",
        help="Single CSV containing both train+validation (e.g., train_data_10.csv). "
        "Will be split randomly by --train_ratio / --val_ratio."
    )
    p.add_argument("--train_csv", help="Optional train CSV for assignment (concept1, concept2[, label ...]). "
                                       "Ignored if --trainval_csv is used.")
    p.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio when using --trainval_csv.")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio when using --trainval_csv.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting when using --trainval_csv.")
    p.add_argument("--sep_token", default="[SEP]", help="Separator token between concept1 and concept2.")
    p.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2", help="SBERT model name.")
    p.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    p.add_argument("--cap", type=int, default=50, help="Maximum cluster size.")
    p.add_argument("--sim_threshold", type=float, default=0.65, help="Merge stops when best sim < threshold.")
    p.add_argument("--output", default="results/clustering/val_clusters.json", help="Output JSON path.")
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare train/val splits
    if args.trainval_csv:
        full_df = pd.read_csv(args.trainval_csv)
        df_shuffled = full_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n = len(df_shuffled)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        if n_train + n_val > n:
            n_val = n - n_train
        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
        print(f"[SPLIT] from {args.trainval_csv}: train={len(train_df)}, val={len(val_df)}, seed={args.seed}")
    else:
        val_df = pd.read_csv(args.val_csv)
        train_df = pd.read_csv(args.train_csv) if args.train_csv else None

    val_texts = build_texts(val_df, sep=args.sep_token)
    print(f"[VAL] {len(val_texts)} samples → embedding with {args.embedding_model}")
    val_emb = embed_texts(args.embedding_model, val_texts, batch_size=args.batch_size, normalize=True)

    print(f"[CLUSTER] cap={args.cap}, sim_threshold={args.sim_threshold}")
    val_cluster_id, clusters = cap_constrained_agglomerative(
        val_emb, cap=args.cap, sim_threshold=args.sim_threshold
    )
    sizes = [c.size for c in clusters] if clusters else []
    if sizes:
        print(f"[CLUSTER] formed {len(clusters)} clusters; size stats -> min={np.min(sizes)}, max={np.max(sizes)}, mean={np.mean(sizes):.2f}, median={np.median(sizes):.2f}")
    else:
        print("[CLUSTER] no clusters formed (unexpected)")

    train_assignments = None
    if args.trainval_csv or args.train_csv:
        if not args.trainval_csv:
            train_df = pd.read_csv(args.train_csv)
        train_texts = build_texts(train_df, sep=args.sep_token)
        print(f"[TRAIN] {len(train_texts)} samples → embedding for assignment")
        train_emb = embed_texts(args.embedding_model, train_texts, batch_size=args.batch_size, normalize=True)
        train_cluster_ids, train_sims = assign_to_clusters(train_emb, clusters)
        train_assignments = {
            "cluster_id": train_cluster_ids,
            "similarity": train_sims,
            "rows": train_df.to_dict(orient="records"),
        }
        print("[TRAIN] assignment complete")

    save_clusters(
        path=args.output,
        val_cluster_id=val_cluster_id,
        clusters=clusters,
        val_df=val_df,
        train_assignments=train_assignments,
    )
    print(f"[DONE] saved clustering payload to {args.output}")


if __name__ == "__main__":
    main()
