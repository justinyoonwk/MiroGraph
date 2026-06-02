"""
Create unlabeled candidate concept pairs from a historical concept graph.

Candidates are all concept pairs whose two endpoints exist in the historical
graph but whose edge has not appeared in that graph.
"""

import argparse
import csv
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _edge(row):
    c1 = str(row["concept1"]).strip()
    c2 = str(row["concept2"]).strip()
    return tuple(sorted((c1, c2)))


def create_candidate_pairs(
    concept_graph_path: str,
    output_path: str,
    max_candidates: int = None,
):
    graph_df = pd.read_csv(concept_graph_path)
    existing_edges = set()
    concepts = set()

    for _, row in graph_df.iterrows():
        edge = _edge(row)
        existing_edges.add(edge)
        concepts.update(edge)

    concepts = sorted(concepts)
    total_possible = len(concepts) * (len(concepts) - 1) // 2
    logger.info(f"Historical concepts: {len(concepts):,}")
    logger.info(f"Historical edges: {len(existing_edges):,}")
    logger.info(f"All possible pairs: {total_possible:,}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept1", "concept2"])
        for c1, c2 in tqdm(combinations(concepts, 2), total=total_possible, desc="Writing candidates"):
            edge = tuple(sorted((c1, c2)))
            if edge in existing_edges:
                continue
            writer.writerow([c1, c2])
            written += 1
            if max_candidates and written >= max_candidates:
                break

    logger.info(f"Candidate pairs saved: {output}")
    logger.info(f"Candidate pairs written: {written:,}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create unlabeled candidate concept pairs from a historical graph")
    parser.add_argument("--concept-graph", "-g", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    create_candidate_pairs(
        args.concept_graph,
        args.output,
        max_candidates=args.max_candidates,
    )
