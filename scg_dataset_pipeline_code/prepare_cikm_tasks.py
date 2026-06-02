"""
Prepare CIKM transductive and inductive datasets with the requested time windows.

Transductive:
- Build graph up to 2023.
- Label candidate pairs from the 2023 graph using 2024-2025 outcomes.
- Split labeled candidates into balanced train/test sets.

Inductive:
- Build graph up to 2021.
- Label candidate pairs from the 2021 graph using 2022-2023 outcomes for balanced training.
- Build graph up to 2023.
- Export unlabeled candidate pairs from the 2023 graph for prediction.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Prepare CIKM graph prediction datasets")
    parser.add_argument(
        "--input",
        default="output/cikm_new_batch_ms1_named_rerun/final_papers_with_concepts.csv",
    )
    parser.add_argument("--venue", default="CIKM")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--max-prediction-candidates", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    trans_dir = Path(f"dataset/{args.venue}/transductive")
    inductive_dir = Path(f"dataset/{args.venue}/inductive")
    trans_dir.mkdir(parents=True, exist_ok=True)
    inductive_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    trans_graph = trans_dir / "concept_graph_2023.csv"
    inductive_graph_2021 = inductive_dir / "concept_graph_2021.csv"
    inductive_graph_2023 = inductive_dir / "concept_graph_2023.csv"

    shared_openalex_cache = Path(f"dataset/{args.venue}/openalex_paper_cache.json")

    if not args.skip_graphs:
        for graph_path, max_year in [
            (trans_graph, 2023),
            (inductive_graph_2021, 2021),
        ]:
            cmd = [
                py,
                "build_concept_graph.py",
                "--input",
                str(input_path),
                "--venue",
                args.venue,
                "--output-dir",
                str(graph_path.parent),
                "--max-year",
                str(max_year),
                "--cache",
                str(shared_openalex_cache),
                "--output-filename",
                graph_path.name,
            ]
            if args.no_openalex:
                cmd.append("--no-openalex")
            run(cmd)
        inductive_graph_2023.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trans_graph, inductive_graph_2023)
        summary_src = trans_graph.parent / "concept_graph_summary.json"
        if summary_src.exists():
            shutil.copy2(summary_src, inductive_graph_2023.parent / "concept_graph_2023_summary.json")

    if not args.skip_labels:
        cmd = [
            py,
            "create_link_prediction_datasets.py",
            "--input",
            str(input_path),
            "--concept-graph",
            str(trans_graph),
            "--output-dir",
            str(trans_dir),
            "--future-years",
            "2024",
            "2025",
            "--citation-years",
            "2024",
            "2025",
            "--thresholds",
            *[str(t) for t in args.thresholds],
            "--train-ratio",
            str(args.train_ratio),
            "--random-seed",
            str(args.random_seed),
            "--cache",
            str(shared_openalex_cache),
            "--eval-remaining",
            "--prefix",
            "transductive",
        ]
        if args.no_openalex:
            cmd.append("--no-openalex")
        run(cmd)

        cmd = [
            py,
            "create_link_prediction_datasets.py",
            "--input",
            str(input_path),
            "--concept-graph",
            str(inductive_graph_2021),
            "--output-dir",
            str(inductive_dir),
            "--future-years",
            "2022",
            "2023",
            "--citation-years",
            "2022",
            "2023",
            "--thresholds",
            *[str(t) for t in args.thresholds],
            "--train-ratio",
            str(args.train_ratio),
            "--random-seed",
            str(args.random_seed),
            "--cache",
            str(shared_openalex_cache),
            "--train-only",
            "--prefix",
            "inductive",
        ]
        if args.no_openalex:
            cmd.append("--no-openalex")
        run(cmd)

    candidate_cmd = [
        py,
        "create_candidate_pairs.py",
        "--concept-graph",
        str(inductive_graph_2023),
        "--output",
        str(inductive_dir / "prediction_candidates_2023.csv"),
    ]
    if args.max_prediction_candidates:
        candidate_cmd.extend(["--max-candidates", str(args.max_prediction_candidates)])
    run(candidate_cmd)


if __name__ == "__main__":
    main()
