"""
Resume LLM validation and naming from saved initial clustering results.

This is useful when HDBSCAN/probability assignment has already completed and
the remaining LLM work should be restarted or parallelized without recomputing
embeddings and membership probabilities.
"""

import argparse
import json
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from concept_namer import ConceptNamer
from llm_cluster_validator import LLMClusterValidator
from pipeline import ConceptClusteringPipeline
from utils import Config, save_intermediate_result, setup_logging


_thread_local = threading.local()


def _get_validator(config_path: str) -> LLMClusterValidator:
    validator = getattr(_thread_local, "validator", None)
    if validator is None:
        validator = LLMClusterValidator(config_path)
        _thread_local.validator = validator
    return validator


def _get_namer(config_path: str) -> ConceptNamer:
    namer = getattr(_thread_local, "namer", None)
    if namer is None:
        namer = ConceptNamer(config_path)
        _thread_local.namer = namer
    return namer


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_concept_probs(path: Path) -> Dict[str, Dict[int, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("concept_to_cluster_probabilities", {})
    return {
        concept: {int(cluster_id): float(prob) for cluster_id, prob in probs.items()}
        for concept, probs in raw.items()
    }


def _validate_one(
    config_path: str,
    max_cluster_size: int,
    item: Tuple[int, List[str]],
) -> Tuple[int, List[List[str]], bool]:
    cluster_id, concepts = item
    if len(concepts) > max_cluster_size:
        return cluster_id, [concepts], True

    validator = _get_validator(config_path)
    are_similar, partitions = validator.validate_cluster(concepts)
    return cluster_id, partitions, are_similar


def _name_one(
    config_path: str,
    item: Tuple[int, int, int, int, List[str], Dict[str, float]],
) -> Tuple[int, int, int, int, List[str], str]:
    final_cluster_id, original_cluster_id, partition_idx, partition_count, partition, keyword_probs = item
    namer = _get_namer(config_path)
    name = namer.generate_concept_name(
        partition,
        cluster_id=original_cluster_id,
        keyword_probs=keyword_probs,
    )
    return final_cluster_id, original_cluster_id, partition_idx, partition_count, partition, name


def _save_final_outputs(
    pipeline: ConceptClusteringPipeline,
    df: pd.DataFrame,
    concept_column: str,
    paper_id_column: str,
    output_dir: Path,
    concept_to_clusters: Dict[str, List[int]],
    final_cluster_to_name: Dict[int, str],
) -> pd.DataFrame:
    result_df = pipeline.mapper.map_papers_to_concepts(
        df,
        concept_column,
        concept_to_clusters,
        final_cluster_to_name,
        paper_id_column,
    )
    result_df = pipeline._filter_assigned_concepts_by_frequency(result_df)

    save_intermediate_result(
        result_df,
        output_dir / "final_papers_with_concepts.pkl",
        "Final Paper-Concept Mapping",
    )

    csv_df = result_df.copy()
    csv_df["assigned_concepts"] = csv_df["assigned_concepts"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) and len(x) > 0 else ""
    )

    output_columns = []
    if "title" in csv_df.columns:
        output_columns.append("title")
    if "year" in csv_df.columns:
        output_columns.append("year")
    output_columns.append("assigned_concepts")

    csv_output = output_dir / "final_papers_with_concepts.csv"
    csv_df[output_columns].to_csv(csv_output, index=False, encoding="utf-8")
    logging.getLogger(__name__).info("CSV saved: %s", csv_output)

    detailed_rows = []
    for idx, row in result_df.iterrows():
        paper_id = row[paper_id_column] if paper_id_column in result_df.columns else idx
        original_concepts = pipeline.data_loader.parse_row_concepts(
            row,
            concept_column,
            normalize=pipeline.normalize_concepts,
        )
        assigned_concepts = row.get("assigned_concepts", [])
        detailed_rows.append(
            {
                "paper_id": paper_id,
                "original_concepts": ", ".join(original_concepts) if original_concepts else "",
                "assigned_concept_count": len(assigned_concepts) if isinstance(assigned_concepts, list) else 0,
                "assigned_concepts": (
                    ", ".join(assigned_concepts)
                    if isinstance(assigned_concepts, list) and len(assigned_concepts) > 0
                    else ""
                ),
            }
        )

    detailed_output = output_dir / "papers_with_detailed_concepts.csv"
    pd.DataFrame(detailed_rows).to_csv(detailed_output, index=False, encoding="utf-8")
    logging.getLogger(__name__).info("Detailed CSV saved: %s", detailed_output)

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Resume LLM validation/naming from initial clusters.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")
    parser.add_argument(
        "--concept-column",
        default="level_2_concepts,level_3_concepts,level_4_concepts,level_5_concepts",
        help="Concept keyword column name(s), comma-separated for multiple columns.",
    )
    parser.add_argument("--paper-id-column", default="paper_id", help="Paper ID column name")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory containing initial_clusters.pkl")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum final concept document frequency")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("LLM_MAX_WORKERS", "8")),
        help="Parallel LLM workers",
    )
    parser.add_argument("--force-validation", action="store_true", help="Recompute validation even if cached")
    parser.add_argument("--force-naming", action="store_true", help="Recompute naming even if cached")
    args = parser.parse_args()

    config = Config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_clusters_path = output_dir / "initial_clusters.pkl"
    clustering_results_path = output_dir / "clustering_results.json"
    validated_clusters_path = output_dir / "validated_clusters.pkl"
    final_names_path = output_dir / "final_cluster_names.pkl"
    concept_to_clusters_path = output_dir / "concept_to_clusters.pkl"

    if not initial_clusters_path.exists():
        raise FileNotFoundError(f"Missing initial clusters: {initial_clusters_path}")
    if not clustering_results_path.exists():
        raise FileNotFoundError(f"Missing clustering probability JSON: {clustering_results_path}")

    workers = max(1, args.workers)
    logger.info("Resuming LLM steps with %s worker(s)", workers)

    cluster_to_concepts = _load_pickle(initial_clusters_path)
    logger.info("Loaded %s initial clusters", len(cluster_to_concepts))

    concept_to_cluster_probs = _load_concept_probs(clustering_results_path)
    logger.info("Loaded probability records for %s concepts", len(concept_to_cluster_probs))

    pipeline = ConceptClusteringPipeline(args.config)
    pipeline.output_dir = output_dir
    pipeline.final_min_frequency = args.min_frequency
    pipeline.normalize_concepts = config.get("pipeline.normalize_concepts", True)
    pipeline.data_loader.normalize_concepts = pipeline.normalize_concepts
    pipeline.mapper.normalize_concepts = pipeline.normalize_concepts

    df = pipeline.data_loader.load_from_csv(args.input, args.concept_column)

    if validated_clusters_path.exists() and not args.force_validation:
        validated_clusters = _load_pickle(validated_clusters_path)
        logger.info("Loaded cached validated clusters: %s", len(validated_clusters))
    else:
        max_validation_size = config.get("processing.max_cluster_size_for_validation", 50)
        items = list(cluster_to_concepts.items())
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_validate_one, args.config, max_validation_size, item)
                for item in items
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parallel cluster validation",
                unit="cluster",
                dynamic_ncols=True,
            ):
                cluster_id, partitions, _are_similar = future.result()
                results[cluster_id] = partitions

        validated_clusters = {
            cluster_id: results[cluster_id]
            for cluster_id, _concepts in items
        }
        save_intermediate_result(validated_clusters, validated_clusters_path, "Validated Clusters")

    if (
        final_names_path.exists()
        and concept_to_clusters_path.exists()
        and not args.force_naming
    ):
        final_cluster_to_name = _load_pickle(final_names_path)
        concept_to_clusters = _load_pickle(concept_to_clusters_path)
        logger.info("Loaded cached final names: %s", len(final_cluster_to_name))
    else:
        naming_tasks = []
        final_cluster_id = 0
        for original_cluster_id, partitions in validated_clusters.items():
            for partition_idx, partition in enumerate(partitions):
                keyword_probs = {
                    kw: concept_to_cluster_probs.get(kw, {}).get(original_cluster_id, 0.0)
                    for kw in partition
                }
                naming_tasks.append(
                    (
                        final_cluster_id,
                        original_cluster_id,
                        partition_idx,
                        len(partitions),
                        partition,
                        keyword_probs,
                    )
                )
                final_cluster_id += 1

        final_cluster_to_name = {}
        partition_by_final_id = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_name_one, args.config, item)
                for item in naming_tasks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parallel LLM naming",
                unit="partition",
                dynamic_ncols=True,
            ):
                (
                    final_id,
                    original_cluster_id,
                    partition_idx,
                    partition_count,
                    partition,
                    name,
                ) = future.result()
                final_cluster_to_name[final_id] = name
                partition_by_final_id[final_id] = partition
                if partition_count > 1:
                    logger.info(
                        "Partition %s/%s of original cluster %s -> final cluster %s: %r (%s concepts)",
                        partition_idx + 1,
                        partition_count,
                        original_cluster_id,
                        final_id,
                        name,
                        len(partition),
                    )

        final_cluster_to_name = dict(sorted(final_cluster_to_name.items()))
        concept_to_clusters = {}
        for final_id in sorted(partition_by_final_id):
            for concept in partition_by_final_id[final_id]:
                concept_to_clusters.setdefault(concept, []).append(final_id)

        logger.info("Final number of clusters: %s", len(final_cluster_to_name))
        save_intermediate_result(final_cluster_to_name, final_names_path, "Final Cluster Names")
        save_intermediate_result(concept_to_clusters, concept_to_clusters_path, "Keyword-Cluster Mapping")

    result_df = _save_final_outputs(
        pipeline,
        df,
        args.concept_column,
        args.paper_id_column,
        output_dir,
        concept_to_clusters,
        final_cluster_to_name,
    )

    logger.info("Pipeline LLM resume completed")
    print("\nLLM resume completed!")
    print(f"Result DataFrame shape: {result_df.shape}")


if __name__ == "__main__":
    main()
