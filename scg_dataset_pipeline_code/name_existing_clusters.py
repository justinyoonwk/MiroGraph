"""
Run LLM validation/naming from an existing clustering_results.json.

This skips CSV parsing for clustering, embedding generation, and HDBSCAN.
Use it after running `main.py --skip-llm`.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from concept_namer import ConceptNamer
from llm_cluster_validator import LLMClusterValidator
from paper_concept_mapper import PaperConceptMapper
from utils import Config, setup_logging, save_intermediate_result


def _load_clusters(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = {
        int(cluster_id): concepts
        for cluster_id, concepts in data["clusters"].items()
    }
    concept_to_probs = {
        concept: {int(cid): float(prob) for cid, prob in probs.items()}
        for concept, probs in data.get("concept_to_cluster_probabilities", {}).items()
    }
    return clusters, concept_to_probs, data


def _validate_clusters(validator, clusters, max_cluster_size: int, checkpoint_path: Path):
    validated = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        validated = {int(k): v for k, v in raw.items()}

    pending_clusters = [
        (cluster_id, concepts)
        for cluster_id, concepts in clusters.items()
        if cluster_id not in validated
    ]
    pbar = tqdm(
        pending_clusters,
        desc="LLM validation",
        total=len(pending_clusters),
        dynamic_ncols=True,
        unit="cluster",
    )
    for cluster_id, concepts in pbar:
        if len(concepts) == 1:
            validated[cluster_id] = [concepts]
            continue
        if len(concepts) > max_cluster_size:
            validated[cluster_id] = [concepts]
            continue

        _, partitions = validator.validate_cluster(concepts)
        validated[cluster_id] = partitions

        if len(validated) % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in validated.items()}, f, indent=2, ensure_ascii=False)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in validated.items()}, f, indent=2, ensure_ascii=False)
    return validated


def _name_partitions(namer, validated_clusters, concept_to_probs, checkpoint_path: Path):
    final_cluster_to_name = {}
    concept_to_clusters = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        final_cluster_to_name = {int(k): v for k, v in raw["final_cluster_to_name"].items()}
        concept_to_clusters = {
            concept: [int(cid) for cid in cluster_ids]
            for concept, cluster_ids in raw["concept_to_clusters"].items()
        }

    total_partitions = sum(len(partitions) for partitions in validated_clusters.values())
    final_cluster_id = 0

    pbar = tqdm(
        total=total_partitions,
        initial=len(final_cluster_to_name),
        desc="LLM naming",
        dynamic_ncols=True,
        unit="partition",
    )
    for original_cluster_id, partitions in validated_clusters.items():
        for partition in partitions:
            if final_cluster_id in final_cluster_to_name:
                final_cluster_id += 1
                continue
            keyword_probs = {
                kw: concept_to_probs.get(kw, {}).get(original_cluster_id, 1.0)
                for kw in partition
            }
            partition_name = namer.generate_concept_name(
                partition,
                cluster_id=original_cluster_id,
                keyword_probs=keyword_probs,
            )
            final_cluster_to_name[final_cluster_id] = partition_name

            for concept in partition:
                concept_to_clusters.setdefault(concept, []).append(final_cluster_id)

            final_cluster_id += 1
            pbar.update(1)

            if final_cluster_id % 50 == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "final_cluster_to_name": {str(k): v for k, v in final_cluster_to_name.items()},
                            "concept_to_clusters": concept_to_clusters,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
    pbar.close()

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_cluster_to_name": {str(k): v for k, v in final_cluster_to_name.items()},
                "concept_to_clusters": concept_to_clusters,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return final_cluster_to_name, concept_to_clusters


def main():
    parser = argparse.ArgumentParser(description="Name existing HDBSCAN clusters with LLM")
    parser.add_argument("--clusters", required=True, help="Path to clustering_results.json")
    parser.add_argument("--input", "-i", required=True, help="Original paper concept CSV")
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parser.add_argument(
        "--concept-column",
        default="level_2_concepts,level_3_concepts,level_4_concepts,level_5_concepts",
    )
    parser.add_argument("--paper-id-column", default="paper_id")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--force", action="store_true", help="Ignore existing checkpoints in the output directory")
    args = parser.parse_args()

    config = Config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for filename in [
            "validated_clusters_checkpoint.json",
            "named_clusters_checkpoint.json",
        ]:
            path = output_dir / filename
            if path.exists():
                path.unlink()

    clusters, concept_to_probs, raw_clustering = _load_clusters(args.clusters)
    logger.info(f"Loaded clusters: {len(clusters):,}")
    logger.info(f"Clustering source concepts: {raw_clustering.get('statistics', {}).get('total_concepts')}")

    validator = LLMClusterValidator(args.config)
    namer = ConceptNamer(args.config)
    mapper = PaperConceptMapper(normalize_concepts=config.get("pipeline.normalize_concepts", True))

    validated = _validate_clusters(
        validator,
        clusters,
        max_cluster_size=config.get("processing.max_cluster_size_for_validation", 50),
        checkpoint_path=output_dir / "validated_clusters_checkpoint.json",
    )
    save_intermediate_result(validated, output_dir / "validated_clusters.pkl", "Validated Clusters")

    final_cluster_to_name, concept_to_clusters = _name_partitions(
        namer,
        validated,
        concept_to_probs,
        checkpoint_path=output_dir / "named_clusters_checkpoint.json",
    )
    save_intermediate_result(final_cluster_to_name, output_dir / "final_cluster_names.pkl", "Final Cluster Names")
    save_intermediate_result(concept_to_clusters, output_dir / "concept_to_clusters.pkl", "Keyword-Cluster Mapping")

    df = pd.read_csv(args.input)
    result_df = mapper.map_papers_to_concepts(
        df,
        args.concept_column,
        concept_to_clusters,
        final_cluster_to_name,
        args.paper_id_column,
    )
    save_intermediate_result(result_df, output_dir / "final_papers_with_concepts.pkl", "Final Paper-Concept Mapping")

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
    csv_df[output_columns].to_csv(output_dir / "final_papers_with_concepts.csv", index=False, encoding="utf-8")

    detailed_rows = []
    for idx, row in result_df.iterrows():
        paper_id = row[args.paper_id_column] if args.paper_id_column in result_df.columns else idx
        assigned = row.get("assigned_concepts", [])
        detailed_rows.append(
            {
                "paper_id": paper_id,
                "assigned_concept_count": len(assigned) if isinstance(assigned, list) else 0,
                "assigned_concepts": ", ".join(assigned) if isinstance(assigned, list) and assigned else "",
            }
        )
    pd.DataFrame(detailed_rows).to_csv(output_dir / "papers_with_detailed_concepts.csv", index=False, encoding="utf-8")
    logger.info(f"Saved final outputs to: {output_dir}")


if __name__ == "__main__":
    main()
