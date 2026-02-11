import logging
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from data_loader import ConceptDataLoader
from embedding_generator import EmbeddingGenerator
from hdbscan_clusterer import HDBSCANProbabilityClusterer
from llm_cluster_validator import LLMClusterValidator
from concept_namer import ConceptNamer
from paper_concept_mapper import PaperConceptMapper
from utils import Config, setup_logging, save_intermediate_result


class ConceptClusteringPipeline:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose HTTP logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Initialize each module
        self.data_loader = ConceptDataLoader(
            min_frequency=self.config.get('pipeline.min_frequency', 2)
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.get('embeddings.model_name', 'allenai/specter2_base'),
            batch_size=self.config.get('embeddings.batch_size', 50),
            max_length=self.config.get('embeddings.max_length', 512)
        )
        self.clusterer = HDBSCANProbabilityClusterer(
            min_cluster_size=self.config.get('pipeline.min_cluster_size', 2),  # Allow smaller clusters
            min_samples=self.config.get('pipeline.min_samples', 1),  # More lenient criteria
            metric='euclidean',
            cluster_selection_method=self.config.get('pipeline.cluster_selection_method', 'leaf'),  # 'leaf' selects smaller clusters too
            cluster_selection_epsilon=self.config.get('pipeline.cluster_selection_epsilon', 0.2),  # Increased to reduce outliers (0.2-0.5 recommended)
            prediction_data=True,
            allow_single_cluster=self.config.get('pipeline.allow_single_cluster', False)  # Set True to eliminate all outliers (extreme)
        )
        self.validator = LLMClusterValidator(config_path)
        self.namer = ConceptNamer(config_path)
        self.mapper = PaperConceptMapper()
        
        # Output directory
        self.output_dir = Path(self.config.get('paths.output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self,
            csv_path: str,
            concept_column: str = 'concepts',
            paper_id_column: str = 'paper_id',
            probability_threshold: float = 0.8,
            save_intermediate: bool = True,
            skip_llm: bool = False,
            output_dir: str = None,
            venue: str = None) -> pd.DataFrame:
        # Override output directory if specified
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using custom output directory: {self.output_dir}")
        elif venue is not None:
            # Use venue-based default path
            self.output_dir = Path(f"output/{venue}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using venue-based output directory: {self.output_dir}")
        
        self.logger.info("=" * 60)
        self.logger.info("Concept Clustering Pipeline Started")
        self.logger.info("=" * 60)
        
        # 1. Data loading
        self.logger.info("\n[Step 1] Data Loading")
        df = self.data_loader.load_from_csv(csv_path, concept_column)
        
        # 2. Unique concept extraction and frequency filtering
        self.logger.info("\n[Step 2] Concept Extraction and Filtering")
        unique_concepts = self.data_loader.extract_all_concepts(df, concept_column)
        filtered_concepts, concept_frequencies = self.data_loader.filter_by_frequency(
            df, concept_column, unique_concepts
        )
        
        if save_intermediate:
            save_intermediate_result(
                {'concepts': filtered_concepts, 'frequencies': concept_frequencies},
                self.output_dir / 'filtered_concepts.pkl',
                "Filtered Concepts"
            )
        
        # 3. Embedding generation
        self.logger.info("\n[Step 3] SPECTER2 Embedding Generation")
        embeddings = self.embedding_generator.generate_embeddings(filtered_concepts)
        
        if save_intermediate:
            save_intermediate_result(
                embeddings,
                self.output_dir / 'concept_embeddings.pkl',
                "Concept Embeddings"
            )
        
        # 4. HDBSCAN clustering
        self.logger.info("\n[Step 4] HDBSCAN Clustering")
        self.clusterer.fit(embeddings)
        
        # Get outlier indices
        outlier_indices_list = self.clusterer.get_outlier_indices()
        outlier_indices = set(outlier_indices_list)
        valid_indices = [i for i in range(len(filtered_concepts)) if i not in outlier_indices]
        
        self.logger.info(f"Outliers detected: {len(outlier_indices)} ({len(outlier_indices)/len(filtered_concepts)*100:.2f}%)")
        
        # Prepare valid concepts and embeddings for cluster assignment (non-outliers only)
        valid_concepts = [filtered_concepts[i] for i in valid_indices]
        valid_embeddings = embeddings[valid_indices]
        
        self.logger.info(f"Valid keywords for cluster assignment: {len(valid_concepts)}")
        
        # Probability-based cluster assignment (supports duplicate assignment) for non-outliers
        cluster_to_concepts, concept_to_cluster_probs = self.clusterer.assign_concepts_to_clusters(
            valid_concepts,
            valid_embeddings,
            probability_threshold=probability_threshold,
            original_indices=valid_indices
        )
        
        # Assign each outlier as an individual cluster
        if len(outlier_indices) > 0:
            self.logger.info("Assigning each outlier as an individual cluster...")
            
            # Find the maximum existing cluster ID
            existing_cluster_ids = [int(cid) for cid in cluster_to_concepts.keys() if str(cid).isdigit()]
            next_cluster_id = max(existing_cluster_ids) + 1 if existing_cluster_ids else 0
            
            # Assign each outlier to its own cluster and set probability to 1.0
            for orig_idx in outlier_indices_list:
                concept = filtered_concepts[orig_idx]
                cluster_to_concepts[next_cluster_id] = [concept]
                # Outliers have probability 1.0 for their individual cluster
                concept_to_cluster_probs[concept] = {next_cluster_id: 1.0}
                next_cluster_id += 1
            
            self.logger.info(f"  - {len(outlier_indices)} outliers assigned as individual clusters")
            self.logger.info(f"  - Cluster IDs range: {min(existing_cluster_ids) if existing_cluster_ids else 0} to {next_cluster_id - 1}")
        
        # All outliers are now assigned as individual clusters
        remaining_outliers = []
        
        # Save clustering results to JSON for inspection
        # Include probability information for each concept in each cluster
        clusters_with_probs = {}
        for cluster_id, concepts in cluster_to_concepts.items():
            clusters_with_probs[str(cluster_id)] = []
            for concept in concepts:
                prob = concept_to_cluster_probs.get(concept, {}).get(cluster_id, 0.0)
                clusters_with_probs[str(cluster_id)].append({
                    "concept": concept,
                    "probability": prob
                })
        
        clustering_results = {
            "clusters": {str(cluster_id): concepts for cluster_id, concepts in cluster_to_concepts.items()},
            "clusters_with_probabilities": clusters_with_probs,
            "concept_to_cluster_probabilities": {
                concept: {str(cid): prob for cid, prob in probs.items()}
                for concept, probs in concept_to_cluster_probs.items()
            },
            "outliers": {
                "total_detected": len(outlier_indices),
                "assigned_as_individual_clusters": len(outlier_indices),
                "remaining": remaining_outliers
            },
            "statistics": {
                "total_concepts": len(filtered_concepts),
                "total_clusters": len(cluster_to_concepts),
                "concepts_in_clusters": sum(len(concepts) for concepts in cluster_to_concepts.values()),
                "outlier_percentage": len(outlier_indices) / len(filtered_concepts) * 100 if len(filtered_concepts) > 0 else 0,
                "outliers_as_individual_clusters": len(outlier_indices),
                "regular_clusters": len(cluster_to_concepts) - len(outlier_indices) if len(outlier_indices) > 0 else len(cluster_to_concepts)
            }
        }
        
        json_output_path = self.output_dir / 'clustering_results.json'
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(clustering_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Clustering results saved to JSON: {json_output_path}")
        
        if save_intermediate:
            save_intermediate_result(
                cluster_to_concepts,
                self.output_dir / 'initial_clusters.pkl',
                "Initial Clusters"
            )
        
        # If skipping LLM, return early with clustering results
        if skip_llm:
            self.logger.info("\nSkipping LLM steps (validation and naming)")
            self.logger.info("Clustering results saved to JSON. Pipeline stopped here.")
            return pd.DataFrame()  # Return empty DataFrame when skipping LLM
        
        # 5. LLM cluster validation and partition
        self.logger.info("\n[Step 5] LLM Cluster Validation and Partition")
        validated_clusters = self.validator.validate_all_clusters(
            cluster_to_concepts,
            max_cluster_size=self.config.get('processing.max_cluster_size_for_validation', 50)
        )
        
        if save_intermediate:
            save_intermediate_result(
                validated_clusters,
                self.output_dir / 'validated_clusters.pkl',
                "Validated Clusters"
            )
        
        # 6. Representative concept name generation
        # Generate names for each partition individually (each partition becomes a separate cluster)
        self.logger.info("\n[Step 6] Representative Concept Name Generation")
        self.logger.info("Generating names for each partition (each partition will be a separate final cluster)")
        
        # 7. Paper-concept mapping
        self.logger.info("\n[Step 7] Paper-Concept Mapping")
        
        # Create concept_to_clusters mapping (keywords can belong to multiple clusters)
        # validated_clusters is in the form {original_cluster_id: [partition1, partition2, ...]}
        # Each partition becomes a new final cluster with its own ID and name
        concept_to_clusters = {}  # {concept: [cluster_id1, cluster_id2, ...]}
        final_cluster_to_name = {}  # {final_cluster_id: concept_name}
        final_cluster_id = 0
        
        # Process each partition as a separate final cluster
        for original_cluster_id, partitions in validated_clusters.items():
            for partition_idx, partition in enumerate(partitions):
                # Get probabilities for this partition
                partition_keyword_probs = None
                if concept_to_cluster_probs:
                    partition_keyword_probs = {
                        kw: concept_to_cluster_probs.get(kw, {}).get(original_cluster_id, 0.0)
                        for kw in partition
                    }
                
                # Generate name for each partition (each partition is a separate cluster)
                partition_name = self.namer.generate_concept_name(
                    partition, 
                    cluster_id=original_cluster_id,
                    keyword_probs=partition_keyword_probs
                )
                
                final_cluster_to_name[final_cluster_id] = partition_name
                
                # Assign each keyword to this final cluster
                for concept in partition:
                    if concept not in concept_to_clusters:
                        concept_to_clusters[concept] = []
                    concept_to_clusters[concept].append(final_cluster_id)
                
                if len(partitions) > 1:
                    self.logger.info(f"  Partition {partition_idx+1}/{len(partitions)} of original cluster {original_cluster_id} -> Final cluster {final_cluster_id}: '{partition_name}' ({len(partition)} concepts)")
                else:
                    self.logger.debug(f"  Original cluster {original_cluster_id} -> Final cluster {final_cluster_id}: '{partition_name}' ({len(partition)} concepts)")
                
                final_cluster_id += 1
        
        self.logger.info(f"Final number of clusters: {len(final_cluster_to_name)}")
        
        if save_intermediate:
            save_intermediate_result(
                final_cluster_to_name,
                self.output_dir / 'final_cluster_names.pkl',
                "Final Cluster Names"
            )
            save_intermediate_result(
                concept_to_clusters,
                self.output_dir / 'concept_to_clusters.pkl',
                "Keyword-Cluster Mapping"
            )
        
        result_df = self.mapper.map_papers_to_concepts(
            df,
            concept_column,
            concept_to_clusters,
            final_cluster_to_name,
            paper_id_column
        )
        
        # Save final results
        if save_intermediate:
            save_intermediate_result(
                result_df,
                self.output_dir / 'final_papers_with_concepts.pkl',
                "Final Paper-Concept Mapping"
            )
            
            # Also save as CSV (convert assigned_concepts to comma-separated string)
            # Only save title, year, and assigned_concepts
            csv_df = result_df.copy()
            csv_df['assigned_concepts'] = csv_df['assigned_concepts'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else ''
            )
            
            # Select only title, year, and assigned_concepts columns
            output_columns = []
            if 'title' in csv_df.columns:
                output_columns.append('title')
            if 'year' in csv_df.columns:
                output_columns.append('year')
            output_columns.append('assigned_concepts')
            
            # Create final output dataframe with only selected columns
            final_csv_df = csv_df[output_columns].copy()
            
            csv_output = self.output_dir / 'final_papers_with_concepts.csv'
            final_csv_df.to_csv(csv_output, index=False, encoding='utf-8')
            self.logger.info(f"CSV saved: {csv_output}")
            
            # Additionally save a detailed CSV showing concepts for each paper
            detailed_output = self.output_dir / 'papers_with_detailed_concepts.csv'
            detailed_rows = []
            for idx, row in result_df.iterrows():
                paper_id = row[paper_id_column] if paper_id_column in result_df.columns else idx
                original_concepts = row.get(concept_column, '')
                assigned_concepts = row.get('assigned_concepts', [])
                
                detailed_rows.append({
                    'paper_id': paper_id,
                    'original_concepts': original_concepts if isinstance(original_concepts, str) else ', '.join(original_concepts) if isinstance(original_concepts, list) else '',
                    'assigned_concept_count': len(assigned_concepts) if isinstance(assigned_concepts, list) else 0,
                    'assigned_concepts': ', '.join(assigned_concepts) if isinstance(assigned_concepts, list) and len(assigned_concepts) > 0 else ''
                })
            
            detailed_df = pd.DataFrame(detailed_rows)
            detailed_df.to_csv(detailed_output, index=False, encoding='utf-8')
            self.logger.info(f"Detailed CSV saved: {detailed_output}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Completed!")
        self.logger.info("=" * 60)
        
        return result_df

