import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

try:
    import hdbscan
except ImportError:
    raise ImportError("hdbscan is not installed. Please install it with 'pip install hdbscan'.")


class HDBSCANProbabilityClusterer:
    
    def __init__(self,
                 min_cluster_size: int = 2,
                 min_samples: Optional[int] = None,
                 metric: str = 'euclidean',
                 cluster_selection_method: str = 'leaf',
                 cluster_selection_epsilon: float = 0.1,
                 prediction_data: bool = True,
                 allow_single_cluster: bool = False):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else 1  # Default to 1 to reduce outliers
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.prediction_data = prediction_data
        self.allow_single_cluster = allow_single_cluster
        
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'HDBSCANProbabilityClusterer':
        self.logger.info(f"HDBSCAN clustering started: {len(embeddings)} data points")
        
        # L2 normalization (preparation for cosine distance)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=self.prediction_data,
            allow_single_cluster=self.allow_single_cluster
        )
        
        self.clusterer.fit(normalized_embeddings)
        self.labels_ = self.clusterer.labels_
        
        # Statistics
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        self.logger.info(f"Clustering completed:")
        self.logger.info(f"  - Number of clusters created: {n_clusters}")
        self.logger.info(f"  - Number of noise points: {n_noise} ({n_noise/len(embeddings)*100:.2f}%)")
        
        return self
    
    def get_outlier_indices(self) -> List[int]:
        if self.labels_ is None:
            raise ValueError("Clustering has not been performed. Call fit() first.")
        
        outlier_indices = np.where(self.labels_ == -1)[0].tolist()
        return outlier_indices
    
    def get_membership_probabilities(self, embeddings: np.ndarray = None) -> np.ndarray:
        if self.clusterer is None or not self.prediction_data:
            raise ValueError("Clustering must be performed with prediction_data=True to calculate probabilities.")
        
        # Calculate cluster membership probabilities for all data used in fit
        # This can be slow for large datasets - add logging
        n_samples = len(self.labels_)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.logger.info(f"Calculating membership probabilities for {n_samples:,} samples across {n_clusters:,} clusters...")
        self.logger.info("This may take several minutes for large datasets...")
        
        membership_probs = hdbscan.all_points_membership_vectors(self.clusterer)
        
        self.logger.info(f"Probability calculation completed: {membership_probs.shape}")
        
        return membership_probs
    
    def assign_concepts_to_clusters(self, 
                                   concepts: List[str],
                                   embeddings: np.ndarray,
                                   probability_threshold: float = 0.8,
                                   original_indices: List[int] = None) -> Tuple[Dict[int, List[str]], Dict[str, Dict[int, float]]]:
        if self.labels_ is None:
            raise ValueError("Clustering has not been performed. Call fit() first.")
        
        self.logger.info(f"Probability-based cluster assignment started (threshold: {probability_threshold})")
        self.logger.info(f"Processing {len(concepts):,} concepts...")
        
        # If original_indices is None, assume sequential from 0
        if original_indices is None:
            # If concepts is already outlier-removed list, find non-outlier indices from all data used in fit
            self.logger.info("Determining original indices for concepts...")
            outlier_indices = set(self.get_outlier_indices())
            all_indices = [i for i in range(len(self.labels_)) if i not in outlier_indices]
            original_indices = all_indices[:len(concepts)]
            self.logger.info(f"Found {len(original_indices):,} valid indices")
        
        # Calculate probabilities for all data used in fit
        # This is the bottleneck - can take 10-30 minutes for large datasets
        self.logger.info("Starting probability calculation (this is the slowest step)...")
        membership_probs = self.get_membership_probabilities()
        self.logger.info("Probability calculation completed, starting cluster assignment...")
        
        # Cluster ID list (excluding noise)
        unique_labels = sorted([l for l in set(self.labels_) if l != -1])
        n_clusters = len(unique_labels)
        
        # Cluster assignment
        cluster_to_concepts = {cluster_id: [] for cluster_id in unique_labels}
        concept_to_cluster_probs = {}  # {concept: {cluster_id: probability, ...}}
        unassigned_concepts = []
        
        for i, orig_idx in enumerate(tqdm(original_indices, desc="Cluster assignment")):
            concept = concepts[i]
            # Use probabilities for original index
            probs = membership_probs[orig_idx]
            
            # Extract only cluster probabilities (excluding noise cluster)
            cluster_probs = probs[:n_clusters] if len(probs) >= n_clusters else probs
            
            # Find max probability
            max_prob = np.max(cluster_probs) if len(cluster_probs) > 0 else 0
            
            # Store probabilities for this concept
            concept_to_cluster_probs[concept] = {}
            for cluster_idx, prob in enumerate(cluster_probs):
                if cluster_idx < len(unique_labels):
                    cluster_id = unique_labels[cluster_idx]
                    concept_to_cluster_probs[concept][cluster_id] = float(prob)
            
            # If max_prob is 0, assign to the first available cluster (should not happen in practice)
            if max_prob == 0:
                if len(unique_labels) > 0:
                    # Assign to first cluster as fallback
                    cluster_id = unique_labels[0]
                    cluster_to_concepts[cluster_id].append(concept)
                    self.logger.warning(f"Concept '{concept}' has zero probability for all clusters, assigned to cluster {cluster_id} as fallback")
                else:
                    unassigned_concepts.append(concept)
                    self.logger.error(f"Concept '{concept}' cannot be assigned: no clusters available and max_prob is 0")
                continue
            
            # Assign to clusters with probability >= threshold * max_probability (allows duplicate assignment)
            threshold = max_prob * probability_threshold
            assigned_cluster_indices = np.where(cluster_probs >= threshold)[0]
            
            if len(assigned_cluster_indices) == 0:
                # If no cluster meets threshold, assign to the cluster with max probability anyway
                # This ensures all valid concepts get assigned
                max_prob_idx = np.argmax(cluster_probs)
                if max_prob_idx < n_clusters and max_prob_idx < len(unique_labels):
                    cluster_id = unique_labels[max_prob_idx]
                    cluster_to_concepts[cluster_id].append(concept)
                else:
                    # Fallback: assign to first cluster if argmax fails
                    if len(unique_labels) > 0:
                        cluster_id = unique_labels[0]
                        cluster_to_concepts[cluster_id].append(concept)
                        self.logger.warning(f"Concept '{concept}' argmax failed, assigned to first cluster {cluster_id} as fallback")
                    else:
                        unassigned_concepts.append(concept)
                        self.logger.error(f"Concept '{concept}' cannot be assigned: no clusters available")
            else:
                for prob_idx in assigned_cluster_indices:
                    if prob_idx < n_clusters:
                        cluster_id = unique_labels[prob_idx]
                        cluster_to_concepts[cluster_id].append(concept)
        
        # Remove empty clusters
        cluster_to_concepts = {k: v for k, v in cluster_to_concepts.items() if len(v) > 0}
        
        self.logger.info(f"Cluster assignment completed: {len(cluster_to_concepts)} clusters")
        self.logger.info(f"  - Assigned concepts: {sum(len(v) for v in cluster_to_concepts.values())}")
        self.logger.info(f"  - Unassigned concepts: {len(unassigned_concepts)}")
        for cluster_id, concept_list in cluster_to_concepts.items():
            self.logger.info(f"  - Cluster {cluster_id}: {len(concept_list)} keywords")
        
        return cluster_to_concepts, concept_to_cluster_probs

