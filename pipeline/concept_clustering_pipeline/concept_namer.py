import json
import logging
from typing import Dict, List
from tqdm import tqdm

from utils import Config, LLMClient, PromptLoader


class ConceptNamer:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.client = LLMClient(self.config)
        self.prompt_loader = PromptLoader(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Default prompt
        self.naming_prompt = self._get_default_naming_prompt()
    
    def _get_default_naming_prompt(self) -> str:
        return """You are an expert in concept naming. Your task is to generate a concise, representative concept name that captures the semantic meaning of a group of keywords.

Given a list of keywords that belong to the same semantic cluster, along with their membership probabilities for this cluster, generate a single, clear concept name that best represents all of them.

Each keyword is provided with its probability of belonging to this cluster (ranging from 0.0 to 1.0). Higher probabilities indicate stronger membership in the cluster. You should:
- Give more weight to keywords with higher probabilities when generating the name
- Consider the overall semantic coherence of the cluster
- Keywords with very low probabilities (< 0.3) may be less representative of the cluster's core meaning

The concept name should be:
- Concise (preferably 1-3 words)
- Representative of all keywords, especially those with higher probabilities
- In English
- Use standard terminology when possible

Return your response as a JSON object:
{
    "concept_name": "the generated concept name",
    "reasoning": "brief explanation of why this name was chosen, considering the probabilities"
}"""
    
    def generate_concept_name(self, keywords: List[str], cluster_id: int = None, 
                             keyword_probs: Dict[str, float] = None) -> str:
        if len(keywords) == 0:
            return "Unknown"
        
        if len(keywords) == 1:
            return keywords[0]
        
        # Normalize probabilities within the cluster (sum to 1.0)
        normalized_probs = None
        if keyword_probs:
            # Calculate sum of probabilities for this cluster
            prob_sum = sum(keyword_probs.get(kw, 0.0) for kw in keywords)
            
            if prob_sum > 0:
                # Normalize so that sum equals 1.0
                normalized_probs = {
                    kw: keyword_probs.get(kw, 0.0) / prob_sum
                    for kw in keywords
                }
            else:
                # If all probabilities are 0, use uniform distribution
                uniform_prob = 1.0 / len(keywords)
                normalized_probs = {kw: uniform_prob for kw in keywords}
        
        # Construct prompt with probabilities if available
        if normalized_probs:
            keywords_str = "\n".join([
                f"- {kw} (normalized probability: {normalized_probs.get(kw, 0.0):.3f})" 
                for kw in keywords
            ])
            input_text = f"""Keywords in this cluster with their normalized membership probabilities (sum = 1.0):
{keywords_str}

Please generate a representative concept name for these keywords, giving more weight to keywords with higher probabilities."""
        else:
            keywords_str = "\n".join([f"- {kw}" for kw in keywords])
            input_text = f"""Keywords in this cluster:
{keywords_str}

Please generate a representative concept name for these keywords."""
        
        # Call LLM
        response = self.client.get_response_with_retry(self.naming_prompt, input_text)
        
        if response is None:
            self.logger.warning(f"LLM response failed. Using first keyword")
            return keywords[0]
        
        if not response or len(response.strip()) == 0:
            self.logger.warning(f"LLM returned empty response. Using first keyword")
            return keywords[0]
        
        try:
            result = json.loads(response)
            concept_name = result.get('concept_name', keywords[0])
            if not concept_name or concept_name.strip() == '':
                self.logger.warning(f"Concept name is empty in response. Using first keyword")
                return keywords[0]
            self.logger.debug(f"Generated concept name: {concept_name}")
            return concept_name.strip()
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.warning(f"Response content (first 200 chars): {response[:200] if response else 'None'}")
            return keywords[0]
    
    def generate_all_concept_names(self, validated_clusters: Dict[int, List[List[str]]], 
                                   concept_to_cluster_probs: Dict[str, Dict[int, float]] = None) -> Dict[int, str]:
        self.logger.info(f"Concept name generation started: {len(validated_clusters)} clusters")
        
        cluster_names = {}
        
        for cluster_id, partitions in tqdm(validated_clusters.items(), desc="Generating concept names"):
            # Generate name for each partition
            partition_names = []
            for partition in partitions:
                # Get probabilities for keywords in this partition for this cluster
                keyword_probs = None
                if concept_to_cluster_probs:
                    keyword_probs = {
                        kw: concept_to_cluster_probs.get(kw, {}).get(cluster_id, 0.0)
                        for kw in partition
                    }
                
                name = self.generate_concept_name(partition, cluster_id=cluster_id, keyword_probs=keyword_probs)
                partition_names.append(name)
            
            # Use that name if single partition
            if len(partition_names) == 1:
                cluster_names[cluster_id] = partition_names[0]
            else:
                # If multiple partitions, each partition will receive a new cluster_id later in pipeline
                # So here generate a representative name for the original cluster_id (may not be used)
                combined_keywords = []
                for partition in partitions:
                    combined_keywords.extend(partition[:5])  # Max 5 from each partition
                
                # Get probabilities for combined keywords
                combined_keyword_probs = None
                if concept_to_cluster_probs:
                    combined_keyword_probs = {
                        kw: concept_to_cluster_probs.get(kw, {}).get(cluster_id, 0.0)
                        for kw in combined_keywords
                    }
                
                cluster_names[cluster_id] = self.generate_concept_name(combined_keywords, 
                                                                        cluster_id=cluster_id, 
                                                                        keyword_probs=combined_keyword_probs)
                self.logger.info(f"Cluster {cluster_id}: {len(partitions)} partitions, representative name: {cluster_names[cluster_id]}")
        
        self.logger.info(f"Concept name generation completed")
        
        return cluster_names

