import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path


class ConceptDataLoader:
    
    def __init__(self, min_frequency: int = 10):
        self.min_frequency = min_frequency
        self.logger = logging.getLogger(__name__)
        
    def load_from_csv(self, csv_path: str, concept_column: str = 'concepts') -> pd.DataFrame:
        self.logger.info(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if concept_column not in df.columns:
            raise ValueError(f"Column '{concept_column}' not found in DataFrame.")
        
        self.logger.info(f"Loaded papers: {len(df)}")
        return df
    
    def extract_all_concepts(self, df: pd.DataFrame, concept_column: str = 'concepts') -> List[str]:
        self.logger.info("Extracting all concept keywords...")
        
        all_concepts = []
        for idx, row in df.iterrows():
            concepts = row[concept_column]
            
            # Parse if string (comma-separated)
            if isinstance(concepts, str):
                # Split by comma and remove whitespace
                concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
                all_concepts.extend(concept_list)
            elif isinstance(concepts, list):
                all_concepts.extend(concepts)
            elif pd.notna(concepts):
                all_concepts.append(str(concepts))
        
        # Remove duplicates
        unique_concepts = list(set(all_concepts))
        
        self.logger.info(f"Total concept keywords: {len(all_concepts)}")
        self.logger.info(f"Unique concept keywords: {len(unique_concepts)}")
        
        return unique_concepts
    
    def filter_by_frequency(self, df: pd.DataFrame, concept_column: str = 'concepts',
                           unique_concepts: List[str] = None) -> Tuple[List[str], Dict[str, int]]:
        if unique_concepts is None:
            unique_concepts = self.extract_all_concepts(df, concept_column)
        
        self.logger.info(f"Calculating concept keyword frequencies... (min frequency: {self.min_frequency})")
        
        # Calculate frequencies
        concept_counter = Counter()
        for idx, row in df.iterrows():
            concepts = row[concept_column]
            
            if isinstance(concepts, str):
                concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
            elif isinstance(concepts, list):
                concept_list = concepts
            elif pd.notna(concepts):
                concept_list = [str(concepts)]
            else:
                concept_list = []
            
            concept_counter.update(concept_list)
        
        # Filter by frequency
        filtered_concepts = [
            concept for concept in unique_concepts 
            if concept_counter[concept] >= self.min_frequency
        ]
        
        filtered_out = len(unique_concepts) - len(filtered_concepts)
        
        self.logger.info(f"Before filtering: {len(unique_concepts)}")
        self.logger.info(f"After filtering: {len(filtered_concepts)} (removed: {filtered_out})")
        
        return filtered_concepts, dict(concept_counter)
    
    def create_concept_to_papers_mapping(self, df: pd.DataFrame, 
                                        concept_column: str = 'concepts',
                                        paper_id_column: str = 'paper_id') -> Dict[str, List[str]]:
        self.logger.info("Creating concept-paper mapping...")
        
        concept_to_papers = {}
        
        for idx, row in df.iterrows():
            paper_id = row[paper_id_column] if paper_id_column in df.columns else idx
            concepts = row[concept_column]
            
            if isinstance(concepts, str):
                concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
            elif isinstance(concepts, list):
                concept_list = concepts
            elif pd.notna(concepts):
                concept_list = [str(concepts)]
            else:
                concept_list = []
            
            for concept in concept_list:
                if concept not in concept_to_papers:
                    concept_to_papers[concept] = []
                concept_to_papers[concept].append(str(paper_id))
        
        self.logger.info(f"Mapping created: {len(concept_to_papers)} concepts")
        
        return concept_to_papers

