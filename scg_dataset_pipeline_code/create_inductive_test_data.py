"""
Create test data for inductive setting
- Find concept pairs from 2022-2024 papers that are NOT in the existing concept_graph_test
- Aggregate citation counts for 2022-2024 period for these new edges
- Create balanced test data with positive/negative based on IR threshold
  - Negative edges include: citation count < threshold OR never co-researched pairs
- No train/test split (all data is test data)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import requests
import time
import json
import os
from typing import Dict, List, Tuple, Set
import logging
from tqdm import tqdm
from itertools import combinations
from openalex_utils import (
    OpenAlexAPIError,
    OpenAlexClient as RobustOpenAlexClient,
    cache_record_to_counts,
    load_citation_cache,
    save_citation_cache,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _trim_citations_to_window(
    by_year: Dict[int, int],
    paper_year: int,
    lo: int,
    hi: int,
) -> Dict[int, int]:
    """Keep only years in [lo, hi] and on/after paper publication."""
    out = {}
    for y, c in by_year.items():
        y = int(y)
        if paper_year and y < paper_year:
            continue
        if lo <= y <= hi:
            out[y] = int(c)
    return out


class OpenAlexClient:
    """Client for OpenAlex API"""
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, rate_limit: float = 0.1, mailto: str = None):
        """
        Args:
            rate_limit: Minimum seconds between API calls (OpenAlex allows 1 request per 0.1 seconds)
            mailto: Contact email for OpenAlex polite pool
        """
        self.rate_limit = rate_limit
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO") or "justinyoonwk@kaist.ac.kr"
        self.last_request_time = 0
        self.session = requests.Session()
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def get_work_by_title(self, title: str, year: int = None) -> Dict:
        """
        Get work information by title from OpenAlex
        
        Args:
            title: Paper title
            year: Publication year (optional, for filtering)
            
        Returns:
            Work information dict or None if not found
        """
        self._wait_for_rate_limit()
        
        try:
            params = {
                "search": title,
                "per_page": 5,
            }
            if self.mailto:
                params["mailto"] = self.mailto
            
            if year:
                params["filter"] = f"publication_year:{year}"
            
            response = self.session.get(
                f"{self.BASE_URL}/works",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                return data["results"][0]
            return None
        except Exception as e:
            logger.warning(f"Error fetching work for title '{title[:50]}...': {e}")
            return None
    
    def get_citation_counts_by_year(self, work: Dict, paper_year: int = None, 
                                    min_year: int = None, max_year: int = None) -> Dict[int, int]:
        """
        Get citation counts by year from a work object
        
        Args:
            work: OpenAlex work object (from get_work_by_title)
            paper_year: Publication year of the paper (to filter citations)
            min_year: Minimum year to include (default: None)
            max_year: Maximum year to include (default: None)
            
        Returns:
            Dictionary mapping year to citation count (only from paper_year to max_year)
        """
        citation_counts = {}
        
        try:
            # Get yearly citation counts from counts_by_year
            counts_by_year = work.get("counts_by_year", [])
            for count_data in counts_by_year:
                year = count_data.get("year")
                citations = count_data.get("cited_by_count", 0)
                
                if year and citations > 0:
                    # Filter by paper_year, min_year, and max_year
                    if paper_year and year < paper_year:
                        continue  # Skip citations before paper publication
                    if min_year and year < min_year:
                        continue  # Skip citations before min_year
                    if max_year and year > max_year:
                        continue  # Skip citations after max_year
                    
                    citation_counts[year] = citations
            
            return citation_counts
        except Exception as e:
            logger.warning(f"Error extracting citation counts from work: {e}")
            return {}


def create_inductive_test_data(final_csv_path: str,
                               concept_graph_path: str,
                               venue: str = "KDD",
                               output_dir: str = None,
                               test_years: Tuple[int, int] = (2022, 2024),
                               citation_years: Tuple[int, int] = (2022, 2024),
                               threshold: int = 1,
                               use_openalex: bool = True,
                               cache_file: str = None,
                               bootstrap_cache_file: str = None,
                               random_seed: int = 42):
    """
    Create test data for inductive setting
    
    Args:
        final_csv_path: Path to final_papers_with_concepts.csv
        concept_graph_path: Path to concept_graph_test.csv (existing edges)
        output_dir: Output directory for test files
        test_years: Tuple of (min_year, max_year) for test papers (default: (2022, 2024))
        citation_years: Tuple of (min_year, max_year) for citation aggregation (default: (2022, 2024))
        threshold: IR threshold for positive/negative classification (default: 1)
        use_openalex: Whether to fetch citation counts from OpenAlex
        cache_file: Path to cache file for citation counts (optional)
        bootstrap_cache_file: Full-year citation JSON; trimmed to citation_years (optional)
        random_seed: Random seed for reproducibility (default: 42)
    """
    
    logger.info("=" * 80)
    logger.info("Creating Test Data for Inductive Setting")
    logger.info("=" * 80)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{venue}/inductive"
    
    # Load existing concept graph edges
    logger.info(f"Loading existing concept graph from: {concept_graph_path}")
    graph_df = pd.read_csv(concept_graph_path)
    
    # Create set of existing edges and concepts
    existing_edges = set()
    existing_concepts = set()
    for _, row in graph_df.iterrows():
        concept1 = str(row['concept1']).strip()
        concept2 = str(row['concept2']).strip()
        edge = tuple(sorted([concept1, concept2]))
        existing_edges.add(edge)
        existing_concepts.add(concept1)
        existing_concepts.add(concept2)
    
    logger.info(f"Existing edges in concept_graph_test: {len(existing_edges)}")
    logger.info(f"Existing concepts in concept_graph_test: {len(existing_concepts)}")
    
    # Load all papers
    logger.info(f"Loading papers from: {final_csv_path}")
    df = pd.read_csv(final_csv_path)
    
    # Filter to test years (2022-2024)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df[df['year'].notna() & 
                (df['year'] >= test_years[0]) & 
                (df['year'] <= test_years[1]) &
                (df['year'] >= 1900)].copy()
        logger.info(f"Filtered to papers from {test_years[0]} to {test_years[1]}: {len(df)} papers")
    
    # Parse assigned concepts from test papers
    logger.info("Parsing assigned concepts from test papers...")
    test_paper_concepts = {}
    for idx, row in df.iterrows():
        paper_id = row.get('paper_id', idx)
        title = row.get('title', '')
        year = row.get('year', None)
        
        if pd.isna(year) or year is None:
            continue
        
        concepts_str = row.get('assigned_concepts', '')
        if pd.isna(concepts_str) or concepts_str == '':
            concepts = []
        elif isinstance(concepts_str, str):
            concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
        else:
            concepts = []
        
        if len(concepts) > 0:
            test_paper_concepts[paper_id] = {
                'title': title,
                'year': int(year),
                'concepts': concepts
            }
    
    logger.info(f"Test papers with concepts: {len(test_paper_concepts)}")
    
    # Find new edges (not in existing concept_graph_test)
    logger.info("Finding new edges from test papers...")
    new_edge_to_papers = defaultdict(list)  # {(concept1, concept2): [paper_id1, paper_id2, ...]}
    
    for paper_id, paper_data in test_paper_concepts.items():
        concepts = paper_data['concepts']
        # Create edges for all pairs of concepts in the same paper
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept1 = concepts[i]
                concept2 = concepts[j]
                edge = tuple(sorted([concept1, concept2]))
                
                # Only include if:
                # 1. NOT in existing edges (new edge)
                # 2. Both concepts exist in existing graph (both nodes must be in graph)
                if edge not in existing_edges:
                    if concept1 in existing_concepts and concept2 in existing_concepts:
                        new_edge_to_papers[edge].append(paper_id)
    
    logger.info(f"New edges (not in concept_graph_test): {len(new_edge_to_papers)}")
    
    if len(new_edge_to_papers) == 0:
        logger.warning("No new edges found! All edges from test papers already exist in concept_graph_test.")
        # Create empty output file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(columns=['concept1', 'concept2', 'total_citations', 'label'])
        output_file = output_path / f"test_data_{threshold}.csv"
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Empty test data saved to: {output_file}")
        return output_file
    
    # Get citation counts if requested
    edge_citation_totals = {}  # {(concept1, concept2): total_citations}
    
    if use_openalex:
        logger.info("Fetching citation counts from OpenAlex API...")
        logger.info(f"Citation aggregation period: {citation_years[0]} to {citation_years[1]}")
        
        citation_cache = load_citation_cache(cache_file)
        if citation_cache:
            logger.info(f"Loaded {len(citation_cache)} cached citation records from: {cache_file}")
        
        bootstrap_cache: Dict[str, dict] = {}
        if bootstrap_cache_file and Path(bootstrap_cache_file).exists():
            logger.info(f"Loading bootstrap citation cache from: {bootstrap_cache_file}")
            bootstrap_cache = load_citation_cache(bootstrap_cache_file)
            logger.info(f"Bootstrap: {len(bootstrap_cache)} cached citation records")
        
        openalex = RobustOpenAlexClient(rate_limit=2.0)
        
        # Get all unique papers
        all_paper_ids = set()
        for paper_ids in new_edge_to_papers.values():
            all_paper_ids.update(paper_ids)
        
        logger.info(f"Fetching citation counts for {len(all_paper_ids)} papers...")
        
        paper_citation_counts = {}  # {paper_id: {year: citation_count}}
        
        n_hits = 0
        n_bootstrap = 0
        for paper_id in tqdm(all_paper_ids, desc="Fetching citations"):
            key = str(paper_id)
            paper_data = test_paper_concepts[paper_id]
            title = paper_data['title']
            year = paper_data['year']
            
            cached_counts = cache_record_to_counts(citation_cache.get(key), year, citation_years[0], citation_years[1])
            if cached_counts is not None:
                paper_citation_counts[paper_id] = cached_counts
                n_hits += 1
                continue
            
            bootstrap_counts = cache_record_to_counts(bootstrap_cache.get(key), year, citation_years[0], citation_years[1])
            if bootstrap_counts is not None:
                paper_citation_counts[paper_id] = bootstrap_counts
                citation_cache[key] = bootstrap_cache[key]
                n_bootstrap += 1
                continue
            
            try:
                record = openalex.lookup_citations_by_title(title, year)
            except OpenAlexAPIError as exc:
                logger.warning(f"OpenAlex API error for paper {paper_id} '{title[:80]}...': {exc}")
                paper_citation_counts[paper_id] = {}
                continue

            citation_cache[key] = record
            paper_citation_counts[paper_id] = (
                cache_record_to_counts(record, year, citation_years[0], citation_years[1]) or {}
            )

            if cache_file and len(citation_cache) % 100 == 0:
                save_citation_cache(citation_cache, cache_file)
        logger.info(f"Inductive-cache hits: {n_hits} / {len(all_paper_ids)}")
        logger.info(f"Bootstrap-cache fills: {n_bootstrap} / {len(all_paper_ids)}")
        
        # Save cache
        if cache_file:
            logger.info(f"Saving citation cache to: {cache_file}")
            save_citation_cache(citation_cache, cache_file)
        
        # Aggregate citation counts by edge (sum across citation_years period)
        logger.info("Aggregating citation counts by edge...")
        for edge, paper_ids in tqdm(new_edge_to_papers.items(), desc="Aggregating"):
            total_citations = 0
            
            for paper_id in paper_ids:
                paper_citations = paper_citation_counts.get(paper_id, {})
                # Sum citations in the citation_years period
                for year, count in paper_citations.items():
                    y = int(year)
                    if citation_years[0] <= y <= citation_years[1]:
                        total_citations += int(count)
            
            edge_citation_totals[edge] = total_citations
    
    # Prepare output data
    logger.info("Preparing output data...")
    output_rows = []
    
    for edge, paper_ids in tqdm(new_edge_to_papers.items(), desc="Preparing output"):
        concept1, concept2 = edge
        
        if use_openalex:
            total_citations = edge_citation_totals.get(edge, 0)
        else:
            total_citations = 0
        
        output_rows.append({
            'concept1': concept1,
            'concept2': concept2,
            'total_citations': int(total_citations)
        })
    
    # Create DataFrame
    df_all = pd.DataFrame(output_rows)
    
    # Classify as positive/negative based on threshold
    logger.info(f"\nClassifying edges with threshold: {threshold}")
    df_all['label'] = (df_all['total_citations'] >= threshold).astype(int)
    # 1 = positive (IR >= threshold), 0 = negative (IR < threshold)
    
    positive_count = (df_all['label'] == 1).sum()
    negative_count = (df_all['label'] == 0).sum()
    
    logger.info(f"Positive edges (IR >= {threshold}): {positive_count:,}")
    logger.info(f"Negative edges (IR < {threshold}): {negative_count:,}")
    
    if positive_count == 0:
        logger.warning(f"No positive edges found with threshold {threshold}!")
        logger.warning("All edges will be treated as negative (label=0).")
        # Still create the dataset, but all will be negative
        df_balanced = df_all.copy()
    else:
        # Get positive edges
        positive_df = df_all[df_all['label'] == 1].copy()
        
        # Create negative candidate pool:
        # 1. Edges from prediction data with citation count < threshold
        # 2. Never co-researched pairs (not in concept_graph_test and not in prediction_data)
        logger.info(f"\nBuilding negative candidate pool...")
        
        # 1. Negative edges from prediction data
        negative_from_prediction = df_all[df_all['label'] == 0].copy()
        logger.info(f"Negative edges from prediction data: {len(negative_from_prediction):,}")
        
        # 2. Get all edges from prediction data (to exclude from never-researched pairs)
        prediction_edges = set()
        for _, row in df_all.iterrows():
            concept1 = str(row['concept1']).strip()
            concept2 = str(row['concept2']).strip()
            edge = tuple(sorted([concept1, concept2]))
            prediction_edges.add(edge)
        
        # 3. Generate never co-researched pairs
        logger.info("Generating never co-researched pairs...")
        existing_concepts_list = sorted(list(existing_concepts))
        never_researched_pairs = []
        
        # Generate all possible pairs from existing concepts
        for concept1, concept2 in combinations(existing_concepts_list, 2):
            edge = tuple(sorted([concept1, concept2]))
            # Only include if not in existing_edges and not in prediction_edges
            if edge not in existing_edges and edge not in prediction_edges:
                never_researched_pairs.append({
                    'concept1': concept1,
                    'concept2': concept2,
                    'total_citations': 0,
                    'label': 0
                })
        
        logger.info(f"Never co-researched pairs: {len(never_researched_pairs):,}")
        
        # Combine negative candidates
        if len(never_researched_pairs) > 0:
            never_researched_df = pd.DataFrame(never_researched_pairs)
            negative_candidates = pd.concat([negative_from_prediction, never_researched_df], ignore_index=True)
        else:
            negative_candidates = negative_from_prediction
        
        logger.info(f"Total negative candidates: {len(negative_candidates):,}")
        
        # Sample negative edges to match positive count
        if len(negative_candidates) > positive_count:
            logger.info(f"Sampling {positive_count:,} negative edges from {len(negative_candidates):,} candidates...")
            np.random.seed(random_seed)
            sampled_negative_df = negative_candidates.sample(n=positive_count, random_state=random_seed)
        else:
            logger.warning(f"Negative candidates ({len(negative_candidates)}) are fewer than positive ({positive_count}). Using all negative candidates.")
            sampled_negative_df = negative_candidates
        
        # Combine positive and sampled negative
        df_balanced = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
        
        # Log statistics about sampled negative data
        sampled_from_prediction_count = 0
        sampled_never_researched_count = 0
        
        for _, row in sampled_negative_df.iterrows():
            concept1 = str(row['concept1']).strip()
            concept2 = str(row['concept2']).strip()
            edge = tuple(sorted([concept1, concept2]))
            if edge in prediction_edges:
                sampled_from_prediction_count += 1
            else:
                sampled_never_researched_count += 1
        
        logger.info(f"Sampled negative edges:")
        logger.info(f"  From prediction data (IR < threshold): {sampled_from_prediction_count:,}")
        logger.info(f"  Never co-researched pairs: {sampled_never_researched_count:,}")
    
    # If no positive edges, still save but warn
    if positive_count == 0:
        logger.warning("Saving test data with all negative edges (no positive edges found).")
    
    logger.info(f"\nBalanced test dataset:")
    logger.info(f"  Total: {len(df_balanced):,}")
    logger.info(f"  Positive: {(df_balanced['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(df_balanced['label'] == 0).sum():,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save test data (with threshold in filename)
    output_file = output_path / f"test_data_{threshold}.csv"
    logger.info(f"\nSaving test data to: {output_file}")
    df_balanced[['concept1', 'concept2', 'total_citations', 'label']].to_csv(
        output_file, index=False, encoding='utf-8'
    )
    
    logger.info("=" * 80)
    logger.info("Test data creation completed!")
    logger.info("=" * 80)
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test data for inductive setting')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to final_papers_with_concepts.csv')
    parser.add_argument('--concept-graph', '-g', required=True,
                       help='Path to concept_graph_test.csv (existing edges)')
    parser.add_argument('--venue', '-v', default='KDD',
                       help='Venue name (default: KDD)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/inductive)')
    parser.add_argument('--test-years', nargs=2, type=int, default=[2022, 2023],
                       help='Future paper years used to label candidate pairs (default: 2022 2023)')
    parser.add_argument('--citation-years', nargs=2, type=int, default=[2022, 2023],
                       help='Citation aggregation years for labels (default: 2022 2023)')
    parser.add_argument('--threshold', type=int, default=1,
                       help='IR threshold for positive/negative classification (default: 1)')
    parser.add_argument('--no-openalex', action='store_true',
                       help='Skip OpenAlex API calls (use for testing)')
    parser.add_argument('--cache', default='citation_cache_inductive.json',
                       help='Path to citation cache file (default: citation_cache_inductive.json)')
    parser.add_argument('--bootstrap-cache', default=None,
                       help='Full-year OpenAlex JSON; trimmed to citation-years (optional)')
    parser.add_argument('--no-bootstrap', action='store_true',
                       help='Do not use bootstrap cache (OpenAlex only)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    bootstrap_path = None
    if not args.no_bootstrap and args.bootstrap_cache:
        if Path(args.bootstrap_cache).exists():
            bootstrap_path = args.bootstrap_cache
        else:
            logger.warning(f"Bootstrap file not found ({args.bootstrap_cache}); continuing without it")
    
    create_inductive_test_data(
        args.input,
        args.concept_graph,
        venue=args.venue,
        output_dir=args.output_dir,
        test_years=tuple(args.test_years),
        citation_years=tuple(args.citation_years),
        threshold=args.threshold,
        use_openalex=not args.no_openalex,
        cache_file=args.cache,
        bootstrap_cache_file=bootstrap_path,
        random_seed=args.random_seed
    )
