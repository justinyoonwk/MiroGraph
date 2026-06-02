"""
Create prediction data for transductive setting
- Find concept pairs from 2019-2021 papers that are NOT in the existing concept_graph
- Aggregate citation counts for 2019-2021 period for these new edges
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
        for attempt in range(5):
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
                if response.status_code == 429:
                    time.sleep(min(2 ** attempt + 1, 60))
                    continue
                response.raise_for_status()
                data = response.json()
                
                if data.get("results") and len(data["results"]) > 0:
                    return data["results"][0]
                return None
            except Exception as e:
                if attempt < 4 and "429" in str(e):
                    time.sleep(min(2**attempt, 30))
                    continue
                logger.warning(f"Error fetching work for title '{title[:50]}...': {e}")
                return None
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


def create_prediction_data(final_csv_path: str,
                          concept_graph_path: str,
                          venue: str = "KDD",
                          output_path: str = None,
                          test_years: Tuple[int, int] = (2019, 2021),
                          citation_years: Tuple[int, int] = (2019, 2021),
                          use_openalex: bool = True,
                          cache_file: str = None,
                          bootstrap_cache_file: str = None):
    """
    Create prediction data for transductive setting
    
    Args:
        final_csv_path: Path to final_papers_with_concepts.csv
        concept_graph_path: Path to concept_graph.csv (existing edges)
        output_path: Output CSV file path
        test_years: Tuple of (min_year, max_year) for test papers (default: (2019, 2021))
        citation_years: Tuple of (min_year, max_year) for citation aggregation (default: (2019, 2021))
        use_openalex: Whether to fetch citation counts from OpenAlex
        cache_file: Path to cache file for citation counts (optional)
    """
    
    logger.info("=" * 80)
    logger.info("Creating Prediction Data for Transductive Setting")
    logger.info("=" * 80)
    
    # Set default output path if not provided
    if output_path is None:
        output_path = f"dataset/{venue}/transductive/prediction_data.csv"
    
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
    
    logger.info(f"Existing edges in concept_graph: {len(existing_edges)}")
    logger.info(f"Existing concepts in concept_graph: {len(existing_concepts)}")
    
    # Load all papers
    logger.info(f"Loading papers from: {final_csv_path}")
    df = pd.read_csv(final_csv_path)
    
    # Filter to test years (2019-2021)
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
    
    # Find new edges (not in existing concept_graph)
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
    
    logger.info(f"New edges (not in concept_graph): {len(new_edge_to_papers)}")
    
    if len(new_edge_to_papers) == 0:
        logger.warning("No new edges found! All edges from test papers already exist in concept_graph.")
        # Create empty output file
        output_df = pd.DataFrame(columns=['concept1', 'concept2', 'total_citations'])
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Empty prediction data saved to: {output_path}")
        return output_path
    
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
        logger.info(f"Prediction-cache hits: {n_hits} / {len(all_paper_ids)}")
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
                        total_citations += count
            
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
    
    # Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    logger.info(f"Saving prediction data to: {output_path}")
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values('total_citations', ascending=False)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved {len(output_rows)} prediction records")
    logger.info(f"Unique new edges: {len(new_edge_to_papers)}")
    
    # Statistics
    if use_openalex:
        edges_with_citations = sum(1 for total in edge_citation_totals.values() if total > 0)
        logger.info(f"Edges with citations > 0: {edges_with_citations}")
        logger.info(f"Total citations aggregated: {sum(edge_citation_totals.values())}")
    
    logger.info("=" * 80)
    logger.info("Prediction data creation completed!")
    logger.info("=" * 80)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create prediction data for transductive setting')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to final_papers_with_concepts.csv')
    parser.add_argument('--concept-graph', '-g', required=True,
                       help='Path to concept_graph.csv (existing edges)')
    parser.add_argument('--venue', '-v', default='KDD',
                       help='Venue name (default: KDD)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output CSV file path (default: dataset/{venue}/transductive/prediction_data.csv)')
    parser.add_argument('--test-years', nargs=2, type=int, default=[2024, 2025],
                       help='Future paper years used to label candidate pairs (default: 2024 2025)')
    parser.add_argument('--citation-years', nargs=2, type=int, default=[2024, 2025],
                       help='Citation aggregation years for labels (default: 2024 2025)')
    parser.add_argument('--no-openalex', action='store_true',
                       help='Skip OpenAlex API calls (use for testing)')
    parser.add_argument('--cache', default='citation_cache.json',
                       help='Path to citation cache file (default: citation_cache.json)')
    parser.add_argument('--bootstrap-cache', default='citation_cache_cikm_level25.json',
                       help='Full-year OpenAlex JSON (e.g. from build_concept_graph); trimmed to citation-years')
    parser.add_argument('--no-bootstrap', action='store_true',
                       help='Do not use bootstrap cache (OpenAlex only)')
    
    args = parser.parse_args()
    
    bootstrap_path = None
    if not args.no_bootstrap and args.bootstrap_cache:
        if Path(args.bootstrap_cache).exists():
            bootstrap_path = args.bootstrap_cache
        else:
            logger.warning(f"Bootstrap file not found ({args.bootstrap_cache}); continuing without it")
    
    create_prediction_data(
        args.input,
        args.concept_graph,
        venue=args.venue,
        output_path=args.output,
        test_years=tuple(args.test_years),
        citation_years=tuple(args.citation_years),
        use_openalex=not args.no_openalex,
        cache_file=args.cache,
        bootstrap_cache_file=bootstrap_path
    )
