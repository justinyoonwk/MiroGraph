import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import requests
import time
import json
from typing import Dict, List, Tuple, Set
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAlexClient:
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, rate_limit: float = 0.1):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
    
    def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def get_work_by_title(self, title: str, year: int = None) -> Dict:
        self._wait_for_rate_limit()
        
        try:
            params = {
                "search": title,
                "per_page": 5,
                "mailto": "concept-clustering@example.com"
            }
            
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


def create_eval_test_data(final_csv_path: str,
                               concept_graph_path: str,
                               venue: str = "SIGIR",
                               output_dir: str = None,
                               test_years: Tuple[int, int] = (2022, 2024),
                               citation_years: Tuple[int, int] = (2022, 2024),
                               threshold: int = 1,
                               use_openalex: bool = True,
                               cache_file: str = None,
                               random_seed: int = 42):
    logger.info("=" * 80)
    logger.info("Creating Test Data for Eval Setting")
    logger.info("=" * 80)

    venue_dir = str(venue).lower()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{venue_dir}/eval"
    
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
        
        # Load cache if exists
        citation_cache = {}
        if cache_file and Path(cache_file).exists():
            logger.info(f"Loading citation cache from: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                citation_cache = json.load(f)
            logger.info(f"Loaded {len(citation_cache)} cached citation counts")
        
        openalex = OpenAlexClient(rate_limit=0.1)
        
        # Get all unique papers
        all_paper_ids = set()
        for paper_ids in new_edge_to_papers.values():
            all_paper_ids.update(paper_ids)
        
        logger.info(f"Fetching citation counts for {len(all_paper_ids)} papers...")
        
        paper_citation_counts = {}  # {paper_id: {year: citation_count}}
        
        for paper_id in tqdm(all_paper_ids, desc="Fetching citations"):
            if paper_id in citation_cache:
                paper_citation_counts[paper_id] = citation_cache[paper_id]
                continue
            
            paper_data = test_paper_concepts[paper_id]
            title = paper_data['title']
            year = paper_data['year']
            
            # Search for work in OpenAlex
            work = openalex.get_work_by_title(title, year)
            
            if work:
                citation_counts = openalex.get_citation_counts_by_year(
                    work, 
                    paper_year=year,
                    min_year=citation_years[0],
                    max_year=citation_years[1]
                )
                paper_citation_counts[paper_id] = citation_counts
                citation_cache[paper_id] = citation_counts
            else:
                paper_citation_counts[paper_id] = {}
                citation_cache[paper_id] = {}
        
        # Save cache
        if cache_file:
            logger.info(f"Saving citation cache to: {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(citation_cache, f, indent=2, ensure_ascii=False)
        
        # Aggregate citation counts by edge (sum across citation_years period)
        logger.info("Aggregating citation counts by edge...")
        for edge, paper_ids in tqdm(new_edge_to_papers.items(), desc="Aggregating"):
            total_citations = 0
            
            for paper_id in paper_ids:
                paper_citations = paper_citation_counts.get(paper_id, {})
                # Sum citations in the citation_years period
                for year, count in paper_citations.items():
                    if citation_years[0] <= year <= citation_years[1]:
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
    elif negative_count > 0:
        negative_df = df_all[df_all['label'] == 0].copy()
        positive_df = df_all[df_all['label'] == 1].copy()
        
        if len(negative_df) > positive_count:
            logger.info(f"Sampling {positive_count:,} negative edges from {negative_count:,} total...")
            np.random.seed(random_seed)
            sampled_negative_df = negative_df.sample(n=positive_count, random_state=random_seed)
        else:
            logger.warning(f"Negative edges ({negative_count}) are fewer than positive ({positive_count}). Using all negative edges.")
            sampled_negative_df = negative_df
        
        # Combine positive and sampled negative
        df_balanced = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    else:
        logger.warning("No negative edges found! All edges are positive.")
        df_balanced = df_all.copy()
    
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
    
    parser = argparse.ArgumentParser(description='Create test data for Eval setting')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to final_papers_with_concepts.csv')
    parser.add_argument('--concept-graph', '-g', required=True,
                       help='Path to concept_graph_test.csv (existing edges)')
    parser.add_argument('--venue', '-v', default='SIGIR',
                       help='Venue name (default: SIGIR)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/eval)')
    parser.add_argument('--test-years', nargs=2, type=int, default=[2022, 2024],
                       help='Test paper years range (default: 2022 2024)')
    parser.add_argument('--citation-years', nargs=2, type=int, default=[2022, 2024],
                       help='Citation aggregation years range (default: 2022 2024)')
    parser.add_argument('--threshold', type=int, default=None,
                       help='Single IR threshold (use either --threshold or --thresholds)')
    parser.add_argument('--thresholds', nargs='+', type=int, default=None,
                       help='Multiple IR thresholds (e.g., 5 10 50 100 200)')
    parser.add_argument('--no-openalex', action='store_true',
                       help='Skip OpenAlex API calls (use for testing)')
    parser.add_argument('--cache', default='citation_cache_eval.json',
                       help='Path to citation cache file (default: citation_cache_eval.json)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()

    # Determine thresholds to run
    if args.thresholds is not None and args.threshold is not None:
        raise SystemExit("Please provide only one of --threshold or --thresholds.")
    if args.thresholds is None and args.threshold is None:
        thresholds = [1, 10, 50, 100, 200]
    elif args.thresholds is not None:
        thresholds = list(args.thresholds)
    else:
        thresholds = [int(args.threshold)]

    for t in thresholds:
        create_eval_test_data(
            args.input,
            args.concept_graph,
            venue=args.venue,
            output_dir=args.output_dir,
            test_years=tuple(args.test_years),
            citation_years=tuple(args.citation_years),
            threshold=int(t),
            use_openalex=not args.no_openalex,
            cache_file=args.cache,
            random_seed=args.random_seed
        )

