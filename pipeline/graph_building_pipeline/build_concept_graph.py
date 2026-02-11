import pandas as pd
import numpy as np
from collections import defaultdict, Counter
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
            # Search by title (OpenAlex search is case-insensitive and handles partial matches)
            params = {
                "search": title,
                "per_page": 5,  # Get top 5 results to find best match
                "mailto": "concept-clustering@example.com"  # Recommended by OpenAlex
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
                # Try to find best match by title similarity
                results = data["results"]
                # Use first result (OpenAlex usually returns best match first)
                # Could add more sophisticated matching here if needed
                return results[0]
            return None
        except Exception as e:
            logger.warning(f"Error fetching work for title '{title[:50]}...': {e}")
            return None
    
    def get_citation_counts_by_year(self, work: Dict, paper_year: int = None, max_year: int = None) -> Dict[int, int]:
        citation_counts = {}
        
        try:
            # Get yearly citation counts from counts_by_year
            counts_by_year = work.get("counts_by_year", [])
            for count_data in counts_by_year:
                year = count_data.get("year")
                citations = count_data.get("cited_by_count", 0)
                
                if year and citations > 0:
                    # Filter by paper_year and max_year
                    if paper_year and year < paper_year:
                        continue  # Skip citations before paper publication
                    if max_year and year > max_year:
                        continue  # Skip citations after max_year
                    
                    citation_counts[year] = citations
            
            return citation_counts
        except Exception as e:
            logger.warning(f"Error extracting citation counts from work: {e}")
            return {}


def build_concept_graph(final_csv_path: str,
                        venue: str = "SIGIR",
                        output_dir: str = None,
                        max_year: int = 2018,
                        use_openalex: bool = True,
                        cache_file: str = None,
                        output_filename: str = "concept_graph.csv"):
    logger.info("=" * 80)
    logger.info("Building Concept Graph")
    logger.info("=" * 80)
    
    venue_dir = str(venue).lower()

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{venue_dir}/test"
    
    # Load data
    logger.info(f"Loading data from: {final_csv_path}")
    df = pd.read_csv(final_csv_path)
    
    # Filter by year
    if 'year' in df.columns:
        # Convert year to numeric, invalid values become NaN
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # Filter: year must be valid (not NaN) and <= max_year
        df = df[df['year'].notna() & (df['year'] <= max_year) & (df['year'] >= 1900)].copy()
        logger.info(f"Filtered to papers up to {max_year}: {len(df)} papers")
    
    # Parse assigned concepts
    logger.info("Parsing assigned concepts...")
    paper_concepts = {}
    for idx, row in df.iterrows():
        paper_id = row.get('paper_id', idx)
        title = row.get('title', '')
        year = row.get('year', None)
        
        # Ensure year is valid
        if pd.isna(year) or year is None:
            logger.warning(f"Skipping paper with invalid year: {title[:50]}...")
            continue
        
        concepts_str = row.get('assigned_concepts', '')
        if pd.isna(concepts_str) or concepts_str == '':
            concepts = []
        elif isinstance(concepts_str, str):
            concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
        else:
            concepts = []
        
        if len(concepts) > 0:
            paper_concepts[paper_id] = {
                'title': title,
                'year': int(year),  # Ensure integer
                'concepts': concepts
            }
    
    logger.info(f"Papers with concepts: {len(paper_concepts)}")
    
    # Build edge list (concept pairs that appear together)
    logger.info("Building edge list...")
    edge_to_papers = defaultdict(list)  # {(concept1, concept2): [paper_id1, paper_id2, ...]}
    
    for paper_id, paper_data in paper_concepts.items():
        concepts = paper_data['concepts']
        # Create edges for all pairs of concepts in the same paper
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept1 = concepts[i]
                concept2 = concepts[j]
                # Sort to ensure consistent edge representation
                edge = tuple(sorted([concept1, concept2]))
                edge_to_papers[edge].append(paper_id)
    
    logger.info(f"Total unique edges: {len(edge_to_papers)}")
    
    # Get citation counts if requested
    edge_citation_counts = {}  # {(concept1, concept2): {year: citation_count}}
    paper_citation_counts = {}  # {paper_id: {year: citation_count}} - Initialize here for use in output preparation
    
    if use_openalex:
        logger.info("Fetching citation counts from OpenAlex API...")
        
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
        for paper_ids in edge_to_papers.values():
            all_paper_ids.update(paper_ids)
        
        logger.info(f"Fetching citation counts for {len(all_paper_ids)} papers...")
        logger.info(f"Citation year range: from paper publication year to {max_year}")
        
        for paper_id in tqdm(all_paper_ids, desc="Fetching citations"):
            if paper_id in citation_cache:
                paper_citation_counts[paper_id] = citation_cache[paper_id]
                continue
            
            paper_data = paper_concepts[paper_id]
            title = paper_data['title']
            year = paper_data['year']
            
            # Search for work in OpenAlex
            work = openalex.get_work_by_title(title, year)
            
            if work:
                citation_counts = openalex.get_citation_counts_by_year(work, paper_year=year, max_year=max_year)
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
        
        # Aggregate citation counts by edge
        logger.info("Aggregating citation counts by edge...")
        for edge, paper_ids in tqdm(edge_to_papers.items(), desc="Aggregating"):
            edge_citations = defaultdict(int)  # {year: total_citations}
            
            for paper_id in paper_ids:
                paper_citations = paper_citation_counts.get(paper_id, {})
                for year, count in paper_citations.items():
                    edge_citations[year] += count
            
            edge_citation_counts[edge] = dict(edge_citations)
    
    # Prepare output data
    logger.info("Preparing output data...")
    output_rows = []
    
    for edge, paper_ids in tqdm(edge_to_papers.items(), desc="Preparing output"):
        concept1, concept2 = edge
        
        # Find the minimum year for papers involving this edge
        edge_years = []
        for paper_id in paper_ids:
            if paper_id in paper_concepts:
                paper_year = paper_concepts[paper_id].get('year')
                # Ensure year is valid (not None, not NaN, within valid range)
                if paper_year is not None and not pd.isna(paper_year) and 1900 <= paper_year <= max_year:
                    edge_years.append(int(paper_year))
        
        min_year = min(edge_years) if edge_years else None
        
        if use_openalex and edge in edge_citation_counts:
            citations_by_year = edge_citation_counts[edge]
            if citations_by_year:
                # Create a row for each year with citations
                for year in sorted(citations_by_year.keys()):
                    citation_count = citations_by_year[year]
                    output_rows.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'year': int(year) if year is not None else None,
                        'citation_count': int(citation_count)
                    })
            else:
                # No citation data: fill from min_year to max_year with 0
                if min_year is not None:
                    for year in range(min_year, max_year + 1):
                        output_rows.append({
                            'concept1': concept1,
                            'concept2': concept2,
                            'year': year,
                            'citation_count': 0
                        })
                else:
                    # No year information, include with year=None
                    output_rows.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'year': None,
                        'citation_count': 0
                    })
        else:
            # No OpenAlex data: fill from min_year to max_year with 0
            if min_year is not None:
                for year in range(min_year, max_year + 1):
                    output_rows.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'year': year,
                        'citation_count': 0
                    })
            else:
                # No year information, include with year=None
                output_rows.append({
                    'concept1': concept1,
                    'concept2': concept2,
                    'year': None,
                    'citation_count': 0
                })
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_path / output_filename
    logger.info(f"Saving graph to: {output_file}")
    
    output_df = pd.DataFrame(output_rows)
    # Ensure year is integer type (not float)
    output_df['year'] = output_df['year'].astype('Int64')  # Nullable integer type
    output_df['citation_count'] = output_df['citation_count'].astype(int)
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info(f"Saved {len(output_rows)} edge-year records")
    logger.info(f"Unique edges: {len(edge_to_papers)}")
    logger.info(f"Unique concepts: {len(set([c for edge in edge_to_papers.keys() for c in edge]))}")
    
    # Also save summary statistics
    summary_file = output_path / "concept_graph_summary.json"
    summary = {
        "total_papers": len(paper_concepts),
        "total_edges": len(edge_to_papers),
        "total_concepts": len(set([c for edge in edge_to_papers.keys() for c in edge])),
        "total_edge_year_records": len(output_rows),
        "papers_with_citations": sum(1 for pid in paper_concepts.keys() 
                                    if use_openalex and paper_citation_counts.get(pid, {})),
        "max_year": max_year
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 80)
    logger.info("Graph building completed!")
    logger.info("=" * 80)
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build concept graph from assigned concepts')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to final_papers_with_concepts.csv')
    parser.add_argument('--venue', '-v', default='SIGIR',
                       help='Venue name (default: SIGIR)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/test)')
    parser.add_argument('--max-year', type=int, default=2018,
                       help='Maximum year to include (default: 2018)')
    parser.add_argument('--no-openalex', action='store_true',
                       help='Skip OpenAlex API calls (use for testing)')
    parser.add_argument('--cache', default='citation_cache.json',
                       help='Path to citation cache file (default: citation_cache.json)')
    parser.add_argument('--output-filename', default='concept_graph.csv',
                       help='Output filename (default: concept_graph.csv)')
    
    args = parser.parse_args()
    
    build_concept_graph(
        args.input,
        venue=args.venue,
        output_dir=args.output_dir,
        max_year=args.max_year,
        use_openalex=not args.no_openalex,
        cache_file=args.cache,
        output_filename=args.output_filename
    )

