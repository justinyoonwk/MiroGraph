#!/usr/bin/env python3
"""
Build concept graph for inductive setting
Creates concept_graph_test.csv in dataset/{venue}/inductive/
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Build concept graph for inductive setting (up to max_year)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build for KDD (default) up to 2021
  python build_inductive_graph.py

  # Build for ICML up to 2020
  python build_inductive_graph.py --venue ICML --max-year 2020

  # Specify custom input and cache
  python build_inductive_graph.py --input output/icml/final_papers_with_concepts.csv --cache citation_cache_icml.json
        """
    )
    
    parser.add_argument('--venue', '-v', default='KDD',
                       help='Venue name (default: KDD)')
    parser.add_argument('--input', '-i', default='output/kdd/final_papers_with_concepts.csv',
                       help='Path to final_papers_with_concepts.csv (default: output/kdd/final_papers_with_concepts.csv)')
    parser.add_argument('--max-year', type=int, default=2021,
                       help='Maximum year to include (default: 2021)')
    parser.add_argument('--cache', default='citation_cache.json',
                       help='Path to citation cache file (default: citation_cache.json)')
    parser.add_argument('--no-openalex', action='store_true',
                       help='Skip OpenAlex API calls (use for testing)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        'build_concept_graph.py',
        '--input', args.input,
        '--venue', args.venue,
        '--output-dir', f'dataset/{args.venue}/inductive',
        '--max-year', str(args.max_year),
        '--cache', args.cache,
        '--output-filename', 'concept_graph_test.csv'
    ]
    
    if args.no_openalex:
        cmd.append('--no-openalex')
    
    print("=" * 80)
    print("Building Inductive Concept Graph")
    print("=" * 80)
    print(f"Venue: {args.venue}")
    print(f"Input: {args.input}")
    print(f"Max year: {args.max_year}")
    print(f"Output: dataset/{args.venue}/inductive/concept_graph_test.csv")
    print(f"Cache: {args.cache}")
    print("=" * 80)
    print()
    
    # Run command
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("Successfully created concept_graph_test.csv")
        print(f"Location: dataset/{args.venue}/inductive/concept_graph_test.csv")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print("Error: Failed to create concept_graph_test.csv")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()

