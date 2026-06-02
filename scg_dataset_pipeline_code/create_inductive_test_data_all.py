"""
Create test data for inductive setting with multiple IR thresholds
"""

import subprocess
import sys
from pathlib import Path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test data for inductive setting with multiple IR thresholds')
    parser.add_argument('--venue', '-v', default='KDD',
                       help='Venue name (default: KDD)')
    parser.add_argument('--input', '-i', default='output/kdd/final_papers_with_concepts.csv',
                       help='Path to final_papers_with_concepts.csv')
    parser.add_argument('--concept-graph', '-g', default=None,
                       help='Path to concept_graph_test.csv (default: dataset/{venue}/inductive/concept_graph_test.csv)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/inductive)')
    parser.add_argument('--test-years', nargs=2, type=int, default=[2022, 2023],
                       help='Test paper years range (default: 2022 2023)')
    parser.add_argument('--citation-years', nargs=2, type=int, default=[2022, 2023],
                       help='Citation aggregation years range (default: 2022 2023)')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[1, 10, 50, 100, 200],
                       help='IR thresholds to process (default: 1 10 50 100 200)')
    parser.add_argument('--cache', default='citation_cache_inductive.json',
                       help='Path to citation cache file (default: citation_cache_inductive.json)')
    
    args = parser.parse_args()
    
    # Set defaults based on venue
    if args.concept_graph is None:
        args.concept_graph = f"dataset/{args.venue}/inductive/concept_graph_test.csv"
    if args.output_dir is None:
        args.output_dir = f"dataset/{args.venue}/inductive"
    
    thresholds = args.thresholds
    
    print("=" * 80)
    print("Creating Inductive Test Data for Multiple IR Thresholds")
    print("=" * 80)
    print(f"Venue: {args.venue}")
    print(f"Thresholds: {thresholds}")
    print()
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Processing threshold: {threshold}")
        print(f"{'='*80}\n")
        
        cmd = [
            sys.executable,
            "create_inductive_test_data.py",
            "--input", args.input,
            "--concept-graph", args.concept_graph,
            "--venue", args.venue,
            "--output-dir", args.output_dir,
            "--test-years", str(args.test_years[0]), str(args.test_years[1]),
            "--citation-years", str(args.citation_years[0]), str(args.citation_years[1]),
            "--threshold", str(threshold),
            "--cache", args.cache
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"\nERROR: Failed to create test data for threshold {threshold}")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("All test data created successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
