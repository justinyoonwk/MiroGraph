import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Direct import
from pipeline import ConceptClusteringPipeline


def main():
    parser = argparse.ArgumentParser(description='Run Concept Clustering Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file path')
    parser.add_argument('--concept-column', default='concepts', help='Concept keyword column name')
    parser.add_argument('--paper-id-column', default='paper_id', help='Paper ID column name')
    parser.add_argument('--probability-threshold', type=float, default=0.8,
                       help='Probability threshold (multiple of max probability, default: 0.8)')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum frequency (default: 2)')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                       help='Minimum cluster size (default: 2, generates more clusters)')
    parser.add_argument('--min-samples', type=int, default=1,
                       help='Minimum samples for cluster core points (default: 1, generates more clusters)')
    parser.add_argument('--cluster-selection-method', type=str, default='leaf',
                       choices=['eom', 'leaf'],
                       help='Cluster selection method: eom(Excess of Mass) or leaf(selects smaller clusters too, default: leaf)')
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.2,
                       help='Cluster selection epsilon value (default: 0.2, higher values reduce outliers, try 0.3-0.5 for minimal outliers)')
    parser.add_argument('--allow-single-cluster', action='store_true',
                       help='Allow all points to form a single cluster (eliminates all outliers but may create one giant cluster)')
    parser.add_argument('--no-save-intermediate', action='store_true',
                       help='Do not save intermediate results')
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM steps (validation and naming), only perform clustering')
    parser.add_argument('--venue', '-v', default='SIGIR',
                       help='Venue name (default: SIGIR)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory path (default: output/{venue})')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"output/{args.venue}"
    
    # Run pipeline
    pipeline = ConceptClusteringPipeline(args.config)
    
    # Override settings
    if args.min_frequency:
        pipeline.data_loader.min_frequency = args.min_frequency
    if args.min_cluster_size:
        pipeline.clusterer.min_cluster_size = args.min_cluster_size
    if args.min_samples:
        pipeline.clusterer.min_samples = args.min_samples
    if args.cluster_selection_method:
        pipeline.clusterer.cluster_selection_method = args.cluster_selection_method
    if args.cluster_selection_epsilon is not None:
        pipeline.clusterer.cluster_selection_epsilon = args.cluster_selection_epsilon
    if args.allow_single_cluster:
        pipeline.clusterer.allow_single_cluster = True
    
    result_df = pipeline.run(
        csv_path=args.input,
        concept_column=args.concept_column,
        paper_id_column=args.paper_id_column,
        probability_threshold=args.probability_threshold,
        save_intermediate=not args.no_save_intermediate,
        skip_llm=args.skip_llm,
        output_dir=args.output_dir,
        venue=args.venue
    )
    
    print("\nPipeline execution completed!")
    if args.skip_llm:
        print("LLM steps were skipped. Check clustering_results.json for clustering results.")
    else:
        print(f"Result DataFrame shape: {result_df.shape}")
        print(f"\nAssigned concepts for first 5 papers:")
        for idx, row in result_df.head().iterrows():
            print(f"  Paper {idx}: {row.get('assigned_concepts', [])}")


if __name__ == "__main__":
    main()

