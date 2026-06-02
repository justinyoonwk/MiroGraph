"""
Create training and test data from prediction data
- Classify edges as positive (IR >= threshold) or negative (IR < threshold)
- Sample negative edges to match positive count
  - Negative edges include: citation count < threshold OR never co-researched pairs
- Split into 85% training and 15% test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_train_test_data(prediction_data_path: str,
                           concept_graph_path: str,
                           venue: str = "KDD",
                           output_dir: str = None,
                           threshold: int = 1,
                           train_ratio: float = 0.85,
                           random_seed: int = 42,
                           train_only: bool = False,
                           train_filename: str = None,
                           test_filename: str = None):
    """
    Create training and test data from prediction data
    
    Args:
        prediction_data_path: Path to prediction_data.csv
        concept_graph_path: Path to concept_graph.csv (to identify existing edges)
        output_dir: Output directory for train/test files
        threshold: IR threshold for positive/negative classification (default: 1)
        train_ratio: Ratio for training data (default: 0.85)
        random_seed: Random seed for reproducibility (default: 42)
    """
    
    logger.info("=" * 80)
    logger.info("Creating Training and Test Data")
    logger.info("=" * 80)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{venue}/transductive"
    
    # Load concept graph to identify existing edges and concepts
    logger.info(f"Loading concept graph from: {concept_graph_path}")
    graph_df = pd.read_csv(concept_graph_path)
    
    # Create set of existing edges and concepts from concept graph
    existing_edges = set()
    existing_concepts = set()
    for _, row in graph_df.iterrows():
        concept1 = str(row['concept1']).strip()
        concept2 = str(row['concept2']).strip()
        edge = tuple(sorted([concept1, concept2]))
        existing_edges.add(edge)
        existing_concepts.add(concept1)
        existing_concepts.add(concept2)
    
    logger.info(f"Existing edges in concept_graph: {len(existing_edges):,}")
    logger.info(f"Existing concepts in concept_graph: {len(existing_concepts):,}")
    
    # Load prediction data
    logger.info(f"Loading prediction data from: {prediction_data_path}")
    df = pd.read_csv(prediction_data_path)
    
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Citation statistics:")
    logger.info(f"  Min: {df['total_citations'].min()}")
    logger.info(f"  Max: {df['total_citations'].max()}")
    logger.info(f"  Mean: {df['total_citations'].mean():.2f}")
    logger.info(f"  Median: {df['total_citations'].median():.2f}")
    
    # Classify as positive/negative based on threshold
    logger.info(f"\nClassifying edges with threshold: {threshold}")
    df['label'] = (df['total_citations'] >= threshold).astype(int)
    # 1 = positive (IR >= threshold), 0 = negative (IR < threshold)
    
    positive_count = (df['label'] == 1).sum()
    negative_count = (df['label'] == 0).sum()
    
    logger.info(f"Positive edges (IR >= {threshold}): {positive_count:,}")
    logger.info(f"Negative edges (IR < {threshold}): {negative_count:,}")
    
    if positive_count == 0:
        logger.error("No positive edges found! Please lower the threshold.")
        return None, None
    
    # Get positive edges
    positive_df = df[df['label'] == 1].copy()
    
    # Create negative candidate pool:
    # 1. Edges from prediction data with citation count < threshold
    # 2. Never co-researched pairs (not in concept_graph and not in prediction_data)
    logger.info(f"\nBuilding negative candidate pool...")
    
    # 1. Negative edges from prediction data
    negative_from_prediction = df[df['label'] == 0].copy()
    logger.info(f"Negative edges from prediction data: {len(negative_from_prediction):,}")
    
    # 2. Get all edges from prediction data (to exclude from never-researched pairs)
    prediction_edges = set()
    for _, row in df.iterrows():
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
    sampled_df = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
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
    
    logger.info(f"\nBalanced dataset:")
    logger.info(f"  Total: {len(sampled_df):,}")
    logger.info(f"  Positive: {(sampled_df['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(sampled_df['label'] == 0).sum():,}")
    
    if train_only:
        logger.info("\nTrain-only mode: using the sampled balanced dataset as training data")
        train_df = sampled_df
        test_df = None
    else:
        # Split into training and test sets
        logger.info(f"\nSplitting into training ({train_ratio*100:.0f}%) and test ({(1-train_ratio)*100:.0f}%)...")
        train_df, test_df = train_test_split(
            sampled_df,
            test_size=1 - train_ratio,
            random_state=random_seed,
            stratify=sampled_df['label']  # Maintain class balance
        )
    
    logger.info(f"Training set: {len(train_df):,}")
    logger.info(f"  Positive: {(train_df['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(train_df['label'] == 0).sum():,}")
    
    if test_df is not None:
        logger.info(f"Test set: {len(test_df):,}")
        logger.info(f"  Positive: {(test_df['label'] == 1).sum():,}")
        logger.info(f"  Negative: {(test_df['label'] == 0).sum():,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save training and test data (with threshold in filename)
    train_file = output_path / (train_filename or f"train_data_{threshold}.csv")
    test_file = output_path / (test_filename or f"test_data_{threshold}.csv")
    
    logger.info(f"\nSaving training data to: {train_file}")
    train_df[['concept1', 'concept2', 'total_citations', 'label']].to_csv(
        train_file, index=False, encoding='utf-8'
    )
    
    if test_df is not None:
        logger.info(f"Saving test data to: {test_file}")
        test_df[['concept1', 'concept2', 'total_citations', 'label']].to_csv(
            test_file, index=False, encoding='utf-8'
        )
    
    logger.info("=" * 80)
    logger.info("Training and test data creation completed!")
    logger.info("=" * 80)
    
    return train_file, test_file if test_df is not None else None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create training and test data from prediction data')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to prediction_data.csv')
    parser.add_argument('--concept-graph', '-g', required=True,
                       help='Path to concept_graph.csv (existing edges)')
    parser.add_argument('--venue', '-v', default='KDD',
                       help='Venue name (default: KDD)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/transductive)')
    parser.add_argument('--threshold', type=int, default=1,
                       help='IR threshold for positive/negative classification (default: 1)')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                       help='Training data ratio (default: 0.85)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--train-only', action='store_true',
                       help='Save all sampled labeled candidates as training data and do not create a held-out test split')
    parser.add_argument('--train-filename', default=None,
                       help='Custom train CSV filename')
    parser.add_argument('--test-filename', default=None,
                       help='Custom test CSV filename')
    
    args = parser.parse_args()
    
    create_train_test_data(
        args.input,
        args.concept_graph,
        venue=args.venue,
        output_dir=args.output_dir,
        threshold=args.threshold,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        train_only=args.train_only,
        train_filename=args.train_filename,
        test_filename=args.test_filename
    )
