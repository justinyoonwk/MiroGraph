"""
Create training and test data from prediction data
- Classify edges as positive (IR >= threshold) or negative (IR < threshold)
- Sample negative edges to match positive count
- Split into 85% training and 15% test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_train_test_data(prediction_data_path: str,
                           venue: str = "SIGIR",
                           output_dir: str = None,
                           threshold: int = 1,
                           train_ratio: float = 0.85,
                           random_seed: int = 42):
    logger.info("=" * 80)
    logger.info("Creating Training and Test Data")
    logger.info("=" * 80)
    
    venue_dir = str(venue).lower()

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{venue_dir}/test"
    
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
    
    if negative_count == 0:
        logger.warning("No negative edges found! All edges are positive.")
        # Use all data
        sampled_df = df.copy()
    else:
        # Sample negative edges to match positive count
        negative_df = df[df['label'] == 0].copy()
        positive_df = df[df['label'] == 1].copy()
        
        if len(negative_df) > positive_count:
            logger.info(f"Sampling {positive_count:,} negative edges from {negative_count:,} total...")
            np.random.seed(random_seed)
            sampled_negative_df = negative_df.sample(n=positive_count, random_state=random_seed)
        else:
            logger.warning(f"Negative edges ({negative_count}) are fewer than positive ({positive_count}). Using all negative edges.")
            sampled_negative_df = negative_df
        
        # Combine positive and sampled negative
        sampled_df = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
        sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
    logger.info(f"\nBalanced dataset:")
    logger.info(f"  Total: {len(sampled_df):,}")
    logger.info(f"  Positive: {(sampled_df['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(sampled_df['label'] == 0).sum():,}")
    
    # Split into training and test sets
    logger.info(f"\nSplitting into training ({train_ratio*100:.0f}%) and test ({(1-train_ratio)*100:.0f}%)...")
    train_df, test_df = train_test_split(
        sampled_df,
        test_size=1 - train_ratio,
        random_state=random_seed,
        stratify=sampled_df['label']  # Maintain class balance
    )
    
    logger.info(f"Training set: {len(train_df):,} ({len(train_df)/len(sampled_df)*100:.1f}%)")
    logger.info(f"  Positive: {(train_df['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(train_df['label'] == 0).sum():,}")
    
    logger.info(f"Test set: {len(test_df):,} ({len(test_df)/len(sampled_df)*100:.1f}%)")
    logger.info(f"  Positive: {(test_df['label'] == 1).sum():,}")
    logger.info(f"  Negative: {(test_df['label'] == 0).sum():,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save training and test data (with threshold in filename)
    train_file = output_path / f"train_data_{threshold}.csv"
    test_file = output_path / f"test_data_{threshold}.csv"
    
    logger.info(f"\nSaving training data to: {train_file}")
    train_df[['concept1', 'concept2', 'total_citations', 'label']].to_csv(
        train_file, index=False, encoding='utf-8'
    )
    
    logger.info(f"Saving test data to: {test_file}")
    test_df[['concept1', 'concept2', 'total_citations', 'label']].to_csv(
        test_file, index=False, encoding='utf-8'
    )
    
    logger.info("=" * 80)
    logger.info("Training and test data creation completed!")
    logger.info("=" * 80)
    
    return train_file, test_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create training and test data from prediction data')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to prediction_data.csv')
    parser.add_argument('--venue', '-v', default='SIGIR',
                       help='Venue name (default: SIGIR)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: dataset/{venue}/test)')
    parser.add_argument('--threshold', type=int, default=1,
                       help='IR threshold for positive/negative classification (default: 1)')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                       help='Training data ratio (default: 0.85)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    create_train_test_data(
        args.input,
        venue=args.venue,
        output_dir=args.output_dir,
        threshold=args.threshold,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )

