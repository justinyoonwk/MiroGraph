import pandas as pd
import ast
from pathlib import Path


def parse_concept_list(concept_str):
    if pd.isna(concept_str) or concept_str == '' or concept_str == '[]':
        return []
    
    try:
        # Try to parse as Python list
        if isinstance(concept_str, str):
            # Remove quotes if present
            concept_str = concept_str.strip()
            if concept_str.startswith('[') and concept_str.endswith(']'):
                return ast.literal_eval(concept_str)
            else:
                # If it's a single string, return as list
                return [concept_str]
        return concept_str if isinstance(concept_str, list) else []
    except:
        return []


def combine_all_concepts(row):
    all_concepts = []
    
    # Process only level 2-5 (excluding level 0 and 1)
    for level in ['level_2_concepts', 'level_3_concepts', 'level_4_concepts', 'level_5_concepts']:
        concepts = parse_concept_list(row.get(level, []))
        if concepts:
            all_concepts.extend(concepts)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for concept in all_concepts:
        concept_clean = str(concept).strip().lower()
        if concept_clean and concept_clean not in seen:
            seen.add(concept_clean)
            unique_concepts.append(str(concept).strip())
    
    return unique_concepts


def preprocess_concepts_file(input_path: str, output_path: str = None):
    print(f"Loading concepts file: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} papers")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create paper_id from title (or use index)
    if 'title' in df.columns:
        df['paper_id'] = df['title'].fillna('').apply(
            lambda x: f"paper_{hash(str(x)) % 100000}" if x else f"paper_{len(df)}"
        )
    else:
        df['paper_id'] = [f"paper_{i}" for i in range(len(df))]
    
    # Combine concepts from level 2-5 only
    print("Combining concepts from level 2-5 (excluding level 0 and 1)...")
    df['concepts'] = df.apply(combine_all_concepts, axis=1)
    
    # Convert to comma-separated string for CSV compatibility
    df['concepts'] = df['concepts'].apply(lambda x: ', '.join(x) if x else '')
    
    # Select columns for output
    output_df = df[['paper_id', 'concepts']].copy()
    
    # Add year and title if available for reference
    if 'year' in df.columns:
        output_df['year'] = df['year']
    if 'title' in df.columns:
        output_df['title'] = df['title']
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_processed.csv"
    
    # Save
    output_path = Path(output_path)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nPreprocessing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total papers: {len(output_df)}")
    print(f"Papers with concepts: {len(output_df[output_df['concepts'] != ''])}")
    
    # Show statistics
    concept_counts = output_df['concepts'].apply(lambda x: len(x.split(', ')) if x else 0)
    print(f"Average concepts per paper: {concept_counts.mean():.2f}")
    print(f"Min concepts: {concept_counts.min()}, Max concepts: {concept_counts.max()}")
    
    # Show sample
    print(f"\nFirst 3 papers:")
    for idx, row in output_df.head(3).iterrows():
        concepts_preview = row['concepts'][:100] + '...' if len(row['concepts']) > 100 else row['concepts']
        print(f"  {row['paper_id']}: {concepts_preview}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess concepts CSV for pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input concepts CSV file')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file path')
    
    args = parser.parse_args()
    
    preprocess_concepts_file(args.input, args.output)

