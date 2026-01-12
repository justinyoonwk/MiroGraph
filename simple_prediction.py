#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple LLM-based prediction: Predict impact using only concept pair and LLM
Supports both single pair and batch processing with parallel execution
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()

def load_simple_prompt(ir: int) -> str:
    """Load simple prompt for direct prediction"""
    prompt = f"""You are a scientific impact prediction expert.

[TASK]
Given a pair of scientific concepts (X, Y), predict whether they will co-occur in papers within the next 3 years and whether their cumulative citation count will be at least {ir} (IR threshold).

[IR DEFINITION]
IR (Impact Range) = {ir} is the threshold for cumulative citation count:
- If concepts (X, Y) do NOT co-occur within 3 years: outcome is "No"
- If concepts (X, Y) co-occur but cumulative citations < {ir}: outcome is "No" (low-impact)
- If concepts (X, Y) co-occur AND cumulative citations ≥ {ir}: outcome is "Yes" (high-impact)

[PREDICTION RULES]
You must predict "Yes" or "No" based on:
- Semantic relationship: Are these concepts related? Could they be combined in research?
- Research trends: Are these areas active? Is there growing interest?
- Potential impact: If combined, would this research likely receive significant citations?

[OUTPUT]
Return a JSON object with:
{{
  "reasoning": "<your reasoning about why this pair will or won't have high impact>",
  "prediction": "Yes" or "No"
}}

The "prediction" field MUST be exactly one of: "Yes", "No".
Do NOT include additional text outside the JSON object.
"""
    return prompt

def predict_pair(client: OpenAI, concept1: str, concept2: str, ir: int, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Predict impact for a concept pair using simple LLM call"""
    prompt = load_simple_prompt(ir)
    
    user_input = f"""Concept Pair:
- Concept X: {concept1}
- Concept Y: {concept2}

Please predict whether this concept pair will have high impact (co-occur within 3 years with cumulative citations ≥ {ir})."""
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Convert prediction to int (1 for "Yes", 0 for "No")
        prediction_str = result.get("prediction", "").strip().lower()
        if prediction_str == "yes":
            prediction_int = 1
        elif prediction_str == "no":
            prediction_int = 0
        else:
            # Try to infer from reasoning if prediction field is missing
            reasoning = result.get("reasoning", "").lower()
            if "yes" in reasoning or "high impact" in reasoning or "will co-occur" in reasoning:
                prediction_int = 1
            else:
                prediction_int = 0
        
        return {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "reasoning": result.get("reasoning", ""),
            "prediction": result.get("prediction", ""),
            "prediction_int": prediction_int
        }
    except Exception as e:
        print(f"Error predicting pair ({concept1}, {concept2}): {e}")
        return {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "reasoning": f"Error: {str(e)}",
            "prediction": "No",
            "prediction_int": 0,
            "error": str(e)
        }

def process_single_pair(args_tuple: Tuple) -> Tuple[int, Dict[str, Any]]:
    """Process a single pair (for parallel execution)"""
    idx, row, ir, model, api_key = args_tuple
    
    concept1 = row['concept1']
    concept2 = row['concept2']
    ground_truth = int(row['label'])
    total_citations = int(row.get('total_citations', 0))
    
    client = OpenAI(api_key=api_key)
    
    try:
        result = predict_pair(client, concept1, concept2, ir, model)
        
        # Add ground truth info
        result['ground_truth_label'] = ground_truth
        result['total_citations'] = total_citations
        result['pair'] = {
            'concept1': concept1,
            'concept2': concept2
        }
        
        return (idx, result)
    except Exception as e:
        print(f"[ERROR] Failed to process pair {idx} ({concept1} <-> {concept2}): {str(e)}")
        return (idx, {
            "concept1": concept1,
            "concept2": concept2,
            "ir": ir,
            "ground_truth_label": ground_truth,
            "total_citations": total_citations,
            "pair": {
                'concept1': concept1,
                'concept2': concept2
            },
            "reasoning": f"Error: {str(e)}",
            "prediction": "No",
            "prediction_int": 0,
            "error": str(e)
        })

def run_batch_prediction(
    test_file: str,
    ir: int,
    model: str = "gpt-4o-mini",
    api_key_name: str = "1",
    output_file: str = None,
    batch_size: int = 100
):
    """Run simple prediction on all test pairs with parallel processing"""
    
    # Load test data
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} test pairs from {test_file}")
    print()
    
    # Get API key
    api_key_name_env = f"OPENAI_API_KEY_{api_key_name}"
    api_key = os.getenv(api_key_name_env)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"API key not found. Set {api_key_name_env} or OPENAI_API_KEY in .env file")
    
    # Set max_workers
    max_workers = batch_size
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"Processing {len(df)} pairs with IR={ir}")
    print(f"Batch size: {batch_size} (동시 처리 수: {max_workers})")
    print(f"Total batches: {num_batches}")
    print()
    
    # Process in batches
    all_results = []
    y_true = []
    y_pred = []
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing pairs {batch_start + 1}-{batch_end} (동시 처리: {max_workers}개)...")
        
        # Prepare arguments for parallel processing
        args_list = [
            (batch_start + i, row, ir, model, api_key)
            for i, (_, row) in enumerate(batch_df.iterrows())
        ]
        
        # Process in parallel
        batch_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_pair, args): args[0] for args in args_list}
            
            with tqdm(total=len(args_list), desc=f"  Batch {batch_idx + 1}") as pbar:
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    batch_results[idx] = result
                    pbar.update(1)
        
        # Sort by index and add to all_results
        for idx in sorted(batch_results.keys()):
            result = batch_results[idx]
            all_results.append(result)
            y_true.append(result['ground_truth_label'])
            y_pred.append(result['prediction_int'])
        
        print(f"  [Saved] 배치 결과 저장됨: {len(batch_results)}개 레코드 추가")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print()
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Total pairs: {len(all_results)}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {cm[1][1]}, TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}")
    print()
    
    # Save results
    if output_file is None:
        output_file = f"results/simple_prediction_ir{ir}.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Save metadata
        metadata = {
            'type': 'metadata',
            'ir': ir,
            'model': model,
            'total_pairs': len(all_results),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tp': int(cm[1][1]),
                'tn': int(cm[0][0]),
                'fp': int(cm[0][1]),
                'fn': int(cm[1][0])
            }
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        # Save individual results
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"[Saved] Results saved to: {output_file}")
    
    return all_results, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Simple LLM-based impact prediction")
    parser.add_argument("--concept1", type=str, default=None, help="First concept (for single prediction)")
    parser.add_argument("--concept2", type=str, default=None, help="Second concept (for single prediction)")
    parser.add_argument("--test_file", type=str, default=None, help="Test data CSV file (for batch prediction)")
    parser.add_argument("--ir", type=int, required=True, help="IR threshold")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api_key_name", type=str, default="1", help="API key name (OPENAI_API_KEY_1, OPENAI_API_KEY_2, etc.)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (direct input)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for parallel processing")
    
    args = parser.parse_args()
    
    # Get API key
    if args.api_key:
        api_key = args.api_key
    else:
        api_key_name = f"OPENAI_API_KEY_{args.api_key_name}"
        api_key = os.getenv(api_key_name)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"API key not found. Set {api_key_name} or OPENAI_API_KEY in .env file")
    
    # Single prediction mode
    if args.concept1 and args.concept2:
        client = OpenAI(api_key=api_key)
        
        print(f"Predicting impact for: {args.concept1} <-> {args.concept2}")
        print(f"IR threshold: {args.ir}")
        print()
        
        result = predict_pair(client, args.concept1, args.concept2, args.ir, args.model)
        
        # Print result
        print("=" * 80)
        print("Prediction Result")
        print("=" * 80)
        print(f"Concept Pair: {result['concept1']} <-> {result['concept2']}")
        print(f"IR Threshold: {result['ir']}")
        print(f"Prediction: {result['prediction']} ({result['prediction_int']})")
        print()
        print("Reasoning:")
        print(result['reasoning'])
        print()
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {args.output}")
        
        return result
    
    # Batch prediction mode
    elif args.test_file:
        run_batch_prediction(
            test_file=args.test_file,
            ir=args.ir,
            model=args.model,
            api_key_name=args.api_key_name,
            output_file=args.output,
            batch_size=args.batch_size
        )
    
    else:
        parser.error("Either --concept1 and --concept2 (for single prediction) or --test_file (for batch prediction) must be provided")

if __name__ == "__main__":
    main()
