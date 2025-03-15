#!/usr/bin/env python3
"""
Consolidated Excel to JSON converter for evaluation datasets.

This script:
1. Converts Excel files with evaluation data to JSON format
2. Extracts ground truth chunks from the Excel file
3. Supports different output formats and options
"""

import os
import json
import pandas as pd
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime


def convert_excel_to_json(
    excel_path: str,
    json_path: str,
    include_ground_truth: bool = True,
    question_column: str = "User Inputs – Question",
    answer_column: str = "LLM generation - Answer",
    max_search_results: int = 40
) -> str:
    """
    Convert the Excel file to the JSON format expected by the RAGEvaluator.
    
    Args:
        excel_path (str): Path to the Excel file.
        json_path (str): Path to save the JSON file.
        include_ground_truth (bool): Whether to include ground truth chunks.
        question_column (str): Column name for questions.
        answer_column (str): Column name for reference answers.
        max_search_results (int): Maximum number of search results to consider.
        
    Returns:
        str: Path to the saved JSON file.
    """
    print(f"Converting Excel file: {excel_path}")
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Create the dataset in the expected format
    dataset = []
    
    for _, row in df.iterrows():
        # Extract question and reference answer
        question = row[question_column]
        reference_answer = row[answer_column]
        
        # Skip if question or answer is missing
        if pd.isna(question) or pd.isna(reference_answer):
            continue
        
        # Create a dataset item
        item = {
            'question': question,
            'answer': reference_answer,
        }
        
        # Extract ground truth chunks if requested
        if include_ground_truth:
            item['ground_truth_chunks'] = []
            
            # Extract ground truth chunks
            for i in range(1, max_search_results + 1):
                paragraph_col = f'Search Result {i} – Paragraph'
                document_col = f'Search Result {i} – Document'
                
                if paragraph_col in row.index and document_col in row.index:
                    paragraph = row[paragraph_col]
                    document = row[document_col]
                    
                    # Skip if paragraph or document is missing
                    if pd.isna(paragraph) or pd.isna(document):
                        continue
                    
                    # Add to ground truth chunks
                    item['ground_truth_chunks'].append({
                        'content': paragraph,
                        'source': document,
                        'id': f"gt_{i}"
                    })
        
        dataset.append(item)
    
    # Save to JSON file
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Converted {len(dataset)} items to JSON: {json_path}")
    return json_path


def main():
    """
    Main function to parse arguments and convert Excel to JSON.
    """
    parser = argparse.ArgumentParser(description="Convert Excel evaluation data to JSON format")
    parser.add_argument("--excel_path", required=True, help="Path to the Excel file")
    parser.add_argument("--json_path", required=True, help="Path to save the JSON file")
    parser.add_argument("--include_ground_truth", action="store_true", default=True, 
                        help="Include ground truth chunks in the output")
    parser.add_argument("--question_column", default="User Inputs – Question", 
                        help="Column name for questions")
    parser.add_argument("--answer_column", default="LLM generation - Answer", 
                        help="Column name for reference answers")
    parser.add_argument("--max_search_results", type=int, default=40, 
                        help="Maximum number of search results to consider")
    
    args = parser.parse_args()
    
    # Convert Excel to JSON
    dataset_path = convert_excel_to_json(
        excel_path=args.excel_path,
        json_path=args.json_path,
        include_ground_truth=args.include_ground_truth,
        question_column=args.question_column,
        answer_column=args.answer_column,
        max_search_results=args.max_search_results
    )
    
    print(f"\nConversion completed. Dataset saved to: {dataset_path}")
    print("You can now run the evaluation using:")
    print(f"python scripts/evaluation/run_evaluation.py --dataset {dataset_path}")


if __name__ == "__main__":
    main()