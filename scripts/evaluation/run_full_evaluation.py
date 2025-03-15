#!/usr/bin/env python3
"""
Script to run the full evaluation pipeline from Excel to results.

This script:
1. Converts the Excel file to JSON with ground truth chunks
2. Runs the evaluation on all chunking methods
3. Generates a comparison report
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the conversion and evaluation functions
from scripts.utils.excel_to_json_converter import convert_excel_to_json
from scripts.evaluation.run_evaluation import evaluate_all_chunkers
from backend.evaluation.visualizer import ResultsVisualizer


async def run_full_evaluation(
    excel_path: str,
    json_path: str,
    output_dir: str,
    include_ground_truth: bool = True,
    question_column: str = "User Inputs – Question",
    answer_column: str = "LLM generation - Answer",
    max_search_results: int = 40
):
    """
    Runs the entire evaluation process from Excel to results.
    
    Args:
        excel_path (str): Path to the Excel file.
        json_path (str): Path to save the JSON file.
        output_dir (str): Directory to save evaluation results.
        include_ground_truth (bool): Whether to include ground truth chunks.
        question_column (str): Column name for questions.
        answer_column (str): Column name for reference answers.
        max_search_results (int): Maximum number of search results to consider.
    """
    print("Starting full evaluation pipeline...")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Convert Excel to JSON with ground truth chunks
    print("\nStep 1: Converting Excel to JSON with ground truth chunks...")
    dataset_path = convert_excel_to_json(
        excel_path=excel_path,
        json_path=json_path,
        include_ground_truth=include_ground_truth,
        question_column=question_column,
        answer_column=answer_column,
        max_search_results=max_search_results
    )
    
    # Step 2: Run the evaluator on all chunking methods
    print("\nStep 2: Running evaluator on all chunking methods...")
    results = await evaluate_all_chunkers(dataset_path, output_dir)
    
    # Step 3: Generate comparison report
    print("\nStep 3: Generating comparison report...")
    if len(results) > 1:
        visualizer = ResultsVisualizer(results_dir=output_dir)
        report_dir = visualizer.create_comparison_report(
            results,
            report_name="chunking_methods_comparison"
        )
        print(f"Comparison report generated: {report_dir}")
    
    print("\nEvaluation process completed.")
    print(f"Results are available in: {output_dir}")
    print(f"Comparison report is available in: {os.path.join(output_dir, 'reports')}")


async def main():
    """
    Main function to parse arguments and run the full evaluation.
    """
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline from Excel to results")
    parser.add_argument("--excel_path", default="data/wfp_evaluation_generisanje_odgovora.xlsx", 
                        help="Path to the Excel file")
    parser.add_argument("--json_path", default="data/evaluation/wfp_evaluation_dataset_with_ground_truth.json", 
                        help="Path to save the JSON file")
    parser.add_argument("--output_dir", default="data/evaluation/results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--include_ground_truth", action="store_true", default=True, 
                        help="Include ground truth chunks in the output")
    parser.add_argument("--question_column", default="User Inputs – Question", 
                        help="Column name for questions")
    parser.add_argument("--answer_column", default="LLM generation - Answer", 
                        help="Column name for reference answers")
    parser.add_argument("--max_search_results", type=int, default=40, 
                        help="Maximum number of search results to consider")
    
    args = parser.parse_args()
    
    await run_full_evaluation(
        excel_path=args.excel_path,
        json_path=args.json_path,
        output_dir=args.output_dir,
        include_ground_truth=args.include_ground_truth,
        question_column=args.question_column,
        answer_column=args.answer_column,
        max_search_results=args.max_search_results
    )


if __name__ == "__main__":
    asyncio.run(main())