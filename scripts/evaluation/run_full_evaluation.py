#!/usr/bin/env python3
"""
Script to run the full evaluation pipeline.

This script:
1. Either converts the Excel file to JSON with ground truth chunks
   or uses the QASPER dataset for evaluation
2. Runs the evaluation on all chunking methods
3. Generates a comparison report
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the conversion and evaluation functions
from scripts.utils.excel_to_json_converter import convert_excel_to_json
from scripts.evaluation.run_evaluation import evaluate_all_chunkers, evaluate_with_qasper
from backend.evaluation.visualizer import ResultsVisualizer

# Check if datasets package is available for QASPER
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' package not found. QASPER dataset evaluation will not be available.")


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


async def run_qasper_evaluation(
    output_dir: str,
    max_samples: int = 50,
    specific_chunker: Optional[str] = None
):
    """
    Runs evaluation using the QASPER dataset.
    
    Args:
        output_dir (str): Directory to save evaluation results.
        max_samples (int): Maximum number of samples to use from QASPER.
        specific_chunker (Optional[str]): Specific chunker to evaluate (None for all).
    """
    if not DATASETS_AVAILABLE:
        print("Error: The 'datasets' package is required to use QASPER dataset")
        print("Install it with: pip install datasets")
        return
    
    print("Starting QASPER dataset evaluation pipeline...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation with QASPER dataset
    print("\nRunning evaluation with QASPER dataset...")
    results = await evaluate_with_qasper(
        max_samples=max_samples,
        output_dir=output_dir,
        chunker_name=specific_chunker
    )
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    if len(results) > 1:
        visualizer = ResultsVisualizer(results_dir=output_dir)
        report_dir = visualizer.create_comparison_report(
            results,
            report_name="qasper_chunking_methods_comparison"
        )
        print(f"Comparison report generated: {report_dir}")
    
    print("\nQASPER evaluation process completed.")
    print(f"Results are available in: {output_dir}")
    print(f"Comparison report is available in: {os.path.join(output_dir, 'reports')}")


async def main():
    """
    Main function to parse arguments and run the full evaluation.
    """
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    
    # Dataset source options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--excel", action="store_true", help="Use Excel file as data source")
    group.add_argument("--qasper", action="store_true", help="Use QASPER dataset as data source")
    
    # Excel-specific options
    parser.add_argument("--excel_path", default="data/wfp_evaluation_generisanje_odgovora.xlsx",
                        help="Path to the Excel file (when using --excel)")
    parser.add_argument("--json_path", default="data/evaluation/wfp_evaluation_dataset_with_ground_truth.json",
                        help="Path to save the JSON file (when using --excel)")
    parser.add_argument("--include_ground_truth", action="store_true", default=True,
                        help="Include ground truth chunks in the output (when using --excel)")
    parser.add_argument("--question_column", default="User Inputs – Question",
                        help="Column name for questions (when using --excel)")
    parser.add_argument("--answer_column", default="LLM generation - Answer",
                        help="Column name for reference answers (when using --excel)")
    parser.add_argument("--max_search_results", type=int, default=40,
                        help="Maximum number of search results to consider (when using --excel)")
    
    # QASPER-specific options
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Maximum number of samples to use from QASPER dataset (when using --qasper)")
    
    # Common options
    parser.add_argument("--output_dir", default="data/evaluation/results",
                        help="Directory to save evaluation results")
    parser.add_argument("--chunker", help="Specific chunker to evaluate (evaluates all if not specified)")
    
    args = parser.parse_args()
    
    if args.excel:
        # Use Excel file as data source
        await run_full_evaluation(
            excel_path=args.excel_path,
            json_path=args.json_path,
            output_dir=args.output_dir,
            include_ground_truth=args.include_ground_truth,
            question_column=args.question_column,
            answer_column=args.answer_column,
            max_search_results=args.max_search_results
        )
    elif args.qasper:
        # Use QASPER dataset as data source
        qasper_output_dir = os.path.join(args.output_dir, "qasper")
        await run_qasper_evaluation(
            output_dir=qasper_output_dir,
            max_samples=args.max_samples,
            specific_chunker=args.chunker
        )


if __name__ == "__main__":
    asyncio.run(main())