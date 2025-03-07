#!/usr/bin/env python3
"""
Quick Start Guide for the Advanced RAG System.

This script demonstrates how to use the system for thesis research on
"The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems."
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.config import *
from backend.indexing import IngestionPipeline
from backend.querying import QueryPipeline
from backend.chunkers.base import ChunkerFactory
from backend.evaluation.evaluator import RAGEvaluator
from backend.evaluation.visualizer import ResultsVisualizer
from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import ExperimentRunner


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_step(step):
    """Print a step."""
    print(f"\n>> {step}\n")


def main():
    """
    Main function demonstrating how to use the system.
    """
    print_section("ADVANCED RAG SYSTEM - QUICK START GUIDE")
    print("This guide demonstrates how to use the system for thesis research on")
    print("'The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems.'")
    
    # Step 1: List available chunkers
    print_step("Step 1: List available chunkers")
    chunkers = ChunkerFactory.list_chunkers()
    print("Available chunkers:")
    for chunker in chunkers:
        print(f"  - {chunker}")
    
    # Step 2: Ingest documents with different chunking methods
    print_step("Step 2: Ingest documents with different chunking methods")
    print("This step would normally ingest documents using different chunking methods.")
    print("For demonstration purposes, we'll just show the code:")
    
    print("""
    # Example: Ingest documents with RecursiveCharacterChunker
    pipeline = IngestionPipeline(
        pdf_dir=PDF_DIR,
        db_dir=DB_DIR,
        db_collection="rag_collection_recursive",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        data_dir=DATA_DIR,
        embedding_model=EMBEDDING_MODEL,
        chunker_name="recursive_character"
    )
    pipeline.run_pipeline()
    
    # Example: Ingest documents with SemanticClusteringChunker
    pipeline = IngestionPipeline(
        pdf_dir=PDF_DIR,
        db_dir=DB_DIR,
        db_collection="rag_collection_semantic",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        data_dir=DATA_DIR,
        embedding_model=EMBEDDING_MODEL,
        chunker_name="semantic_clustering"
    )
    pipeline.run_pipeline()
    """)
    
    # Step 3: Create an experiment configuration
    print_step("Step 3: Create an experiment configuration")
    config = ExperimentConfig(
        name="demo_experiment",
        description="Demonstration experiment for the quick start guide"
    )
    
    # Add chunkers to the experiment
    config.add_chunker(
        chunker_name="recursive_character",
        chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
        experiment_name="recursive_character_demo"
    )
    
    config.add_chunker(
        chunker_name="semantic_clustering",
        chunker_params={"max_chunk_size": 200},
        experiment_name="semantic_clustering_demo"
    )
    
    # Set dataset and output directory
    config.set_dataset("./data/evaluation/evaluation_dataset_chatgpt_unique.json")
    config.set_output_dir("./data/evaluation/results")
    
    # Save the configuration
    config_path = config.save()
    print(f"Experiment configuration saved to: {config_path}")
    
    # Step 4: Run an experiment
    print_step("Step 4: Run an experiment")
    print("This step would normally run the experiment.")
    print("For demonstration purposes, we'll just show the code:")
    
    print("""
    # Run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    runner.save_results()
    """)
    
    # Step 5: Analyze results
    print_step("Step 5: Analyze results")
    print("This step would normally analyze the experiment results.")
    print("For demonstration purposes, we'll just show the code:")
    
    print("""
    # Analyze results
    from backend.experiments.results import ResultsAnalyzer, ResultsStorage
    
    storage = ResultsStorage(results_dir="./data/evaluation/results")
    analyzer = ResultsAnalyzer(storage=storage)
    
    # Load results
    results = storage.load_all_results()
    
    # Create metrics DataFrame
    metrics_df = analyzer.create_metrics_dataframe(results)
    print(metrics_df)
    
    # Find the best chunker for semantic similarity
    best_chunker = analyzer.find_best_chunker(results, metric="semantic_similarity")
    print(f"Best chunker for semantic similarity: {best_chunker['chunker_name']}")
    
    # Generate a report
    report_dir = analyzer.generate_report(results, report_name="demo_analysis")
    print(f"Analysis report generated: {report_dir}")
    """)
    
    # Step 6: Generate visualizations
    print_step("Step 6: Generate visualizations")
    print("This step would normally generate visualizations of the results.")
    print("For demonstration purposes, we'll just show the code:")
    
    print("""
    # Generate visualizations
    visualizer = ResultsVisualizer(results_dir="./data/evaluation/results")
    
    # Create metrics comparison plot
    metrics_plot = visualizer.plot_metrics_comparison(
        results,
        title="Metrics Comparison",
        save_path="./data/visualizations/metrics_comparison.png"
    )
    
    # Create efficiency comparison plot
    efficiency_plot = visualizer.plot_efficiency_comparison(
        results,
        title="Efficiency Comparison",
        save_path="./data/visualizations/efficiency_comparison.png"
    )
    
    # Create chunk statistics plot
    chunk_stats_plot = visualizer.plot_chunk_statistics(
        results,
        title="Chunk Statistics",
        save_path="./data/visualizations/chunk_statistics.png"
    )
    """)
    
    # Step 7: Generate a thesis report
    print_step("Step 7: Generate a thesis report")
    print("This step would normally generate a comprehensive report for the thesis.")
    print("For demonstration purposes, we'll just show the code:")
    
    print("""
    # Generate a thesis report
    import subprocess
    
    subprocess.run([
        "python", "scripts/generate_report.py",
        "--results-dir", "./data/evaluation/results",
        "--output-dir", "./data/thesis",
        "--report-name", "thesis_report"
    ])
    """)
    
    # Conclusion
    print_section("CONCLUSION")
    print("This quick start guide demonstrated how to use the Advanced RAG System")
    print("for thesis research on 'The Impact of Chunking Methods on Information")
    print("Retrieval Quality in RAG Systems.'")
    print("\nFor more detailed information, please refer to the README.md file.")
    print("\nHappy researching!")


if __name__ == "__main__":
    main()