#!/usr/bin/env python3
"""
Script to run experiments comparing different chunking methods.

This script provides a command-line interface for running experiments
to compare different chunking methods in the RAG system.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import ExperimentRunner, run_experiment_from_config, run_default_experiment
from backend.experiments.results import ResultsAnalyzer, ResultsStorage
from backend.chunkers.base import ChunkerFactory


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments comparing different chunking methods."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run default experiment
    default_parser = subparsers.add_parser(
        "default",
        help="Run the default experiment with all chunkers"
    )
    
    # Run experiment from config
    config_parser = subparsers.add_parser(
        "from-config",
        help="Run an experiment from a configuration file"
    )
    config_parser.add_argument(
        "config_path",
        help="Path to the experiment configuration file"
    )
    
    # Create a new experiment configuration
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new experiment configuration"
    )
    create_parser.add_argument(
        "--name",
        required=True,
        help="Name of the experiment"
    )
    create_parser.add_argument(
        "--description",
        help="Description of the experiment"
    )
    create_parser.add_argument(
        "--dataset",
        default="./data/evaluation/evaluation_dataset_chatgpt_unique.json",
        help="Path to the evaluation dataset"
    )
    create_parser.add_argument(
        "--output-dir",
        default="./data/evaluation/results",
        help="Directory to store experiment results"
    )
    create_parser.add_argument(
        "--chunkers",
        nargs="+",
        choices=ChunkerFactory.list_chunkers(),
        help="List of chunkers to include in the experiment"
    )
    
    # List available chunkers
    subparsers.add_parser(
        "list-chunkers",
        help="List all available chunkers"
    )
    
    # Analyze results
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze experiment results"
    )
    analyze_parser.add_argument(
        "--results-dir",
        default="./data/evaluation/results",
        help="Directory containing experiment results"
    )
    analyze_parser.add_argument(
        "--report-name",
        default="chunking_methods_analysis",
        help="Name of the analysis report"
    )
    
    return parser.parse_args()


def run_default_experiment_command():
    """
    Run the default experiment with all chunkers.
    """
    print("Running default experiment with all chunkers...")
    results = run_default_experiment()
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    report_dir = analyzer.generate_report(
        results,
        report_name="default_experiment"
    )
    
    print(f"Default experiment completed. Report generated: {report_dir}")


def run_experiment_from_config_command(config_path: str):
    """
    Run an experiment from a configuration file.
    
    Args:
        config_path (str): Path to the experiment configuration file.
    """
    print(f"Running experiment from configuration: {config_path}")
    results = run_experiment_from_config(config_path)
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    report_dir = analyzer.generate_report(
        results,
        report_name=os.path.basename(config_path).split(".")[0]
    )
    
    print(f"Experiment completed. Report generated: {report_dir}")


def create_experiment_config_command(args):
    """
    Create a new experiment configuration.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print(f"Creating experiment configuration: {args.name}")
    
    # Create the experiment configuration
    config = ExperimentConfig(
        name=args.name,
        description=args.description or f"Experiment: {args.name}"
    )
    
    # Set dataset and output directory
    config.set_dataset(args.dataset)
    config.set_output_dir(args.output_dir)
    
    # Add chunkers
    if args.chunkers:
        for chunker_name in args.chunkers:
            config.add_chunker(
                chunker_name=chunker_name,
                experiment_name=f"{chunker_name}_experiment"
            )
    else:
        # Add all available chunkers
        for chunker_name in ChunkerFactory.list_chunkers():
            config.add_chunker(
                chunker_name=chunker_name,
                experiment_name=f"{chunker_name}_experiment"
            )
    
    # Save the configuration
    config_path = config.save()
    print(f"Experiment configuration saved to: {config_path}")
    
    return config_path


def list_chunkers_command():
    """
    List all available chunkers.
    """
    chunkers = ChunkerFactory.list_chunkers()
    print("Available chunkers:")
    for chunker in chunkers:
        print(f"  - {chunker}")


def analyze_results_command(args):
    """
    Analyze experiment results.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print(f"Analyzing experiment results in: {args.results_dir}")
    
    # Load results
    storage = ResultsStorage(results_dir=args.results_dir)
    results = storage.load_all_results()
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} experiment results.")
    
    # Analyze results
    analyzer = ResultsAnalyzer(storage=storage)
    report_dir = analyzer.generate_report(
        results,
        report_name=args.report_name
    )
    
    print(f"Analysis completed. Report generated: {report_dir}")


def main():
    """
    Main function.
    """
    args = parse_args()
    
    if args.command == "default":
        run_default_experiment_command()
    elif args.command == "from-config":
        run_experiment_from_config_command(args.config_path)
    elif args.command == "create":
        config_path = create_experiment_config_command(args)
        
        # Ask if the user wants to run the experiment
        response = input("Do you want to run this experiment now? (y/n): ")
        if response.lower() in ["y", "yes"]:
            run_experiment_from_config_command(config_path)
    elif args.command == "list-chunkers":
        list_chunkers_command()
    elif args.command == "analyze":
        analyze_results_command(args)
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main()