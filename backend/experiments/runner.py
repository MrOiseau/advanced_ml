"""
Experiment Runner module for the RAG system.

This module contains the ExperimentRunner class for running experiments
to compare different chunking methods in the RAG system.
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.config import *
from backend.chunkers.base import ChunkerFactory
from backend.querying import QueryPipeline
from backend.evaluation.evaluator import RAGEvaluator
from backend.evaluation.visualizer import ResultsVisualizer
from backend.experiments.config import ExperimentConfig
from backend.utils import NumpyEncoder


class ExperimentRunner:
    """
    Runner for RAG system experiments.
    
    This class provides methods for running experiments to compare
    different chunking methods in the RAG system.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the ExperimentRunner.
        
        Args:
            config (ExperimentConfig): Experiment configuration.
            results_dir (Optional[str]): Directory to store experiment results.
        """
        self.config = config
        self.results_dir = results_dir or config.output_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize the query pipeline
        self.query_pipeline = self._initialize_query_pipeline()
        
        # Initialize the evaluator
        self.evaluator = RAGEvaluator(
            dataset_path=config.dataset_path,
            query_pipeline=self.query_pipeline,
            output_dir=self.results_dir
        )
        
        # Initialize the visualizer
        self.visualizer = ResultsVisualizer(results_dir=self.results_dir)
        
        # Store experiment results
        self.results = []
    
    def _initialize_query_pipeline(self) -> QueryPipeline:
        """
        Initialize the query pipeline for the experiment.
        
        Returns:
            QueryPipeline: The initialized query pipeline.
        """
        # Validate environment variables
        validate_environment()
        
        # Initialize the query pipeline using constants from config
        query_pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT
        )
        
        return query_pipeline
    
    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run the experiment for all configured chunkers.
        
        Returns:
            List[Dict[str, Any]]: List of experiment results.
        """
        print(f"Starting experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Number of chunkers to evaluate: {len(self.config.chunker_configs)}")
        
        start_time = time.time()
        
        # Run each chunker configuration
        for i, chunker_config in enumerate(self.config.chunker_configs):
            chunker_name = chunker_config["chunker_name"]
            chunker_params = chunker_config["chunker_params"]
            experiment_name = chunker_config["experiment_name"]
            
            print(f"\n[{i+1}/{len(self.config.chunker_configs)}] Evaluating chunker: {chunker_name}")
            print(f"Parameters: {chunker_params}")
            print(f"Experiment name: {experiment_name}")
            
            try:
                # Evaluate the chunker
                result = self.evaluator.evaluate_chunker(
                    chunker_name=chunker_name,
                    chunker_params=chunker_params,
                    experiment_name=experiment_name
                )
                
                self.results.append(result)
                
                print(f"Evaluation completed. Results saved to: {result['results_path']}")
                print("Average metrics:")
                for metric, value in result["average_metrics"].items():
                    print(f"  {metric}: {value:.4f}")
            
            except Exception as e:
                print(f"Error evaluating chunker {chunker_name}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nExperiment completed in {total_time:.2f} seconds")
        
        # Generate comparison report
        if len(self.results) > 1:
            report_dir = self.visualizer.create_comparison_report(
                self.results,
                report_name=self.config.name
            )
            print(f"Comparison report generated: {report_dir}")
        
        return self.results
    
    def save_results(self) -> str:
        """
        Save the experiment results to a JSON file.
        
        Returns:
            str: Path to the saved results file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare results data
        results_data = {
            "experiment_name": self.config.name,
            "experiment_description": self.config.description,
            "timestamp": timestamp,
            "chunker_configs": self.config.chunker_configs,
            "results": self.results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        
        return filepath


def run_experiment_from_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Run an experiment from a configuration file.
    
    Args:
        config_path (str): Path to the experiment configuration file.
        
    Returns:
        List[Dict[str, Any]]: List of experiment results.
    """
    # Load the experiment configuration
    config = ExperimentConfig.load(config_path)
    
    # Create and run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    
    # Save the results
    runner.save_results()
    
    return results


def run_default_experiment() -> List[Dict[str, Any]]:
    """
    Run the default experiment with all chunkers.
    
    Returns:
        List[Dict[str, Any]]: List of experiment results.
    """
    # Create the default experiment configuration
    config = ExperimentConfig.create_default_experiment()
    
    # Save the configuration
    config_path = config.save()
    print(f"Default experiment configuration saved to: {config_path}")
    
    # Run the experiment
    return run_experiment_from_config(config_path)