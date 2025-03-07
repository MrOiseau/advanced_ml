"""
Results Storage and Analysis module for the RAG system.

This module contains classes for storing and analyzing the results of
experiments comparing different chunking methods in the RAG system.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.
    
    This encoder converts NumPy types to their Python equivalents
    so they can be properly serialized to JSON.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from backend.evaluation.visualizer import ResultsVisualizer


class ResultsStorage:
    """
    Storage for RAG experiment results.
    
    This class provides methods for storing, loading, and managing
    the results of experiments comparing different chunking methods.
    """
    
    def __init__(self, results_dir: str = "./data/evaluation/results"):
        """
        Initialize the ResultsStorage.
        
        Args:
            results_dir (str): Directory to store experiment results.
        """
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_result(self, result: Dict[str, Any], experiment_name: str) -> str:
        """
        Save an experiment result to a JSON file.
        
        Args:
            result (Dict[str, Any]): The experiment result to save.
            experiment_name (str): Name of the experiment.
            
        Returns:
            str: Path to the saved result file.
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def load_result(self, filepath: str) -> Dict[str, Any]:
        """
        Load an experiment result from a JSON file.
        
        Args:
            filepath (str): Path to the result file.
            
        Returns:
            Dict[str, Any]: The loaded result.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all experiment results from the results directory.
        
        Returns:
            List[Dict[str, Any]]: List of loaded results.
        """
        results = []
        
        # Find all JSON files in the results directory
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        for filepath in json_files:
            try:
                result = self.load_result(filepath)
                results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return results
    
    def find_results_by_chunker(self, chunker_name: str) -> List[Dict[str, Any]]:
        """
        Find all results for a specific chunker.
        
        Args:
            chunker_name (str): Name of the chunker.
            
        Returns:
            List[Dict[str, Any]]: List of results for the chunker.
        """
        all_results = self.load_all_results()
        return [r for r in all_results if r.get("chunker_name") == chunker_name]
    
    def find_results_by_experiment(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        Find all results for a specific experiment.
        
        Args:
            experiment_name (str): Name of the experiment.
            
        Returns:
            List[Dict[str, Any]]: List of results for the experiment.
        """
        all_results = self.load_all_results()
        return [r for r in all_results if r.get("experiment_name") == experiment_name]
    
    def get_latest_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the latest n experiment results.
        
        Args:
            n (int): Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of the latest results.
        """
        # Find all JSON files in the results directory
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        # Sort by modification time (newest first)
        json_files.sort(key=os.path.getmtime, reverse=True)
        
        # Load the latest n results
        results = []
        for filepath in json_files[:n]:
            try:
                result = self.load_result(filepath)
                results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return results
    
    def delete_result(self, filepath: str) -> bool:
        """
        Delete an experiment result file.
        
        Args:
            filepath (str): Path to the result file.
            
        Returns:
            bool: True if the file was deleted, False otherwise.
        """
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
            return False


class ResultsAnalyzer:
    """
    Analyzer for RAG experiment results.
    
    This class provides methods for analyzing and comparing the results
    of experiments comparing different chunking methods.
    """
    
    def __init__(self, storage: Optional[ResultsStorage] = None):
        """
        Initialize the ResultsAnalyzer.
        
        Args:
            storage (Optional[ResultsStorage]): Storage for experiment results.
                If None, a new ResultsStorage will be created.
        """
        self.storage = storage or ResultsStorage()
        self.visualizer = ResultsVisualizer(results_dir=self.storage.results_dir)
    
    def create_metrics_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of metrics from experiment results.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            
        Returns:
            pd.DataFrame: DataFrame of metrics.
        """
        data = []
        
        for result in results:
            experiment_name = result.get("experiment_name", "Unknown")
            chunker_name = result.get("chunker_name", "Unknown")
            avg_metrics = result.get("average_metrics", {})
            
            row = {
                "Experiment": experiment_name,
                "Chunker": chunker_name
            }
            
            # Add all metrics to the row
            for metric, value in avg_metrics.items():
                row[metric] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def create_efficiency_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of efficiency metrics from experiment results.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            
        Returns:
            pd.DataFrame: DataFrame of efficiency metrics.
        """
        data = []
        
        for result in results:
            experiment_name = result.get("experiment_name", "Unknown")
            chunker_name = result.get("chunker_name", "Unknown")
            timing_data = result.get("timing_data", {})
            chunk_data = result.get("chunk_data", {})
            
            row = {
                "Experiment": experiment_name,
                "Chunker": chunker_name,
                "Chunking Time (s)": timing_data.get("chunking_time", 0),
                "Retrieval Time (s)": timing_data.get("retrieval_time", 0),
                "Generation Time (s)": timing_data.get("generation_time", 0),
                "Total Time (s)": timing_data.get("total_time", 0),
                "Num Chunks": chunk_data.get("num_chunks", 0),
                "Avg Chunk Size (chars)": chunk_data.get("avg_chunk_size_chars", 0),
                "Avg Chunk Size (words)": chunk_data.get("avg_chunk_size_words", 0)
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def find_best_chunker(
        self,
        results: List[Dict[str, Any]],
        metric: str = "semantic_similarity"
    ) -> Dict[str, Any]:
        """
        Find the best chunker based on a specific metric.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            metric (str): Metric to use for comparison.
            
        Returns:
            Dict[str, Any]: The result for the best chunker.
        """
        if not results:
            return {}
        
        # Create a DataFrame of metrics
        df = self.create_metrics_dataframe(results)
        
        if metric not in df.columns:
            print(f"Metric {metric} not found in results")
            return {}
        
        # Find the best chunker
        best_idx = df[metric].idxmax()
        best_row = df.iloc[best_idx]
        
        # Find the corresponding result
        best_chunker = best_row["Chunker"]
        best_experiment = best_row["Experiment"]
        
        for result in results:
            if (result.get("chunker_name") == best_chunker and
                result.get("experiment_name") == best_experiment):
                return result
        
        return {}
    
    def compare_chunkers(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare chunkers across multiple metrics.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            metrics (Optional[List[str]]): List of metrics to compare.
                If None, uses all available metrics.
            
        Returns:
            pd.DataFrame: DataFrame comparing chunkers across metrics.
        """
        # Create a DataFrame of metrics
        df = self.create_metrics_dataframe(results)
        
        if metrics is not None:
            # Filter to only include specified metrics
            columns = ["Experiment", "Chunker"] + [m for m in metrics if m in df.columns]
            df = df[columns]
        
        return df
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of experiment results.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            
        Returns:
            Dict[str, Any]: Analysis results.
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Create DataFrames
        metrics_df = self.create_metrics_dataframe(results)
        efficiency_df = self.create_efficiency_dataframe(results)
        
        # Find the best chunker for each metric
        best_chunkers = {}
        for metric in metrics_df.columns:
            if metric not in ["Experiment", "Chunker"]:
                best_idx = metrics_df[metric].idxmax()
                best_chunker = metrics_df.iloc[best_idx]["Chunker"]
                best_value = metrics_df.iloc[best_idx][metric]
                best_chunkers[metric] = {
                    "chunker": best_chunker,
                    "value": best_value
                }
        
        # Find the most efficient chunker
        fastest_idx = efficiency_df["Total Time (s)"].idxmin()
        fastest_chunker = efficiency_df.iloc[fastest_idx]["Chunker"]
        fastest_time = efficiency_df.iloc[fastest_idx]["Total Time (s)"]
        
        # Find the chunker with the best balance of performance and efficiency
        # We'll use semantic_similarity / total_time as a simple balance metric
        if "semantic_similarity" in metrics_df.columns:
            balance_df = pd.merge(
                metrics_df[["Chunker", "semantic_similarity"]],
                efficiency_df[["Chunker", "Total Time (s)"]],
                on="Chunker"
            )
            balance_df["balance"] = balance_df["semantic_similarity"] / balance_df["Total Time (s)"]
            best_balance_idx = balance_df["balance"].idxmax()
            best_balance_chunker = balance_df.iloc[best_balance_idx]["Chunker"]
        else:
            best_balance_chunker = "N/A"
        
        # Create visualizations
        metrics_plot_path = os.path.join(self.storage.results_dir, "metrics_comparison.png")
        self.visualizer.plot_metrics_comparison(
            results,
            title="Metrics Comparison",
            save_path=metrics_plot_path
        )
        
        efficiency_plot_path = os.path.join(self.storage.results_dir, "efficiency_comparison.png")
        self.visualizer.plot_efficiency_comparison(
            results,
            title="Efficiency Comparison",
            save_path=efficiency_plot_path
        )
        
        chunk_stats_plot_path = os.path.join(self.storage.results_dir, "chunk_statistics.png")
        self.visualizer.plot_chunk_statistics(
            results,
            title="Chunk Statistics",
            save_path=chunk_stats_plot_path
        )
        
        # Return analysis results
        return {
            "best_chunkers": best_chunkers,
            "fastest_chunker": {
                "chunker": fastest_chunker,
                "time": fastest_time
            },
            "best_balance_chunker": best_balance_chunker,
            "metrics_df": metrics_df.to_dict(orient="records"),
            "efficiency_df": efficiency_df.to_dict(orient="records"),
            "plots": {
                "metrics_comparison": metrics_plot_path,
                "efficiency_comparison": efficiency_plot_path,
                "chunk_statistics": chunk_stats_plot_path
            }
        }
    
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        report_name: str = "chunking_methods_analysis"
    ) -> str:
        """
        Generate a comprehensive report of experiment results.
        
        Args:
            results (List[Dict[str, Any]]): List of experiment results.
            output_dir (Optional[str]): Directory to save the report.
            report_name (str): Name of the report.
            
        Returns:
            str: Path to the generated report.
        """
        # Create the comparison report using the visualizer
        return self.visualizer.create_comparison_report(
            results,
            output_dir=output_dir,
            report_name=report_name
        )