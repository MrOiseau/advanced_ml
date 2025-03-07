"""
Results Visualizer module for the RAG system.

This module contains the ResultsVisualizer class for visualizing and comparing
the results of different chunking methods in the RAG system.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple


class ResultsVisualizer:
    """
    Visualizer for RAG evaluation results.
    
    This class provides methods for visualizing and comparing the results
    of different chunking methods in the RAG system.
    """
    
    def __init__(self, results_dir: str = "./data/evaluation/results"):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            results_dir (str): Directory containing evaluation results.
        """
        self.results_dir = results_dir
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from a JSON file.
        
        Args:
            filepath (str): Path to the results file.
            
        Returns:
            Dict[str, Any]: The loaded results.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all evaluation results from the results directory.
        
        Returns:
            List[Dict[str, Any]]: List of loaded results.
        """
        results = []
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    result = self.load_results(filepath)
                    results.append(result)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        return results
    
    def plot_metrics_comparison(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        title: str = "Metrics Comparison",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a comparison of metrics across different chunking methods.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            metrics (Optional[List[str]]): List of metrics to compare.
                If None, uses a default set of metrics.
            title (str): Plot title.
            figsize (Tuple[int, int]): Figure size.
            save_path (Optional[str]): Path to save the figure.
            
        Returns:
            plt.Figure: The generated figure.
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                'bleu', 'f1_score', 'semantic_similarity',
                'rouge-1', 'rouge-l', 'precision@3', 'recall@3'
            ]
        
        # Extract experiment names and average metrics
        experiment_names = []
        metrics_data = []
        
        for result in results:
            experiment_name = result.get('experiment_name', 'Unknown')
            experiment_names.append(experiment_name)
            
            avg_metrics = result.get('average_metrics', {})
            # Ensure all metrics are float values
            metrics_dict = {}
            for metric in metrics:
                value = avg_metrics.get(metric, 0)
                # Convert to float to ensure it's numeric
                metrics_dict[metric] = float(value) if value is not None else 0.0
            metrics_data.append(metrics_dict)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data, index=experiment_names)
        
        # Check if we have numeric data to plot
        if df.empty or not df.select_dtypes(include=['number']).columns.any():
            # Create a figure with a message instead of plotting
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No numeric data available to plot",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=16)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax)
        
        # Customize plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Chunking Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_efficiency_comparison(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        title: str = "Efficiency Comparison",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a comparison of efficiency metrics across different chunking methods.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            metrics (Optional[List[str]]): List of efficiency metrics to compare.
                If None, uses a default set of metrics.
            title (str): Plot title.
            figsize (Tuple[int, int]): Figure size.
            save_path (Optional[str]): Path to save the figure.
            
        Returns:
            plt.Figure: The generated figure.
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                'chunking_time', 'retrieval_time', 'generation_time', 'total_time'
            ]
        
        # Extract experiment names and timing data
        experiment_names = []
        timing_data = []
        
        for result in results:
            experiment_name = result.get('experiment_name', 'Unknown')
            experiment_names.append(experiment_name)
            
            timing = result.get('timing_data', {})
            # Ensure all metrics are float values
            timing_dict = {}
            for metric in metrics:
                value = timing.get(metric, 0)
                # Convert to float to ensure it's numeric
                timing_dict[metric] = float(value) if value is not None else 0.0
            timing_data.append(timing_dict)
        
        # Create DataFrame
        df = pd.DataFrame(timing_data, index=experiment_names)
        
        # Check if we have numeric data to plot
        if df.empty or not df.select_dtypes(include=['number']).columns.any():
            # Create a figure with a message instead of plotting
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No numeric data available to plot",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=16)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax)
        
        # Customize plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Chunking Method', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.legend(title='Metrics', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_chunk_statistics(
        self,
        results: List[Dict[str, Any]],
        title: str = "Chunk Statistics",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot chunk statistics across different chunking methods.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            title (str): Plot title.
            figsize (Tuple[int, int]): Figure size.
            save_path (Optional[str]): Path to save the figure.
            
        Returns:
            plt.Figure: The generated figure.
        """
        # Extract experiment names and chunk data
        experiment_names = []
        num_chunks = []
        avg_chunk_size_chars = []
        avg_chunk_size_words = []
        
        for result in results:
            experiment_name = result.get('experiment_name', 'Unknown')
            experiment_names.append(experiment_name)
            
            chunk_data = result.get('chunk_data', {})
            # Convert to float to ensure it's numeric
            num_chunks.append(float(chunk_data.get('num_chunks', 0)) if chunk_data.get('num_chunks') is not None else 0.0)
            avg_chunk_size_chars.append(float(chunk_data.get('avg_chunk_size_chars', 0)) if chunk_data.get('avg_chunk_size_chars') is not None else 0.0)
            avg_chunk_size_words.append(float(chunk_data.get('avg_chunk_size_words', 0)) if chunk_data.get('avg_chunk_size_words') is not None else 0.0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Number of Chunks': num_chunks,
            'Avg. Chunk Size (chars)': avg_chunk_size_chars,
            'Avg. Chunk Size (words)': avg_chunk_size_words
        }, index=experiment_names)
        
        # Check if we have numeric data to plot
        if df.empty or not df.select_dtypes(include=['number']).columns.any() or df['Number of Chunks'].sum() == 0:
            # Create a figure with a message instead of plotting
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No chunk data available to plot",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=16)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot number of chunks on primary y-axis
        color = 'tab:blue'
        ax.set_xlabel('Chunking Method', fontsize=12)
        ax.set_ylabel('Number of Chunks', color=color, fontsize=12)
        ax.bar(experiment_names, df['Number of Chunks'], color=color, alpha=0.7)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Create secondary y-axis for average chunk sizes
        ax2 = ax.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Chunk Size', color=color, fontsize=12)
        ax2.plot(experiment_names, df['Avg. Chunk Size (chars)'], 'o-', color=color, label='Chars')
        ax2.plot(experiment_names, df['Avg. Chunk Size (words)'], 's-', color='tab:green', label='Words')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, ['Number of Chunks'] + labels2, loc='upper right')
        
        # Customize plot
        ax.set_title(title, fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(df['Number of Chunks']):
            ax.text(i, v + 5, str(int(v)), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metric_distribution(
        self,
        results: Dict[str, Any],
        metric: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the distribution of a metric across all questions for a single experiment.
        
        Args:
            results (Dict[str, Any]): Evaluation results for a single experiment.
            metric (str): The metric to plot.
            title (Optional[str]): Plot title.
            figsize (Tuple[int, int]): Figure size.
            save_path (Optional[str]): Path to save the figure.
            
        Returns:
            plt.Figure: The generated figure.
        """
        # Extract metric values for all questions
        metric_values = []
        
        for result in results.get('results', []):
            metrics = result.get('metrics', {})
            if metric in metrics:
                metric_values.append(metrics[metric])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        sns.histplot(metric_values, kde=True, ax=ax)
        
        # Add mean line
        mean_value = np.mean(metric_values)
        ax.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
        
        # Customize plot
        if title is None:
            experiment_name = results.get('experiment_name', 'Unknown')
            title = f"{metric} Distribution for {experiment_name}"
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        report_name: str = "chunking_methods_comparison"
    ) -> str:
        """
        Create a comprehensive comparison report of different chunking methods.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            output_dir (Optional[str]): Directory to save the report.
            report_name (str): Name of the report.
            
        Returns:
            str: Path to the generated report.
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "reports")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{report_name}_{timestamp}"
        report_dir = os.path.join(output_dir, report_filename)
        os.makedirs(report_dir, exist_ok=True)
        
        # Create plots
        metrics_plot = self.plot_metrics_comparison(
            results,
            title="Metrics Comparison",
            save_path=os.path.join(report_dir, "metrics_comparison.png")
        )
        
        efficiency_plot = self.plot_efficiency_comparison(
            results,
            title="Efficiency Comparison",
            save_path=os.path.join(report_dir, "efficiency_comparison.png")
        )
        
        chunk_stats_plot = self.plot_chunk_statistics(
            results,
            title="Chunk Statistics",
            save_path=os.path.join(report_dir, "chunk_statistics.png")
        )
        
        # Create summary table
        summary_data = []
        
        for result in results:
            experiment_name = result.get('experiment_name', 'Unknown')
            chunker_name = result.get('chunker_name', 'Unknown')
            avg_metrics = result.get('average_metrics', {})
            timing_data = result.get('timing_data', {})
            chunk_data = result.get('chunk_data', {})
            
            summary_data.append({
                'Experiment': experiment_name,
                'Chunker': chunker_name,
                'BLEU': avg_metrics.get('bleu', 0),
                'F1 Score': avg_metrics.get('f1_score', 0),
                'Semantic Similarity': avg_metrics.get('semantic_similarity', 0),
                'ROUGE-1': avg_metrics.get('rouge-1', 0),
                'ROUGE-L': avg_metrics.get('rouge-l', 0),
                'Precision@3': avg_metrics.get('precision@3', 0),
                'Recall@3': avg_metrics.get('recall@3', 0),
                'Chunking Time (s)': timing_data.get('chunking_time', 0),
                'Retrieval Time (s)': timing_data.get('retrieval_time', 0),
                'Generation Time (s)': timing_data.get('generation_time', 0),
                'Total Time (s)': timing_data.get('total_time', 0),
                'Num Chunks': chunk_data.get('num_chunks', 0),
                'Avg Chunk Size (chars)': chunk_data.get('avg_chunk_size_chars', 0),
                'Avg Chunk Size (words)': chunk_data.get('avg_chunk_size_words', 0)
            })
        
        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(report_dir, "summary_table.csv"), index=False)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chunking Methods Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Chunking Methods Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Experiment</th>
                        <th>Chunker</th>
                        <th>BLEU</th>
                        <th>F1 Score</th>
                        <th>Semantic Similarity</th>
                        <th>ROUGE-1</th>
                        <th>ROUGE-L</th>
                        <th>Precision@3</th>
                        <th>Recall@3</th>
                    </tr>
                    {"".join(f"<tr><td>{row['Experiment']}</td><td>{row['Chunker']}</td><td>{row['BLEU']:.4f}</td><td>{row['F1 Score']:.4f}</td><td>{row['Semantic Similarity']:.4f}</td><td>{row['ROUGE-1']:.4f}</td><td>{row['ROUGE-L']:.4f}</td><td>{row['Precision@3']:.4f}</td><td>{row['Recall@3']:.4f}</td></tr>" for row in summary_data)}
                </table>
            </div>
            
            <div class="section">
                <h2>Metrics Comparison</h2>
                <img src="metrics_comparison.png" alt="Metrics Comparison">
            </div>
            
            <div class="section">
                <h2>Efficiency Comparison</h2>
                <img src="efficiency_comparison.png" alt="Efficiency Comparison">
                <table>
                    <tr>
                        <th>Experiment</th>
                        <th>Chunking Time (s)</th>
                        <th>Retrieval Time (s)</th>
                        <th>Generation Time (s)</th>
                        <th>Total Time (s)</th>
                    </tr>
                    {"".join(f"<tr><td>{row['Experiment']}</td><td>{row['Chunking Time (s)']:.2f}</td><td>{row['Retrieval Time (s)']:.2f}</td><td>{row['Generation Time (s)']:.2f}</td><td>{row['Total Time (s)']:.2f}</td></tr>" for row in summary_data)}
                </table>
            </div>
            
            <div class="section">
                <h2>Chunk Statistics</h2>
                <img src="chunk_statistics.png" alt="Chunk Statistics">
                <table>
                    <tr>
                        <th>Experiment</th>
                        <th>Number of Chunks</th>
                        <th>Avg Chunk Size (chars)</th>
                        <th>Avg Chunk Size (words)</th>
                    </tr>
                    {"".join(f"<tr><td>{row['Experiment']}</td><td>{row['Num Chunks']}</td><td>{row['Avg Chunk Size (chars)']:.2f}</td><td>{row['Avg Chunk Size (words)']:.2f}</td></tr>" for row in summary_data)}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = os.path.join(report_dir, "report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return report_dir