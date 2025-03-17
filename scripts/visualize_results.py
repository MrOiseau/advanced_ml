#!/usr/bin/env python3
"""
Script to visualize experiment results.

This script provides a command-line interface for creating visualizations
of experiment results for the thesis.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.experiments.results import ResultsStorage
from backend.evaluation.visualizer import ResultsVisualizer


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create visualizations of experiment results."
    )
    
    parser.add_argument(
        "--results-dir",
        default="./data/evaluation/results",
        help="Directory containing experiment results"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./data/visualizations",
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--include-metrics",
        nargs="+",
        default=["bleu", "f1_score", "semantic_similarity", "rouge-1", "rouge-l", "precision@3", "recall@3"],
        help="Metrics to include in visualizations"
    )
    
    parser.add_argument(
        "--visualization-type",
        choices=["bar", "line", "radar", "heatmap", "scatter", "all"],
        default="all",
        help="Type of visualization to create"
    )
    
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[12, 8],
        help="Figure size (width, height) in inches"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures"
    )
    
    return parser.parse_args()


def create_bar_chart(
    metrics_df: pd.DataFrame,
    metric: str,
    output_dir: str,
    figsize: List[int],
    dpi: int
):
    """
    Create a bar chart for a specific metric.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of metrics.
        metric (str): Metric to visualize.
        output_dir (str): Directory to save the visualization.
        figsize (List[int]): Figure size [width, height] in inches.
        dpi (int): DPI for the saved figure.
    """
    if metric not in metrics_df.columns:
        print(f"Metric {metric} not found in results")
        return
    
    plt.figure(figsize=tuple(figsize))
    ax = sns.barplot(x="Chunker", y=metric, data=metrics_df)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df[metric]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    
    plt.title(f"{metric} Across Chunking Methods", fontsize=16)
    plt.xlabel("Chunking Method", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.join(output_dir, "bar_charts"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "bar_charts", f"{metric}_bar.png"), dpi=dpi)
    plt.close()


def create_line_chart(
    metrics_df: pd.DataFrame,
    include_metrics: List[str],
    output_dir: str,
    figsize: List[int],
    dpi: int
):
    """
    Create a line chart comparing all metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of metrics.
        include_metrics (List[str]): Metrics to include.
        output_dir (str): Directory to save the visualization.
        figsize (List[int]): Figure size [width, height] in inches.
        dpi (int): DPI for the saved figure.
    """
    # Filter metrics that exist in the DataFrame
    metrics = [m for m in include_metrics if m in metrics_df.columns]
    
    if not metrics:
        print("No valid metrics found for line chart")
        return
    
    # Normalize metrics to 0-1 scale for fair comparison
    normalized_df = metrics_df.copy()
    for metric in metrics:
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (metrics_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 0
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(
        normalized_df,
        id_vars=["Chunker"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Normalized Value"
    )
    
    plt.figure(figsize=tuple(figsize))
    sns.lineplot(
        x="Chunker",
        y="Normalized Value",
        hue="Metric",
        style="Metric",
        markers=True,
        dashes=False,
        data=melted_df
    )
    
    plt.title("Normalized Metrics Comparison", fontsize=16)
    plt.xlabel("Chunking Method", fontsize=12)
    plt.ylabel("Normalized Value (0-1)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Metric", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.join(output_dir, "line_charts"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "line_charts", "metrics_line.png"), dpi=dpi)
    plt.close()

def create_radar_chart(
    metrics_df: pd.DataFrame,
    include_metrics: List[str],
    output_dir: str,
    figsize: List[int],
    dpi: int
):
    """
    Create a radar chart for each chunker.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of metrics.
        include_metrics (List[str]): Metrics to include.
        output_dir (str): Directory to save the visualization.
        figsize (List[int]): Figure size [width, height] in inches.
        dpi (int): DPI for the saved figure.
    """
    # Filter metrics that exist in the DataFrame
    metrics = [m for m in include_metrics if m in metrics_df.columns]
    
    if not metrics:
        print("No valid metrics found for radar chart")
        return
    
    # Print available metrics for debugging
    print(f"Available metrics for radar chart: {metrics}")
    print(f"Metrics DataFrame columns: {metrics_df.columns.tolist()}")
    
    # Normalize metrics to 0-1 scale
    normalized_df = metrics_df.copy()
    for metric in metrics:
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (metrics_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 0
    
    # Create radar chart for each chunker
    chunkers = normalized_df["Chunker"].unique()
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "radar_charts"), exist_ok=True)
    
    # Create a radar chart comparing all chunkers
    plt.figure(figsize=tuple(figsize))
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax = plt.subplot(111, polar=True)
    
    # Add labels
    plt.xticks(angles[:-1], metrics, fontsize=10)
    
    # Draw the chart for each chunker
    for chunker in chunkers:
        # Get values for this chunker
        chunker_df = normalized_df[normalized_df["Chunker"] == chunker]
        
        # Check if we have data for this chunker
        if chunker_df.empty:
            print(f"No data for chunker {chunker}, skipping")
            continue
        
        # Get values for the metrics
        try:
            values = chunker_df[metrics].values.flatten().tolist()
            
            # Check if values has the right length
            if len(values) != len(metrics):
                print(f"Warning: Values for chunker {chunker} has length {len(values)}, expected {len(metrics)}. Skipping.")
                continue
                
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=chunker)
            ax.fill(angles, values, alpha=0.1)
        except Exception as e:
            print(f"Error plotting chunker {chunker}: {e}")
            continue
    
    plt.title("Normalized Metrics by Chunking Method", fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "radar_charts", "metrics_radar.png"), dpi=dpi)
    plt.close()
    plt.close()


def create_heatmap(
    metrics_df: pd.DataFrame,
    include_metrics: List[str],
    output_dir: str,
    figsize: List[int],
    dpi: int
):
    """
    Create a heatmap of metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of metrics.
        include_metrics (List[str]): Metrics to include.
        output_dir (str): Directory to save the visualization.
        figsize (List[int]): Figure size [width, height] in inches.
        dpi (int): DPI for the saved figure.
    """
    # Filter metrics that exist in the DataFrame
    metrics = [m for m in include_metrics if m in metrics_df.columns]
    
    if not metrics:
        print("No valid metrics found for heatmap")
        return
    
    # Normalize metrics to 0-1 scale
    normalized_df = metrics_df.copy()
    for metric in metrics:
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (metrics_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 0
    
    # Create pivot table for heatmap
    # Instead of using pivot, we'll just set the index to "Chunker" and select the metrics columns
    pivot_df = normalized_df.set_index("Chunker")[metrics]
    
    plt.figure(figsize=tuple(figsize))
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        linewidths=.5
    )
    
    plt.title("Metrics Heatmap (Normalized Values)", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "heatmaps", "metrics_heatmap.png"), dpi=dpi)
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=tuple(figsize))
    corr = metrics_df[metrics].corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=.5
    )
    
    plt.title("Metrics Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "heatmaps", "correlation_heatmap.png"), dpi=dpi)
    plt.close()


def create_scatter_plots(
    metrics_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    include_metrics: List[str],
    output_dir: str,
    figsize: List[int],
    dpi: int
):
    """
    Create scatter plots of metrics vs. efficiency.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of metrics.
        efficiency_df (pd.DataFrame): DataFrame of efficiency metrics.
        include_metrics (List[str]): Metrics to include.
        output_dir (str): Directory to save the visualization.
        figsize (List[int]): Figure size [width, height] in inches.
        dpi (int): DPI for the saved figure.
    """
    # Filter metrics that exist in the DataFrame
    metrics = [m for m in include_metrics if m in metrics_df.columns]
    
    if not metrics:
        print("No valid metrics found for scatter plots")
        return
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "scatter_plots"), exist_ok=True)
    
    # Create scatter plots for each metric vs. efficiency
    for metric in metrics:
        plt.figure(figsize=tuple(figsize))
        
        # Merge metrics and efficiency DataFrames
        merged_df = pd.merge(
            metrics_df[["Chunker", metric]],
            efficiency_df[["Chunker", "Total Time (s)"]],
            on="Chunker"
        )
        
        # Create scatter plot
        sns.scatterplot(
            x="Total Time (s)",
            y=metric,
            data=merged_df,
            s=100
        )
        
        # Add labels for each point
        for i, row in merged_df.iterrows():
            plt.text(
                row["Total Time (s)"] + 0.1,
                row[metric],
                row["Chunker"],
                fontsize=10
            )
        
        plt.title(f"{metric} vs. Efficiency", fontsize=16)
        plt.xlabel("Total Time (seconds)", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "scatter_plots", f"{metric}_vs_efficiency.png"), dpi=dpi)
        plt.close()


def visualize_results(args):
    """
    Create visualizations of experiment results.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print(f"Creating visualizations of experiment results")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results directly from JSON files
    results = []
    json_files = [f for f in os.listdir(args.results_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No results found.")
        return
    
    print(f"Found {len(json_files)} experiment results.")
    
    # Create analyzer and visualizer
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    
    # Create DataFrames for metrics and efficiency
    metrics_data = []
    efficiency_data = []
    
    for json_file in json_files:
        file_path = os.path.join(args.results_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Handle the case where the result might be a list
            if isinstance(result, list):
                print(f"Warning: Result in {json_file} is a list, not a dictionary. Skipping.")
                continue
            
            results.append(result)
            
            experiment_name = result.get("experiment_name", "Unknown")
            chunker_name = result.get("chunker_name", "Unknown")
            avg_metrics = result.get("average_metrics", {})
            timing_data = result.get("timing_data", {})
            chunk_data = result.get("chunk_data", {})
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
        
        # Metrics data
        metrics_row = {
            "Experiment": experiment_name,
            "Chunker": chunker_name
        }
        
        # Add all metrics to the row
        for metric, value in avg_metrics.items():
            metrics_row[metric] = value
        
        metrics_data.append(metrics_row)
        
        # Efficiency data
        efficiency_row = {
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
        
        efficiency_data.append(efficiency_row)
    
    # Create DataFrames
    metrics_df = pd.DataFrame(metrics_data)
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Save DataFrames to CSV
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    efficiency_df.to_csv(os.path.join(args.output_dir, "efficiency.csv"), index=False)
    
    # Create visualizations based on the specified type
    if args.visualization_type in ["bar", "all"]:
        print("Creating bar charts...")
        for metric in args.include_metrics:
            if metric in metrics_df.columns:
                create_bar_chart(metrics_df, metric, args.output_dir, args.figsize, args.dpi)
    
    if args.visualization_type in ["line", "all"]:
        print("Creating line chart...")
        create_line_chart(metrics_df, args.include_metrics, args.output_dir, args.figsize, args.dpi)
    
    if args.visualization_type in ["radar", "all"]:
        print("Creating radar chart...")
        create_radar_chart(metrics_df, args.include_metrics, args.output_dir, args.figsize, args.dpi)
    
    if args.visualization_type in ["heatmap", "all"]:
        print("Creating heatmaps...")
        create_heatmap(metrics_df, args.include_metrics, args.output_dir, args.figsize, args.dpi)
    
    if args.visualization_type in ["scatter", "all"]:
        print("Creating scatter plots...")
        create_scatter_plots(metrics_df, efficiency_df, args.include_metrics, args.output_dir, args.figsize, args.dpi)
    
    # Create standard visualizations using the visualizer
    print("Creating standard visualizations...")
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(
        results,
        metrics=args.include_metrics,
        title="Metrics Comparison Across Chunking Methods",
        figsize=tuple(args.figsize),
        save_path=os.path.join(args.output_dir, "metrics_comparison.png")
    )
    
    # Efficiency comparison
    visualizer.plot_efficiency_comparison(
        results,
        title="Efficiency Comparison Across Chunking Methods",
        figsize=tuple(args.figsize),
        save_path=os.path.join(args.output_dir, "efficiency_comparison.png")
    )
    
    # Chunk statistics
    visualizer.plot_chunk_statistics(
        results,
        title="Chunk Statistics Across Chunking Methods",
        figsize=tuple(args.figsize),
        save_path=os.path.join(args.output_dir, "chunk_statistics.png")
    )
    
    print(f"Visualizations created in: {args.output_dir}")


def main():
    """
    Main function.
    """
    args = parse_args()
    visualize_results(args)


if __name__ == "__main__":
    main()
