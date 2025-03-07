"""
Experiment Configuration module for the RAG system.

This module contains the ExperimentConfig class for configuring experiments
to compare different chunking methods in the RAG system.
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from backend.experiments.results import NumpyEncoder


class ExperimentConfig:
    """
    Configuration for RAG system experiments.
    
    This class provides methods for configuring and managing experiments
    to compare different chunking methods in the RAG system.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        config_dir: str = "./data/experiments/configs"
    ):
        """
        Initialize the ExperimentConfig.
        
        Args:
            name (str): Name of the experiment.
            description (Optional[str]): Description of the experiment.
            config_dir (str): Directory to store experiment configurations.
        """
        self.name = name
        self.description = description or f"Experiment: {name}"
        self.config_dir = config_dir
        self.chunker_configs = []
        self.dataset_path = None
        self.output_dir = "./data/evaluation/results"
        self.created_at = datetime.now().isoformat()
    
    def add_chunker(
        self,
        chunker_name: str,
        chunker_params: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ) -> 'ExperimentConfig':
        """
        Add a chunker configuration to the experiment.
        
        Args:
            chunker_name (str): Name of the chunker.
            chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
            experiment_name (Optional[str]): Name for this specific experiment.
            
        Returns:
            ExperimentConfig: Self for method chaining.
        """
        chunker_params = chunker_params or {}
        experiment_name = experiment_name or f"{chunker_name}_experiment"
        
        self.chunker_configs.append({
            "chunker_name": chunker_name,
            "chunker_params": chunker_params,
            "experiment_name": experiment_name
        })
        
        return self
    
    def set_dataset(self, dataset_path: str) -> 'ExperimentConfig':
        """
        Set the dataset path for the experiment.
        
        Args:
            dataset_path (str): Path to the evaluation dataset.
            
        Returns:
            ExperimentConfig: Self for method chaining.
        """
        self.dataset_path = dataset_path
        return self
    
    def set_output_dir(self, output_dir: str) -> 'ExperimentConfig':
        """
        Set the output directory for experiment results.
        
        Args:
            output_dir (str): Directory to store experiment results.
            
        Returns:
            ExperimentConfig: Self for method chaining.
        """
        self.output_dir = output_dir
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration.
        """
        return {
            "name": self.name,
            "description": self.description,
            "chunker_configs": self.chunker_configs,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "created_at": self.created_at
        }
    
    def save(self) -> str:
        """
        Save the experiment configuration to a JSON file.
        
        Returns:
            str: Path to the saved configuration file.
        """
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.json"
        filepath = os.path.join(self.config_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load an experiment configuration from a JSON file.
        
        Args:
            filepath (str): Path to the configuration file.
            
        Returns:
            ExperimentConfig: The loaded configuration.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(
            name=config_dict["name"],
            description=config_dict.get("description"),
            config_dir=os.path.dirname(filepath)
        )
        
        config.chunker_configs = config_dict.get("chunker_configs", [])
        config.dataset_path = config_dict.get("dataset_path")
        config.output_dir = config_dict.get("output_dir")
        config.created_at = config_dict.get("created_at")
        
        return config
    
    @classmethod
    def create_default_experiment(cls) -> 'ExperimentConfig':
        """
        Create a default experiment configuration with all chunkers.
        
        Returns:
            ExperimentConfig: The default experiment configuration.
        """
        config = cls(
            name="all_chunkers_comparison",
            description="Comparison of all available chunking methods"
        )
        
        # Add all chunkers with default parameters
        config.add_chunker(
            chunker_name="recursive_character",
            chunker_params={"chunk_size": 1000, "chunk_overlap": 200},
            experiment_name="recursive_character_default"
        )
        
        config.add_chunker(
            chunker_name="semantic_clustering",
            chunker_params={"max_chunk_size": 200, "min_clusters": 2, "max_clusters": 10},
            experiment_name="semantic_clustering_default"
        )
        
        config.add_chunker(
            chunker_name="sentence_transformers",
            chunker_params={"max_chunk_size": 200, "similarity_threshold": 0.6},
            experiment_name="sentence_transformers_default"
        )
        
        config.add_chunker(
            chunker_name="topic_based",
            chunker_params={"num_topics": 5, "max_chunk_size": 200},
            experiment_name="topic_based_default"
        )
        
        config.add_chunker(
            chunker_name="hierarchical",
            chunker_params={"max_chunk_size": 200},
            experiment_name="hierarchical_default"
        )
        
        # Set default dataset path
        config.set_dataset("./data/evaluation/evaluation_dataset_chatgpt_unique.json")
        
        return config