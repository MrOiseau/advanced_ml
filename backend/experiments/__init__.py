"""
Experiments module for the RAG system.

This module contains tools for configuring, running, and analyzing experiments
to compare different chunking methods in the RAG system.
"""

from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import ExperimentRunner
from backend.experiments.results import ResultsStorage, ResultsAnalyzer

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "ResultsStorage",
    "ResultsAnalyzer",
]