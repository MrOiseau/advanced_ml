"""
Evaluation module for the RAG system.

This module contains evaluation metrics and tools for assessing the performance
of different chunking methods in the RAG system.
"""

from backend.evaluation.metrics import (
    calculate_bleu,
    calculate_rouge,
    calculate_f1_score,
    calculate_semantic_similarity,
    calculate_exact_match,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
)
from backend.evaluation.evaluator import RAGEvaluator
from backend.evaluation.visualizer import ResultsVisualizer

__all__ = [
    "calculate_bleu",
    "calculate_rouge",
    "calculate_f1_score",
    "calculate_semantic_similarity",
    "calculate_exact_match",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_mrr",
    "RAGEvaluator",
    "ResultsVisualizer",
]