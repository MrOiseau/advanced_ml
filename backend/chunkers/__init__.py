"""
Chunkers module for the RAG system.

This module contains various document chunking strategies for the RAG system.
"""

from backend.chunkers.base import BaseChunker, ChunkerFactory
from backend.chunkers.recursive_character import RecursiveCharacterChunker
from backend.chunkers.semantic_clustering import SemanticClusteringChunker
from backend.chunkers.sentence_transformers import SentenceTransformersSplitter
from backend.chunkers.topic_based import TopicBasedChunker
from backend.chunkers.hierarchical import HierarchicalChunker

# Register all chunkers here
ChunkerFactory.register("recursive_character", RecursiveCharacterChunker)
ChunkerFactory.register("semantic_clustering", SemanticClusteringChunker)
ChunkerFactory.register("sentence_transformers", SentenceTransformersSplitter)
ChunkerFactory.register("topic_based", TopicBasedChunker)
ChunkerFactory.register("hierarchical", HierarchicalChunker)

__all__ = [
    "BaseChunker",
    "ChunkerFactory",
    "RecursiveCharacterChunker",
    "SemanticClusteringChunker",
    "SentenceTransformersSplitter",
    "TopicBasedChunker",
    "HierarchicalChunker",
]