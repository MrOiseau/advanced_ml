"""
Chunkers module for the RAG system.

This module contains various document chunking strategies for the RAG system.
"""

from backend.chunkers.base import BaseChunker, ChunkerFactory, ChunkingMethod
from backend.chunkers.recursive_character import RecursiveCharacterChunker
from backend.chunkers.semantic_clustering import SemanticClusteringChunker
from backend.chunkers.sentence_transformers import SentenceTransformersSplitter
from backend.chunkers.topic_based import TopicBasedChunker
from backend.chunkers.hierarchical import HierarchicalChunker
from backend.chunkers.hybrid import HybridChunker

# Register all chunkers here with their corresponding ChunkingMethod enum values
ChunkerFactory.register("recursive_character", RecursiveCharacterChunker, ChunkingMethod.RECURSIVE_CHARACTER)
ChunkerFactory.register("semantic_clustering", SemanticClusteringChunker, ChunkingMethod.GLOBAL_SEMANTIC)
ChunkerFactory.register("sentence_transformers", SentenceTransformersSplitter, ChunkingMethod.LOCAL_SEMANTIC)
ChunkerFactory.register("topic_based", TopicBasedChunker, ChunkingMethod.THEMATIC)
ChunkerFactory.register("hierarchical", HierarchicalChunker, ChunkingMethod.HIERARCHICAL)
# HybridChunker is registered in its own file, but we'll also register it with the enum here
ChunkerFactory.register("hybrid", HybridChunker, ChunkingMethod.HYBRID)

__all__ = [
    "BaseChunker",
    "ChunkerFactory",
    "RecursiveCharacterChunker",
    "SemanticClusteringChunker",
    "SentenceTransformersSplitter",
    "TopicBasedChunker",
    "HierarchicalChunker",
    "HybridChunker",
]