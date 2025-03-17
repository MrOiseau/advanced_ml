"""
HybridChunker module for the RAG system.

This module implements the HybridChunker class, which combines multiple
chunking strategies based on document characteristics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from langchain.schema import Document
from backend.chunkers.base import BaseChunker, ChunkerFactory
from backend.chunkers.semantic_clustering import SemanticClusteringChunker
from backend.chunkers.sentence_transformers import SentenceTransformersSplitter
from backend.chunkers.topic_based import TopicBasedChunker


class HybridChunker(BaseChunker):
    """
    A hybrid chunker that combines multiple chunking strategies.
    
    This chunker analyzes document characteristics and selects the most
    appropriate chunking strategy for each document or document section.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 200,
        preserve_order: bool = True,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        **kwargs
    ):
        """
        Initialize the HybridChunker.
        
        Args:
            max_chunk_size (int): Maximum number of words in a single chunk.
            preserve_order (bool): Whether to preserve chronological order.
            embedding_model_name (str): The sentence-transformers model name.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            preserve_order=preserve_order,
            embedding_model_name=embedding_model_name,
            **kwargs
        )
        
        # Initialize component chunkers
        self.topic_chunker = TopicBasedChunker(
            max_chunk_size=max_chunk_size,
            preserve_order=preserve_order,
            **kwargs
        )
        
        self.semantic_chunker = SemanticClusteringChunker(
            max_chunk_size=max_chunk_size,
            preserve_order=preserve_order,
            embedding_model_name=embedding_model_name,
            **kwargs
        )
        
        self.sentence_chunker = SentenceTransformersSplitter(
            max_chunk_size=max_chunk_size,
            embedding_model_name=embedding_model_name,
            **kwargs
        )
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.preserve_order = preserve_order
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents using a hybrid approach that selects the best
        chunking strategy based on content characteristics.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects representing chunks.
        """
        all_chunks = []
        
        for doc in docs:
            # Analyze document complexity
            complexity_score, complexity_type = self._analyze_complexity(doc)
            
            if complexity_type == "semantic_diversity":
                # Use semantic clustering for documents with diverse content
                chunks = self.semantic_chunker.chunk_documents([doc])
            elif complexity_type == "topic_structure":
                # Use topic-based chunking for documents with clear topic structure
                chunks = self.topic_chunker.chunk_documents([doc])
            else:
                # Use sentence transformers for simpler, more linear documents
                chunks = self.sentence_chunker.chunk_documents([doc])
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _analyze_complexity(self, doc: Document) -> Tuple[float, str]:
        """
        Analyze document complexity to determine the best chunking strategy.
        
        Args:
            doc (Document): Document to analyze.
            
        Returns:
            Tuple[float, str]: (complexity_score, complexity_type)
                complexity_score: A score between 0 (simple) and 1 (complex)
                complexity_type: One of "semantic_diversity", "topic_structure", "linear"
        """
        text = doc.page_content
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        
        if len(sentences) <= 5:
            return 0.0, "linear"  # Very simple document
        
        # Factors that contribute to complexity:
        
        # 1. Vocabulary diversity (Type-Token Ratio)
        words = ' '.join(sentences).split()
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # 2. Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        normalized_sentence_length = min(avg_sentence_length / 30, 1.0)  # Normalize to 0-1
        
        # 3. Topic diversity (using embeddings variance as proxy)
        try:
            # Use sentence transformer model to get embeddings
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate variance of embeddings
            variance = np.var(embeddings, axis=0).mean()
            normalized_variance = min(variance * 10, 1.0)  # Scale and cap at 1.0
            
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings)
            
            # Calculate average similarity between adjacent sentences
            adjacent_similarities = [similarities[i, i+1] for i in range(len(sentences)-1)]
            avg_adjacent_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 0.5
            
            # Calculate average similarity between all sentences
            all_similarities = similarities.sum() / (len(sentences) * len(sentences))
            
            # Detect topic shifts
            topic_shifts = 0
            for i in range(len(sentences) - 1):
                if similarities[i, i+1] < 0.5:  # Threshold for topic shift
                    topic_shifts += 1
            normalized_topic_shifts = min(topic_shifts / (len(sentences) / 5), 1.0)
            
        except Exception as e:
            print(f"Error in complexity analysis: {e}")
            normalized_variance = 0.5
            avg_adjacent_similarity = 0.5
            all_similarities = 0.5
            normalized_topic_shifts = 0.5
        
        # Determine document type based on metrics
        semantic_diversity_score = normalized_variance * 0.7 + (1 - all_similarities) * 0.3
        topic_structure_score = normalized_topic_shifts * 0.7 + (1 - avg_adjacent_similarity) * 0.3
        linear_score = avg_adjacent_similarity * 0.7 + (1 - normalized_variance) * 0.3
        
        # Get the dominant characteristic
        scores = {
            "semantic_diversity": semantic_diversity_score,
            "topic_structure": topic_structure_score,
            "linear": linear_score
        }
        
        dominant_type = max(scores.items(), key=lambda x: x[1])[0]
        complexity_score = max(scores.values())
        
        return complexity_score, dominant_type


# Register the HybridChunker with the ChunkerFactory
ChunkerFactory.register("hybrid", HybridChunker)