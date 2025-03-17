"""
Base chunker module for the RAG system.

This module defines the BaseChunker abstract class that all chunkers must implement,
the ChunkerFactory for creating chunker instances, and the ChunkingMethod enum
for a unified API across all chunking methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Optional
import time
from enum import Enum, auto
from langchain.schema import Document


class ChunkingMethod(Enum):
    """
    Enumeration of available chunking methods.
    
    This provides a unified API for referring to different chunking strategies.
    """
    RECURSIVE_CHARACTER = auto()  # Basic recursive character chunking
    GLOBAL_SEMANTIC = auto()      # SemanticClusteringChunker
    LOCAL_SEMANTIC = auto()       # SentenceTransformersSplitter
    THEMATIC = auto()             # TopicBasedChunker
    HIERARCHICAL = auto()         # HierarchicalChunker
    HYBRID = auto()               # HybridChunker (combines multiple strategies)


class ChunkingConfig:
    """
    Configuration class for chunking parameters.
    
    This class provides a standardized way to configure chunking parameters
    across different chunking methods.
    """
    def __init__(
        self,
        max_chunk_size: int = 200,
        chunk_overlap: int = 50,
        preserve_order: bool = True,
        semantic_threshold: float = 0.6,
        **kwargs
    ):
        """
        Initialize chunking configuration.
        
        Args:
            max_chunk_size (int): Maximum number of words in a single chunk.
            chunk_overlap (int): Number of words to overlap between chunks.
            preserve_order (bool): Whether to preserve chronological order.
            semantic_threshold (float): Threshold for semantic similarity (0.0 to 1.0).
            **kwargs: Additional configuration parameters.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_order = preserve_order
        self.semantic_threshold = semantic_threshold
        self.additional_params = kwargs


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    
    All chunking strategies should inherit from this class and implement
    the chunk_documents method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the chunker with optional parameters.
        
        Args:
            **kwargs: Arbitrary keyword arguments for specific chunker implementations.
        """
        self.params = kwargs
        self.name = self.__class__.__name__
    
    @abstractmethod
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects where each document
            represents a chunk with the original metadata preserved.
        """
        pass
    
    def measure_performance(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Measure the performance of the chunking method.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            Dict[str, Any]: A dictionary containing performance metrics such as
            processing time, number of chunks created, average chunk size, etc.
        """
        start_time = time.time()
        chunks = self.chunk_documents(docs)
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        num_chunks = len(chunks)
        
        # Calculate average chunk size (in characters)
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / num_chunks if num_chunks > 0 else 0
        
        # Calculate average chunk size (in words)
        total_words = sum(len(chunk.page_content.split()) for chunk in chunks)
        avg_chunk_words = total_words / num_chunks if num_chunks > 0 else 0
        
        # Calculate chronological coherence (if applicable)
        chronological_coherence = self._calculate_chronological_coherence(chunks, docs)
        
        # Calculate semantic coherence (if applicable)
        semantic_coherence = self._calculate_semantic_coherence(chunks)
        
        return {
            "chunker_name": self.name,
            "processing_time_seconds": processing_time,
            "num_chunks": num_chunks,
            "avg_chunk_size_chars": avg_chunk_size,
            "avg_chunk_size_words": avg_chunk_words,
            "total_chars": total_chars,
            "total_words": total_words,
            "chronological_coherence": chronological_coherence,
            "semantic_coherence": semantic_coherence,
            "params": self.params
        }
    
    def _calculate_chronological_coherence(self, chunks: List[Document], original_docs: List[Document]) -> Optional[float]:
        """
        Calculate how well chunks preserve the original document order.
        
        This is a placeholder method that can be overridden by subclasses.
        
        Args:
            chunks (List[Document]): The generated chunks.
            original_docs (List[Document]): The original documents.
            
        Returns:
            Optional[float]: A score between 0 and 1, where 1 means perfect
            preservation of chronological order, or None if not applicable.
        """
        return None
    
    def _calculate_semantic_coherence(self, chunks: List[Document]) -> Optional[float]:
        """
        Calculate semantic coherence within chunks.
        
        This is a placeholder method that can be overridden by subclasses.
        
        Args:
            chunks (List[Document]): The generated chunks.
            
        Returns:
            Optional[float]: A score between 0 and 1, where 1 means perfect
            semantic coherence, or None if not applicable.
        """
        return None


class ChunkerFactory:
    """
    Factory class for creating chunker instances.
    
    This class maintains a registry of chunker types and provides methods
    for creating instances of registered chunkers.
    """
    
    _registry: Dict[str, Type[BaseChunker]] = {}
    _method_map: Dict[ChunkingMethod, str] = {}
    
    @classmethod
    def register(cls, name: str, chunker_class: Type[BaseChunker], method: Optional[ChunkingMethod] = None) -> None:
        """
        Register a chunker class with a name and optional method enum.
        
        Args:
            name (str): The name to register the chunker under.
            chunker_class (Type[BaseChunker]): The chunker class to register.
            method (Optional[ChunkingMethod]): The corresponding ChunkingMethod enum value.
        """
        cls._registry[name] = chunker_class
        if method is not None:
            cls._method_map[method] = name
    
    @classmethod
    def create(cls, name_or_method: Any, **kwargs) -> BaseChunker:
        """
        Create an instance of a registered chunker.
        
        Args:
            name_or_method: Either a string name or a ChunkingMethod enum value.
            **kwargs: Arguments to pass to the chunker's constructor.
            
        Returns:
            BaseChunker: An instance of the requested chunker.
            
        Raises:
            ValueError: If the requested chunker is not registered.
        """
        name = name_or_method
        
        # If a ChunkingMethod enum is provided, convert it to a name
        if isinstance(name_or_method, ChunkingMethod):
            if name_or_method not in cls._method_map:
                registered_methods = ", ".join(str(m) for m in cls._method_map.keys())
                raise ValueError(
                    f"ChunkingMethod '{name_or_method}' not mapped to any chunker. "
                    f"Available methods: {registered_methods}"
                )
            name = cls._method_map[name_or_method]
        
        # Create the chunker by name
        if name not in cls._registry:
            registered_chunkers = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Chunker '{name}' not found in registry. "
                f"Available chunkers: {registered_chunkers}"
            )
        
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_chunkers(cls) -> List[str]:
        """
        List all registered chunker names.
        
        Returns:
            List[str]: A list of registered chunker names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def list_methods(cls) -> List[ChunkingMethod]:
        """
        List all registered chunking methods.
        
        Returns:
            List[ChunkingMethod]: A list of registered ChunkingMethod enum values.
        """
        return list(cls._method_map.keys())
    
    @classmethod
    def create_from_config(cls, method: ChunkingMethod, config: ChunkingConfig) -> BaseChunker:
        """
        Create a chunker from a ChunkingMethod enum and ChunkingConfig.
        
        This provides a unified way to create chunkers with standardized configuration.
        
        Args:
            method (ChunkingMethod): The chunking method to use.
            config (ChunkingConfig): Configuration parameters.
            
        Returns:
            BaseChunker: An instance of the requested chunker.
        """
        # Convert config to kwargs
        kwargs = {
            "max_chunk_size": config.max_chunk_size,
            "preserve_order": config.preserve_order,
        }
        
        # Add method-specific parameters
        if method == ChunkingMethod.LOCAL_SEMANTIC:
            kwargs["similarity_threshold"] = config.semantic_threshold
            
        # Add any additional parameters
        kwargs.update(config.additional_params)
        
        return cls.create(method, **kwargs)