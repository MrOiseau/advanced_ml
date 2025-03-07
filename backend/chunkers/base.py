"""
Base chunker module for the RAG system.

This module defines the BaseChunker abstract class that all chunkers must implement,
and the ChunkerFactory for creating chunker instances.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type
import time
from langchain.schema import Document


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
        
        return {
            "chunker_name": self.name,
            "processing_time_seconds": processing_time,
            "num_chunks": num_chunks,
            "avg_chunk_size_chars": avg_chunk_size,
            "avg_chunk_size_words": avg_chunk_words,
            "total_chars": total_chars,
            "total_words": total_words,
            "params": self.params
        }


class ChunkerFactory:
    """
    Factory class for creating chunker instances.
    
    This class maintains a registry of chunker types and provides methods
    for creating instances of registered chunkers.
    """
    
    _registry: Dict[str, Type[BaseChunker]] = {}
    
    @classmethod
    def register(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        """
        Register a chunker class with a name.
        
        Args:
            name (str): The name to register the chunker under.
            chunker_class (Type[BaseChunker]): The chunker class to register.
        """
        cls._registry[name] = chunker_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseChunker:
        """
        Create an instance of a registered chunker.
        
        Args:
            name (str): The name of the chunker to create.
            **kwargs: Arguments to pass to the chunker's constructor.
            
        Returns:
            BaseChunker: An instance of the requested chunker.
            
        Raises:
            ValueError: If the requested chunker is not registered.
        """
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