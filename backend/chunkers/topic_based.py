"""
TopicBasedChunker module for the RAG system.

This module implements the TopicBasedChunker class, which uses
BERTopic to group content by topics, replacing the previous
Latent Dirichlet Allocation (LDA) implementation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from bertopic import BERTopic
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from langchain.schema import Document
from backend.chunkers.base import BaseChunker

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TopicBasedChunker(BaseChunker):
    """
    A chunker that uses BERTopic to group content by topics.
    
    This chunker first splits documents into sentences, then applies BERTopic to group
    sentences by topic, and finally forms chunks based on topic assignments.
    
    BERTopic is more effective than LDA for shorter texts like sentences, as it uses
    transformer-based embeddings that better capture semantic meaning.
    """
    
    def __init__(
        self,
        num_topics: int = 5,
        max_chunk_size: int = 200,
        min_topic_prob: float = 0.3,
        random_state: int = 42,
        language: str = 'english',
        preserve_order: bool = True,
        **kwargs
    ):
        """
        Initialize the TopicBasedChunker.
        
        Args:
            num_topics (int): Number of topics to extract.
            max_chunk_size (int): Maximum number of words in a single chunk.
            min_topic_prob (float): Minimum probability for a sentence to belong to a topic.
            random_state (int): For reproducible results.
            language (str): Language for stopwords removal.
            preserve_order (bool): Whether to preserve the original order of sentences within topics.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            num_topics=num_topics,
            max_chunk_size=max_chunk_size,
            min_topic_prob=min_topic_prob,
            random_state=random_state,
            language=language,
            preserve_order=preserve_order,
            **kwargs
        )
        
        self.num_topics = num_topics
        self.max_chunk_size = max_chunk_size
        self.min_topic_prob = min_topic_prob
        self.random_state = random_state
        self.language = language
        self.preserve_order = preserve_order
        self.stop_words = set(stopwords.words(language))
        
        # Initialize BERTopic model
        self.topic_model = BERTopic(
            nr_topics=num_topics,
            calculate_probabilities=True,
            verbose=True
            # random_state parameter is not supported by BERTopic
        )
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents by grouping sentences by topic using BERTopic.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects where each document
            represents a chunk with sentences grouped by topic.
        """
        all_chunks = []
        
        for doc in docs:
            # Split document into sentences
            text = doc.page_content
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
            
            if not sentences:
                continue
                
            if len(sentences) <= 3:  # Too few sentences for meaningful topic modeling
                all_chunks.append(Document(page_content=text, metadata=doc.metadata))
                continue
            
            try:
                # Apply BERTopic to sentences
                topics, probs = self.topic_model.fit_transform(sentences)
                
                # Group sentences by topic, preserving original indices
                topic_sentences = {}
                sentence_indices = {}  # To preserve original order
                
                for i, (sentence, topic) in enumerate(zip(sentences, topics)):
                    if topic not in topic_sentences:
                        topic_sentences[topic] = []
                        sentence_indices[topic] = []
                    
                    topic_sentences[topic].append(sentence)
                    sentence_indices[topic].append(i)
                
                # Create chunks from topic groups, preserving original order if needed
                for topic, topic_sents in topic_sentences.items():
                    if self.preserve_order:
                        # Sort sentences within each topic by their original position
                        sorted_pairs = sorted(zip(sentence_indices[topic], topic_sents))
                        topic_sents = [sent for _, sent in sorted_pairs]
                    
                    chunks = self._create_chunks_from_sentences(topic_sents, doc.metadata)
                    all_chunks.extend(chunks)
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in topic modeling: {e}\nDetails: {error_details}")
                
                # Add error information to metadata
                error_metadata = doc.metadata.copy()
                error_metadata["chunking_error"] = str(e)
                error_metadata["chunking_fallback"] = "simple_paragraphs"
                
                # More intelligent fallback: try paragraph-based chunking instead of one big chunk
                try:
                    # Split by paragraphs (double newlines)
                    paragraphs = text.split("\n\n")
                    
                    # Process each paragraph
                    for para in paragraphs:
                        if not para.strip():
                            continue
                            
                        para_words = len(para.split())
                        
                        # If paragraph fits in chunk size, add it directly
                        if para_words <= self.max_chunk_size:
                            all_chunks.append(Document(page_content=para, metadata=error_metadata))
                        else:
                            # Otherwise, split into sentences
                            para_sentences = sent_tokenize(para)
                            current_chunk = []
                            current_chunk_words = 0
                            
                            for sentence in para_sentences:
                                sentence_words = len(sentence.split())
                                
                                if current_chunk_words + sentence_words <= self.max_chunk_size:
                                    current_chunk.append(sentence)
                                    current_chunk_words += sentence_words
                                else:
                                    # Create a chunk with the current sentences
                                    if current_chunk:
                                        chunk_text = " ".join(current_chunk).strip()
                                        all_chunks.append(Document(page_content=chunk_text, metadata=error_metadata))
                                    
                                    # Start a new chunk with this sentence
                                    current_chunk = [sentence]
                                    current_chunk_words = sentence_words
                            
                            # Add the last chunk if there's anything left
                            if current_chunk:
                                chunk_text = " ".join(current_chunk).strip()
                                all_chunks.append(Document(page_content=chunk_text, metadata=error_metadata))
                
                except Exception as fallback_error:
                    print(f"Fallback chunking also failed: {fallback_error}. Using document as single chunk.")
                    # Ultimate fallback: treat the whole document as one chunk
                    all_chunks.append(Document(page_content=text, metadata=error_metadata))
        
        return all_chunks
    
    def _create_chunks_from_sentences(
        self, 
        sentences: List[str], 
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Create chunks from a list of sentences, respecting the max chunk size.
        
        Args:
            sentences (List[str]): List of sentences to chunk.
            metadata (Dict[str, Any]): Metadata to preserve in the chunks.
            
        Returns:
            List[Document]: List of Document objects representing chunks.
        """
        chunks = []
        current_chunk = []
        current_chunk_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_chunk_words + sentence_words <= self.max_chunk_size:
                current_chunk.append(sentence)
                current_chunk_words += sentence_words
            else:
                # Create a chunk with the current sentences
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
                
                # Start a new chunk with this sentence
                current_chunk = [sentence]
                current_chunk_words = sentence_words
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
        return chunks