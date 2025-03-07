"""
TopicBasedChunker module for the RAG system.

This module implements the TopicBasedChunker class, which uses
Latent Dirichlet Allocation (LDA) to group content by topics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
    A chunker that uses Latent Dirichlet Allocation (LDA) to group content by topics.
    
    This chunker first splits documents into sentences, then applies LDA to group
    sentences by topic, and finally forms chunks based on topic assignments.
    """
    
    def __init__(
        self,
        num_topics: int = 5,
        max_chunk_size: int = 200,
        min_topic_prob: float = 0.3,
        random_state: int = 42,
        language: str = 'english',
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
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            num_topics=num_topics,
            max_chunk_size=max_chunk_size,
            min_topic_prob=min_topic_prob,
            random_state=random_state,
            language=language,
            **kwargs
        )
        
        self.num_topics = num_topics
        self.max_chunk_size = max_chunk_size
        self.min_topic_prob = min_topic_prob
        self.random_state = random_state
        self.language = language
        self.stop_words = set(stopwords.words(language))
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents by grouping sentences by topic.
        
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
            
            # Create a document-term matrix
            vectorizer = CountVectorizer(
                stop_words=self.stop_words,
                min_df=2,  # Ignore terms that appear in less than 2 sentences
                max_df=0.9  # Ignore terms that appear in more than 90% of sentences
            )
            
            try:
                X = vectorizer.fit_transform(sentences)
                
                # Check if we have enough features for topic modeling
                if X.shape[1] < 5:  # Not enough features
                    all_chunks.append(Document(page_content=text, metadata=doc.metadata))
                    continue
                
                # Apply LDA
                lda = LatentDirichletAllocation(
                    n_components=min(self.num_topics, len(sentences) - 1),
                    random_state=self.random_state,
                    max_iter=10
                )
                
                # Get topic distributions for each sentence
                sentence_topics = lda.fit_transform(X)
                
                # Group sentences by dominant topic
                topic_sentences = self._group_by_topic(sentences, sentence_topics)
                
                # Create chunks from topic groups
                for topic_idx, topic_sents in topic_sentences.items():
                    chunks = self._create_chunks_from_sentences(topic_sents, doc.metadata)
                    all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error in topic modeling: {e}")
                # Fallback to treating the whole document as one chunk
                all_chunks.append(Document(page_content=text, metadata=doc.metadata))
        
        return all_chunks
    
    def _group_by_topic(
        self, 
        sentences: List[str], 
        sentence_topics: np.ndarray
    ) -> Dict[int, List[str]]:
        """
        Group sentences by their dominant topic.
        
        Args:
            sentences (List[str]): List of sentences.
            sentence_topics (np.ndarray): Topic distributions for each sentence.
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping topic indices to lists of sentences.
        """
        topic_sentences = {}
        
        for i, (sentence, topic_dist) in enumerate(zip(sentences, sentence_topics)):
            # Get the dominant topic for this sentence
            dominant_topic = np.argmax(topic_dist)
            dominant_prob = topic_dist[dominant_topic]
            
            # Only assign if the probability is above the threshold
            if dominant_prob >= self.min_topic_prob:
                if dominant_topic not in topic_sentences:
                    topic_sentences[dominant_topic] = []
                topic_sentences[dominant_topic].append(sentence)
            else:
                # For sentences without a clear topic, create a separate "topic"
                misc_topic = len(sentence_topics[0]) + i  # Ensure unique topic ID
                topic_sentences[misc_topic] = [sentence]
        
        return topic_sentences
    
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