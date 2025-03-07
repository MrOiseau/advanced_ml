"""
SemanticClusteringChunker module for the RAG system.

This module implements the SemanticClusteringChunker class, which uses
sentence embeddings and K-means clustering to group semantically similar
sentences into chunks.
"""

import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from langchain.schema import Document
from backend.chunkers.base import BaseChunker

# Download the punkt tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SemanticClusteringChunker(BaseChunker):
    """
    A chunker that uses sentence-transformers embeddings + k-means
    to group semantically similar sentences into chunk(s).
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 200,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the SemanticClusteringChunker.
        
        Args:
            embedding_model_name (str): The sentence-transformers model name.
            max_chunk_size (int): Maximum number of words in a single chunk (heuristic).
            min_clusters (int): Lower bound for searching the optimal number of clusters.
            max_clusters (int): Upper bound for searching the optimal number of clusters.
            random_state (int): For reproducible clustering.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            embedding_model_name=embedding_model_name,
            max_chunk_size=max_chunk_size,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_state=random_state,
            **kwargs
        )
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Main method: for each Document, splits into sentences, embeds them,
        finds the best K (clusters), groups by cluster, forms final chunk(s).
        
        Uses batch processing for embeddings to improve performance.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects where each document
            represents a semantically coherent chunk with the original metadata preserved.
        """
        all_chunks = []
        
        # Process documents in parallel using batch processing for embeddings
        # Collect all sentences from all documents first
        all_sentences = []
        doc_sentence_map = []  # To keep track of which sentences belong to which document
        
        for doc_idx, doc in enumerate(docs):
            # 1) Split into sentences using NLTK
            text = doc.page_content
            sentences_text = [s.strip() for s in sent_tokenize(text) if s.strip()]
            
            if not sentences_text:
                continue
                
            # Add sentences to the batch
            all_sentences.extend(sentences_text)
            # Keep track of which sentences belong to which document
            doc_sentence_map.extend([(doc_idx, i) for i in range(len(sentences_text))])
        
        # Process all sentences in one batch for efficiency
        if not all_sentences:
            return all_chunks
            
        # Batch size for embedding generation
        batch_size = 32
        all_embeddings = []
        
        # Process embeddings in batches
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:min(i+batch_size, len(all_sentences))]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)
        
        # Now process each document's sentences
        doc_sentence_indices = {}
        for doc_idx, sent_idx in doc_sentence_map:
            if doc_idx not in doc_sentence_indices:
                doc_sentence_indices[doc_idx] = []
            doc_sentence_indices[doc_idx].append(sent_idx)
        
        # Process each document
        for doc_idx, sent_indices in doc_sentence_indices.items():
            doc = docs[doc_idx]
            doc_sentences = [all_sentences[doc_sentence_map.index((doc_idx, i))] for i in range(len(sent_indices))]
            doc_embeddings = np.array([all_embeddings[doc_sentence_map.index((doc_idx, i))] for i in range(len(sent_indices))])
            
            if len(doc_sentences) == 1:
                # Only 1 sentence, just make it a chunk
                all_chunks.append(Document(page_content=doc_sentences[0], metadata=doc.metadata))
                continue

            # 3) Find optimal K in [min_clusters, max_clusters] via silhouette
            best_k = self._select_optimal_k(doc_embeddings)

            # 4) K-means clustering
            kmeans = KMeans(n_clusters=best_k, random_state=self.random_state)
            labels = kmeans.fit_predict(doc_embeddings)

            # 5) Group sentences per cluster
            cluster_dict = {}
            for label, sentence in zip(labels, doc_sentences):
                cluster_dict.setdefault(label, []).append(sentence)

            # 6) Form final chunks with max_chunk_size limit
            for cl_label, sentence_list in cluster_dict.items():
                chunk_words = 0
                current_chunk = []
                for sent in sentence_list:
                    sent_len = len(sent.split())
                    if (chunk_words + sent_len) <= self.max_chunk_size:
                        current_chunk.append(sent)
                        chunk_words += sent_len
                    else:
                        # close current chunk
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            all_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
                        # start a new chunk
                        current_chunk = [sent]
                        chunk_words = sent_len

                # any leftover
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        all_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

        return all_chunks

    def _select_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Select the optimal number of clusters (k) using silhouette scoring.
        
        Args:
            embeddings (np.ndarray): Array of sentence embeddings.
            
        Returns:
            int: The optimal number of clusters.
        """
        n_samples = embeddings.shape[0]
        # If we have 2 or fewer sentences, clustering doesn't make sense
        if n_samples <= 2:
            return 1

        max_k = min(self.max_clusters, n_samples - 1)
        if max_k < 2:
            return 1

        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(embeddings)
            # Check if it makes sense to calculate silhouette_score
            # (there must be at least 2 unique labels)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k