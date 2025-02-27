import spacy
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from langchain.schema import Document

nlp = spacy.load("en_core_web_sm")


class SemanticClusteringChunker:
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
        random_state: int = 42
    ):
        """
        Args:
            embedding_model_name (str): The sentence-transformers model name.
            max_chunk_size (int): Maximum number of words in a single chunk (heuristic).
            min_clusters (int): Lower bound for searching the optimal number of clusters.
            max_clusters (int): Upper bound for searching the optimal number of clusters.
            random_state (int): For reproducible clustering.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.max_chunk_size = max_chunk_size
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Main method: for each Document, splits into sentences, embeds them,
        finds the best K (clusters), groups by cluster, forms final chunk(s).
        """
        all_chunks = []
        for doc in docs:
            # 1) Split into sentences
            sents = list(nlp(doc.page_content).sents)
            if not sents:
                continue

            # 2) Get text & embeddings
            sentences_text = [s.text.strip() for s in sents if s.text.strip()]
            if not sentences_text:
                continue

            embeddings = self.embedding_model.encode(sentences_text, show_progress_bar=False)

            if len(sentences_text) == 1:
                # Only 1 sentence, just make it a chunk
                all_chunks.append(Document(page_content=sentences_text[0], metadata=doc.metadata))
                continue

            # 3) Find optimal K in [min_clusters, max_clusters] via silhouette
            best_k = self._select_optimal_k(embeddings)

            # 4) K-means clustering
            kmeans = KMeans(n_clusters=best_k, random_state=self.random_state)
            labels = kmeans.fit_predict(embeddings)

            # 5) Group sentences per cluster
            cluster_dict = {}
            for label, sentence in zip(labels, sentences_text):
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
        """
        n_samples = embeddings.shape[0]
        # Ako imamo 2 ili manje rečenica, klasterisanje nema smisla
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
            # Proverimo da li ima smisla računati silhouette_score
            # (moraju postojati bar 2 unikatne etikete)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k

