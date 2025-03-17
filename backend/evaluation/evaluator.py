"""
RAG Evaluator module for the RAG system.

This module contains the RAGEvaluator class for evaluating the performance
of different chunking methods in the RAG system.
"""

import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

from backend.chunkers.base import BaseChunker, ChunkerFactory
from backend.querying import QueryPipeline
from backend.evaluation.metrics import (
    calculate_all_metrics,
    calculate_bleu,
    calculate_rouge,
    calculate_f1_score,
    calculate_semantic_similarity,
    calculate_exact_match,
)
from backend.utils import NumpyEncoder


class RAGEvaluator:
    """
    Evaluator for the RAG system.
    
    This class evaluates the performance of different chunking methods
    in the RAG system using various metrics.
    """
    
    def __init__(
        self,
        dataset_path: str,
        query_pipeline: QueryPipeline,
        chunker: Optional[BaseChunker] = None,
        output_dir: str = "./data/evaluation/results"
    ) -> None:
        """
        Initializes the evaluator with a dataset and query pipeline.
        
        Args:
            dataset_path (str): Path to the evaluation dataset in JSON format.
            query_pipeline (QueryPipeline): Query pipeline used to retrieve documents and generate answers.
            chunker (Optional[BaseChunker]): Chunker to evaluate. If None, the default chunker from the pipeline is used.
            output_dir (str): Directory to save evaluation results.
        """
        self.dataset = self.load_dataset(dataset_path)
        self.sentence_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.query_pipeline = query_pipeline
        self.chunker = chunker
        self.output_dir = output_dir
        
        # Track timing information
        self.timing_data = {
            'chunking_time': 0.0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'total_time': 0.0
        }
        
        # Track chunk information
        self.chunk_data = {
            'num_chunks': 0,
            'avg_chunk_size_chars': 0.0,
            'avg_chunk_size_words': 0.0
        }
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Loads the evaluation dataset from a JSON file.
        
        Args:
            dataset_path (str): Path to the evaluation dataset in JSON format.
            
        Returns:
            List[Dict[str, Any]]: The loaded dataset.
        """
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def is_relevant(self, doc: Document, answer: str) -> bool:
        """
        Checks if a document is relevant to the answer using semantic similarity.
        
        This method uses sentence embeddings to compute the semantic similarity
        between the document content and the reference answer. Documents with
        similarity scores above a threshold are considered relevant.
        
        Args:
            doc (Document): Document to check.
            answer (str): Reference answer string.
            
        Returns:
            bool: True if the document is semantically relevant to the answer.
        """
        # Use a threshold for semantic similarity (0.4 is a reasonable starting point)
        SIMILARITY_THRESHOLD = 0.4
        
        try:
            # Generate embeddings for the document and answer
            doc_embedding = self.sentence_model.encode(doc.page_content, convert_to_tensor=True)
            answer_embedding = self.sentence_model.encode(answer, convert_to_tensor=True)
            
            # Move tensors to CPU before converting to numpy arrays
            doc_embedding = doc_embedding.cpu().numpy()
            answer_embedding = answer_embedding.cpu().numpy()
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(doc_embedding.reshape(1, -1),
                                          answer_embedding.reshape(1, -1))[0][0]
            
            return similarity > SIMILARITY_THRESHOLD
        except Exception as e:
            # Fallback to word overlap method if there's an error
            print(f"Error in semantic similarity calculation: {e}. Falling back to word overlap.")
            answer_words = set(answer.lower().split())
            doc_words = set(doc.page_content.lower().split())
            return len(answer_words.intersection(doc_words)) > 0
    
    async def retrieve_documents_async(self, question: str) -> Tuple[List[Document], float]:
        """
        Asynchronously retrieves documents for a given query and measures retrieval time.
        
        Args:
            question (str): The query to retrieve documents for.
            
        Returns:
            Tuple[List[Document], float]: Retrieved documents and retrieval time.
        """
        start_time = time.time()
        docs = await asyncio.to_thread(self.query_pipeline.retrieve_documents, question)
        retrieval_time = time.time() - start_time
        return docs, retrieval_time
    
    def generate_answer(self, question: str, docs: List[Document]) -> Tuple[str, float]:
        """
        Generates an answer for a given question and documents and measures generation time.
        
        Args:
            question (str): The question to answer.
            docs (List[Document]): The documents to use for answering.
            
        Returns:
            Tuple[str, float]: Generated answer and generation time.
        """
        start_time = time.time()
        formatted_docs = [{'content': doc.page_content} for doc in docs]
        answer = self.query_pipeline.generate_summary(
            question, 
            "\n".join([doc['content'] for doc in formatted_docs])
        )
        generation_time = time.time() - start_time
        return answer, generation_time
    
    async def evaluate_async(self) -> List[Dict[str, Any]]:
        """
        Asynchronously evaluates the query pipeline using the dataset and computes various metrics.
        
        Returns:
            List[Dict[str, Any]]: List of evaluation results for each question.
        """
        all_results = []
        questions = [item['question'] for item in self.dataset]
        reference_answers = [item['answer'] for item in self.dataset]
        
        # Reset timing and chunk data
        self.timing_data = {
            'chunking_time': 0.0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'total_time': 0.0
        }
        
        # Fetch documents for all questions asynchronously
        retrieval_tasks = [self.retrieve_documents_async(question) for question in questions]
        retrieval_results = await asyncio.gather(*retrieval_tasks)
        
        # Process each question
        for idx, (question, reference_answer, (retrieved_docs, retrieval_time)) in enumerate(
            zip(questions, reference_answers, retrieval_results)
        ):
            if not retrieved_docs:
                continue
            
            # Update retrieval time
            self.timing_data['retrieval_time'] += retrieval_time
            
            # Generate answer and measure time
            generated_answer, generation_time = self.generate_answer(question, retrieved_docs)
            
            # Update generation time
            self.timing_data['generation_time'] += generation_time
            
            if not generated_answer:
                continue
            
            # Determine which documents are relevant to the reference answer
            relevant_docs = [doc for doc in retrieved_docs if self.is_relevant(doc, reference_answer)]
            
            # Compute metrics
            efficiency_data = {
                'chunking_time': self.timing_data['chunking_time'] / len(questions) if questions else 0,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'num_chunks': self.chunk_data.get('num_chunks', 0),
                'avg_chunk_size': self.chunk_data.get('avg_chunk_size_chars', 0)
            }
            
            metrics = calculate_all_metrics(
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                relevant_docs=relevant_docs,
                retrieved_docs=retrieved_docs,
                efficiency_data=efficiency_data,
                model=self.sentence_model
            )
            
            # Add question and answers to results
            results = {
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'metrics': metrics
            }
            
            all_results.append(results)
        
        # Update total time
        self.timing_data['total_time'] = (
            self.timing_data['chunking_time'] +
            self.timing_data['retrieval_time'] +
            self.timing_data['generation_time']
        )
        
        return all_results
    
    def compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes the average of all evaluation metrics across all results.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            
        Returns:
            Dict[str, float]: Dictionary of average metrics.
        """
        if not results:
            return {}
        
        # Extract all metric names from the first result
        metric_names = []
        for key, value in results[0]['metrics'].items():
            if isinstance(value, dict):
                # For nested metrics like efficiency
                for nested_key in value.keys():
                    metric_names.append(f"{key}.{nested_key}")
            else:
                metric_names.append(key)
        
        # Compute averages
        averages = {}
        for metric in metric_names:
            if '.' in metric:
                # Handle nested metrics
                parent, child = metric.split('.')
                values = [result['metrics'][parent][child] for result in results if parent in result['metrics'] and child in result['metrics'][parent]]
            else:
                # Handle top-level metrics
                values = [result['metrics'][metric] for result in results if metric in result['metrics']]
            
            if values:
                averages[metric] = np.mean(values)
        
        return averages
    
    def save_results(self, results: List[Dict[str, Any]], experiment_name: str) -> str:
        """
        Saves evaluation results to a JSON file.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results.
            experiment_name (str): Name of the experiment.
            
        Returns:
            str: Path to the saved results file.
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Add timing and chunk data to results
        output = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'timing_data': self.timing_data,
            'chunk_data': self.chunk_data,
            'results': results,
            'average_metrics': self.compute_average_metrics(results)
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def evaluate_chunker(
        self,
        chunker_name: str,
        chunker_params: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a specific chunker with the given parameters.
        
        Args:
            chunker_name (str): Name of the chunker to evaluate.
            chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
            experiment_name (Optional[str]): Name of the experiment.
            
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        # Create the chunker
        chunker_params = chunker_params or {}
        chunker = ChunkerFactory.create(chunker_name, **chunker_params)
        self.chunker = chunker
        
        # Set experiment name if not provided
        if experiment_name is None:
            experiment_name = f"{chunker_name}_evaluation"
        
        # Measure chunking time and get chunk data
        # This would typically be done during indexing, but we simulate it here
        from backend.indexing import IngestionPipeline
        import os
        
        # Create a temporary ingestion pipeline to measure chunking performance
        ingestion = IngestionPipeline(
            pdf_dir=os.getenv("PDF_DIR", "./data/pdfs"),
            db_dir=os.getenv("DB_DIR", "./data/db"),
            db_collection=os.getenv("DB_COLLECTION", "rag_collection_advanced"),
            chunk_size=chunker_params.get("chunk_size", 1000),
            chunk_overlap=chunker_params.get("chunk_overlap", 200),
            data_dir=os.getenv("DATA_DIR", "./data"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        
        # Parse PDFs
        docs = ingestion.parse_pdfs()
        
        # Measure chunking performance
        start_time = time.time()
        chunks = chunker.chunk_documents(docs)
        self.timing_data['chunking_time'] = time.time() - start_time
        
        # Get chunk data
        self.chunk_data = {
            'num_chunks': len(chunks),
            'avg_chunk_size_chars': np.mean([len(chunk.page_content) for chunk in chunks]) if chunks else 0,
            'avg_chunk_size_words': np.mean([len(chunk.page_content.split()) for chunk in chunks]) if chunks else 0
        }
        
        # Run evaluation
        results = asyncio.run(self.evaluate_async())
        
        # Save results
        results_path = self.save_results(results, experiment_name)
        
        # Return summary
        return {
            'experiment_name': experiment_name,
            'chunker_name': chunker_name,
            'chunker_params': chunker_params,
            'results_path': results_path,
            'average_metrics': self.compute_average_metrics(results),
            'timing_data': self.timing_data,
            'chunk_data': self.chunk_data
        }