#!/usr/bin/env python3
"""
Consolidated evaluation script for RAG system.

This script:
1. Loads the evaluation dataset with ground truth chunks
2. Evaluates all chunking methods or a specific chunker
3. Generates comparison reports and visualizations
"""

import os
import sys
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import *
from backend.chunkers.base import ChunkerFactory
from backend.querying import QueryPipeline
from backend.evaluation.metrics import calculate_all_metrics
from backend.evaluation.visualizer import ResultsVisualizer


class RAGEvaluator:
    """
    Evaluator for the RAG system that uses ground truth chunks.
    """
    
    def __init__(
        self,
        dataset_path: str,
        query_pipeline: QueryPipeline,
        chunker_name: str,
        chunker_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "./data/evaluation/results"
    ) -> None:
        """
        Initializes the evaluator with a dataset and query pipeline.
        
        Args:
            dataset_path (str): Path to the evaluation dataset in JSON format.
            query_pipeline (QueryPipeline): Query pipeline used to retrieve documents and generate answers.
            chunker_name (str): Name of the chunker to evaluate.
            chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
            output_dir (str): Directory to save evaluation results.
        """
        self.dataset = self.load_dataset(dataset_path)
        self.query_pipeline = query_pipeline
        self.chunker_name = chunker_name
        self.chunker_params = chunker_params or {}
        self.output_dir = output_dir
        
        # Create the chunker
        self.chunker = ChunkerFactory.create(chunker_name, **self.chunker_params)
        
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
    
    def convert_ground_truth_to_documents(self, item: Dict[str, Any]) -> List[Document]:
        """
        Converts ground truth chunks to Document objects.
        
        Args:
            item (Dict[str, Any]): Dataset item containing ground truth chunks.
            
        Returns:
            List[Document]: List of Document objects.
        """
        from langchain.schema import Document
        ground_truth_docs = []
        
        # Use ground_truth_chunks to create Document objects
        if 'ground_truth_chunks' in item:
            for i, chunk in enumerate(item['ground_truth_chunks']):
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'id': chunk.get('id', f"gt_{i}"),
                        'source': chunk.get('source', ''),
                        'title': chunk.get('source', ''),
                        'is_ground_truth': True
                    }
                )
                ground_truth_docs.append(doc)
        
        return ground_truth_docs
    
    def is_document_in_ground_truth(self, doc, ground_truth_docs) -> bool:
        """
        Checks if a document matches any ground truth document.
        
        Args:
            doc: Document to check.
            ground_truth_docs: List of ground truth documents.
            
        Returns:
            bool: True if the document matches a ground truth document.
        """
        # First try to match by content (most reliable)
        for gt_doc in ground_truth_docs:
            if doc.page_content.strip() == gt_doc.page_content.strip():
                return True
        
        # If no content match, try to match by ID
        doc_id = doc.metadata.get('id')
        if doc_id:
            for gt_doc in ground_truth_docs:
                if doc_id == gt_doc.metadata.get('id'):
                    return True
        
        # If still no match, try semantic similarity as a fallback
        # This is less reliable but can catch cases where the content is slightly different
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            doc_embedding = model.encode(doc.page_content, convert_to_tensor=True).cpu().numpy()
            
            for gt_doc in ground_truth_docs:
                gt_embedding = model.encode(gt_doc.page_content, convert_to_tensor=True).cpu().numpy()
                similarity = cosine_similarity(doc_embedding.reshape(1, -1), gt_embedding.reshape(1, -1))[0][0]
                if similarity > 0.9:  # High threshold for similarity
                    return True
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
        
        return False
    
    async def retrieve_documents_async(self, question: str) -> Tuple[List, float]:
        """
        Asynchronously retrieves documents for a given query and measures retrieval time.
        
        Args:
            question (str): The query to retrieve documents for.
            
        Returns:
            Tuple[List, float]: Retrieved documents and retrieval time.
        """
        start_time = time.time()
        docs = await asyncio.to_thread(self.query_pipeline.retrieve_documents, question)
        retrieval_time = time.time() - start_time
        return docs, retrieval_time
    
    def generate_answer(self, question: str, docs) -> Tuple[str, float]:
        """
        Generates an answer for a given question and documents and measures generation time.
        
        Args:
            question (str): The question to answer.
            docs: The documents to use for answering.
            
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
        
        # Reset timing data
        self.timing_data = {
            'chunking_time': 0.0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'total_time': 0.0
        }
        
        # Process each question
        for idx, item in enumerate(self.dataset):
            question = item['question']
            reference_answer = item['answer']
            ground_truth_docs = self.convert_ground_truth_to_documents(item)
            
            print(f"Processing question {idx+1}/{len(self.dataset)}: {question[:50]}...")
            
            # Retrieve documents
            retrieved_docs, retrieval_time = await self.retrieve_documents_async(question)
            
            if not retrieved_docs:
                print(f"No documents retrieved for question {idx+1}. Skipping.")
                continue
            
            # Update retrieval time
            self.timing_data['retrieval_time'] += retrieval_time
            
            # Generate answer
            generated_answer, generation_time = self.generate_answer(question, retrieved_docs)
            
            # Update generation time
            self.timing_data['generation_time'] += generation_time
            
            if not generated_answer:
                print(f"No answer generated for question {idx+1}. Skipping.")
                continue
            
            # Determine which documents are relevant (match ground truth)
            relevant_docs = [
                doc for doc in retrieved_docs 
                if self.is_document_in_ground_truth(doc, ground_truth_docs)
            ]
            
            print(f"Found {len(relevant_docs)} relevant documents out of {len(retrieved_docs)} retrieved.")
            
            # Compute metrics
            efficiency_data = {
                'chunking_time': self.timing_data['chunking_time'] / len(self.dataset),
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
                k_values=[1, 3, 5, 10]  # Explicitly include k=10
            )
            
            # Add question and answers to results
            results = {
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'num_retrieved': len(retrieved_docs),
                'num_relevant': len(relevant_docs),
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
        import numpy as np
        
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
            'chunker_name': self.chunker_name,
            'chunker_params': self.chunker_params,
            'timing_data': self.timing_data,
            'chunk_data': self.chunk_data,
            'results': results,
            'average_metrics': self.compute_average_metrics(results)
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def measure_chunking_performance(self) -> None:
        """
        Measures chunking performance and updates chunk data.
        """
        from backend.indexing import IngestionPipeline
        
        # Create a temporary ingestion pipeline to measure chunking performance
        ingestion = IngestionPipeline(
            pdf_dir=os.getenv("PDF_DIR", "./data/pdfs"),
            db_dir=os.getenv("DB_DIR", "./data/db"),
            db_collection=os.getenv("DB_COLLECTION", "rag_collection_advanced"),
            chunk_size=self.chunker_params.get("chunk_size", 1000),
            chunk_overlap=self.chunker_params.get("chunk_overlap", 200),
            data_dir=os.getenv("DATA_DIR", "./data"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        
        # Parse PDFs
        docs = ingestion.parse_pdfs()
        
        # Measure chunking performance
        start_time = time.time()
        chunks = self.chunker.chunk_documents(docs)
        self.timing_data['chunking_time'] = time.time() - start_time
        
        # Get chunk data
        import numpy as np
        self.chunk_data = {
            'num_chunks': len(chunks),
            'avg_chunk_size_chars': np.mean([len(chunk.page_content) for chunk in chunks]) if chunks else 0,
            'avg_chunk_size_words': np.mean([len(chunk.page_content.split()) for chunk in chunks]) if chunks else 0
        }
    
    async def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the chunker and returns the results.
        
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        # Set experiment name
        experiment_name = f"{self.chunker_name}_evaluation"
        
        # Measure chunking performance
        self.measure_chunking_performance()
        
        # Run evaluation
        results = await self.evaluate_async()
        
        # Save results
        results_path = self.save_results(results, experiment_name)
        
        # Return summary
        return {
            'experiment_name': experiment_name,
            'chunker_name': self.chunker_name,
            'chunker_params': self.chunker_params,
            'results_path': results_path,
            'average_metrics': self.compute_average_metrics(results),
            'timing_data': self.timing_data,
            'chunk_data': self.chunk_data
        }


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.
    """
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


async def evaluate_chunker(
    dataset_path: str,
    chunker_name: str,
    chunker_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "./data/evaluation/results"
) -> Dict[str, Any]:
    """
    Evaluates a specific chunker with the given parameters.
    
    Args:
        dataset_path (str): Path to the evaluation dataset.
        chunker_name (str): Name of the chunker to evaluate.
        chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
        output_dir (str): Directory to save evaluation results.
        
    Returns:
        Dict[str, Any]: Evaluation results.
    """
    # Initialize the query pipeline
    query_pipeline = QueryPipeline(
        db_dir=DB_DIR,
        db_collection=f"wfp_collection_{chunker_name}",
        embedding_model=EMBEDDING_MODEL,
        chat_model=CHAT_MODEL,
        chat_temperature=CHAT_TEMPERATURE,
        search_results_num=SEARCH_RESULTS_NUM,
        langsmith_project=LANGSMITH_PROJECT
    )
    
    # Initialize the evaluator
    evaluator = RAGEvaluator(
        dataset_path=dataset_path,
        query_pipeline=query_pipeline,
        chunker_name=chunker_name,
        chunker_params=chunker_params,
        output_dir=output_dir
    )
    
    # Run evaluation
    return await evaluator.evaluate()


async def evaluate_all_chunkers(
    dataset_path: str,
    output_dir: str = "./data/evaluation/results"
) -> List[Dict[str, Any]]:
    """
    Evaluates all available chunkers.
    
    Args:
        dataset_path (str): Path to the evaluation dataset.
        output_dir (str): Directory to save evaluation results.
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results for each chunker.
    """
    # Get all available chunkers
    chunkers = ChunkerFactory.list_chunkers()
    print(f"Found {len(chunkers)} chunking methods: {', '.join(chunkers)}")
    
    # Evaluate each chunker
    results = []
    for chunker_name in chunkers:
        print(f"\nEvaluating chunker: {chunker_name}")
        
        try:
            # Evaluate the chunker
            result = await evaluate_chunker(
                dataset_path=dataset_path,
                chunker_name=chunker_name,
                output_dir=output_dir
            )
            
            results.append(result)
            
            print(f"Evaluation completed. Results saved to: {result['results_path']}")
            print("Average metrics:")
            for metric, value in result["average_metrics"].items():
                print(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            print(f"Error evaluating chunker {chunker_name}: {e}")
    
    return results


async def main():
    """
    Main function to parse arguments and run evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG system with different chunking methods")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset")
    parser.add_argument("--output_dir", default="./data/evaluation/results", help="Directory to save evaluation results")
    parser.add_argument("--chunker", help="Specific chunker to evaluate (evaluates all if not specified)")
    parser.add_argument("--chunker_params", help="JSON string of parameters for the chunker")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse chunker_params if provided
    chunker_params = None
    if args.chunker_params:
        try:
            chunker_params = json.loads(args.chunker_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing chunker_params: {e}")
            return
    
    # Evaluate specific chunker or all chunkers
    if args.chunker:
        print(f"Evaluating chunker: {args.chunker}")
        result = await evaluate_chunker(
            dataset_path=args.dataset,
            chunker_name=args.chunker,
            chunker_params=chunker_params,
            output_dir=args.output_dir
        )
        results = [result]
    else:
        print("Evaluating all chunkers")
        results = await evaluate_all_chunkers(
            dataset_path=args.dataset,
            output_dir=args.output_dir
        )
    
    # Generate comparison report if multiple results
    if len(results) > 1:
        visualizer = ResultsVisualizer(results_dir=args.output_dir)
        report_dir = visualizer.create_comparison_report(
            results,
            report_name="chunking_methods_comparison"
        )
        print(f"Comparison report generated: {report_dir}")
    
    print("\nEvaluation completed.")


if __name__ == "__main__":
    asyncio.run(main())