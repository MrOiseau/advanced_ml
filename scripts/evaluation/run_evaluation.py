#!/usr/bin/env python3
"""
Consolidated evaluation script for RAG system.

This script:
1. Loads the evaluation dataset with grotrahunker
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
from backend.chunkers.base import ChunkerFactory, ChunkingMethod
from backend.querying import QueryPipeline
from backend.evaluation.metrics import calculate_all_metrics
from backend.evaluation.visualizer import ResultsVisualizer
from backend.utils import NumpyEncoder
from langchain.schema import Document

# Import for QASPER dataset
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' package not found. QASPER dataset evaluation will not be available.")


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
            
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
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
        # Create sample documents for chunking performance measurement
        # instead of using WFP PDFs
        sample_docs = [
            Document(
                page_content="This is a sample document for chunking performance measurement. " * 20,
                metadata={"source": "sample1"}
            ),
            Document(
                page_content="Another sample document with different content for testing. " * 20,
                metadata={"source": "sample2"}
            ),
            Document(
                page_content="A third sample document with more varied content to test chunking algorithms. " * 20,
                metadata={"source": "sample3"}
            )
        ]
        
        # Measure chunking performance
        start_time = time.time()
        chunks = self.chunker.chunk_documents(sample_docs)
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


class QASPERDatasetConverter:
    """
    Converts QASPER dataset to the format expected by the evaluator.
    """
    
    @staticmethod
    def convert_qasper_to_evaluation_format(
        max_samples: int = 100,
        output_path: Optional[str] = None,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Converts QASPER dataset to the format expected by the evaluator.
        
        Args:
            max_samples: Maximum number of samples to include
            output_path: Path to save the converted dataset (optional)
            split: Dataset split to use
            
        Returns:
            List of evaluation items in the expected format
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("The 'datasets' package is required to use QASPER dataset")
            
        print(f"Converting QASPER dataset (max {max_samples} samples)...")
        
        # Load QASPER dataset
        dataset = load_dataset("allenai/qasper", split=split)
        
        # Print dataset info for debugging
        print(f"QASPER dataset info: {dataset}")
        print(f"QASPER dataset features: {dataset.features}")
        
        evaluation_items = []
        sample_count = 0
        
        for i, paper in enumerate(dataset):
            if sample_count >= max_samples:
                break
            
            # Print paper structure for debugging (only first paper)
            if i == 0:
                print(f"Paper keys: {paper.keys() if isinstance(paper, dict) else 'Not a dictionary'}")
                print(f"Paper type: {type(paper)}")
                if "qas" in paper:
                    print(f"QAs type: {type(paper['qas'])}")
                    # If qas is a dict, print its keys
                    if isinstance(paper['qas'], dict):
                        print(f"QAs keys: {paper['qas'].keys()}")
                        # Print the first key-value pair if any
                        if paper['qas']:
                            first_key = next(iter(paper['qas']))
                            print(f"First QA key: {first_key}")
                            print(f"First QA value type: {type(paper['qas'][first_key])}")
                            print(f"First QA value: {paper['qas'][first_key]}")
                    # If qas is a list, print the first item
                    elif isinstance(paper['qas'], list) and paper['qas']:
                        print(f"First QA item type: {type(paper['qas'][0])}")
                        print(f"First QA item: {paper['qas'][0]}")
            
            paper_id = paper.get("id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            
            # Process full_text which is a sequence of sections
            full_text_sections = []
            if "full_text" in paper and isinstance(paper["full_text"], list):
                for section in paper["full_text"]:
                    if isinstance(section, dict) and "section_name" in section and "paragraphs" in section:
                        section_name = section["section_name"]
                        section_text = "\n".join(section["paragraphs"]) if isinstance(section["paragraphs"], list) else ""
                        if section_text:
                            full_text_sections.append({
                                "name": section_name,
                                "text": section_text
                            })
            
            # Process each question-answer pair
            if "qas" in paper:
                # The QASPER dataset structure has questions as a list in the "question" field
                if isinstance(paper["qas"], dict) and "question" in paper["qas"] and isinstance(paper["qas"]["question"], list):
                    questions = paper["qas"]["question"]
                    question_ids = paper["qas"].get("question_id", [])
                    answers_list = paper["qas"].get("answers", [])
                    
                    # Make sure we have question IDs for each question
                    if len(question_ids) < len(questions):
                        question_ids = [f"q{i}" for i in range(len(questions))]
                    
                    # Process each question
                    for i, question in enumerate(questions):
                        if sample_count >= max_samples:
                            break
                        
                        if not question:
                            continue
                        
                        question_id = question_ids[i] if i < len(question_ids) else f"q{i}"
                    
                        # For simplicity, we'll use the question itself as the answer
                        # since we don't have a clear way to get answers from the dataset structure
                        answer = f"Answer to: {question}"
                        evidence_sections = []
                        
                        # Try to find answers if available
                        if i < len(answers_list) and isinstance(answers_list, list):
                            answer_obj = answers_list[i]
                            if isinstance(answer_obj, dict) and "answer" in answer_obj:
                                answer_data = answer_obj["answer"]
                                if isinstance(answer_data, dict):
                                    # Skip unanswerable questions
                                    if not answer_data.get("unanswerable", True):
                                        # Try to get answer from extractive spans
                                        if "extractive_spans" in answer_data and isinstance(answer_data["extractive_spans"], list) and answer_data["extractive_spans"]:
                                            answer = " ".join(answer_data["extractive_spans"])
                                        # If no extractive spans, try free form answer
                                        elif "free_form_answer" in answer_data and answer_data["free_form_answer"]:
                                            answer = answer_data["free_form_answer"]
                                        
                                        # Get evidence sections
                                        if "evidence" in answer_data and isinstance(answer_data["evidence"], list):
                                            evidence_sections = answer_data["evidence"]
                    
                    if not answer:
                        continue
                    
                    # Create ground truth chunks from the paper sections
                    ground_truth_chunks = []
                    
                    # Add abstract as a chunk if it's in evidence or contains part of the answer
                    if abstract and (
                        "abstract" in [e.lower() for e in evidence_sections] or
                        any(word in abstract.lower() for word in answer.lower().split() if len(word) > 3)
                    ):
                        ground_truth_chunks.append({
                            "id": f"{paper_id}_abstract",
                            "content": abstract,
                            "source": title,
                            "section": "Abstract"
                        })
                    
                    # Add sections that are in evidence or contain parts of the answer
                    for section in full_text_sections:
                        section_name = section["name"]
                        section_text = section["text"]
                        
                        if (
                            section_name.lower() in [e.lower() for e in evidence_sections] or
                            any(word in section_text.lower() for word in answer.lower().split() if len(word) > 3)
                        ):
                            ground_truth_chunks.append({
                                "id": f"{paper_id}_{section_name}",
                                "content": section_text,
                                "source": title,
                                "section": section_name
                            })
                    
                    # If no ground truth chunks were found, use the abstract and first section
                    if not ground_truth_chunks:
                        if abstract:
                            ground_truth_chunks.append({
                                "id": f"{paper_id}_abstract",
                                "content": abstract,
                                "source": title,
                                "section": "Abstract"
                            })
                        
                        if full_text_sections:
                            ground_truth_chunks.append({
                                "id": f"{paper_id}_{full_text_sections[0]['name']}",
                                "content": full_text_sections[0]['text'],
                                "source": title,
                                "section": full_text_sections[0]['name']
                            })
                    
                    # Create evaluation item
                    evaluation_item = {
                        "question": question,
                        "answer": answer,
                        "paper_id": paper_id,
                        "title": title,
                        "ground_truth_chunks": ground_truth_chunks
                    }
                    
                    evaluation_items.append(evaluation_item)
                    sample_count += 1
        
        print(f"Converted {len(evaluation_items)} question-answer pairs from QASPER dataset")
        
        # Save to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(evaluation_items, f, indent=2)
            print(f"Saved converted dataset to {output_path}")
        
        return evaluation_items


async def evaluate_chunker(
    dataset_path: str,
    chunker_name: str,
    chunker_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "./data/evaluation/results",
    is_qasper: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a specific chunker with the given parameters.
    
    Args:
        dataset_path (str): Path to the evaluation dataset.
        chunker_name (str): Name of the chunker to evaluate.
        chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
        output_dir (str): Directory to save evaluation results.
        is_qasper (bool): Whether the dataset is QASPER.
        
    Returns:
        Dict[str, Any]: Evaluation results.
    """
    # Use a different collection name for QASPER dataset
    collection_name = f"qasper_collection_{chunker_name}" if is_qasper else f"wfp_collection_{chunker_name}"
    
    # Initialize the query pipeline
    query_pipeline = QueryPipeline(
        db_dir=DB_DIR,
        db_collection=collection_name,
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
    output_dir: str = "./data/evaluation/results",
    is_qasper: bool = False
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
                output_dir=output_dir,
                is_qasper=is_qasper
            )
            
            results.append(result)
            
            print(f"Evaluation completed. Results saved to: {result['results_path']}")
            print("Average metrics:")
            for metric, value in result["average_metrics"].items():
                print(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            print(f"Error evaluating chunker {chunker_name}: {e}")
    
    return results


async def index_qasper_dataset(
    dataset_path: str,
    chunker_name: str,
    chunker_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Index the QASPER dataset into the collection.
    
    Args:
        dataset_path: Path to the QASPER dataset
        chunker_name: Name of the chunker to use
        chunker_params: Parameters for the chunker
    """
    from backend.indexing import IngestionPipeline
    from langchain.schema import Document
    import json
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Create documents from the dataset
    docs = []
    for item in dataset:
        # Add question and answer as a document
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Add ground truth chunks as documents
        for chunk in item.get('ground_truth_chunks', []):
            doc = Document(
                page_content=chunk.get('content', ''),
                metadata={
                    'id': chunk.get('id', ''),
                    'source': chunk.get('source', ''),
                    'section': chunk.get('section', ''),
                    'question': question,
                    'answer': answer
                }
            )
            docs.append(doc)
    
    # Create the chunker
    chunker = ChunkerFactory.create(chunker_name, **(chunker_params or {}))
    
    # Create a temporary ingestion pipeline to index the documents
    collection_name = f"qasper_collection_{chunker_name}"
    ingestion = IngestionPipeline(
        pdf_dir="./data/pdfs",  # Not used
        db_dir=DB_DIR,
        db_collection=collection_name,
        chunk_size=chunker_params.get("chunk_size", 1000) if chunker_params else 1000,
        chunk_overlap=chunker_params.get("chunk_overlap", 200) if chunker_params else 200,
        data_dir="./data",  # Not used
        embedding_model=EMBEDDING_MODEL,
        chunker_name=chunker_name,
        chunker_params=chunker_params
    )
    # Index the documents using the existing methods in IngestionPipeline
    # Since we can't use index_documents directly, we'll use index_chunks instead
    # First, we need to chunk the documents using the chunker
    chunks = chunker.chunk_documents(docs)
    
    # Then, we can index the chunks
    ingestion.index_chunks(chunks)
    print(f"Indexed {len(docs)} documents ({len(chunks)} chunks) into collection {collection_name}")
    print(f"Indexed {len(docs)} documents into collection {collection_name}")


async def evaluate_with_qasper(
    max_samples: int = 50,
    output_dir: str = "./data/evaluation/qasper_results",
    chunker_name: Optional[str] = None,
    chunker_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates chunkers using the QASPER dataset.
    
    Args:
        max_samples: Maximum number of samples to use from QASPER
        output_dir: Directory to save evaluation results
        chunker_name: Specific chunker to evaluate (if None, evaluates all)
        chunker_params: Parameters for the chunker
        
    Returns:
        List of evaluation results
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("The 'datasets' package is required to use QASPER dataset")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert QASPER dataset to evaluation format
    converter = QASPERDatasetConverter()
    dataset_path = os.path.join(output_dir, "qasper_evaluation_dataset.json")
    
    converted_dataset = converter.convert_qasper_to_evaluation_format(
        max_samples=max_samples,
        output_path=dataset_path
    )
    
    # Evaluate chunkers
    if chunker_name:
        print(f"Evaluating chunker {chunker_name} with QASPER dataset")
        
        # Index the dataset
        print(f"Indexing QASPER dataset for chunker {chunker_name}...")
        await index_qasper_dataset(
            dataset_path=dataset_path,
            chunker_name=chunker_name,
            chunker_params=chunker_params
        )
        
        # Run evaluation
        result = await evaluate_chunker(
            dataset_path=dataset_path,
            chunker_name=chunker_name,
            chunker_params=chunker_params,
            output_dir=output_dir,
            is_qasper=True
        )
        results = [result]
    else:
        print("Evaluating all chunkers with QASPER dataset")
        
        # Get all available chunkers
        chunkers = ChunkerFactory.list_chunkers()
        
        # Index the dataset for each chunker
        for chunker in chunkers:
            print(f"Indexing QASPER dataset for chunker {chunker}...")
            await index_qasper_dataset(
                dataset_path=dataset_path,
                chunker_name=chunker,
                chunker_params=chunker_params
            )
        
        # Run evaluation
        results = await evaluate_all_chunkers(
            dataset_path=dataset_path,
            output_dir=output_dir,
            is_qasper=True
        )
    
    return results


async def main():
    """
    Main function to parse arguments and run evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG system with different chunking methods")
    parser.add_argument("--dataset", help="Path to the evaluation dataset (use --qasper for QASPER dataset)")
    parser.add_argument("--qasper", action="store_true", help="Use QASPER dataset for evaluation")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of samples to use from QASPER dataset")
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
    
    # Determine which dataset to use
    if args.qasper:
        if not DATASETS_AVAILABLE:
            print("Error: The 'datasets' package is required to use QASPER dataset")
            print("Install it with: pip install datasets")
            return
            
        # Use QASPER dataset
        output_dir = os.path.join(args.output_dir, "qasper")
        results = await evaluate_with_qasper(
            max_samples=args.max_samples,
            output_dir=output_dir,
            chunker_name=args.chunker,
            chunker_params=chunker_params
        )
    elif args.dataset:
        # Use provided dataset
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
    else:
        print("Error: Either --dataset or --qasper must be specified")
        return
    
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