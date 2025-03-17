"""
QASPER Evaluation module for the RAG system.

This module implements the QASPERChunkerEvaluator class, which evaluates
chunking methods using the QASPER dataset.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.schema import Document
from backend.chunkers.base import BaseChunker, ChunkingMethod, ChunkerFactory


class QASPERChunkerEvaluator:
    """
    Evaluation framework for chunking methods using the QASPER dataset.
    
    This class loads the QASPER dataset, evaluates different chunking methods,
    and provides metrics and visualizations for comparison.
    """
    
    def __init__(
        self,
        chunkers_to_evaluate: Dict[str, BaseChunker],
        max_samples: int = 100,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize the evaluator with chunkers to evaluate.
        
        Args:
            chunkers_to_evaluate: Dictionary mapping chunker names to chunker instances
            max_samples: Maximum number of samples to use from QASPER
            embedding_model_name: Model to use for semantic similarity calculations
        """
        self.chunkers = chunkers_to_evaluate
        self.max_samples = max_samples
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dataset = self._load_qasper_dataset()
        self.results = {}
    
    def _load_qasper_dataset(self, split="train"):
        """
        Load and prepare the QASPER dataset.
        
        Args:
            split: Dataset split to use
            
        Returns:
            Processed dataset
        """
        print(f"Loading QASPER dataset ({split} split)...")
        
        # Load dataset
        dataset = load_dataset("allenai/qasper", split=split)
        
        # Limit dataset size if needed
        if self.max_samples and self.max_samples < len(dataset):
            dataset = dataset.select(range(self.max_samples))
            print(f"Using {self.max_samples} samples from QASPER dataset")
        else:
            print(f"Using all {len(dataset)} samples from QASPER dataset")
        
        return dataset
    
    def evaluate_all_chunkers(self):
        """
        Evaluate all chunkers and store results.
        
        Returns:
            Dictionary mapping chunker names to evaluation results
        """
        for chunker_name, chunker in self.chunkers.items():
            print(f"Evaluating {chunker_name}...")
            self.results[chunker_name] = self._evaluate_chunker(chunker)
        
        return self.results
    
    def _evaluate_chunker(self, chunker: BaseChunker) -> Dict[str, Any]:
        """
        Evaluate a single chunker on the dataset.
        
        Args:
            chunker: Chunker instance to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Metrics to track
        metrics = {
            "retrieval_accuracy": [],
            "chronological_coherence": [],
            "semantic_coherence": [],
            "chunk_count": [],
            "avg_chunk_size": [],
            "processing_time": []
        }
        
        # Process each document in the dataset
        for i, sample in enumerate(self.dataset):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(self.dataset)}...")
                
            # Extract full text from the paper
            full_text = sample["full_text"]
            
            # Create a Document object
            doc = Document(page_content=full_text)
            
            # Measure chunking performance
            performance = chunker.measure_performance([doc])
            
            # Store basic metrics
            metrics["chunk_count"].append(performance["num_chunks"])
            metrics["avg_chunk_size"].append(performance["avg_chunk_size_words"])
            metrics["processing_time"].append(performance["processing_time_seconds"])
            
            # Get chunks
            chunks = chunker.chunk_documents([doc])
            
            # Evaluate retrieval accuracy using questions and answers
            retrieval_accuracy = self._evaluate_retrieval(chunks, sample)
            metrics["retrieval_accuracy"].append(retrieval_accuracy)
            
            # Evaluate chronological coherence
            chronological_coherence = self._evaluate_chronological_coherence(chunks, full_text)
            metrics["chronological_coherence"].append(chronological_coherence)
            
            # Evaluate semantic coherence
            semantic_coherence = self._evaluate_semantic_coherence(chunks)
            metrics["semantic_coherence"].append(semantic_coherence)
        
        # Calculate aggregate metrics
        result = {}
        for metric_name, values in metrics.items():
            result[f"avg_{metric_name}"] = np.mean(values)
            result[f"std_{metric_name}"] = np.std(values)
            result[f"min_{metric_name}"] = np.min(values)
            result[f"max_{metric_name}"] = np.max(values)
        
        return result
    
    def _evaluate_retrieval(self, chunks: List[Document], sample: Dict) -> float:
        """
        Evaluate retrieval accuracy using questions and answers from QASPER.
        
        Args:
            chunks: List of chunks
            sample: QASPER sample with questions and answers
            
        Returns:
            Retrieval accuracy score (0-1)
        """
        if not sample.get("qas") or not chunks:
            return 0.0
        
        # Extract questions and answers
        qas = sample["qas"]
        
        # Calculate retrieval accuracy for each question
        question_scores = []
        
        for qa in qas:
            question = qa.get("question", "")
            if not question:
                continue
                
            # Get the best answer if available
            answers = []
            for answer_obj in qa.get("answers", []):
                if answer_obj.get("answer") and answer_obj["answer"].get("unanswerable") is False:
                    if answer_obj["answer"].get("extractive_spans"):
                        answers.extend(answer_obj["answer"]["extractive_spans"])
                    elif answer_obj["answer"].get("free_form_answer"):
                        answers.append(answer_obj["answer"]["free_form_answer"])
            
            if not answers:
                continue
                
            # Check if any chunk contains the answer
            answer_found = False
            for answer in answers:
                for chunk in chunks:
                    if answer.lower() in chunk.page_content.lower():
                        answer_found = True
                        break
                if answer_found:
                    break
            
            question_scores.append(1.0 if answer_found else 0.0)
        
        # Return average score across all questions
        return np.mean(question_scores) if question_scores else 0.0
    
    def _evaluate_chronological_coherence(self, chunks: List[Document], original_text: str) -> float:
        """
        Evaluate how well chunks preserve the original document order.
        
        Args:
            chunks: List of chunks
            original_text: Original document text
            
        Returns:
            Chronological coherence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Extract chunk texts
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Calculate position of each chunk in the original text
        chunk_positions = []
        for chunk_text in chunk_texts:
            # Find the position of the first sentence of the chunk in the original text
            first_sentence = chunk_text.split('.')[0] + '.'
            position = original_text.find(first_sentence)
            if position == -1:  # If not found exactly, use a fuzzy approach
                # Try with the first 50 characters
                first_chars = chunk_text[:min(50, len(chunk_text))]
                position = original_text.find(first_chars)
                if position == -1:
                    # If still not found, use a very approximate position
                    position = 0
            chunk_positions.append(position)
        
        # Check if chunks are in chronological order
        is_ordered = all(chunk_positions[i] <= chunk_positions[i+1] for i in range(len(chunk_positions)-1))
        
        if is_ordered:
            return 1.0
        
        # Calculate degree of disorder
        inversions = 0
        for i in range(len(chunk_positions)):
            for j in range(i+1, len(chunk_positions)):
                if chunk_positions[i] > chunk_positions[j]:
                    inversions += 1
        
        max_inversions = (len(chunk_positions) * (len(chunk_positions) - 1)) // 2
        disorder_ratio = inversions / max_inversions if max_inversions > 0 else 0
        
        # Return coherence score (1 - disorder)
        return 1.0 - disorder_ratio
    
    def _evaluate_semantic_coherence(self, chunks: List[Document]) -> float:
        """
        Evaluate semantic coherence within chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Semantic coherence score (0-1)
        """
        if not chunks or len(chunks) == 1:
            return 1.0  # Perfect coherence for a single chunk
        
        # Calculate intra-chunk coherence
        chunk_coherence_scores = []
        
        for chunk in chunks:
            # Split chunk into sentences
            sentences = [s.strip() for s in chunk.page_content.split('.') if s.strip()]
            
            if len(sentences) <= 1:
                # Single sentence chunks are perfectly coherent
                chunk_coherence_scores.append(1.0)
                continue
            
            # Calculate embeddings for sentences
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Calculate average similarity between adjacent sentences
            adjacent_similarities = [similarities[i, i+1] for i in range(len(sentences)-1)]
            avg_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 1.0
            
            chunk_coherence_scores.append(avg_similarity)
        
        # Return average coherence across all chunks
        return np.mean(chunk_coherence_scores)
    
    def visualize_results(self, save_path=None):
        """
        Visualize evaluation results.
        
        Args:
            save_path: Path to save visualization
        """
        if not self.results:
            print("No results to visualize. Run evaluate_all_chunkers first.")
            return
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chunking Methods Evaluation Results', fontsize=16)
        
        # Extract metrics for visualization
        chunker_names = list(self.results.keys())
        retrieval_scores = [self.results[name]["avg_retrieval_accuracy"] for name in chunker_names]
        chronological_scores = [self.results[name]["avg_chronological_coherence"] for name in chunker_names]
        semantic_scores = [self.results[name]["avg_semantic_coherence"] for name in chunker_names]
        processing_times = [self.results[name]["avg_processing_time"] for name in chunker_names]
        chunk_counts = [self.results[name]["avg_chunk_count"] for name in chunker_names]
        
        # Plot retrieval accuracy
        axes[0, 0].bar(chunker_names, retrieval_scores)
        axes[0, 0].set_title('Retrieval Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_ylabel('Score (0-1)')
        axes[0, 0].set_xticklabels(chunker_names, rotation=45, ha='right')
        
        # Plot chronological coherence
        axes[0, 1].bar(chunker_names, chronological_scores)
        axes[0, 1].set_title('Chronological Coherence')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_ylabel('Score (0-1)')
        axes[0, 1].set_xticklabels(chunker_names, rotation=45, ha='right')
        
        # Plot semantic coherence
        axes[1, 0].bar(chunker_names, semantic_scores)
        axes[1, 0].set_title('Semantic Coherence')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel('Score (0-1)')
        axes[1, 0].set_xticklabels(chunker_names, rotation=45, ha='right')
        
        # Plot processing time vs. chunk count
        scatter = axes[1, 1].scatter(processing_times, chunk_counts, 
                                     s=100, alpha=0.7, 
                                     c=range(len(chunker_names)), 
                                     cmap='viridis')
        axes[1, 1].set_title('Processing Time vs. Chunk Count')
        axes[1, 1].set_xlabel('Processing Time (seconds)')
        axes[1, 1].set_ylabel('Average Chunk Count')
        
        # Add labels to scatter plot points
        for i, name in enumerate(chunker_names):
            axes[1, 1].annotate(name, 
                               (processing_times[i], chunk_counts[i]),
                               xytext=(5, 5),
                               textcoords='offset points')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self):
        """
        Generate a detailed report of evaluation results.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results available. Run evaluate_all_chunkers first."
        
        report = "# Chunking Methods Evaluation Report\n\n"
        report += f"Dataset: QASPER (samples: {self.max_samples})\n\n"
        
        # Overall performance table
        report += "## Overall Performance\n\n"
        report += "| Chunker | Retrieval Accuracy | Chronological Coherence | Semantic Coherence | Avg. Chunks | Avg. Chunk Size | Processing Time (s) |\n"
        report += "|---------|-------------------|-------------------------|-------------------|------------|----------------|---------------------|\n"
        
        for name, result in self.results.items():
            report += f"| {name} | {result['avg_retrieval_accuracy']:.3f} | {result['avg_chronological_coherence']:.3f} | "
            report += f"{result['avg_semantic_coherence']:.3f} | {result['avg_chunk_count']:.1f} | "
            report += f"{result['avg_chunk_size']:.1f} | {result['avg_processing_time']:.3f} |\n"
        
        # Detailed metrics
        report += "\n## Detailed Metrics\n\n"
        
        for name, result in self.results.items():
            report += f"### {name}\n\n"
            report += "| Metric | Average | Std Dev | Min | Max |\n"
            report += "|--------|---------|---------|-----|-----|\n"
            
            metrics = [
                ("Retrieval Accuracy", "retrieval_accuracy"),
                ("Chronological Coherence", "chronological_coherence"),
                ("Semantic Coherence", "semantic_coherence"),
                ("Chunk Count", "chunk_count"),
                ("Chunk Size (words)", "avg_chunk_size"),
                ("Processing Time (s)", "processing_time")
            ]
            
            for metric_name, metric_key in metrics:
                avg = result[f"avg_{metric_key}"]
                std = result[f"std_{metric_key}"]
                min_val = result[f"min_{metric_key}"]
                max_val = result[f"max_{metric_key}"]
                
                report += f"| {metric_name} | {avg:.3f} | {std:.3f} | {min_val:.3f} | {max_val:.3f} |\n"
            
            report += "\n"
        
        # Conclusions
        report += "## Conclusions\n\n"
        
        # Find best chunker for each metric
        best_retrieval = max(self.results.items(), key=lambda x: x[1]["avg_retrieval_accuracy"])
        best_chronological = max(self.results.items(), key=lambda x: x[1]["avg_chronological_coherence"])
        best_semantic = max(self.results.items(), key=lambda x: x[1]["avg_semantic_coherence"])
        fastest = min(self.results.items(), key=lambda x: x[1]["avg_processing_time"])
        
        report += f"- Best for retrieval accuracy: **{best_retrieval[0]}** ({best_retrieval[1]['avg_retrieval_accuracy']:.3f})\n"
        report += f"- Best for chronological coherence: **{best_chronological[0]}** ({best_chronological[1]['avg_chronological_coherence']:.3f})\n"
        report += f"- Best for semantic coherence: **{best_semantic[0]}** ({best_semantic[1]['avg_semantic_coherence']:.3f})\n"
        report += f"- Fastest processing: **{fastest[0]}** ({fastest[1]['avg_processing_time']:.3f}s)\n\n"
        
        # Overall recommendation
        # Simple scoring: 0.4 * retrieval + 0.3 * chronological + 0.2 * semantic + 0.1 * (1/processing_time)
        scores = {}
        for name, result in self.results.items():
            score = (
                0.4 * result["avg_retrieval_accuracy"] +
                0.3 * result["avg_chronological_coherence"] +
                0.2 * result["avg_semantic_coherence"] +
                0.1 * (1.0 / (result["avg_processing_time"] + 0.001))  # Avoid division by zero
            )
            scores[name] = score
        
        best_overall = max(scores.items(), key=lambda x: x[1])
        report += f"**Overall recommendation:** {best_overall[0]} (score: {best_overall[1]:.3f})\n"
        
        return report


def run_evaluation(output_dir="./data/evaluation"):
    """
    Run a full evaluation of all chunking methods.
    
    Args:
        output_dir: Directory to save results
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all registered chunkers
    chunkers = {}
    for name in ChunkerFactory.list_chunkers():
        chunkers[name] = ChunkerFactory.create(name)
    
    # Create evaluator
    evaluator = QASPERChunkerEvaluator(chunkers, max_samples=50)
    
    # Run evaluation
    results = evaluator.evaluate_all_chunkers()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"chunker_comparison_{timestamp}.png")
    evaluator.visualize_results(save_path=viz_path)
    
    # Generate and save report
    report = evaluator.generate_report()
    report_path = os.path.join(output_dir, f"chunker_evaluation_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    print(f"Report: {report_path}")
    print(f"Visualization: {viz_path}")
    
    return results


if __name__ == "__main__":
    run_evaluation()