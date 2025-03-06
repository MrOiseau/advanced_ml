# Import config first to ensure environment variables are set before other imports
from backend.config import *
import json
from typing import List, Dict, Any
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import asyncio
import matplotlib.pyplot as plt

from backend.querying import QueryPipeline


class RAGEvaluator:
    def __init__(self, dataset_path: str, query_pipeline: QueryPipeline) -> None:
        """
        Initializes the evaluator with a dataset and query pipeline.
        
        Args:
            dataset_path (str): Path to the evaluation dataset in JSON format.
            query_pipeline (QueryPipeline): Query pipeline used to retrieve documents and generate answers.
        """
        self.dataset = self.load_dataset(dataset_path)
        self.rouge = Rouge()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model once
        self.query_pipeline = query_pipeline

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Loads the evaluation dataset from a JSON file.
        """
        with open(dataset_path, 'r') as f:
            return json.load(f)

    def is_relevant(self, doc: Dict[str, Any], answer: str) -> bool:
        """
        Checks if a document is relevant to the answer using semantic similarity.
        
        This method uses sentence embeddings to compute the semantic similarity
        between the document content and the reference answer. Documents with
        similarity scores above a threshold are considered relevant.
        
        Args:
            doc: Document dictionary with 'content' key
            answer: Reference answer string
            
        Returns:
            bool: True if the document is semantically relevant to the answer
        """
        # Use a threshold for semantic similarity (0.4 is a reasonable starting point)
        SIMILARITY_THRESHOLD = 0.4
        
        try:
            # Generate embeddings for the document and answer
            doc_embedding = self.sentence_model.encode(doc['content'], convert_to_tensor=True)
            answer_embedding = self.sentence_model.encode(answer, convert_to_tensor=True)
            
            # Move tensors to CPU before converting to numpy arrays
            doc_embedding = doc_embedding.cpu().numpy()
            answer_embedding = answer_embedding.cpu().numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(doc_embedding.reshape(1, -1),
                                          answer_embedding.reshape(1, -1))[0][0]
            
            return similarity > SIMILARITY_THRESHOLD
        except Exception as e:
            # Fallback to word overlap method if there's an error
            print(f"Error in semantic similarity calculation: {e}. Falling back to word overlap.")
            answer_words = set(answer.lower().split())
            doc_words = set(doc['content'].lower().split())
            return len(answer_words.intersection(doc_words)) > 0

    def calculate_rouge(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Computes ROUGE scores between the generated and reference answers.
        
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap
        of n-grams between the generated and reference texts:
        - ROUGE-1: Overlap of unigrams (single words)
        - ROUGE-2: Overlap of bigrams (word pairs)
        - ROUGE-L: Longest Common Subsequence, measuring the longest matching sequence
        
        Higher scores indicate better overlap between generated and reference answers.
        """
        scores = self.rouge.get_scores(generated_answer, reference_answer)[0]
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }

    def calculate_bleu(self, generated_answer: str, reference_answer: str) -> float:
        """
        Computes the BLEU score between the generated and reference answers.
        
        BLEU (Bilingual Evaluation Understudy) is a precision-based metric that measures
        the n-gram overlap between the generated text and reference text. It was originally
        designed for machine translation evaluation but is also used for text generation tasks.
        
        The score ranges from 0 to 1, where:
        - 0 indicates no overlap
        - 1 indicates perfect overlap (identical texts)
        
        This implementation uses smoothing to handle cases where there are no n-gram matches.
        """
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu([reference_answer.split()], generated_answer.split(), smoothing_function=smoothing_function)

    def exact_match(self, generated_answer: str, reference_answer: str) -> int:
        """
        Computes the exact match score between the generated and reference answers.
        
        Exact Match (EM) is a binary metric that returns 1 if the generated answer
        exactly matches the reference answer after normalization (lowercasing and
        stripping whitespace), and 0 otherwise.
        
        This is a strict metric that requires perfect matching and doesn't account
        for partial correctness or semantic similarity.
        """
        return int(generated_answer.strip().lower() == reference_answer.strip().lower())

    def f1_score(self, generated_answer: str, reference_answer: str) -> float:
        """
        Computes the F1 score between the generated and reference answers.
        
        F1 score is the harmonic mean of precision and recall at the word level:
        - Precision: The fraction of words in the generated answer that are also in the reference answer
        - Recall: The fraction of words in the reference answer that are also in the generated answer
        - F1 = 2 * (precision * recall) / (precision + recall)
        
        This metric balances precision and recall, providing a measure of partial correctness
        that is more flexible than exact match. The score ranges from 0 to 1, where higher
        values indicate better performance.
        """
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        common = gen_tokens & ref_tokens
        if not common:
            return 0.0
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def semantic_similarity(self, generated_answer: str, reference_answer: str) -> float:
        """
        Computes the cosine similarity between sentence embeddings of the generated and reference answers.
        
        Semantic similarity uses sentence embeddings to capture the meaning of texts beyond
        simple word overlap. This method:
        1. Encodes both the generated and reference answers into dense vector representations
        2. Computes the cosine similarity between these vectors
        
        The score ranges from -1 to 1, where:
        - 1 indicates perfect semantic similarity (identical meaning)
        - 0 indicates no semantic relationship
        - -1 indicates opposite meanings (rarely occurs in practice)
        
        This metric is particularly valuable for evaluating RAG systems as it can detect
        when answers are semantically correct even if they use different wording.
        """
        gen_embedding = self.sentence_model.encode([generated_answer], convert_to_tensor=True)
        ref_embedding = self.sentence_model.encode([reference_answer], convert_to_tensor=True)

        # Move tensors to CPU before converting to numpy arrays
        gen_embedding = gen_embedding.cpu().numpy()
        ref_embedding = ref_embedding.cpu().numpy()

        return cosine_similarity(gen_embedding, ref_embedding)[0][0]

    async def retrieve_documents_async(self, question: str):
        """
        Asynchronously retrieves documents for a given query.
        """
        return await asyncio.to_thread(self.query_pipeline.retrieve_documents, question)
    
    async def evaluate_async(self) -> List[Dict[str, Any]]:
        """
        Asynchronously evaluates the query pipeline using the dataset and computes various metrics.
        """
        all_results = []
        questions = [item['question'] for item in self.dataset]
        reference_answers = [item['answer'] for item in self.dataset]

        # Fetch documents for all questions asynchronously
        tasks = [self.retrieve_documents_async(question) for question in questions]
        retrieved_docs_batch = await asyncio.gather(*tasks)
        
        # Generate answers and compute metrics
        for idx, (question, reference_answer, retrieved_docs) in enumerate(zip(questions, reference_answers, retrieved_docs_batch)):
            if not retrieved_docs:
                continue

            formatted_docs = [{'content': doc.page_content} for doc in retrieved_docs]
            generated_answer = self.query_pipeline.generate_summary(question, "\n".join([doc['content'] for doc in formatted_docs]))

            if not generated_answer:
                continue

            # Compute metrics
            results = {
                'question': question,
                'bleu': self.calculate_bleu(generated_answer, reference_answer),
                'exact_match': self.exact_match(generated_answer, reference_answer),
                'f1_score': self.f1_score(generated_answer, reference_answer),
                'semantic_similarity': self.semantic_similarity(generated_answer, reference_answer)
            }

            # Add ROUGE metrics
            rouge_scores = self.calculate_rouge(generated_answer, reference_answer)
            results.update(rouge_scores)

            all_results.append(results)

        return all_results

    def compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes the average of all evaluation metrics across all results.
        """
        metrics = ['bleu', 'exact_match', 'f1_score', 'semantic_similarity', 'rouge-1', 'rouge-2', 'rouge-l']
        averages = {metric: np.mean([result[metric] for result in results]) for metric in metrics}
        return averages

    def save_metrics_chart(self, average_metrics: Dict[str, float], output_path: str) -> None:
        """
        Save the average metrics as a bar chart.
        """
        metrics_names = list(average_metrics.keys())
        metrics_values = list(average_metrics.values())

        plt.figure(figsize=(10, 6))
        plt.barh(metrics_names, metrics_values, color='skyblue')
        plt.xlabel('Scores')
        plt.title('Average Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

async def main():
    # Environment variables are already loaded by config module
    try:
        # Validate environment variables
        validate_environment()
        
        # Initialize the query pipeline using constants from config
        query_pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT
        )
        
        # Create evaluator using constants from config
        evaluator = RAGEvaluator(EVALUATION_DATASET, query_pipeline)

        # Evaluate asynchronously
        results = await evaluator.evaluate_async()

        # Compute average metrics
        average_metrics = evaluator.compute_average_metrics(results)

        print("Average Metrics:")
        for metric, value in average_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics chart
        evaluator.save_metrics_chart(average_metrics, RESULTS_EVALUATION_AVERAGE_METRICS)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
