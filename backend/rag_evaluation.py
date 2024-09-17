import json
from typing import List, Dict, Any
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import asyncio
import matplotlib.pyplot as plt

from backend.querying_17_09_24 import QueryPipeline


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
        Checks if a document is relevant by comparing word overlaps with the answer.
        """
        answer_words = set(answer.lower().split())
        doc_words = set(doc['content'].lower().split())
        return len(answer_words.intersection(doc_words)) > 0

    def calculate_rouge(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Computes ROUGE scores between the generated and reference answers.
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
        """
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu([reference_answer.split()], generated_answer.split(), smoothing_function=smoothing_function)

    def exact_match(self, generated_answer: str, reference_answer: str) -> int:
        """
        Computes the exact match score between the generated and reference answers.
        """
        return int(generated_answer.strip().lower() == reference_answer.strip().lower())

    def f1_score(self, generated_answer: str, reference_answer: str) -> float:
        """
        Computes the F1 score between the generated and reference answers.
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
    import os

    # Load environment variables
    query_pipeline = QueryPipeline(
        db_dir=os.getenv("DB_DIR"),
        db_collection=os.getenv("DB_COLLECTION"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        chat_temperature=0.7,
        search_results_num=5,
        langsmith_project=os.getenv("LANGSMITH_PROJECT")
    )
    
    EVALUATION_DATASET = os.getenv("EVALUATION_DATASET")
    evaluator = RAGEvaluator(EVALUATION_DATASET, query_pipeline)

    # Evaluate asynchronously
    results = await evaluator.evaluate_async()

    # Compute average metrics
    average_metrics = evaluator.compute_average_metrics(results)

    print("Average Metrics:")
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics chart
    evaluator.save_metrics_chart(average_metrics, os.getenv("RESULTS_EVALUATION_AVERAGE_METRICS"))

if __name__ == "__main__":
    asyncio.run(main())
