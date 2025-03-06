"""
Configuration module for the RAG system.

This module should be imported before any other modules that might use tokenizers
to ensure environment variables are properly set.
"""
import os
from dotenv import load_dotenv

# Set tokenizers parallelism to false to avoid deadlocks with process forking
# This must be done before importing any libraries that use tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Constants for the application
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
DB_DIR = os.getenv("DB_DIR", "./data/db")
DB_COLLECTION = os.getenv("DB_COLLECTION", "rag_collection_advanced")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
SEARCH_RESULTS_NUM = int(os.getenv("SEARCH_RESULTS_NUM", "5"))
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "advanced_ml")
EVALUATION_DATASET = os.getenv("EVALUATION_DATASET", "./data/evaluation/evaluation_dataset_chatgpt_unique.json")
RESULTS_EVALUATION_AVERAGE_METRICS = os.getenv("RESULTS_EVALUATION_AVERAGE_METRICS", "./data/evaluation/results_evaluation_average_metrics.png")
ADVANCED_CHUNKING = os.getenv("ADVANCED_CHUNKING", "false").lower() == "true"

# Validate required environment variables
def validate_environment():
    """
    Validate that all required environment variables are set.
    Raises ValueError if any required variables are missing.
    """
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "CHAT_MODEL": CHAT_MODEL,
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")