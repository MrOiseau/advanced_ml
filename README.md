
# AI Document Retrieval System

## Overview

This project implements an AI-powered document retrieval system using Python, LangChain, OpenAI models, and Streamlit. The system processes PDF documents, indexes them using vector embeddings, and provides a user-friendly interface for querying and retrieving relevant information. The system aims to improve retrieval-augmented generation (RAG) pipelines, combining document retrieval with large language models (LLMs) for enhanced accuracy in generated answers by accessing relevant documents efficiently.

## Features

- PDF document ingestion and indexing using ChromaDB
- Vector-based document retrieval for efficient search
- Query rewriting for improved search accuracy
- Document reranking with FlashrankRerank for relevance optimization
- Semantic document chunking using sentence embeddings and k-means clustering
- Summarization of retrieved documents
- Streamlit-based web interface for easy interaction
- Features like query rewriting and user feedback integration using LangSmith

## Components

### Backend

1. **`config.py`**: Centralizes all configuration settings and environment variables for the application. It loads variables from the .env file and provides them to other modules, ensuring consistent configuration across the system.
2. **`indexing.py`**: Handles PDF document ingestion and indexing into ChromaDB. It processes documents into chunks, generates vector embeddings, and stores them in the database for efficient retrieval.
3. **`querying.py`**: Manages query processing, query rewriting, document retrieval, reranking, and summarization. It uses vector embeddings to retrieve documents and applies query expansion and reranking for better results.
4. **`chunker_advanced.py`**: Implements semantic text chunking using sentence embeddings and k-means clustering for more intelligent document splitting. This module:
   - Uses sentence-transformers to create embeddings for each sentence
   - Determines the optimal number of clusters using silhouette scoring
   - Groups semantically similar sentences into coherent chunks
   - Respects maximum chunk size constraints while preserving semantic relationships
   - Provides more context-aware document splitting compared to traditional character-based methods
5. **`rag_evaluation.py`**: Provides comprehensive evaluation metrics (ROUGE, BLEU, F1, semantic similarity) and visualization of system performance.
The evaluation process:
   - Loading the evaluation dataset
   - Processing each question
   - Retrieving relevant documents
   - Generating answers
   - Computing evaluation metrics
   - Outputting average metrics across all questions
Evaluation Metrics:
   - BLEU (precision-based overlap)
   - Exact Match (strict matching)
   - F1 Score (balance of precision and recall)
   - Semantic Similarity (meaning-based comparison)
   - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
6. **`utils.py`**: Contains utility functions for logging and generating a deduplicated evaluation dataset.

### Frontend

- **`app.py`**: Implements the Streamlit web interface for user interactions. It allows users to input queries, apply filters, and view summarized results and retrieved documents.
  
  ![Streamlit App Demo](data/demo_files/streamlit_app.gif)

### Prompts

- **`query_expansion.jinja2`**: Template for query expansion to improve the precision of searches.
- **`summarize.jinja2`**: Template for generating concise summaries of retrieved documents.
- **`generate_evaluation_set.jinja2`**: Template for creating an evaluation dataset by extracting questions and answers from PDF documents using GPT-4o.

## Setup

### Prerequisites

Before running the system, ensure you have:
- Git installed
- Conda/Miniconda installed (strongly recommended for cross-platform compatibility)
- An OpenAI API key
- A LangSmith API key (optional, for tracking and evaluation)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/MrOiseau/advanced_ml.git
   cd advanced_ml
   ```

2. **Create and activate a Conda environment** (strongly recommended for ensuring compatibility across different operating systems and Python versions):
   
   ```bash
   # For Windows, macOS, and Linux
   conda create -n advanced_ml python=3.11.11 --y
   ```
   
   **Activate the environment:**
   
   ```bash
   # For Windows/macOS/Linux
   conda activate advanced_ml

   
   > **Note:** Using Conda ensures that your project runs consistently regardless of the user's OS or Python version. This project specifically requires Python 3.11.11 for optimal compatibility with all dependencies. If you don't have Conda installed, you can download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
   >
   > **Alternative:** If you prefer not to use Conda, you can use a virtual environment with:
   > ```bash
   > # For Windows
   > python -m venv venv
   > venv\Scripts\activate
   >
   > # For macOS/Linux
   > python -m venv venv
   > source venv/bin/activate
   > ```

3. Install dependencies:
   ```bash
   # For Windows, macOS, and Linux
   pip install -r requirements.txt
   ```
   
   > **Note:** This will install all required packages listed in requirements.txt. If you encounter any issues, try upgrading pip first with `pip install --upgrade pip`.

4. **Set up environment variables** by copying `.env_example` to `.env`:
   
   ```bash
   # For Windows
   copy .env_example .env
   
   # For macOS/Linux
   cp .env_example .env
   ```
   
   Edit the `.env` file and add your values for these essential environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key (required)
   - `LANGSMITH_API_KEY`: Your LangSmith API key (optional, for tracking and evaluation)
   - `DB_DIR`: Directory for ChromaDB storage (default: "./data/db")
   - `DB_COLLECTION`: Name for the ChromaDB collection (default: "rag_collection_advanced")
   - `EMBEDDING_MODEL`: OpenAI embedding model name (default: "text-embedding-3-small")
   - `CHAT_MODEL`: OpenAI chat model name (default: "gpt-4o-mini")
   - `ADVANCED_CHUNKING`: Set to "true" to enable semantic chunking (default: true)
   - `CHUNK_SIZE`: Size of text chunks (default: 1000)
   - `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
   
   > **Important:** The `.env` file contains sensitive information like API keys. Never commit this file to version control. The `.env_example` file is included in the repository as a template, but the actual `.env` file is listed in `.gitignore`.

5. **Create the vector index** by processing and indexing PDF documents:
   ```bash
   # Make sure you're in the project root directory and your environment is activated
   python backend/indexing.py
   ```
   
   > **Note:** This step will process all PDF documents located in the `data/pdfs` folder and create a vector index using ChromaDB. The process may take some time depending on the number and size of your PDF files.
   >
   > **Troubleshooting:** If you encounter a "ModuleNotFoundError", ensure your PYTHONPATH is set correctly. The .env file includes `PYTHONPATH=$PYTHONPATH:`pwd`` which should handle this, but if you're on Windows, you might need to set it manually or use:
   > ```bash
   > # For Windows
   > set PYTHONPATH=%PYTHONPATH%;%cd%
   > ```

6. **Start the Streamlit app**:
   ```bash
   # Make sure your environment is activated
   streamlit run frontend/app.py
   ```

7. Open the provided URL in your web browser to access the AI Document Retrieval Interface.
   > **Note:** By default, Streamlit will run on http://localhost:8501

## Usage

### Querying

- Enter a query in the search box and hit enter.
- The system will retrieve relevant documents, rank them by relevance, and display summarized results.
- You can apply filters and provide feedback on the relevance of the results via the sidebar.

### Evaluation

To evaluate the system:

1. Run the evaluation script:
   ```bash
   python backend/rag_evaluation.py
   ```

2. The evaluation results will include metrics like precision@k, MRR, BLEU, F1 score, ROUGE, and semantic similarity.

## Evaluation and Results

The system is evaluated using a dataset stored in `data/evaluation/evaluation_dataset.json`. Below are the metrics used in evaluation:

### Metrics Used

- **Precision@k**: Measures how many of the top k retrieved documents are relevant.
- **Mean Reciprocal Rank (MRR)**: Evaluates how early the first relevant document appears in the ranked list.
- **BLEU**: Measures n-gram overlap between generated and reference answers.
- **Exact Match (EM)**: Evaluates whether the generated answer exactly matches the reference answer.
- **F1 Score**: Balances precision and recall at the word level between generated and reference answers.
- **Semantic Similarity**: Uses sentence embeddings to evaluate how semantically close the generated answer is to the reference answer.
- **ROUGE-1, ROUGE-2, ROUGE-L**: Measures unigram, bigram, and longest common subsequence overlap between generated and reference answers.

### Example Results

Example evaluation result for the question: "What is the purpose of multi-head attention in transformers?"

```json
{
  "question": "What is the purpose of multi-head attention in transformers?",
  "precision@1": 1.0,
  "precision@3": 1.0,
  "precision@5": 0.6,
  "mrr": 1.0,
  "bleu": 0.0011,
  "exact_match": 0,
  "f1_score": 0.0462,
  "semantic_similarity": 0.5160,
  "rouge-1": 0.0822,
  "rouge-2": 0.0,
  "rouge-l": 0.0822
}
```

### Average Metrics Across All Questions

![Results Evaluation Average Metrics](data/evaluation/results_evaluation_average_metrics.png)

**Note**: The dataset was not created properly, and "context" was not exactly cut text from PDFs. This will be fixed in future versions.

## Evaluation Dataset Creation

The evaluation dataset is created by generating questions, context, and answers from the PDF documents using large language models. This process ensures comprehensive coverage of the document content while maintaining high-quality question-answer pairs for testing the system's performance.

### Steps to Create the Evaluation Dataset

1. **Document Processing**: Process the PDF documents to extract meaningful content.
2. **Dataset Generation**: Use language models to generate diverse questions and answers from the document content.
3. **Dataset Formatting**: Structure the generated dataset into a JSON format with unique IDs, questions, context, and answers.
4. **Quality Verification**: Ensure the generated pairs accurately reflect the document content and maintain high quality.

Example dataset entry:

```json
{
  "id": 1,
  "question": "What is the main goal of the Rewrite-Retrieve-Read framework introduced in the paper?",
  "context": ["This work introduces a new framework, Rewrite-Retrieve-Read, which aims to improve retrieval-augmented large language models by focusing on query rewriting rather than just adapting the retriever or reader components."],
  "answer": "The main goal of the Rewrite-Retrieve-Read framework is to improve retrieval-augmented LLMs by focusing on query rewriting."
}
```

## Advanced Features

- **Query Expansion**: Improves the precision of document retrieval by expanding user queries.
- **Document Reranking**: Enhances the relevance of retrieved documents using a reranking algorithm.
- **User Feedback Integration**: Collects feedback through LangSmith for continuous improvement of the retrieval and summarization pipeline.

## Logging

Logging is configured via the `utils.py` file, using Python's logging module. The logs can be customized for different levels (INFO, DEBUG, ERROR).

## Contributing

Contributions are welcome! Please follow the code style, add tests for new features, and ensure proper documentation.

## License

This project is licensed under the MIT License. See the [LICENSE file](LICENSE) for details.

## Troubleshooting

If you encounter issues while setting up or running the project, here are some common solutions:

### Environment Setup Issues

- **ModuleNotFoundError**: If you encounter module import errors, ensure your PYTHONPATH is set correctly:
  ```bash
  # For macOS/Linux
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  
  # For Windows
  set PYTHONPATH=%PYTHONPATH%;%cd%
  ```

- **Conda environment issues**: If you have problems with the Conda environment:
  ```bash
  # List all environments to verify creation
  conda env list
  
  # Remove and recreate if necessary
  conda remove --name advanced_ml --all
  conda create -n advanced_ml python=3.11.11
  ```

### API and Environment Variable Issues

- **API Key errors**: Ensure your OpenAI API key is correctly set in the `.env` file and has sufficient credits
- **Environment variables not loading**: Verify that python-dotenv is installed and your `.env` file is in the project root

### Database and Indexing Issues

- **ChromaDB errors**: If you encounter issues with the vector database:
  ```bash
  # Remove the existing database and reindex
  rm -rf ./data/db
  python backend/indexing.py
  ```

- **PDF processing errors**: Ensure your PDF files are valid and readable

### Streamlit App Issues

- **Streamlit not starting**: Try running with the debug flag:
  ```bash
  streamlit run --debug frontend/app.py
  ```

- **Browser not opening automatically**: Manually navigate to http://localhost:8501
