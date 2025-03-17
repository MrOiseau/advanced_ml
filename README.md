
# Advanced ML Repository

This repository contains a Retrieval-Augmented Generation (RAG) system for document processing, indexing, and querying. It was developed as part of a university master thesis titled "The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems".

## Project Overview

The system implements various document chunking strategies and evaluates their impact on retrieval quality in RAG applications. It provides a framework for:

- Processing and indexing PDF documents using different chunking methods
- Querying the indexed documents with semantic search
- Comparing the effectiveness of different chunking strategies
- Evaluating retrieval quality with various metrics
- Visualizing results through a web interface

## System Architecture

The following diagram illustrates the end-to-end flow of how documents are processed, indexed, queried, and evaluated in the system:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│             │     │             │     │                     │     │             │
│  PDF Files  │────▶│  Parsing    │────▶│  Chunking Methods   │────▶│  Embedding  │
│             │     │  (PyPDF)    │     │                     │     │  Generation │
└─────────────┘     └─────────────┘     └─────────────────────┘     └─────────────┘
                                          │         │         │            │
                                          │         │         │            │
                                          ▼         ▼         ▼            ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│             │     │             │     │                     │     │             │
│  Evaluation │◀────│  Answer     │◀────│  Document Retrieval │◀────│  Vector     │
│  Framework  │     │  Generation │     │  & Reranking        │     │  Database   │
│             │     │             │     │                     │     │  (ChromaDB) │
└─────────────┘     └─────────────┘     └─────────────────────┘     └─────────────┘
       │                                          ▲                        │
       │                                          │                        │
       │                  ┌─────────────┐         │                        │
       │                  │             │         │                        │
       └─────────────────▶│  Metrics &  │         │                        │
                          │  Reports    │         │                        │
                          │             │         │                        │
                          └─────────────┘         │                        │
                                                  │                        │
                          ┌─────────────┐         │                        │
                          │             │         │                        │
                          │  User Query │─────────┘                        │
                          │             │                                  │
                          └─────────────┘                                  │
                                                                           │
                          ┌─────────────┐                                  │
                          │             │                                  │
                          │  Web UI     │◀─────────────────────────────────┘
                          │  (Streamlit)│
                          │             │
                          └─────────────┘
```

### Processing Flow

1. **Document Ingestion**: PDF files are loaded and parsed using PyPDF
   - Script: `backend/indexing.py`
   - Method: `IngestionPipeline.load_documents()`

2. **Chunking**: Documents are split into chunks using different strategies:
   - Script: `backend/chunkers/` directory
   - Methods: Each chunker implements `chunk_document()` method

3. **Vectorization**: Chunks are converted to vector embeddings using OpenAI or HuggingFace models
   - Script: `backend/indexing.py`
   - Method: `IngestionPipeline.index_documents()`

4. **Indexing**: Vectors are stored in ChromaDB collections (one per chunking method)
   - Script: `backend/indexing.py`
   - Method: `IngestionPipeline.index_documents()`

5. **Querying**: User queries are processed, expanded, and used to retrieve relevant documents
   - Script: `backend/querying.py`
   - Methods: `QueryPipeline.expand_query()`, `QueryPipeline.retrieve_documents()`

6. **Reranking**: Retrieved documents are reranked for relevance
   - Script: `backend/querying.py`
   - Method: `QueryPipeline._initialize_reranker()`

7. **Answer Generation**: LLM generates answers based on retrieved context
   - Script: `backend/querying.py`
   - Method: `QueryPipeline.generate_summary()`

8. **Evaluation**: System performance is measured using various metrics
   - Script: `backend/evaluation/evaluator.py`
   - Method: `RAGEvaluator.evaluate()`

9. **Visualization**: Results are presented through the Streamlit web interface
   - Script: `frontend/app.py`
   - Method: Main Streamlit application

### Chunking Methods in Detail

As this repository is focused on "The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems", the choice and implementation of chunking methods is critical. Here's a detailed overview of the implemented methods:

1. **Recursive Character Chunking** (`recursive_character.py`)
   - **Approach**: Splits text based on character count with overlap
   - **Strengths**: Simple, fast, and works with any text
   - **Weaknesses**: Ignores semantic boundaries, may split mid-sentence or paragraph
   - **Use case**: Baseline method, good for homogeneous text

2. **Semantic Clustering Chunking** (`semantic_clustering.py`)
   - **Approach**: Groups sentences by semantic similarity using embeddings
   - **Strengths**: Creates semantically coherent chunks
   - **Weaknesses**: Computationally intensive, may create uneven chunk sizes
   - **Use case**: Documents with varied content where semantic coherence is important

3. **Topic-Based Chunking** (`topic_based.py`)
   - **Approach**: Identifies topics in the text and chunks around them
   - **Strengths**: Preserves topical coherence, good for long documents
   - **Weaknesses**: Topic detection can be imprecise
   - **Use case**: Long documents with distinct topical sections

4. **Hierarchical Chunking** (`hierarchical.py`)
   - **Approach**: Respects document structure (headings, sections)
   - **Strengths**: Preserves document hierarchy, maintains context
   - **Weaknesses**: Depends on well-structured documents
   - **Use case**: Formal documents with clear section hierarchy

5. **Sentence Transformers Chunking** (`sentence_transformers.py`)
   - **Approach**: Uses sentence embeddings to determine chunk boundaries
   - **Strengths**: Balances semantic coherence with size constraints
   - **Weaknesses**: Requires good sentence boundary detection
   - **Use case**: General-purpose chunking with semantic awareness

#### Additional Chunking Methods to Consider

For future work, these additional chunking methods could enhance the system:

1. **Sliding Window with Discourse Markers**
   - Identifies discourse markers (e.g., "however", "therefore") to create better chunk boundaries
   - Preserves logical flow and argumentative structure

2. **Entity-Based Chunking**
   - Groups text around named entities or key concepts
   - Useful for information extraction and question answering about specific entities

3. **Multi-level Chunking**
   - Creates a hierarchy of chunks at different granularities
   - Allows for more flexible retrieval based on query complexity

4. **Adaptive Chunking**
   - Dynamically adjusts chunk size based on content density and complexity
   - Smaller chunks for dense, complex content; larger chunks for simpler content

5. **Cross-Document Chunking**
   - Creates chunks that span multiple documents when they discuss the same topic
   - Useful for synthesizing information across a corpus

6. **LLM-Guided Chunking**
   - Uses an LLM to identify natural semantic boundaries
   - Can create more human-like divisions of text

## Features
- **Multiple Chunking Methods**: Implements various document chunking strategies:
  - Recursive Character Chunking
  - Semantic Clustering Chunking (with time-sensitivity)
  - Topic-Based Chunking (using BERTopic)
  - Hierarchical Chunking
  - Sentence Transformers Chunking (with sliding window)
  - Hybrid Chunking (intelligent strategy selection)
  - Sentence Transformers Chunking

- **Vector Database Integration**: Uses ChromaDB for efficient vector storage and retrieval

- **Advanced Retrieval**: Implements query expansion and reranking for improved results

- **Evaluation Framework**: Comprehensive evaluation system with metrics like precision, recall, and more

- **Interactive UI**: Streamlit-based web interface for querying and comparing chunking methods

- **Experiment Management**: Tools for running and analyzing experiments across chunking methods

## Repository Structure

```
advanced_ml/
├── backend/                  # Core system components
│   ├── chunkers/             # Chunking methods implementation
│   │   ├── base.py           # Base chunker class
│   │   ├── hierarchical.py   # Hierarchical chunking
│   │   ├── recursive_character.py # Basic recursive character chunking
│   │   ├── semantic_clustering.py # Semantic clustering chunking
│   │   ├── sentence_transformers.py # Sentence transformers chunking
│   │   ├── topic_based.py    # Topic-based chunking
│   │   └── hybrid.py         # Hybrid chunking
│   ├── evaluation/           # Evaluation framework
│   │   ├── evaluator.py      # Main evaluation logic
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualizer.py     # Results visualization
│   ├── experiments/          # Experiment management
│   │   ├── config.py         # Experiment configuration
│   │   ├── results.py        # Results processing
│   │   └── runner.py         # Experiment runner
│   ├── config.py             # System configuration
│   ├── indexing.py           # Document indexing
│   ├── querying.py           # Query pipeline
│   └── utils.py              # Utility functions
├── data/                     # Data directory (gitignored)
│   ├── analysis/             # Analysis results and statistics
│   ├── db/                   # Vector database files
│   ├── evaluation/           # Evaluation datasets
│   ├── pdfs/                 # Source PDF documents
│   ├── experiments/          # Experiment results
│   └── visualizations/       # Data visualizations
├── notebooks/                # Jupyter notebooks for analysis
│   ├── README.md             # Notebooks documentation
│   └── qasper_dataset_analysis.ipynb # QASPER dataset analysis
├── frontend/                 # Web interface
│   └── app.py                # Streamlit application
├── prompts/                  # Prompt templates
│   ├── generate_evaluation_set.jinja2 # Evaluation set generation
│   ├── query_expansion.jinja2 # Query expansion
│   └── summarize.jinja2      # Document summarization
├── scripts/                  # Utility scripts
│   ├── create_collections.py # Create vector collections
│   ├── generate_report.py    # Generate evaluation reports
│   ├── run_experiments.py    # Run experiments
│   ├── visualize_results.py  # Visualize results
│   ├── evaluation/           # Evaluation scripts
│   │   ├── run_evaluation.py # Main evaluation script
│   │   └── run_full_evaluation.py # Full evaluation pipeline
│   └── utils/                # Utility scripts
│       └── excel_to_json_converter.py # Convert Excel to JSON
├── .env_example              # Example environment variables
├── .gitignore                # Git ignore file
├── code_organization_plan.md # Code organization documentation
├── LICENSE                   # MIT License
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── run_app.sh                # Script to run the application
```

## Installation and Setup

### Prerequisites

- Git
- Miniconda (recommended for environment management)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced_ml.git
   cd advanced_ml
   ```

2. Install Miniconda (if not already installed):

   **For Windows:**
   ```
   # Download the installer
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
   
   # Run the installer (GUI will open)
   # Follow the installation instructions in the GUI
   ```
   After installation, open Anaconda Prompt from the Start menu.

   **For macOS:**
   ```bash
   # Download the installer
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   
   # Make the installer executable
   chmod +x Miniconda3-latest-MacOSX-x86_64.sh
   
   # Run the installer
   ./Miniconda3-latest-MacOSX-x86_64.sh
   
   # Follow the prompts to complete installation
   # Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)
   ```

   **For Linux:**
   ```bash
   # Download the installer
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   
   # Make the installer executable
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   
   # Run the installer
   ./Miniconda3-latest-Linux-x86_64.sh
   
   # Follow the prompts to complete installation
   # Restart your terminal or run: source ~/.bashrc
   ```

3. Create and activate a Conda environment:
   ```bash
   # Create a new environment with Python 3.11.11
   conda create -n advanced_ml python=3.11.11
   
   # Activate the environment
   # On Windows (Anaconda Prompt):
   conda activate advanced_ml
   
   # On macOS/Linux:
   conda activate advanced_ml
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the provided `.env_example`:
   ```bash
   cp .env_example .env
   ```

5. Edit the `.env` file to add your API keys and customize settings:
   - Add your OpenAI API key
   - Add your LangSmith API key (optional, for tracing)
   - Adjust model settings if needed
   - Configure paths and other parameters

## Usage

### Running the Web Interface

The easiest way to use the system is through the web interface:

```bash
./run_app.sh
```

This script:
- Sets the correct PYTHONPATH
- Loads environment variables from the `.env` file
- Starts the Streamlit application

The web interface allows you to:
- Enter queries and retrieve relevant documents
- Compare results from different chunking methods side by side
- Filter results by document title
- View highlighted common content between different chunking methods

### Creating Vector Collections

Before querying, you need to create vector collections from your documents:

```bash
python scripts/create_collections.py
```

This script processes PDF documents using all available chunking methods and creates a separate vector collection for each method.

## Configuration

The system is configured through environment variables in the `.env` file:

### Core Settings
- `OPENAI_API_KEY`: Your OpenAI API key
- `CHAT_MODEL`: The OpenAI model to use for chat (default: "gpt-4o-mini")
- `EMBEDDING_MODEL`: The OpenAI model to use for embeddings (default: "text-embedding-3-small")

### Path Settings
- `PDF_DIR`: Directory containing PDF documents (default: "./data/pdfs")
- `DB_DIR`: Directory for vector database (default: "./data/db")
- `DB_COLLECTION`: Default collection name (default: "rag_collection_advanced")
- `DATA_DIR`: Base data directory (default: "./data")

### Chunking Settings
- `CHUNK_SIZE`: Default chunk size in characters (default: 1000)
- `CHUNK_OVERLAP`: Default chunk overlap in characters (default: 200)
- `ADVANCED_CHUNKING`: Use advanced chunking methods (default: true)

### LangSmith Settings (Optional)
- `LANGSMITH_API_KEY`: Your LangSmith API key
- `LANGSMITH_PROJECT`: LangSmith project name (default: "advanced_ml")
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing (default: true)
- `LANGCHAIN_ENDPOINT`: LangSmith API endpoint

### Performance Settings
- `USE_LOCAL_EMBEDDINGS`: Use local HuggingFace embeddings instead of OpenAI (default: false)
- `MIN_COMMON_WORDS`: Minimum consecutive words for highlighting common sequences (default: 5)

## Development

### Project Structure Overview

The project follows a modular structure:

- **Backend**: Core system components
  - `chunkers/`: Different chunking strategies
  - `evaluation/`: Evaluation framework
  - `experiments/`: Experiment management
  - `config.py`: System configuration
  - `indexing.py`: Document ingestion and indexing
  - `querying.py`: Query processing and retrieval

- **Frontend**: User interface
  - `app.py`: Streamlit application

- **Scripts**: Utility scripts for various tasks
  - `create_collections.py`: Create vector collections
  - `run_experiments.py`: Run experiments
  - `generate_report.py`: Generate evaluation reports

### Adding a New Chunking Method

To add a new chunking method:

1. Create a new file in `backend/chunkers/` (e.g., `new_chunker.py`)
2. Implement a class that inherits from `BaseChunker` in `backend/chunkers/base.py`
3. Implement the required methods, especially `chunk_document()`
4. Register your chunker in the chunker factory

Example:
```python
from backend.chunkers.base import BaseChunker

class NewChunker(BaseChunker):
    """New chunking method implementation."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(chunk_size, chunk_overlap)
        
    def chunk_document(self, document):
        # Implement your chunking logic here
        chunks = []
        # ...
        return chunks
```

## Evaluation

The system includes a comprehensive evaluation framework to assess the quality of different chunking methods.

### Running Evaluations

#### Using Excel Data

To run a full evaluation using Excel data:

```bash
python scripts/evaluation/run_full_evaluation.py --excel --excel_path data/your_evaluation_data.xlsx
```

This script:
1. Converts Excel evaluation data to JSON format
2. Runs evaluation on all chunking methods
3. Generates comparison reports

#### Using QASPER Dataset

To evaluate chunking methods using the standardized QASPER dataset:

```bash
python scripts/evaluation/run_full_evaluation.py --qasper --max_samples 50
```

This will:
1. Load the QASPER dataset from Hugging Face
2. Convert it to the format expected by the evaluator
3. Run evaluation on all chunking methods
4. Generate comparison reports

To evaluate a specific chunker (e.g., the hybrid chunker):

```bash
python scripts/evaluation/run_full_evaluation.py --qasper --chunker hybrid
```

The QASPER dataset provides a standardized benchmark for evaluating RAG systems, with academic papers and corresponding question-answer pairs.

### QASPER Dataset Analysis

For a detailed analysis of the QASPER dataset, check out the Jupyter notebook:

```bash
jupyter notebook notebooks/qasper_dataset_analysis.ipynb
```

This notebook provides comprehensive analysis of:
- Total number of documents, questions, and answers
- Ground truth retrieved chunks for each question
- Structure and characteristics of ground truth generated answers
- Statistical analysis and visualizations

The analysis results are saved in `data/analysis/qasper_dataset_stats.json` and visualizations in `data/visualizations/`.

### Generating Reports

To generate evaluation reports:

```bash
python scripts/generate_report.py
```

### Visualizing Results

To visualize evaluation results:

```bash
python scripts/visualize_results.py
```

### Available Chunking Methods

The system now includes the following chunking methods:

1. **Recursive Character Chunking** - Basic chunking by character count
2. **Semantic Clustering Chunking** - Groups sentences by semantic similarity with time-sensitivity
3. **Topic-Based Chunking** - Uses BERTopic for improved topic modeling
4. **Hierarchical Chunking** - Respects document structure
5. **Sentence Transformers Chunking** - Uses sliding window for better context
6. **Hybrid Chunking** - Intelligently selects the best strategy based on document characteristics

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

### Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and methods
- Add type hints where appropriate
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Nikola Bijanic**  
- Email: nikolabijanic@yahoo.com  
- GitHub: [@MrOiseau](https://github.com/MrOiseau)  
- LinkedIn: [Nikola Bijanic](https://www.linkedin.com/in/nikola-bijanic/)

## Acknowledgments

- This project was developed as part of a university master thesis
- Uses OpenAI models for embeddings and generation
- Built with LangChain, ChromaDB, and Streamlit