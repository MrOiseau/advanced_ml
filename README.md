# Advanced RAG System for Chunking Methods Research

This repository implements a comprehensive Retrieval-Augmented Generation (RAG) framework for evaluating how different document chunking strategies impact information retrieval quality. The system supports rigorous empirical analysis for research on optimal chunking methods in RAG applications.

## Overview

Modern RAG systems rely heavily on effective document chunking to create meaningful context windows for retrieval and generation. This framework enables systematic comparison of various chunking approaches through:

1. **Modular Implementation** of multiple chunking strategies
2. **Standardized Evaluation** using established NLP metrics
3. **Reproducible Experimentation** with configurable parameters
4. **Performance Optimization** for processing large document collections
5. **Comprehensive Visualization** of experimental results

## System Architecture

The framework consists of four main components that work together to enable systematic evaluation of chunking methods:

### Core Components

1. **Document Processing Pipeline**
   - PDF loading and parsing
   - Metadata extraction
   - Document preparation

2. **Chunking Methods**
   - Recursive character-based chunking
   - Semantic clustering
   - Sentence transformers
   - Topic-based chunking
   - Hierarchical chunking

3. **Indexing & Retrieval**
   - Vector database (ChromaDB)
   - Embedding models
   - Query processing

4. **Evaluation Framework**
   - Metrics calculation
   - Results analysis
   - Visualization tools

### Experiment Management

The system includes a comprehensive experiment management layer that handles:
- Experiment configuration
- Experiment execution
- Results storage
- Report generation

### Data Flow

```
Document Processing → Chunking → Vector Database → Query Pipeline
                                                        ↓
Report Generation ← Results Analysis ← Evaluation Framework
```

This architecture enables end-to-end evaluation of chunking methods, from document ingestion to final analysis.

## Key Features

### Advanced Chunking Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| **RecursiveCharacter** | Traditional approach that splits text by character count | `chunk_size`, `chunk_overlap` |
| **SemanticClustering** | Groups sentences by semantic similarity using embeddings and K-means | `max_chunk_size`, `min/max_clusters` |
| **SentenceTransformers** | Splits by sentences and merges similar ones based on embedding similarity | `max_chunk_size`, `similarity_threshold` |
| **TopicBased** | Groups content by topics using Latent Dirichlet Allocation | `num_topics`, `max_chunk_size` |
| **Hierarchical** | Creates a hierarchical structure of chunks based on document structure | `max_chunk_size`, `heading_patterns` |

### Comprehensive Evaluation Metrics

- **Semantic Quality**: Semantic similarity, ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L), BLEU score, F1 score
- **Retrieval Performance**: Precision@k, Recall@k, Mean Reciprocal Rank (MRR)
- **Efficiency**: Processing time, memory usage, chunk statistics

### Experiment Management

- **Configuration**: JSON-based experiment configuration with parameter customization
- **Execution**: Parallel experiment execution with progress tracking
- **Analysis**: Statistical analysis of results with significance testing
- **Visualization**: Automated generation of publication-quality charts and tables

### Performance Optimization

- **Batch Processing**: Optimized embedding generation with configurable batch sizes
- **Parallel Processing**: Multi-core utilization for document chunking
- **Local Embeddings**: Option to use local models for faster development cycles

## Project Structure

```
advanced_ml/
├── backend/
│   ├── chunkers/               # Chunking methods
│   │   ├── base.py             # Base chunker class and factory
│   │   ├── recursive_character.py
│   │   ├── semantic_clustering.py
│   │   ├── sentence_transformers.py
│   │   ├── topic_based.py
│   │   └── hierarchical.py
│   ├── evaluation/             # Evaluation framework
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── evaluator.py        # RAG evaluator
│   │   └── visualizer.py       # Results visualization
│   ├── experiments/            # Experiment management
│   │   ├── config.py           # Experiment configuration
│   │   ├── runner.py           # Experiment runner
│   │   └── results.py          # Results storage and analysis
│   ├── config.py               # System configuration
│   ├── indexing.py             # Document indexing
│   ├── querying.py             # Query pipeline
│   ├── rag_evaluation.py       # RAG evaluation script
│   └── utils.py                # Utility functions
├── data/
│   ├── pdfs/                   # Source documents
│   ├── db/                     # Vector database
│   ├── evaluation/             # Evaluation datasets and results
│   └── thesis/                 # Thesis reports and visualizations
├── frontend/
│   └── app.py                  # Streamlit web interface
├── prompts/                    # Prompt templates
├── scripts/
│   ├── run_experiments.py      # Script to run experiments
│   ├── generate_report.py      # Script to generate thesis report
│   └── visualize_results.py    # Script to create visualizations
├── .env                        # Environment variables
└── requirements.txt            # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd advanced_ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env_example` and add your API keys:
```bash
cp .env_example .env
# Edit .env to add your API keys
```

## Usage

The framework supports a complete research workflow from document ingestion to result analysis.

### Quick Start

```bash
# Set up environment
cp .env_example .env  # Edit to add your API keys

# Ingest documents with default settings
python -m backend.indexing

# Run default experiment with all chunking methods
python scripts/run_experiments.py default

# Generate comprehensive report
python scripts/generate_report.py

# Launch web interface
streamlit run frontend/app.py
```

### Core Workflows

#### 1. Document Ingestion

```python
from backend.indexing import IngestionPipeline
from backend.config import *

# Configure custom chunking method
pipeline = IngestionPipeline(
    pdf_dir=PDF_DIR,
    db_dir=DB_DIR,
    db_collection="custom_collection",
    chunker_name="semantic_clustering",
    chunker_params={
        "max_chunk_size": 200,
        "min_clusters": 2,
        "max_clusters": 10
    }
)
pipeline.run_pipeline()
```

#### 2. Experiment Configuration

```bash
# Create custom experiment
python scripts/run_experiments.py create \
    --name "semantic_comparison" \
    --chunkers semantic_clustering sentence_transformers \
    --dataset ./data/evaluation/custom_dataset.json

# Run experiment from configuration
python scripts/run_experiments.py from-config ./data/evaluation/configs/semantic_comparison.json
```

#### 3. Results Analysis

```bash
# Generate visualizations
python scripts/visualize_results.py --visualization-type radar heatmap bar

# Create comprehensive report
python scripts/generate_report.py \
    --results-dir ./data/evaluation/results/semantic_comparison \
    --output-dir ./data/thesis \
    --include-metrics semantic_similarity rouge-1 precision@3
```

## Extending the System

The framework is designed for extensibility, allowing researchers to add new components and customize existing ones.

### Adding a New Chunking Method

```python
# 1. Create a new chunker class in backend/chunkers/custom_chunker.py
from typing import List
from langchain.schema import Document
from backend.chunkers.base import BaseChunker

class CustomChunker(BaseChunker):
    """
    Custom chunking method that implements a novel approach.
    """
    def __init__(self, custom_param: int = 100, **kwargs):
        super().__init__(custom_param=custom_param, **kwargs)
        self.custom_param = custom_param
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        # Implement your chunking logic here
        chunks = []
        # Process documents according to your algorithm
        return chunks

# 2. Register the chunker in backend/chunkers/__init__.py
from backend.chunkers.custom_chunker import CustomChunker
ChunkerFactory.register("custom_chunker", CustomChunker)
```

### Adding a New Evaluation Metric

```python
# In backend/evaluation/metrics.py
def calculate_custom_metric(generated_answer: str, reference_answer: str) -> float:
    """
    Calculate a custom evaluation metric.
    
    Args:
        generated_answer (str): The answer generated by the system
        reference_answer (str): The reference (ground truth) answer
        
    Returns:
        float: The metric score
    """
    # Implement your metric calculation
    score = ...
    return score

# Update the calculate_all_metrics function
def calculate_all_metrics(...):
    # Add your metric to the results
    metrics['custom_metric'] = calculate_custom_metric(
        generated_answer, reference_answer
    )
    return metrics
```

## Performance Optimization

The framework implements several optimization techniques to efficiently process large document collections:

| Technique | Implementation | Performance Impact |
|-----------|----------------|-------------------|
| **Batch Processing** | Processes embeddings in configurable batches (default: 32) | 3-5x faster embedding generation |
| **Parallel Processing** | Uses `ProcessPoolExecutor` for CPU-bound chunking tasks | Near-linear scaling with CPU cores |
| **Local Embeddings** | Option to use HuggingFace models instead of API calls | Eliminates API latency during development |
| **Optimized Vector Storage** | ChromaDB with efficient indexing | Fast similarity search for large collections |

### Configuration Options

Performance can be tuned through environment variables and configuration parameters:

```python
# Environment variables
USE_LOCAL_EMBEDDINGS=true  # Use local embedding models
CHUNK_SIZE=1000            # Set default chunk size
CHUNK_OVERLAP=200          # Set default chunk overlap

# Runtime configuration
pipeline = IngestionPipeline(
    # ... other parameters ...
    chunker_params={
        "embedding_model_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Faster model
        "batch_size": 64  # Larger batch size for faster processing
    }
)
```

## Research Applications

This framework enables several types of research investigations:

1. **Comparative Analysis**: Systematically compare different chunking methods across various document types and retrieval tasks
2. **Parameter Optimization**: Identify optimal parameters for each chunking method based on document characteristics
3. **Domain-Specific Tuning**: Evaluate which chunking methods perform best for specific domains (legal, medical, technical, etc.)
4. **Efficiency Studies**: Analyze the trade-offs between retrieval quality and computational efficiency

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
@misc{AdvancedRAGSystem2025,
  author = {Your Name},
  title = {Advanced RAG System for Chunking Methods Research},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/advanced_ml}}
}
```

## Acknowledgments

This project was developed as part of a master thesis research on "The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems."
