# Document Ingestion Pipeline

This project contains a modular document ingestion pipeline to parse, chunk, enrich with metadata, and index PDF documents into a ChromaDB vector store. All configurations are handled via environment variables, ensuring that sensitive details are abstracted away.

## Features
- **PDF Parsing**: Automatically parse PDF documents from the specified directory.
- **Document Chunking**: Chunk the documents into smaller, more manageable pieces for efficient indexing.
- **Metadata Enrichment**: Add metadata such as document titles to each chunk.
- **Vector Store Indexing**: Index the chunks using OpenAI embeddings and store them in ChromaDB.

## Project Structure
- `backend/ingestion_pipeline.py`: Contains the main pipeline class that handles the document ingestion process, including parsing, chunking, metadata enrichment, and indexing.
- `index.py`: The main script that runs the ingestion pipeline, fetching settings and configuration from environment variables.

## Requirements
- Python 3.11.10
- The following Python packages:
  - `langchain`
  - `openai`
  - `chromadb`
  - `python-dotenv`
  - `requests`
  - `pandas`
  - `pickle`
  - `openpyxl`

## Installation

1. Open terminal & Clone the repository. Then open cloned project in VSCode:
    ```bash
    git clone git@github.com:MrOiseau/advanced_ml.git
    cd advanced_ml
    ```

2. Create and activate a virtual environment:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Copy `.env_example` to `.env` and add missing values for the following variables: OPENAI_API_KEY


## Usage

### 1. Ingestion Pipeline

The main pipeline script is `backend/indexing.py`. This script:
- Parses PDF documents from the `PDF_DIR`.
- Chunks the parsed documents based on the specified chunk size and overlap.
- Enriches each chunk with metadata such as the document title.
- Indexes the chunks using the OpenAI `text-embedding-3-small` model and stores them in ChromaDB.

### 2. Running the Indexing Script

To run ingestion pipeline:
```bash
python run_indexing.py
```
This will:
- Parse and chunk the documents.
- Enrich them with metadata.
- Index and store them in ChromaDB.

### 3. Pipeline Configuration
All configurations are handled via environment variables, which are set in the .env file. The following variables are used:
- PDF_DIR: Directory where the PDF documents are stored.
- DB_DIR: Directory where the ChromaDB vector store will be saved.
- DB_COLLECTION: Name of the collection in ChromaDB.
- CHUNK_SIZE: Size of each document chunk.
- CHUNK_OVERLAP: Overlap between chunks for context continuity.

### License
This project is licensed under the terms of MIT 2.0 licence.
