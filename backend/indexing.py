# Import config first to ensure environment variables are set before other imports
from backend.config import *
import json
from typing import List, Optional, Dict, Any
import os
import time
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from backend.utils import setup_logging
from backend.chunkers.base import ChunkerFactory
import chromadb

# Configure logging
logger = setup_logging(__name__)


class IngestionPipeline:
    """
    A pipeline to ingest PDF documents by following several steps:
    - Loading environment variables
    - Verifying/creating directories for database and data storage
    - Finding PDF files in ./data/pdfs
    - Parsing the PDF files by using PyPDFLoader from LangChain
    - Chunking the documents (using the specified chunker from ChunkerFactory)
    - Enriching chunks with metadata (prepends the document title to each chunk)
    - Saving data to a JSON file (./data/preview_data_to_ingest_to_db.json)
    - Indexing the chunks using ChromaDB with OpenAI embeddings
    - Verifying the index
    """
    def __init__(
        self,
        pdf_dir: str,
        db_dir: str,
        db_collection: str,
        chunk_size: int,
        chunk_overlap: int,
        data_dir: str,
        embedding_model: str,
        chunker_name: Optional[str] = None,
        chunker_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the ingestion pipeline with necessary directories and configurations.

        Args:
            pdf_dir (str): Directory containing PDF files.
            db_dir (str): Directory to store the ChromaDB data.
            db_collection (str): Name of the ChromaDB collection.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between text chunks.
            data_dir (str): Directory for additional data storage.
            embedding_model (str): Model name for generating embeddings.
            chunker_name (Optional[str]): Name of the chunker to use.
                If None, uses the value from ADVANCED_CHUNKING config.
            chunker_params (Optional[Dict[str, Any]]): Parameters for the chunker.
        """
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.db_collection = db_collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        
        # Determine which chunker to use
        if chunker_name is None:
            # Use ADVANCED_CHUNKING from config module
            self.chunker_name = "semantic_clustering" if ADVANCED_CHUNKING else "recursive_character"
        else:
            self.chunker_name = chunker_name
        
        # Set default chunker parameters if none provided
        self.chunker_params = chunker_params or {}
        
        # Set default parameters based on chunker type
        if self.chunker_name == "recursive_character" and "chunk_size" not in self.chunker_params:
            self.chunker_params["chunk_size"] = self.chunk_size
            self.chunker_params["chunk_overlap"] = self.chunk_overlap
        elif self.chunker_name == "semantic_clustering" and "max_chunk_size" not in self.chunker_params:
            self.chunker_params["max_chunk_size"] = self.chunk_size  # re-using chunk_size as max words
        
        logger.info(f"Using chunker: {self.chunker_name} with params: {self.chunker_params}")
        
        # Create the chunker
        try:
            self.chunker = ChunkerFactory.create(self.chunker_name, **self.chunker_params)
            logger.info(f"Chunker created successfully: {self.chunker.name}")
        except Exception as e:
            logger.error(f"Error creating chunker: {e}")
            raise

        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Directories verified or created: DB_DIR={self.db_dir}, DATA_DIR={self.data_dir}")
        
        # Initialize performance metrics
        self.performance_metrics = {
            "chunking_time": 0.0,
            "indexing_time": 0.0,
            "num_chunks": 0,
            "avg_chunk_size_chars": 0.0,
            "avg_chunk_size_words": 0.0
        }

    def parse_pdfs(self) -> List[Document]:
        """
        Parse PDF documents from the specified directory and add document name as metadata.
        Uses parallel processing for faster parsing of multiple PDFs.

        Returns:
            List[Document]: A list of parsed document dictionaries.
        """
        docs = []
        try:
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
            logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}.")

            # Define a function to process a single PDF file
            def process_pdf(pdf_file):
                try:
                    logger.info(f"Parsing PDF: {pdf_file}")
                    file_path = os.path.join(self.pdf_dir, pdf_file)
                    loader = PyPDFLoader(file_path)
                    sub_docs = loader.load()

                    # Add document name as metadata to each sub_doc
                    for doc in sub_docs:
                        doc.metadata["title"] = os.path.splitext(pdf_file)[0]
                    return sub_docs
                except Exception as e:
                    logger.error(f"Error parsing {pdf_file}: {e}")
                    return []

            # Use ThreadPoolExecutor for parallel processing
            # This is ideal for I/O bound tasks like PDF parsing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(pdf_files))) as executor:
                # Submit all PDF files for processing
                future_to_pdf = {executor.submit(process_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    try:
                        sub_docs = future.result()
                        docs.extend(sub_docs)
                        logger.info(f"Completed parsing: {pdf_file} - got {len(sub_docs)} documents")
                    except Exception as e:
                        logger.error(f"Exception processing {pdf_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to list PDF files in {self.pdf_dir}: {e}")

        logger.info(f"Total documents parsed: {len(docs)}")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents using the specified chunker with parallel processing.

        Args:
            docs (List[Document]): List of document dictionaries to be chunked.

        Returns:
            List[Document]: A list of chunked document dictionaries.
        """
        logger.info(f"Chunking {len(docs)} documents using {self.chunker.name} with parallel processing.")
        
        try:
            # Measure chunking time
            start_time = time.time()
            
            # If there are too few documents, don't use parallel processing
            if len(docs) <= 4:
                chunks = self.chunker.chunk_documents(docs)
            else:
                # Determine optimal batch size based on number of documents
                # For very large documents, we want smaller batches
                batch_size = max(1, min(10, len(docs) // (os.cpu_count() or 4)))
                logger.info(f"Using parallel processing with batch size: {batch_size}")
                
                # Split documents into batches for parallel processing
                doc_batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
                
                # Define a function to process a batch of documents
                def process_batch(batch):
                    try:
                        return self.chunker.chunk_documents(batch)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        return []
                
                # Use ProcessPoolExecutor for CPU-bound chunking tasks
                # This is more efficient than ThreadPoolExecutor for CPU-intensive operations
                all_chunks = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    # Submit all batches for processing
                    future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(doc_batches)}
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_chunks = future.result()
                            all_chunks.extend(batch_chunks)
                            logger.info(f"Completed chunking batch {batch_idx+1}/{len(doc_batches)} - got {len(batch_chunks)} chunks")
                        except Exception as e:
                            logger.error(f"Exception processing batch {batch_idx+1}: {e}")
                
                chunks = all_chunks
            
            chunking_time = time.time() - start_time
            
            # Calculate chunk statistics
            num_chunks = len(chunks)
            avg_chunk_size_chars = sum(len(chunk.page_content) for chunk in chunks) / num_chunks if num_chunks > 0 else 0
            avg_chunk_size_words = sum(len(chunk.page_content.split()) for chunk in chunks) / num_chunks if num_chunks > 0 else 0
            
            # Update performance metrics
            self.performance_metrics["chunking_time"] = chunking_time
            self.performance_metrics["num_chunks"] = num_chunks
            self.performance_metrics["avg_chunk_size_chars"] = avg_chunk_size_chars
            self.performance_metrics["avg_chunk_size_words"] = avg_chunk_size_words
            
            logger.info(f"Total chunks created: {num_chunks}")
            logger.info(f"Average chunk size: {avg_chunk_size_chars:.2f} chars, {avg_chunk_size_words:.2f} words")
            logger.info(f"Chunking time: {chunking_time:.2f} seconds")
            
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return []

    def enrich_with_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Enrich chunks by prepending the document title to page_content.

        Args:
            chunks (List[Document]): List of chunked document dictionaries.

        Returns:
            List[Document]: Enriched list of chunked document dictionaries.
        """
        logger.info("Enriching chunks with metadata.")
        try:
            for chunk in chunks:
                title = chunk.metadata.get("title", "")
                if title:
                    chunk.page_content = f"{title}\n\n{chunk.page_content}"
            logger.info("Chunks enriched with metadata.")
            return chunks
        except Exception as e:
            logger.error(f"Error during metadata enrichment: {e}")
            return chunks
            
    def _save_data_to_json(self, chunks: List[Document], file_path: str) -> None:
        """
        Save chunked document data to a JSON file.

        Args:
            chunks (List[Document]): List of enriched chunked document dictionaries.
            file_path (str): File path where the JSON will be saved.
        """
        try:
            logger.info(f"Saving data to JSON file: {file_path}")
            data_to_save = [
                {
                    "metadata": chunk.metadata,
                    "page_content": chunk.page_content
                } for chunk in chunks
            ]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            logger.info("Data successfully saved to JSON.")
        except Exception as e:
            logger.error(f"Error saving data to JSON: {e}")

    def index_chunks(self, chunks: List[Document]) -> None:
        """
        Index the document chunks using ChromaDB with the specified embedding model.
        Uses batch processing to optimize embedding generation.

        Args:
            chunks (List[Document]): List of enriched chunked document dictionaries.
        """
        try:
            logger.info(f"Indexing {len(chunks)} chunks.")
            
            # Measure indexing time
            start_time = time.time()

            # Check if we should use a local embedding model (faster for development)
            use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
            
            if use_local_embeddings:
                # Use HuggingFace embeddings for local processing (no API calls)
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    logger.info("Using local HuggingFace embeddings model")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                except ImportError:
                    logger.warning("HuggingFaceEmbeddings not available, falling back to OpenAI")
                    embeddings = OpenAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        disallowed_special=(),
                        chunk_size=20  # Process 20 documents at a time
                    )
            else:
                # Use OpenAI embeddings with batch processing
                logger.info(f"Using OpenAI embeddings model: {EMBEDDING_MODEL}")
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    disallowed_special=(),
                    chunk_size=20  # Process 20 documents at a time
                )
            
            # Process in batches for better performance
            batch_size = 20  # Adjust based on your needs
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            logger.info(f"Processing in {total_batches} batches of size {batch_size}")
            
            # Initialize Chroma client
            chroma_client = chromadb.PersistentClient(path=self.db_dir)
            
            # Get or create collection
            collection_name = self.db_collection
            
            # Use LangChain's Chroma with batching
            vector_store = Chroma(
                persist_directory=self.db_dir,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                # Add documents in batch
                vector_store.add_documents(documents=batch)
                
                logger.info(f"Completed batch {batch_num}/{total_batches}")
            
            # Persist the vector store
            vector_store.persist()
            
            indexing_time = time.time() - start_time
            self.performance_metrics["indexing_time"] = indexing_time

            # Chroma automatically persists the data
            logger.info(f"Indexing completed and persisted to: {self.db_dir}")
            logger.info(f"Indexing time: {indexing_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")

    def verify_index(self, query: Optional[str] = None) -> None:
        """
        Verify that the index is populated and inspect its contents.

        Args:
            query (Optional[str]): Query string to test the index. Defaults to a sample query.
        """
        query = query or "What is Query Rewriting?"
        try:
            logger.info("Verifying the index.")
            db_path = self.db_dir
            # Check if we should use a local embedding model (faster for development)
            use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
            
            if use_local_embeddings:
                # Use HuggingFace embeddings for local processing (no API calls)
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    logger.info("Using local HuggingFace embeddings model for verification")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                except ImportError:
                    logger.warning("HuggingFaceEmbeddings not available, falling back to OpenAI")
                    embeddings = OpenAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        disallowed_special=(),
                    )
            else:
                # Use OpenAI embeddings
                logger.info(f"Using OpenAI embeddings model for verification: {EMBEDDING_MODEL}")
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    disallowed_special=(),
                )
            vector_store = Chroma(
                persist_directory=db_path,
                collection_name=self.db_collection,
                embedding_function=embeddings,
            )
            # Load existing index
            retriever = vector_store.as_retriever()

            # Perform a test query
            results = retriever.invoke(query)
            logger.info(f"Number of documents retrieved: {len(results)}")
            for doc in results:
                logger.info(f"Title: {doc.metadata.get('title', 'N/A')}")
                logger.info(f"Content snippet: {doc.page_content[:100]}...\n")
        except Exception as e:
            logger.error(f"Error during index verification: {e}")

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            Dict[str, Any]: Performance metrics for the ingestion pipeline.
        """
        logger.info("Running the ingestion pipeline.")
        docs = self.parse_pdfs()
        if not docs:
            logger.warning("No documents to process. Exiting pipeline.")
            return self.performance_metrics

        chunks = self.chunk_documents(docs)
        if not chunks:
            logger.warning("No chunks created. Exiting pipeline.")
            return self.performance_metrics

        enriched_chunks = self.enrich_with_metadata(chunks)
        
        # Save the enriched chunks to a JSON file before indexing
        json_file_path = os.path.join(self.data_dir, "preview_data_to_ingest_to_db.json")
        self._save_data_to_json(enriched_chunks, json_file_path)
    
        self.index_chunks(enriched_chunks)
        self.verify_index()
        logger.info("Ingestion pipeline finished.")
        
        # Log performance metrics
        logger.info("Performance metrics:")
        for metric, value in self.performance_metrics.items():
            logger.info(f"  {metric}: {value}")
        
        return self.performance_metrics

def main() -> None:
    """
    Main function to initialize and run the ingestion pipeline.
    """
    try:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Ingest documents into the RAG system.")
        parser.add_argument("--pdf_dir", default=PDF_DIR, help="Directory containing PDF files.")
        parser.add_argument("--db_dir", default=DB_DIR, help="Directory to store the ChromaDB database.")
        parser.add_argument("--db_collection", default=DB_COLLECTION, help="Name of the ChromaDB collection.")
        parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Size of each text chunk.")
        parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between text chunks.")
        parser.add_argument("--data_dir", default=DATA_DIR, help="Directory for additional data storage.")
        parser.add_argument("--embedding_model", default=EMBEDDING_MODEL, help="Model name for generating embeddings.")
        parser.add_argument("--chunker_name", help="Name of the chunker to use.")
        parser.add_argument("--chunker_params", help="JSON string of parameters for the chunker.")
        args = parser.parse_args()
        
        # Log the start of the script
        logger.info("Start of the ingestion pipeline.")
        
        # Environment variables are already loaded by config module
        logger.info("Environment variables loaded.")

        # Log loaded environment variables and command-line arguments
        logger.debug(f"PDF_DIR: {args.pdf_dir}")
        logger.debug(f"DB_DIR: {args.db_dir}")
        logger.debug(f"DB_COLLECTION: {args.db_collection}")
        logger.debug(f"CHUNK_SIZE: {args.chunk_size}")
        logger.debug(f"CHUNK_OVERLAP: {args.chunk_overlap}")
        logger.debug(f"EMBEDDING_MODEL: {args.embedding_model}")
        logger.debug(f"DATA_DIR: {args.data_dir}")
        logger.debug(f"CHUNKER_NAME: {args.chunker_name}")

        # Validate environment variables
        try:
            validate_environment()
        except ValueError as e:
            logger.error(str(e))
            return

        # Parse chunker_params if provided
        chunker_params = {}
        if args.chunker_params:
            import json
            try:
                chunker_params = json.loads(args.chunker_params)
                logger.debug(f"CHUNKER_PARAMS: {chunker_params}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing chunker_params: {e}")
                return

        # Initialize the pipeline
        pipeline = IngestionPipeline(
            pdf_dir=args.pdf_dir,
            db_dir=args.db_dir,
            db_collection=args.db_collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            data_dir=args.data_dir,
            embedding_model=args.embedding_model,
            chunker_name=args.chunker_name,
            chunker_params=chunker_params
        )
        pipeline.run_pipeline()
        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        
        
if __name__ == "__main__":
    main()
