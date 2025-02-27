import os
import json
from typing import List, Optional

from dotenv import load_dotenv
from backend.chunker_advanced import SemanticClusteringChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from backend.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)


class IngestionPipeline:
    """
    A pipeline to ingest PDF documents, process them into chunks, enrich with metadata,
    index using ChromaDB with OpenAI embeddings, and verify the indexing.
    """
    def __init__(self, pdf_dir: str, db_dir: str, db_collection: str, chunk_size: int, chunk_overlap: int, data_dir: str, embedding_model: str,) -> None:
        """
        Initialize the ingestion pipeline with necessary directories and configurations, plus advanced chunking toggle.

        Args:
            pdf_dir (str): Directory containing PDF files.
            db_dir (str): Directory to store the ChromaDB data.
            db_collection (str): Name of the ChromaDB collection.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between text chunks.
            data_dir (str): Directory for additional data storage.
            embedding_model (str): Model name for generating embeddings.
        """
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.db_collection = db_collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = data_dir
        # Read env variable if we want advanced chunking
        use_advanced_str = os.getenv("ADVANCED_CHUNKING", "false").lower()
        self.use_advanced_chunking = (use_advanced_str == "true")
        logger.info(f"Advanced chunking enabled = {self.use_advanced_chunking}")
        self.embedding_model = embedding_model

        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Directories verified or created: DB_DIR={self.db_dir}, DATA_DIR={self.data_dir}")

    def parse_pdfs(self) -> List[dict]:
        """
        Parse PDF documents from the specified directory and add document name as metadata.

        Returns:
            List[dict]: A list of parsed document dictionaries.
        """
        docs = []
        try:
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
            logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}.")

            for pdf_file in pdf_files:
                try:
                    logger.info(f"Parsing PDF: {pdf_file}")
                    file_path = os.path.join(self.pdf_dir, pdf_file)
                    loader = PyPDFLoader(file_path)
                    sub_docs = loader.load()

                    # Add document name as metadata to each sub_doc
                    for doc in sub_docs:
                        doc.metadata["title"] = os.path.splitext(pdf_file)[0]
                    docs.extend(sub_docs)
                except Exception as e:
                    logger.error(f"Error parsing {pdf_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to list PDF files in {self.pdf_dir}: {e}")

        logger.info(f"Total documents parsed: {len(docs)}")
        return docs

    def chunk_documents(self, docs: List[dict]) -> List[dict]:
        """
        Chunk documents using RecursiveCharacterTextSplitter.

        Args:
            docs (List[dict]): List of document dictionaries to be chunked.

        Returns:
            List[dict]: A list of chunked document dictionaries.
        """
        logger.info(f"Chunking {len(docs)} documents.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # used only if we do default chunking
            chunk_overlap=self.chunk_overlap,
            separators=[
                ".\n\n", ".\n", ". ", "!\n\n", "!\n", "? ",
                "\n\n", "\n", " - ", ": ", "; ", ", ", " "
            ],
        )

        # If advanced chunking, call the advanced chunker
        if self.use_advanced_chunking:
            adv_chunker = SemanticClusteringChunker(
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                max_chunk_size=self.chunk_size  # re-using chunk_size as max words
            )
            return adv_chunker.chunk_documents(docs)

        try:
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Total chunks created: {len(chunks)}")
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return []

    def enrich_with_metadata(self, chunks: List[dict]) -> List[dict]:
        """
        Enrich chunks by prepending the document title to page_content.

        Args:
            chunks (List[dict]): List of chunked document dictionaries.

        Returns:
            List[dict]: Enriched list of chunked document dictionaries.
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
    def _save_data_to_json(self, chunks: List[dict], file_path: str) -> None:
        """
        Save chunked document data to a JSON file.

        Args:
            chunks (List[dict]): List of enriched chunked document dictionaries.
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

    def index_chunks(self, chunks: List[dict]) -> None:
        """
        Index the document chunks using ChromaDB with the specified embedding model.

        Args:
            chunks (List[dict]): List of enriched chunked document dictionaries.
        """
        try:
            logger.info(f"Indexing {len(chunks)} chunks.")

            embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.db_dir,  # Save directly in DB_DIR
                collection_name=self.db_collection,
            )

            # Chroma automatically persists the data
            logger.info(f"Indexing completed and persisted to: {self.db_dir}")
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
            embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
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

    def run_pipeline(self) -> None:
        """
        Run the complete ingestion pipeline.
        """
        logger.info("Running the ingestion pipeline.")
        docs = self.parse_pdfs()
        if not docs:
            logger.warning("No documents to process. Exiting pipeline.")
            return

        chunks = self.chunk_documents(docs)
        if not chunks:
            logger.warning("No chunks created. Exiting pipeline.")
            return

        enriched_chunks = self.enrich_with_metadata(chunks)
        
        # Save the enriched chunks to a JSON file before indexing
        json_file_path = os.path.join(self.data_dir, "preview_data_to_ingest_to_db.json")
        self._save_data_to_json(enriched_chunks, json_file_path)
    
        self.index_chunks(enriched_chunks)
        self.verify_index()
        logger.info("Ingestion pipeline finished.")

def main() -> None:
    """
    Main function to initialize and run the ingestion pipeline.
    """
    try:
        # Log the start of the script
        logger.info("Start of the ingestion pipeline.")

        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded.")

        # Constants
        PDF_DIR = os.getenv("PDF_DIR")
        DB_DIR = os.getenv("DB_DIR")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        DATA_DIR = os.getenv("DATA_DIR")

        # Log loaded environment variables for verification
        logger.debug(f"PDF_DIR: {PDF_DIR}")
        logger.debug(f"DB_DIR: {DB_DIR}")
        logger.debug(f"DB_COLLECTION: {DB_COLLECTION}")
        logger.debug(f"CHUNK_SIZE: {CHUNK_SIZE}")
        logger.debug(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
        logger.debug(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        logger.debug(f"DATA_DIR: {DATA_DIR}")

        # Check essential environment variables
        required_vars = {
            "PDF_DIR": PDF_DIR,
            "DB_DIR": DB_DIR,
            "DB_COLLECTION": DB_COLLECTION,
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "DATA_DIR": DATA_DIR,
        }
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return
        
        # Convert CHUNK_SIZE and CHUNK_OVERLAP to integers with defaults
        try:
            chunk_size = int(CHUNK_SIZE) if CHUNK_SIZE else 1000
            chunk_overlap = int(CHUNK_OVERLAP) if CHUNK_OVERLAP else 200
        except ValueError as ve:
            logger.error(f"Invalid chunk size or overlap values: {ve}")
            return

        # Initialize the pipeline
        pipeline = IngestionPipeline(
            pdf_dir=PDF_DIR,
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            data_dir=DATA_DIR,
            embedding_model=EMBEDDING_MODEL,
        )
        pipeline.run_pipeline()
        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        
        
if __name__ == "__main__":
    main()
