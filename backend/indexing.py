# backend/indexing.py
import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from backend.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

class IngestionPipeline:
    def __init__(self, pdf_dir, db_dir, db_collection, chunk_size, chunk_overlap):
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.db_collection = db_collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(os.getenv("DATA_DIR"), exist_ok=True)
        logger.info(f"Directories verified or created: DB_DIR={self.db_dir}, DATA_DIR={os.getenv('DATA_DIR')}")

    def parse_pdfs(self):
        """Parse PDF documents from the specified directory and add document name as metadata."""
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

    def chunk_documents(self, docs):
        """Chunk documents using RecursiveCharacterTextSplitter."""
        logger.info(f"Chunking {len(docs)} documents.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        try:
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Total chunks created: {len(chunks)}")
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return []

    def enrich_with_metadata(self, chunks):
        """Enrich chunks by prepending the document title to page_content."""
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

    def index_chunks(self, chunks):
        """Index the document chunks using ChromaDB with the specified embedding model."""
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
            logger.info(f"Indexing completed and persisted to: {db_path}")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")

    def verify_index(self):
        """Verify that the index is populated and inspect its contents."""
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
            query = "What is Query Rewriting?"
            results = retriever.invoke(query)
            logger.info(f"Number of documents retrieved: {len(results)}")
            for doc in results:
                logger.info(f"Title: {doc.metadata.get('title', 'N/A')}")
                logger.info(f"Content snippet: {doc.page_content[:100]}...\n")
        except Exception as e:
            logger.error(f"Error during index verification: {e}")

    def run_pipeline(self):
        """Run the complete ingestion pipeline."""
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
        self.index_chunks(enriched_chunks)
        self.verify_index()
        logger.info("Ingestion pipeline finished.")

def main():
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
        missing_vars = []
        for var_name, var_value in [("PDF_DIR", PDF_DIR), ("DB_DIR", DB_DIR), 
                                    ("DB_COLLECTION", DB_COLLECTION), ("EMBEDDING_MODEL", EMBEDDING_MODEL),
                                    ("DATA_DIR", DATA_DIR)]:
            if not var_value:
                missing_vars.append(var_name)
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return

        # Initialize the pipeline
        pipeline = IngestionPipeline(
            pdf_dir=PDF_DIR,
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        pipeline.run_pipeline()
        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        
        
if __name__ == "__main__":
    main()
