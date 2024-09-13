import hashlib
import os
import sys
from uuid import uuid4
from typing import Dict, List

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain import callbacks, chat_models, hub, prompts
from langchain.output_parsers import PydanticToolsParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain.retrievers.document_compressors import FlashrankRerank, LLMChainFilter
from langchain.schema import Document, StrOutputParser
from langchain_chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Removed the traceable import
# from langsmith import traceable

from backend.utils import setup_logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = setup_logging(__name__)

# Constants and Supported Models
SUPPORTED_CHAT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
SUPPORTED_EMBEDDING_MODELS = ["text-embedding-3-large", "text-embedding-3-small"]
QUERY_EXPANSION_TEMPLATE = "query_expansion.jinja2"
SUMMARY_TEMPLATE = "summarize.jinja2"

class ExpandedQuery(BaseModel):
    """
    Pydantic model for expanded query
    """
    expanded_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

class QueryPipeline:
    def __init__(
        self,
        db_dir: str,
        db_collection: str,
        embedding_model: str,
        chat_model: str,
        chat_temperature: float,
        search_results_num: int,
        langsmith_project: str,
        prompt_templates_dir: str = "./prompts",
        query_expansion: bool = True,
        rerank: bool = True
    ):
        self.db_dir = db_dir
        self.db_collection = db_collection
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chat_temperature = chat_temperature
        self.search_results_num = search_results_num
        self.langsmith_project = langsmith_project
        self.prompt_templates_dir = prompt_templates_dir
        self.query_expansion = query_expansion
        self.rerank = rerank

        logger.info("Initializing QueryPipeline")

        # Set up Jinja2 templates
        file_loader = FileSystemLoader(self.prompt_templates_dir)
        self.template_env = Environment(loader=file_loader)
        try:
            self.summary_prompt_template = self.template_env.get_template(SUMMARY_TEMPLATE).render()
            if self.query_expansion:
                self.query_expansion_prompt_template = self.template_env.get_template(QUERY_EXPANSION_TEMPLATE).render()
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise

        # Initialize Embeddings
        if self.embedding_model in SUPPORTED_EMBEDDING_MODELS:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            logger.info(f"Using embedding model: {self.embedding_model}")
        else:
            logger.error(f"Unsupported embedding_model: {self.embedding_model}")
            raise ValueError(f"Unsupported embedding_model: {self.embedding_model}")

        # Initialize Vector Store
        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
                collection_name=self.db_collection,
            )
            logger.info(f"Connected to Chroma vector store at {self.db_dir} with collection {self.db_collection}")
        except Exception as e:
            logger.error(f"Error initializing Chroma vector store: {e}")
            raise

        # Initialize LLM for chat
        if self.chat_model in SUPPORTED_CHAT_MODELS:
            try:
                self.llm = ChatOpenAI(
                    model=self.chat_model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=self.chat_temperature,
                )
                logger.info(f"Using chat model: {self.chat_model}")
            except Exception as e:
                logger.error(f"Error initializing ChatOpenAI: {e}")
                raise
        else:
            logger.error(f"Unsupported chat_model: {self.chat_model}")
            raise ValueError(f"Unsupported chat_model: {self.chat_model}")

        # Initialize query expansion chain if enabled
        if self.query_expansion:
            self.query_expansion_chain = self._build_query_expansion_chain()

        # Initialize base retriever
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_results_num})

        # Initialize reranker if enabled
        if self.rerank:
            try:
                self.reranker = FlashrankRerank()
                logger.info("Reranker initialized")
            except Exception as e:
                logger.error(f"Error initializing reranker: {e}")
                raise

            # Use ContextualCompressionRetriever with reranker
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever

        logger.info("Query pipeline initialization completed")

    def _build_query_expansion_chain(self) -> callable:
        """Builds the query expansion chain using the expansion prompt."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.query_expansion_prompt_template),
                ("human", "{user_input}"),
            ])
            query_expansion_chain = (
                prompt
                | self.llm
                | PydanticToolsParser(tools=[ExpandedQuery])
            )
            logger.info("Query expansion chain built")
            return query_expansion_chain
        except Exception as e:
            logger.error(f"Error building query expansion chain: {e}")
            raise

    def retrieve_documents(self, query: str, metadata_filter=None):
        """Retrieve documents based on the query with optional metadata filter."""
        logger.info(f"Retrieving documents for query: {query} with metadata filter: {metadata_filter}")
        try:
            search_kwargs = {"k": self.search_results_num}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter

            # Create a base retriever with updated search_kwargs
            base_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

            if self.rerank:
                retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=base_retriever
                )
            else:
                retriever = base_retriever

            # Use invoke instead of deprecated method
            docs = retriever.invoke(query, **search_kwargs)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def expand_query(self, query: str) -> str:
        """Expand the user query to improve retrieval."""
        logger.info(f"Expanding query: {query}")
        try:
            expanded_results = self.query_expansion_chain.invoke({"user_input": query})
            if isinstance(expanded_results, list) and len(expanded_results) > 0:
                expanded_query = expanded_results[0].expanded_query
                logger.info(f"Expanded query: {expanded_query}")
                return expanded_query
            else:
                logger.warning("Query expansion returned no results. Using original query.")
                return query  # Fallback to original query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query  # Fallback to original query

    def deduplicate_documents(self, docs) -> list:
        """Remove duplicate documents based on content."""
        logger.info("Deduplicating documents")
        unique_docs = []
        seen_hashes = set()
        for doc in docs:
            if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                logger.warning("Document missing 'page_content' or 'metadata'. Skipping.")
                continue
            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(doc_hash)
        logger.info(f"Deduplicated documents: {len(unique_docs)} unique documents out of {len(docs)}")
        return unique_docs

    def format_documents(self, docs: List) -> str:
        """Format documents to include metadata and content."""
        logger.info("Formatting documents")
        formatted_docs = []
        for doc in docs:
            if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                logger.warning("Document missing 'metadata' or 'page_content'. Skipping.")
                continue
            title = doc.metadata.get("title", "Untitled")
            formatted_docs.append(f"**{title}**\n{doc.page_content}\n\n---\n\n")
        return "\n".join(formatted_docs)

    def generate_summary(self, question: str, context: str) -> str:
        """Generate a summarized answer based on the context."""
        logger.info("Generating summarized answer")
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.summary_prompt_template),
                ("human", "{question}\n\n{context}")
            ])
            messages = prompt.format_messages(question=question, context=context)
            response = self.llm.invoke(messages)
            logger.info("Summary generated")
            return response.content
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "I'm sorry, I couldn't generate a summary based on the provided context."

    def get_unique_titles(self) -> list:
        """Retrieve all unique document titles from the vector store."""
        logger.info("Retrieving unique document titles from vector store.")
        try:
            # Fetch all documents from the vector store
            collection = self.vectorstore.get()

            if "metadatas" not in collection:
                logger.error("No metadatas key found in Chroma response")
                return []

            metadatas = collection["metadatas"]

            titles = set()
            for metadata in metadatas:
                title = metadata.get("title", "Untitled")
                titles.add(title)

            unique_titles = sorted(list(titles))
            logger.info(f"Retrieved {len(unique_titles)} unique titles.")
            return unique_titles
        except Exception as e:
            logger.error(f"Error retrieving unique titles: {e}")
            return []

    def run_query(self, question: str, metadata_filter=None) -> Dict[str, str]:
        """Complete query pipeline with optional metadata filter."""
        logger.info(f"Running query pipeline for question: {question}")

        # Generate a unique run ID
        run_id = str(uuid4())

        # Step 1: Query Expansion
        if self.query_expansion:
            expanded_query = self.expand_query(question)
        else:
            expanded_query = question

        # Step 2: Retrieve Documents
        docs = self.retrieve_documents(expanded_query, metadata_filter=metadata_filter)
        if not docs:
            logger.warning("No documents retrieved.")
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "documents": "",
                "run_id": run_id,  # Include run_id in the result
            }

        # Step 3: Deduplicate Documents
        unique_docs = self.deduplicate_documents(docs)

        # Step 4: Format Documents
        formatted_context = self.format_documents(unique_docs)

        # Step 5: Generate Summary
        answer = self.generate_summary(question, formatted_context)

        return {
            "question": question,
            "answer": answer,
            "documents": formatted_context,
            "run_id": run_id,  # Include run_id in the result
        }

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
        CHAT_MODEL = os.getenv("CHAT_MODEL")
        CHAT_TEMPERATURE = os.getenv("CHAT_TEMPERATURE")
        SEARCH_RESULTS_NUM = os.getenv("SEARCH_RESULTS_NUM", 10)
        LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

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
        pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT,
            query_expansion=True,
            rerank=True,
        )
        pipeline.run_query("Example question")
        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
