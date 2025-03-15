# Import config first to ensure environment variables are set before other imports
from backend.config import *
import hashlib
import sys
from uuid import uuid4
from typing import Dict, List, Optional, Any
from jinja2 import Environment, FileSystemLoader
from langchain import callbacks, chat_models, hub, prompts
from langchain.output_parsers import PydanticToolsParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain.retrievers.document_compressors import FlashrankRerank, LLMChainFilter
from langchain.schema import Document, StrOutputParser
from langchain_chroma import Chroma
from pydantic.v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    Pydantic model for expanded query.
    """
    expanded_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )


class QueryPipeline:
    """
    A pipeline to handle query expansion, document retrieval, deduplication,
    formatting, and summary generation.
    """

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
    ) -> None:
        """
        Initializes the QueryPipeline with necessary configurations.

        Args:
            db_dir (str): Directory of the vector store.
            db_collection (str): Collection name in the vector store.
            embedding_model (str): Model name for embeddings.
            chat_model (str): Model name for chat.
            chat_temperature (float): Temperature setting for the chat model.
            search_results_num (int): Number of search results to retrieve.
            langsmith_project (str): Langsmith project identifier.
            prompt_templates_dir (str, optional): Directory for prompt templates.
            query_expansion (bool, optional): Enable query expansion.
            rerank (bool, optional): Enable reranking of documents.
        """
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
        self.template_env = Environment(loader=FileSystemLoader(self.prompt_templates_dir))
        self._load_templates()

        # Initialize Embeddings
        self.embeddings = self._initialize_embeddings()

        # Initialize Vector Store
        self.vectorstore = self._initialize_vectorstore()

        # Initialize LLM for chat
        self.llm = self._initialize_llm()

        # Initialize query expansion chain if enabled
        if self.query_expansion:
            self.query_expansion_chain = self._build_query_expansion_chain()

        # Initialize base retriever
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_results_num})

        # Check if reranking should be disabled based on environment variable
        rerank_env = os.getenv("RERANK", "true").lower() == "true"
        self.rerank = self.rerank and rerank_env
        
        if not self.rerank:
            self.retriever = self.base_retriever
            logger.info("Reranker disabled (either by parameter or environment variable)")
        else:
            try:
                reranker = self._initialize_reranker()
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=self.base_retriever
                )
                logger.info("Reranker initialized and enabled")
            except Exception as e:
                logger.error(f"Error initializing reranker: {e}. Falling back to base retriever.")
                self.rerank = False
                self.retriever = self.base_retriever

        logger.info("Query pipeline initialization completed")

    def _load_templates(self) -> None:
        """
        Loads Jinja2 templates for query expansion and summarization.
        """
        try:
            self.summary_prompt_template = self.template_env.get_template(SUMMARY_TEMPLATE).render()
            if self.query_expansion:
                self.query_expansion_prompt_template = self.template_env.get_template(QUERY_EXPANSION_TEMPLATE).render()
            logger.info("Templates loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise

    def _initialize_embeddings(self):
        """
        Initializes the embeddings model.

        Returns:
            Embeddings: Initialized embeddings model (OpenAI or HuggingFace).
        """
        # Check if we should use local embeddings
        use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        
        if use_local_embeddings:
            # Use HuggingFace embeddings for local processing (no API calls)
            try:
                # Use the updated import path for HuggingFaceEmbeddings
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("Using local HuggingFace embeddings model")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                return embeddings
            except ImportError:
                logger.warning("HuggingFaceEmbeddings not available, falling back to OpenAI")
                # Fall back to OpenAI embeddings
        
        # Use OpenAI embeddings
        if self.embedding_model in SUPPORTED_EMBEDDING_MODELS:
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                disallowed_special=(),
            )
            logger.info(f"Using embedding model: {self.embedding_model}")
            return embeddings
        else:
            logger.error(f"Unsupported embedding_model: {self.embedding_model}")
            raise ValueError(f"Unsupported embedding_model: {self.embedding_model}")

    def _initialize_vectorstore(self) -> Chroma:
        """
        Initializes the Chroma vector store.

        Returns:
            Chroma: Initialized vector store.
        """
        try:
            # Check if the collection is one of the WFP collections
            if self.db_collection.startswith("wfp_collection_"):
                # Use the collection-specific directory
                collection_dir = os.path.join(self.db_dir, self.db_collection)
                vectorstore = Chroma(
                    persist_directory=collection_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.db_collection,
                )
                logger.info(f"Connected to Chroma vector store at {collection_dir} with collection {self.db_collection}")
            else:
                # Use the standard approach for other collections
                vectorstore = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.db_collection,
                )
                logger.info(f"Connected to Chroma vector store at {self.db_dir} with collection {self.db_collection}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error initializing Chroma vector store: {e}")
            raise

    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initializes the ChatOpenAI model.

        Returns:
            ChatOpenAI: Initialized chat model.
        """
        if self.chat_model in SUPPORTED_CHAT_MODELS:
            try:
                llm = ChatOpenAI(
                    model=self.chat_model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=self.chat_temperature,
                )
                logger.info(f"Using chat model: {self.chat_model}")
                return llm
            except Exception as e:
                logger.error(f"Error initializing ChatOpenAI: {e}")
                raise
        else:
            logger.error(f"Unsupported chat_model: {self.chat_model}")
            raise ValueError(f"Unsupported chat_model: {self.chat_model}")

    def _initialize_reranker(self) -> FlashrankRerank:
        """
        Initializes the Flashrank reranker.

        Returns:
            FlashrankRerank: Initialized reranker.
        """
        try:
            reranker = FlashrankRerank()
            logger.info("Reranker initialized")
            return reranker
        except Exception as e:
            logger.error(f"Error initializing reranker: {e}")
            raise

    def _build_query_expansion_chain(self) -> Any:
        """
        Builds the query expansion chain using the expansion prompt.

        Returns:
            Callable: The query expansion chain callable.
        """
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

    def retrieve_documents(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve documents based on the query with optional metadata filter.

        Args:
            query (str): The search query.
            metadata_filter (Optional[Dict[str, Any]]): Optional metadata filters.

        Returns:
            List[Document]: List of retrieved documents.
        """
        logger.info(f"Retrieving documents for query: {query} with metadata filter: {metadata_filter}")
        try:
            # Log collection information
            logger.info(f"Using collection: {self.db_collection}")
            
            # Check if collection has documents
            collection = self.vectorstore.get()
            if "metadatas" not in collection or len(collection["metadatas"]) == 0:
                logger.warning(f"Collection '{self.db_collection}' is empty. No documents to retrieve from.")
                return []
                
            search_kwargs = {"k": self.search_results_num}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter

            # Use the existing retriever (which might be a base retriever or a contextual compression retriever)
            # but update its search parameters if needed
            if hasattr(self.retriever, "search_kwargs"):
                self.retriever.search_kwargs.update(search_kwargs)
            
            # Use invoke instead of deprecated method
            docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def expand_query(self, query: str) -> str:
        """
        Expand the user query to improve retrieval.

        Args:
            query (str): The original query.

        Returns:
            str: The expanded query.
        """
        logger.info(f"Expanding query: {query}")
        try:
            expanded_results = self.query_expansion_chain.invoke({"user_input": query})
            if isinstance(expanded_results, list) and expanded_results:
                expanded_query = expanded_results[0].expanded_query
                logger.info(f"Expanded query: {expanded_query}")
                return expanded_query
            else:
                logger.warning("Query expansion returned no results. Using original query.")
                return query  # Fallback to original query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query  # Fallback to original query

    def deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content.

        Args:
            docs (List[Document]): List of documents to deduplicate.

        Returns:
            List[Document]: List of unique documents.
        """
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

    def format_documents(self, docs: List[Document]) -> str:
        """
        Format documents to include metadata and content.

        Args:
            docs (List[Document]): List of documents to format.

        Returns:
            str: Formatted string of documents.
        """
        logger.info("Formatting documents")
        formatted_docs = []
        for doc in docs:
            if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                logger.warning("Document missing 'metadata' or 'page_content'. Skipping.")
                continue
            # The title is already in the page_content from the indexing process
            # Just add the document content without repeating the title
            formatted_docs.append(f"{doc.page_content}\n\n---\n\n")
        return "\n".join(formatted_docs)

    def generate_summary(self, question: str, context: str) -> str:
        """
        Generate a summarized answer based on the context.

        Args:
            question (str): The original question.
            context (str): The context from retrieved documents.

        Returns:
            str: The summarized answer.
        """
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

    def get_unique_titles(self) -> List[str]:
        """
        Retrieve all unique document titles from the vector store.

        Returns:
            List[str]: List of unique titles.
        """
        logger.info("Retrieving unique document titles from vector store.")
        try:
            # Fetch all documents from the vector store
            collection = self.vectorstore.get()

            if "metadatas" not in collection:
                logger.error("No metadatas key found in Chroma response")
                return []

            metadatas = collection["metadatas"]
            
            # Log more details about the collection
            logger.info(f"Collection name: {self.db_collection}")
            logger.info(f"Total documents in collection: {len(metadatas)}")
            
            # Check if collection is empty
            if len(metadatas) == 0:
                logger.warning(f"Collection '{self.db_collection}' is empty. No documents have been indexed.")
                return []

            titles = {metadata.get("title", "Untitled") for metadata in metadatas}
            unique_titles = sorted(titles)
            logger.info(f"Retrieved {len(unique_titles)} unique titles.")
            return unique_titles
        except Exception as e:
            logger.error(f"Error retrieving unique titles: {e}")
            return []

    def run_query(self, question: str, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete query pipeline with optional metadata filter.

        Args:
            question (str): The user's question.
            metadata_filter (Optional[Dict[str, Any]]): Optional metadata filters.

        Returns:
            Dict[str, Any]: Dictionary containing the question, answer, documents, and run_id.
        """
        logger.info(f"Running query pipeline for question: {question}")

        # Generate a unique run ID
        run_id = str(uuid4())

        # Step 1: Query Expansion
        expanded_query = self.expand_query(question) if self.query_expansion else question

        # Step 2: Retrieve Documents
        docs = self.retrieve_documents(expanded_query, metadata_filter=metadata_filter)
        if not docs:
            logger.warning("No documents retrieved.")
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "documents": "",
                "run_id": run_id,
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
            "run_id": run_id,
        }


def main() -> None:
    """
    Main function to initialize and run the query pipeline.
    """
    try:
        # Log the start of the script
        logger.info("Start of the query pipeline.")
        
        # Environment variables are already loaded by config module
        logger.info("Environment variables loaded.")

        # Log loaded environment variables for verification
        logger.debug(f"PDF_DIR: {PDF_DIR}")
        logger.debug(f"DB_DIR: {DB_DIR}")
        logger.debug(f"DB_COLLECTION: {DB_COLLECTION}")
        logger.debug(f"CHUNK_SIZE: {CHUNK_SIZE}")
        logger.debug(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
        logger.debug(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        logger.debug(f"DATA_DIR: {DATA_DIR}")
        logger.debug(f"CHAT_MODEL: {CHAT_MODEL}")
        logger.debug(f"CHAT_TEMPERATURE: {CHAT_TEMPERATURE}")
        logger.debug(f"SEARCH_RESULTS_NUM: {SEARCH_RESULTS_NUM}")
        logger.debug(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")

        # Validate environment variables
        try:
            validate_environment()
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)  # Exit with error code

        # Get rerank setting from environment variable
        rerank = os.getenv("RERANK", "true").lower() == "true"
        logger.debug(f"RERANK: {rerank}")

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
            rerank=rerank,
        )

        # Example query run with a specific question about transformers
        result = pipeline.run_query("What is the purpose of multi-head attention in transformers?")
        logger.info(f"Query Result: {result}")

        logger.info("Query pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
