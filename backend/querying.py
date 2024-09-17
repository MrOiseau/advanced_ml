import hashlib
import os
import sys
from uuid import uuid4
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain import prompts
from langchain.output_parsers import PydanticToolsParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
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

        # Initialize reranker if enabled
        if self.rerank:
            self.reranker = self._initialize_reranker()
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.base_retriever
            )
        else:
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

    def _initialize_embeddings(self) -> OpenAIEmbeddings:
        """
        Initializes the embeddings model.

        Returns:
            OpenAIEmbeddings: Initialized embeddings model.
        """
        if self.embedding_model in SUPPORTED_EMBEDDING_MODELS:
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
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
            title = doc.metadata.get("title", "Untitled")
            formatted_docs.append(f"**{title}**\n{doc.page_content}\n\n---\n\n")
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
    Main function to initialize and run the ingestion pipeline.
    """
    try:
        # Log the start of the script
        logger.info("Start of the ingestion pipeline.")

        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded.")

        # Constants
        DB_DIR = os.getenv("DB_DIR")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")  # Default model
        CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7))  # Default temperature
        SEARCH_RESULTS_NUM = int(os.getenv("SEARCH_RESULTS_NUM", 10))
        LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

        # Log loaded environment variables for verification
        logger.debug(f"DB_DIR: {DB_DIR}")
        logger.debug(f"DB_COLLECTION: {DB_COLLECTION}")
        logger.debug(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        logger.debug(f"CHAT_MODEL: {CHAT_MODEL}")
        logger.debug(f"CHAT_TEMPERATURE: {CHAT_TEMPERATURE}")
        logger.debug(f"SEARCH_RESULTS_NUM: {SEARCH_RESULTS_NUM}")
        logger.debug(f"LANGSMITH_PROJECT: {LANGSMITH_PROJECT}")

        # Check essential environment variables
        essential_vars = {
            "DB_DIR": DB_DIR,
            "DB_COLLECTION": DB_COLLECTION,
            "EMBEDDING_MODEL": EMBEDDING_MODEL
        }
        missing_vars = [var for var, value in essential_vars.items() if not value]
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            sys.exit(1)  # Exit with error code

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

        # Example query run (You might want to replace this with actual input handling)
        result = pipeline.run_query("Example question")
        logger.info(f"Query Result: {result}")

        logger.info("Ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
