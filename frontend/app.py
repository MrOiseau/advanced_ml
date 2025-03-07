# Import config first to ensure environment variables are set before other imports
from backend.config import *
import logging
import streamlit as st
import uuid
from backend.querying import QueryPipeline
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled
from backend.utils import setup_logging

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="AI Document Retrieval", layout="wide")

# Configure logging
logger = setup_logging(__name__)

# Use constants from config module
FEEDBACK_OPTION = "faces"  # Options can be "thumbs" or "faces"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Initialize LangSmith client for feedback
if not LANGSMITH_API_KEY:
    st.error("Missing LANGSMITH_API_KEY environment variable.")
    st.stop()
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# Validate required environment variables
required_vars = [
    "DB_DIR",
    "DB_COLLECTION",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "LANGSMITH_PROJECT",
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

@st.cache_resource
def initialize_query_pipeline() -> QueryPipeline:
    """
    Initialize the QueryPipeline and cache it to avoid redundant setups.

    Returns:
        QueryPipeline: An initialized instance of QueryPipeline.
    """
    try:
        # Use constants from config module
        pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT,
            query_expansion=True,  # Enable if desired
            rerank=True,           # Enable if desired
        )
        logger.info("QueryPipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize QueryPipeline: {e}")
        st.error(f"Initialization error: {e}")
        st.stop()

# Define available collections
available_collections = [
    {"name": "rag_collection_recursive", "description": "Recursive Character Chunking"},
    {"name": "rag_collection_semantic", "description": "Semantic Clustering Chunking"},
    {"name": "rag_collection_topic", "description": "Topic-Based Chunking"},
    {"name": "rag_collection_hierarchical", "description": "Hierarchical Chunking"},
    {"name": "rag_collection_advanced", "description": "Default Collection"}
]

# Create a mapping for easy lookup
collection_descriptions = {c["name"]: c["description"] for c in available_collections}

# Modify the query pipeline initialization to accept a collection name
@st.cache_resource(hash_funcs={dict: lambda _: None})
def initialize_query_pipeline(collection_name=DB_COLLECTION):
    """
    Initialize the QueryPipeline and cache it to avoid redundant setups.

    Args:
        collection_name (str): The name of the ChromaDB collection to use.

    Returns:
        QueryPipeline: An initialized instance of QueryPipeline.
    """
    try:
        # Use constants from config module but override the collection name
        pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=collection_name,  # Use the provided collection name
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT,
            query_expansion=True,  # Enable if desired
            rerank=True,           # Enable if desired
        )
        logger.info(f"QueryPipeline initialized successfully with collection: {collection_name}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize QueryPipeline: {e}")
        st.error(f"Initialization error: {e}")
        st.stop()

# Streamlit UI Setup
st.title("📄 AI Document Retrieval Interface")

# Sidebar for Filters and Settings
st.sidebar.title("Settings & Filters")

# Collection Selection
st.sidebar.subheader("Vector Database Selection")
selected_collection = st.sidebar.selectbox(
    "Select Vector Database:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{x} ({collection_descriptions.get(x, 'Unknown')})",
    help="Choose which vector database to query. Each database contains documents chunked with a different method."
)

# Initialize Query Pipeline with selected collection
if "query_pipeline" not in st.session_state or st.session_state.get("current_collection") != selected_collection:
    with st.spinner(f"Loading {collection_descriptions.get(selected_collection, 'Unknown')} database..."):
        st.session_state["query_pipeline"] = initialize_query_pipeline(selected_collection)
        st.session_state["current_collection"] = selected_collection
        st.success(f"Loaded {collection_descriptions.get(selected_collection, 'Unknown')} database")

query_pipeline = st.session_state["query_pipeline"]

# Fetch unique titles for filtering
unique_titles = query_pipeline.get_unique_titles()

# Title Filter
st.sidebar.subheader("Document Filters")
selected_titles = st.sidebar.multiselect(
    "Filter by Title:",
    options=unique_titles,
    help="Select one or more document titles to restrict the search.",
)

# User input for query within a form
with st.form(key="query_form"):
    user_input = st.text_input("Enter your query:", placeholder="e.g., What is Query Rewriting?")
    submit_button = st.form_submit_button(label="Search")

# Initialize state fields
for field in ["run_id", "last_run_id", "last_result", "last_user_input"]:
    if field not in st.session_state:
        st.session_state[field] = None

# Run the query when user submits input
if submit_button and user_input.strip():
    with st.spinner("Processing your query..."):
        try:
            metadata_filter = {}
            if selected_titles:
                metadata_filter = {"title": {"$in": selected_titles}}

            # Wrap the run_query invocation with tracing_v2_enabled
            with tracing_v2_enabled() as cb:
                result = query_pipeline.run_query(user_input, metadata_filter=metadata_filter)
                run_id = result.get("run_id")
                trace_url = cb.get_run_url()  # Get the run URL from the callback

                if not run_id:
                    run_id = str(uuid.uuid4())  # Generate a unique run_id if not provided
                    logger.warning("No run_id returned from run_query. Generated a new run_id.")

            st.session_state["run_id"] = run_id
            st.session_state["last_result"] = result
            st.session_state["last_user_input"] = user_input

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
            logger.error(f"Query processing error: {e}")
            result = None

    if result:
        # Create tabs for different views
        summary_tab, documents_tab, system_tab = st.tabs(["Summary", "Documents", "System"])

        with summary_tab:
            st.subheader("📚 Summary of Retrieved Information")
            summary = query_pipeline.generate_summary(user_input, result.get("documents", ""))
            st.write(summary)
            
            # # Add feedback buttons
            # st.write("Was this summary helpful?")
            # col1, col2, col3 = st.columns([1, 1, 3])
            # with col1:
            #     if st.button("👍 Yes"):
            #         try:
            #             langsmith_client.create_feedback(
            #                 run_id=run_id,
            #                 key="user_feedback",
            #                 score=1.0,
            #                 comment="User found the summary helpful"
            #             )
            #             st.success("Thank you for your feedback!")
            #         except Exception as e:
            #             st.error(f"Error recording feedback: {e}")
            #             logger.error(f"Feedback error: {e}")
            
            # with col2:
            #     if st.button("👎 No"):
            #         try:
            #             langsmith_client.create_feedback(
            #                 run_id=run_id,
            #                 key="user_feedback",
            #                 score=0.0,
            #                 comment="User did not find the summary helpful"
            #             )
            #             st.success("Thank you for your feedback!")
            #         except Exception as e:
            #             st.error(f"Error recording feedback: {e}")
            #             logger.error(f"Feedback error: {e}")

        with documents_tab:
            st.subheader("📄 Retrieved Documents")
            documents = result.get("documents", "")
            if documents:
                doc_list = documents.split("\n\n---\n\n")
                for idx, doc in enumerate(doc_list, 1):
                    if doc.strip():  # Skip empty documents
                        st.markdown(f"**Document {idx}**")
                        st.write(doc)
                        st.markdown("---")  # Add a separator between documents
            else:
                st.write("No documents retrieved.")

        with system_tab:
            st.subheader("🔍 System Information")
            
            # Add collection/chunking method info
            system_info = {
                "Collection": selected_collection,
                "Chunking Method": collection_descriptions.get(selected_collection, "Unknown"),
                "Embedding Model": query_pipeline.embedding_model,
                "Chat Model": query_pipeline.chat_model,
                "Temperature": query_pipeline.chat_temperature,
                "Search Results Number": query_pipeline.search_results_num,
                "Language Smith Project": query_pipeline.langsmith_project,
            }
            st.json(system_info)

            # Display LangSmith trace link using the trace_url from the callback
            if trace_url:
                st.markdown(f"[View LangSmith Trace]({trace_url})")
                
            # Add information about the different chunking methods
            st.subheader("📊 Chunking Methods Comparison")
            st.markdown("""
            | Collection | Chunking Method | Best For |
            |------------|-----------------|----------|
            | rag_collection_recursive | Recursive Character | Simple text with uniform content |
            | rag_collection_semantic | Semantic Clustering | Content with varying topics and themes |
            | rag_collection_topic | Topic-Based | Documents with distinct topical sections |
            | rag_collection_hierarchical | Hierarchical | Structured documents with headings and sections |
            """)
            
            # Add a note about creating collections
            st.info("""
            **Note:** To create a new collection with a specific chunking method, run:
            ```bash
            python -m backend.indexing --chunker_name="method_name" --db_collection="collection_name"
            ```
            """)
    else:
        st.warning("No results found. Try a different query.")

elif submit_button:
    st.warning("Please enter a valid query.")

# Footer with additional information or links
st.markdown("---")
st.markdown("Developed for the university subject 'Advanced ML' using Streamlit and LangChain (OpenAI models for embedding and generation, query rewriting, FlashrankRerank, ChromaDB).")
