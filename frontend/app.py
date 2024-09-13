import os
import logging
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import uuid
from streamlit_feedback import streamlit_feedback
from backend.querying import QueryPipeline
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled  # Import tracing_v2_enabled

# Load environment variables
load_dotenv()

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="AI Document Retrieval", layout="wide")

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables with Validation
DB_DIR = os.getenv("DB_DIR")
DB_COLLECTION = os.getenv("DB_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7))
SEARCH_RESULTS_NUM = int(os.getenv("SEARCH_RESULTS_NUM", 5))
FEEDBACK_OPTION = "faces"  # Options can be "thumbs" or "faces"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    "LANGCHAIN_API_KEY",
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# Define the feedback scoring system
SCORE_MAPPINGS = {
    "thumbs": {"👍": 1, "👎": 0},
    "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
}

# Feedback settings
score_mappings = SCORE_MAPPINGS[FEEDBACK_OPTION]

def _submit_feedback(feedback_data, run_id, **kwargs):
    """Handle feedback submission to LangSmith."""
    try:
        if not run_id:
            logger.error("No run_id provided. Cannot submit feedback.")
            st.error("Feedback submission failed: No run ID found.")
            return

        feedback_type_str = f"{FEEDBACK_OPTION} {feedback_data['score']}"
        score = score_mappings[feedback_data['score']]

        # Formulate comment as a JSON string with additional metadata
        comment = {
            "tag": kwargs.get('tag', ''),
            "user_comment": feedback_data.get("text", ""),
            "metadata": kwargs,
        }

        # Record the feedback with the formulated feedback type string and comment
        feedback_record = langsmith_client.create_feedback(
            run_id=run_id,
            key=feedback_type_str,
            score=score,
            comment=str(comment),  # Convert comment dict to string
        )
        st.session_state["feedback"] = {
            "feedback_id": str(feedback_record.id),
            "score": score,
        }

        st.toast(f"Feedback submitted: {feedback_data.get('score', 'Thank you!')}")
        logger.info(f"Sent feedback for run ID: {run_id}")
    except Exception as e:
        st.error(f"Failed to submit feedback: {e}")
        logger.error(f"Feedback submission error: {e}")

def handle_feedback(tag, run_id, **kwargs):
    """
    Display feedback component and handle user input.
    """
    feedback_key = f"feedback_{tag}"
    feedback = streamlit_feedback(
        feedback_type=FEEDBACK_OPTION,
        optional_text_label="[Optional] Please provide an explanation",
        key=feedback_key,
        on_submit=lambda feedback: _submit_feedback(feedback, run_id, tag=tag, **kwargs),
    )

@st.cache_resource
def initialize_query_pipeline():
    """
    Initialize the QueryPipeline and cache it to avoid redundant setups.
    """
    try:
        pipeline = QueryPipeline(
            db_dir=DB_DIR,
            db_collection=DB_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
            chat_temperature=CHAT_TEMPERATURE,
            search_results_num=SEARCH_RESULTS_NUM,
            langsmith_project=LANGSMITH_PROJECT,
            query_expansion=True,  # Enable if desired
            rerank=True,          # Enable if desired
        )
        logger.info("QueryPipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize QueryPipeline: {e}")
        st.error(f"Initialization error: {e}")
        st.stop()

# Initialize Query Pipeline
if "query_pipeline" not in st.session_state:
    st.session_state["query_pipeline"] = initialize_query_pipeline()

query_pipeline = st.session_state["query_pipeline"]

# Fetch unique titles for filtering
unique_titles = query_pipeline.get_unique_titles()

# Streamlit UI Setup
st.title("📄 AI Document Retrieval Interface")

# Sidebar for Filters
st.sidebar.title("Set up Filters")

# Title Filter
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
            handle_feedback(tag="rag_summary", run_id=run_id, query=user_input)

        with documents_tab:
            st.subheader("📄 Retrieved Documents")
            documents = result.get("documents", "")
            if documents:
                doc_list = documents.split("\n\n---\n\n")
                for idx, doc in enumerate(doc_list, 1):
                    if doc.strip():  # Skip empty documents
                        st.markdown(f"**Document {idx}**")
                        st.write(doc)
                        # Use the same run_id for all documents, but differentiate with metadata
                        handle_feedback(tag=f"search_{idx-1}", run_id=run_id, query=user_input, doc_index=idx)
                        st.markdown("---")  # Add a separator between documents
            else:
                st.write("No documents retrieved.")

        with system_tab:
            st.subheader("🔍 System Information")
            system_info = {
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
    else:
        st.warning("No results found. Try a different query.")

elif submit_button:
    st.warning("Please enter a valid query.")

# Footer with additional information or links
st.markdown("---")
st.markdown("Developed for the university subject 'Advanced ML' using Streamlit and LangChain (OpenAI models for embedding and generation, query rewriting, FlashrankRerank, ChromaDB).")
