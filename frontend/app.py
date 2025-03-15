# Import config first to ensure environment variables are set before other imports
from backend.config import *
import logging
import streamlit as st
import uuid
import concurrent.futures
import re
import hashlib
import random
from backend.querying import QueryPipeline
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled
from backend.utils import setup_logging

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="AI Document Retrieval", layout="wide")

# Configure logging
logger = setup_logging(__name__)

# Helper functions for finding and highlighting common text chunks
def preprocess_text(text: str) -> list:
    """
    Preprocess the text by removing non-word characters, extra spaces, and converting to lowercase.
    Returns a list of words.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        list: List of preprocessed words
    """
    # Remove non-word characters and extra spaces
    text = re.sub(r'[\W_]+', ' ', text, flags=re.UNICODE)  # Remove symbols that are not words or numbers
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.lower().strip()
    words = text.split()
    return words

def find_longest_common_substring(text1, text2, min_words=5):
    """
    Find the longest common substring between two texts using dynamic programming.
    This is optimized to find the single longest common substring.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        min_words (int): Minimum number of consecutive words to consider a match
        
    Returns:
        str: The longest common substring, or empty string if none found
    """
    # Normalize texts (lowercase, preserve punctuation but normalize whitespace)
    text1_norm = re.sub(r'\s+', ' ', text1.lower()).strip()
    text2_norm = re.sub(r'\s+', ' ', text2.lower()).strip()
    
    # Split into words
    words1 = text1_norm.split()
    words2 = text2_norm.split()
    
    # Initialize the DP table
    dp = [[0 for _ in range(len(words2) + 1)] for _ in range(len(words1) + 1)]
    
    # Variables to keep track of the maximum length and ending position
    max_length = 0
    end_pos = 0
    
    # Fill the DP table
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    
    # If we found a common substring of at least min_words
    if max_length >= min_words:
        # Extract the original text (with original case and punctuation)
        start_pos = end_pos - max_length
        # Find the corresponding position in the original text
        word_positions = []
        current_pos = 0
        for i, word in enumerate(text1.split()):
            word_positions.append(current_pos)
            current_pos += len(word) + 1  # +1 for the space
        
        # Get the start and end positions in the original text
        if start_pos < len(word_positions):
            orig_start = word_positions[start_pos]
            if end_pos < len(word_positions):
                orig_end = word_positions[end_pos]
            else:
                orig_end = len(text1)
            
            # Extract the substring from the original text
            return text1[orig_start:orig_end].strip()
    
    return ""

def find_common_sequences(text1, text2, min_words=5):
    """
    Find sequences of at least min_words consecutive words that appear in both texts.
    Prioritizes finding the longest common substring.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        min_words (int): Minimum number of consecutive words to consider a match
        
    Returns:
        list: List of common sequences found in both texts
    """
    # Check for exact match first (optimization for identical texts)
    if text1.strip() == text2.strip():
        logger.info("Exact match found between documents")
        return [text1.strip()]
    
    # Normalize texts (lowercase, preserve punctuation but normalize whitespace)
    text1_norm = re.sub(r'\s+', ' ', text1.lower()).strip()
    text2_norm = re.sub(r'\s+', ' ', text2.lower()).strip()
    
    # Check for exact match after normalization
    if text1_norm == text2_norm:
        logger.info("Exact match found after normalization")
        return [text1.strip()]
    
    # Find the longest common substring using dynamic programming
    longest_common = find_longest_common_substring(text1, text2, min_words)
    
    if longest_common:
        logger.info(f"Found longest common substring: {len(longest_common.split())} words")
        return [longest_common]
    
    # If no long common substring found, fall back to the previous approach
    # Split into words
    words1 = text1_norm.split()
    words2 = text2_norm.split()
    
    common_sequences = []
    
    # Fall back to word-by-word comparison for smaller matches
    for i in range(len(words1) - min_words + 1):
        current_sequence = words1[i:i + min_words]
        sequence_str = ' '.join(current_sequence)
        
        # Check if this sequence exists in the second text
        if sequence_str in text2_norm:
            # Try to extend the sequence as much as possible
            j = i + min_words
            while j < len(words1):
                extended_sequence = words1[i:j+1]
                extended_str = ' '.join(extended_sequence)
                if extended_str in text2_norm:
                    j += 1
                else:
                    break
            
            # Add the longest matching sequence
            final_sequence = ' '.join(words1[i:j])
            common_sequences.append(final_sequence)
    
    # Remove duplicates and sort by length (longest first)
    unique_sequences = list(set(common_sequences))
    unique_sequences.sort(key=len, reverse=True)
    
    return unique_sequences

def generate_consistent_color(text):
    """
    Generate a consistent color based on the text content.
    
    Args:
        text (str): Text to generate color for
        
    Returns:
        str: Hex color code
    """
    # Use hash of text to generate a consistent color
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    
    # Ensure the color is light enough to see text on it
    r = min(255, r + 100)
    g = min(255, g + 100)
    b = min(255, b + 100)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def highlight_common_sequences(text, common_sequences, color_map):
    """
    Highlight common sequences in the text with their assigned colors.
    The function first finds all match intervals (non-overlapping) for the provided sequences
    (case-insensitively) and then reconstructs the text with HTML spans inserted.
    
    Args:
        text (str): Text to highlight.
        common_sequences (list): List of sequences to highlight.
        color_map (dict): Mapping of sequences to their colors.
    
    Returns:
        str: HTML-formatted text with highlights.
    """
    if not common_sequences:
        return text

    # List to hold all found match intervals: (start_index, end_index, color)
    intervals = []
    lower_text = text.lower()

    # Process each sequence (prioritizing longer ones by sorting them)
    for seq in sorted(common_sequences, key=lambda s: -len(s)):
        seq_lower = seq.lower().strip()
        if not seq_lower:
            continue
        start = 0
        while True:
            idx = lower_text.find(seq_lower, start)
            if idx == -1:
                break
            end = idx + len(seq_lower)
            # Add the interval if it does not overlap an already recorded interval.
            overlap = any(not (end <= existing[0] or idx >= existing[1]) for existing in intervals)
            if not overlap:
                intervals.append((idx, end, color_map[seq]))
            start = idx + 1

    # Sort intervals by their starting position
    intervals.sort(key=lambda x: x[0])

    # Rebuild the highlighted text by iterating over intervals
    highlighted_text = ""
    last_index = 0
    for start, end, color in intervals:
        # Append unchanged text between the last match and current match
        highlighted_text += text[last_index:start]
        # Append highlighted match
        highlighted_text += f'<span style="background-color: {color};">{text[start:end]}</span>'
        last_index = end
    # Append any remaining text after the last match
    highlighted_text += text[last_index:]

    return highlighted_text

# Use constants from config module
FEEDBACK_OPTION = "faces"  # Options can be "thumbs" or "faces"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Get minimum words for common sequence highlighting from environment variable
# Default to 5 if not specified
MIN_COMMON_WORDS = int(os.getenv("MIN_COMMON_WORDS", "5"))

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
        # Get rerank setting from environment variable
        rerank = os.getenv("RERANK", "true").lower() == "true"
        logger.info(f"RERANK setting: {rerank}")

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
            rerank=rerank,         # Use environment variable setting
        )
        logger.info("QueryPipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize QueryPipeline: {e}")
        st.error(f"Initialization error: {e}")
        st.stop()

# Define available collections
# available_collections = [
#     {"name": "rag_collection_recursive_character", "description": "Recursive Character Chunking"},
#     {"name": "rag_collection_semantic_clustering", "description": "Semantic Clustering Chunking"},
#     {"name": "rag_collection_topic_based", "description": "Topic-Based Chunking"},
#     {"name": "rag_collection_hierarchical", "description": "Hierarchical Chunking"},
#     {"name": "rag_collection_sentence_transformers", "description": "Sentence Transformers Chunking"}
# ]
available_collections = [
    {"name": "wfp_collection_recursive_character", "description": "Recursive Character Chunking"},
    {"name": "wfp_collection_semantic_clustering", "description": "Semantic Clustering Chunking"},
    {"name": "wfp_collection_topic_based", "description": "Topic-Based Chunking"},
    {"name": "wfp_collection_hierarchical", "description": "Hierarchical Chunking"},
    {"name": "wfp_collection_sentence_transformers", "description": "Sentence Transformers Chunking"}
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
        # Get rerank setting from environment variable
        rerank = os.getenv("RERANK", "true").lower() == "true"
        logger.info(f"RERANK setting: {rerank}")

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
            rerank=rerank,         # Use environment variable setting
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

# Collection Selection for two databases
st.sidebar.subheader("Vector Database Selection")
st.sidebar.markdown("### Database 1")
selected_collection_1 = st.sidebar.selectbox(
    "Select Vector Database 1:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{x} ({collection_descriptions.get(x, 'Unknown')})",
    help="Choose the first vector database to query.",
    key="db1"
)

st.sidebar.markdown("### Database 2")
selected_collection_2 = st.sidebar.selectbox(
    "Select Vector Database 2:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{x} ({collection_descriptions.get(x, 'Unknown')})",
    help="Choose the second vector database to query.",
    key="db2",
    index=1 if len(available_collections) > 1 else 0  # Select a different default if possible
)

# Initialize Query Pipelines for both selected collections
# Pipeline 1
if "query_pipeline_1" not in st.session_state or st.session_state.get("current_collection_1") != selected_collection_1:
    with st.spinner(f"Loading {collection_descriptions.get(selected_collection_1, 'Unknown')} database..."):
        st.session_state["query_pipeline_1"] = initialize_query_pipeline(selected_collection_1)
        st.session_state["current_collection_1"] = selected_collection_1
        st.success(f"Loaded {collection_descriptions.get(selected_collection_1, 'Unknown')} database")

# Pipeline 2
if "query_pipeline_2" not in st.session_state or st.session_state.get("current_collection_2") != selected_collection_2:
    with st.spinner(f"Loading {collection_descriptions.get(selected_collection_2, 'Unknown')} database..."):
        st.session_state["query_pipeline_2"] = initialize_query_pipeline(selected_collection_2)
        st.session_state["current_collection_2"] = selected_collection_2
        st.success(f"Loaded {collection_descriptions.get(selected_collection_2, 'Unknown')} database")

# Get both query pipelines
query_pipeline_1 = st.session_state["query_pipeline_1"]
query_pipeline_2 = st.session_state["query_pipeline_2"]

# Fetch unique titles for filtering (using the first pipeline for simplicity)
unique_titles = query_pipeline_1.get_unique_titles()

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

            # Process queries for both databases in parallel
            results = {}
            trace_urls = {}
            
            # Define functions to run each query with its own tracing context
            def run_query_1():
                with tracing_v2_enabled() as cb1:
                    result = query_pipeline_1.run_query(user_input, metadata_filter=metadata_filter)
                    run_id = result.get("run_id")
                    trace_url = cb1.get_run_url()
                    
                    if not run_id:
                        run_id = str(uuid.uuid4())
                        logger.warning("No run_id returned from run_query for DB1. Generated a new run_id.")
                    
                    return result, run_id, trace_url
            
            def run_query_2():
                with tracing_v2_enabled() as cb2:
                    result = query_pipeline_2.run_query(user_input, metadata_filter=metadata_filter)
                    run_id = result.get("run_id")
                    trace_url = cb2.get_run_url()
                    
                    if not run_id:
                        run_id = str(uuid.uuid4())
                        logger.warning("No run_id returned from run_query for DB2. Generated a new run_id.")
                    
                    return result, run_id, trace_url
            
            # Execute both queries in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both query tasks
                future1 = executor.submit(run_query_1)
                future2 = executor.submit(run_query_2)
                
                # Get results as they complete
                result1, run_id1, trace_url1 = future1.result()
                result2, run_id2, trace_url2 = future2.result()
                
                # Store results
                results[1] = result1
                results[2] = result2
                trace_urls[1] = trace_url1
                trace_urls[2] = trace_url2

            # Store results in session state
            st.session_state["run_id_1"] = run_id1
            st.session_state["run_id_2"] = run_id2
            st.session_state["last_result_1"] = result1
            st.session_state["last_result_2"] = result2
            st.session_state["last_user_input"] = user_input
            st.session_state["trace_urls"] = trace_urls

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
            logger.error(f"Query processing error: {e}")
            results = {}

    if results:
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Database 1 Results (Left Column)
        with col1:
            st.header(f"Database 1: {collection_descriptions.get(selected_collection_1, 'Unknown')}")
            
            # Create tabs for different views
            summary_tab1, documents_tab1, system_tab1 = st.tabs(["Summary", "Documents", "System"])
            
            # Generate summaries in parallel
            def generate_summary_1():
                return query_pipeline_1.generate_summary(user_input, results[1].get("documents", ""))
                
            def generate_summary_2():
                return query_pipeline_2.generate_summary(user_input, results[2].get("documents", ""))
                
            # Start summary generation in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_summary1 = executor.submit(generate_summary_1)
                future_summary2 = executor.submit(generate_summary_2)
                
                # We'll get the results when we need them in the UI
            
            with summary_tab1:
                st.subheader("📚 Summary")
                # Get summary result when needed
                summary1 = future_summary1.result()
                st.write(summary1)
            
            # Process documents for highlighting common sequences
            documents1 = results[1].get("documents", "")
            documents2 = results[2].get("documents", "")
            
            # Split documents into individual chunks
            doc_list1 = documents1.split("\n\n---\n\n") if documents1 else []
            doc_list2 = documents2.split("\n\n---\n\n") if documents2 else []
            
            # Extract titles and content separately from documents
            doc_titles1 = []
            doc_titles2 = []
            doc_contents1 = []
            doc_contents2 = []
            
            for doc in doc_list1:
                lines = doc.strip().split('\n')
                if len(lines) >= 3:  # Document number, blank line, title
                    doc_titles1.append(lines[2].strip())
                    # Extract content (skip document number, blank line, and title)
                    content = '\n'.join(lines[3:]) if len(lines) > 3 else ""
                    doc_contents1.append(content)
                else:
                    doc_titles1.append("")
                    doc_contents1.append(doc)
            
            for doc in doc_list2:
                lines = doc.strip().split('\n')
                if len(lines) >= 3:  # Document number, blank line, title
                    doc_titles2.append(lines[2].strip())
                    # Extract content (skip document number, blank line, and title)
                    content = '\n'.join(lines[3:]) if len(lines) > 3 else ""
                    doc_contents2.append(content)
                else:
                    doc_titles2.append("")
                    doc_contents2.append(doc)
            
            logger.info(f"Extracted titles from DB1: {doc_titles1}")
            logger.info(f"Extracted titles from DB2: {doc_titles2}")
            
            # Find common sequences across document pairs, prioritizing same titles
            all_common_sequences = []
            title_matched_pairs = []
            
            # First, match documents with the same titles
            for i, content1 in enumerate(doc_contents1):
                for j, content2 in enumerate(doc_contents2):
                    if content1.strip() and content2.strip():
                        title1 = doc_titles1[i] if i < len(doc_titles1) else ""
                        title2 = doc_titles2[j] if j < len(doc_titles2) else ""
                        
                        if title1 and title2 and title1 == title2:
                            logger.info(f"Found documents with matching title: '{title1}'")
                            title_matched_pairs.append((content1, content2))
            
            # For documents with matching titles, find the longest common sequence
            for content1, content2 in title_matched_pairs:
                longest_common = find_longest_common_substring(content1, content2, min_words=MIN_COMMON_WORDS)
                if longest_common:
                    all_common_sequences.append(longest_common)
                    logger.info(f"Found longest common sequence between title-matched documents: {len(longest_common.split())} words")
                    logger.info(f"Sequence: '{longest_common[:100]}{'...' if len(longest_common) > 100 else ''}'")
            
            # If no matches found with title matching, fall back to comparing all document contents
            if not all_common_sequences:
                logger.info("No matches found with title matching, comparing all documents")
                for content1 in doc_contents1:
                    for content2 in doc_contents2:
                        if content1.strip() and content2.strip():
                            common_sequences = find_common_sequences(content1, content2, min_words=MIN_COMMON_WORDS)
                            all_common_sequences.extend(common_sequences)
            
            # Remove duplicates
            all_common_sequences = list(set(all_common_sequences))
            
            # Log information about found sequences
            if all_common_sequences:
                logger.info(f"Found {len(all_common_sequences)} common sequences with at least {MIN_COMMON_WORDS} consecutive words")
                
                # Log the top 3 longest sequences
                for i, seq in enumerate(sorted(all_common_sequences, key=len, reverse=True)[:3], 1):
                    word_count = len(seq.split())
                    logger.info(f"Example {i}: {word_count} words, '{seq[:100]}{'...' if len(seq) > 100 else ''}'")
                
                # Log exact matches if any
                exact_matches = []
                for doc1 in doc_list1:
                    for doc2 in doc_list2:
                        if doc1.strip() and doc2.strip() and (doc1.strip() == doc2.strip() or doc1.lower().strip() == doc2.lower().strip()):
                            exact_matches.append(doc1.strip())
                
                if exact_matches:
                    logger.info(f"Found {len(exact_matches)} exact document matches")
                    for i, match in enumerate(exact_matches[:1], 1):  # Log just the first one to avoid verbosity
                        logger.info(f"Exact match {i}: '{match[:50]}{'...' if len(match) > 50 else ''}'")
            else:
                logger.info(f"No common sequences with at least {MIN_COMMON_WORDS} consecutive words found")
                
                # Log a sample of documents to help debug
                if doc_list1 and doc_list2:
                    logger.info(f"Sample document from DB1: '{doc_list1[0][:100]}{'...' if len(doc_list1[0]) > 100 else ''}'")
                    logger.info(f"Sample document from DB2: '{doc_list2[0][:100]}{'...' if len(doc_list2[0]) > 100 else ''}'")
            
            # Create a color map for consistent highlighting
            color_map = {seq: generate_consistent_color(seq) for seq in all_common_sequences}
            
            with documents_tab1:
                st.subheader("📄 Retrieved Documents")
                if doc_list1:
                    for idx, doc in enumerate(doc_list1, 1):
                        if doc.strip():  # Skip empty documents
                            st.markdown(f"**Document {idx}**")
                            
                            # Split document into title and content for highlighting
                            lines = doc.strip().split('\n')
                            title_part = ""
                            content_part = doc
                            
                            if len(lines) >= 3:  # Document number, blank line, title
                                title_part = lines[0] + '\n' + lines[1] + '\n' + lines[2]
                                content_part = '\n'.join(lines[3:]) if len(lines) > 3 else ""
                            
                            # Display title without highlighting
                            st.markdown(title_part)
                            
                            # Apply highlighting only to the content part
                            if content_part.strip():
                                highlighted_content = highlight_common_sequences(content_part, all_common_sequences, color_map)
                                # Use unsafe_allow_html to render the HTML with highlighting
                                st.markdown(highlighted_content, unsafe_allow_html=True)
                            
                            st.markdown("---")  # Add a separator between documents
                else:
                    st.write("No documents retrieved.")
            
            with system_tab1:
                st.subheader("🔍 System Information")
                
                # Add collection/chunking method info
                system_info1 = {
                    "Collection": selected_collection_1,
                    "Chunking Method": collection_descriptions.get(selected_collection_1, "Unknown"),
                    "Embedding Model": query_pipeline_1.embedding_model,
                    "Chat Model": query_pipeline_1.chat_model,
                    "Temperature": query_pipeline_1.chat_temperature,
                    "Search Results Number": query_pipeline_1.search_results_num,
                }
                st.json(system_info1)
                
                # Display LangSmith trace link
                if trace_urls.get(1):
                    st.markdown(f"[View LangSmith Trace]({trace_urls[1]})")
        
        # Database 2 Results (Right Column)
        with col2:
            st.header(f"Database 2: {collection_descriptions.get(selected_collection_2, 'Unknown')}")
            
            # Create tabs for different views
            summary_tab2, documents_tab2, system_tab2 = st.tabs(["Summary", "Documents", "System"])
            
            with summary_tab2:
                st.subheader("📚 Summary")
                # Get summary result when needed
                summary2 = future_summary2.result()
                st.write(summary2)
            
            with documents_tab2:
                st.subheader("📄 Retrieved Documents")
                if doc_list2:
                    for idx, doc in enumerate(doc_list2, 1):
                        if doc.strip():  # Skip empty documents
                            st.markdown(f"**Document {idx}**")
                            
                            # Split document into title and content for highlighting
                            lines = doc.strip().split('\n')
                            title_part = ""
                            content_part = doc
                            
                            if len(lines) >= 3:  # Document number, blank line, title
                                title_part = lines[0] + '\n' + lines[1] + '\n' + lines[2]
                                content_part = '\n'.join(lines[3:]) if len(lines) > 3 else ""
                            
                            # Display title without highlighting
                            st.markdown(title_part)
                            
                            # Apply highlighting only to the content part
                            if content_part.strip():
                                highlighted_content = highlight_common_sequences(content_part, all_common_sequences, color_map)
                                # Use unsafe_allow_html to render the HTML with highlighting
                                st.markdown(highlighted_content, unsafe_allow_html=True)
                            
                            st.markdown("---")  # Add a separator between documents
                else:
                    st.write("No documents retrieved.")
            
            with system_tab2:
                st.subheader("🔍 System Information")
                
                # Add collection/chunking method info
                system_info2 = {
                    "Collection": selected_collection_2,
                    "Chunking Method": collection_descriptions.get(selected_collection_2, "Unknown"),
                    "Embedding Model": query_pipeline_2.embedding_model,
                    "Chat Model": query_pipeline_2.chat_model,
                    "Temperature": query_pipeline_2.chat_temperature,
                    "Search Results Number": query_pipeline_2.search_results_num,
                }
                st.json(system_info2)
                
                # Display LangSmith trace link
                if trace_urls.get(2):
                    st.markdown(f"[View LangSmith Trace]({trace_urls[2]})")
        
        # Add highlighting explanation if common sequences were found
        if all_common_sequences:
            st.markdown("---")
            st.subheader("🔍 Common Content Highlighting")
            st.markdown(f"""
            Text highlighted with the same color in both columns represents identical content chunks
            (sequences of at least {MIN_COMMON_WORDS} consecutive words) found in both databases. This helps identify:
            
            - Common information retrieved by different chunking methods
            - How different chunking strategies affect context preservation
            - Which chunks contain the most relevant information (appearing in both results)
            
            The highlighting algorithm prioritizes:
            1. Documents with the same titles (e.g., "evaluation_of_the_wfp_response_to_the_covid19_pandemic_615_cut")
            2. The longest common text sequences between documents with matching titles
            3. Falls back to comparing all documents if no title matches are found
            
            *Note: The minimum number of consecutive words can be configured using the MIN_COMMON_WORDS environment variable.*
            """)
            
            # Display a few examples of common sequences if available
            if len(all_common_sequences) > 0:
                with st.expander("View examples of common sequences"):
                    # Show up to 5 examples, prioritizing longer sequences
                    examples = sorted(all_common_sequences, key=len, reverse=True)[:5]
                    for i, seq in enumerate(examples, 1):
                        color = color_map[seq]
                        st.markdown(f"**Example {i}:** <span style='background-color: {color};'>{seq}</span>", unsafe_allow_html=True)
        
        # Add chunking methods comparison at the bottom
        st.markdown("---")
        with st.expander("📊 Chunking Methods Comparison"):
            st.markdown("""
            | Collection | Chunking Method | Best For |
            |------------|-----------------|----------|
            | wfp_collection_recursive_character | Recursive Character | Simple text with uniform content |
            | wfp_collection_semantic_clustering | Semantic Clustering | Content with varying topics and themes |
            | wfp_collection_topic_based | Topic-Based | Documents with distinct topical sections |
            | wfp_collection_hierarchical | Hierarchical | Structured documents with headings and sections |
            | wfp_collection_sentence_transformers | Sentence Transformers | Semantic similarity and context preservation |
            """)
    else:
        st.warning("No results found. Try a different query.")

elif submit_button:
    st.warning("Please enter a valid query.")

# Add a note about the comparison feature
if not submit_button or not user_input.strip():
    st.info("""
    **Side-by-Side Comparison Feature**
    This interface allows you to compare results from two different vector databases simultaneously.
    Select different chunking methods in the sidebar to see how they affect retrieval and summarization quality.
    
    Enter your query above and click 'Search' to see the comparison in action.
    """)

# Footer with additional information or links
st.markdown("---")
st.markdown("Developed for the university master thesis 'The Impact of Chunking Methods on Information Retrieval Quality in RAG Systems' using Streamlit and LangChain (OpenAI models for embedding and generation, query rewriting, FlashrankRerank, ChromaDB).")
