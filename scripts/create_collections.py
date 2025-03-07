#!/usr/bin/env python3
"""
Script to create multiple ChromaDB collections with different chunking methods.

This script automates the process of creating multiple vector database collections,
each using a different chunking method, to enable comparison in the frontend app.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import *
from backend.chunkers.base import ChunkerFactory


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create multiple ChromaDB collections with different chunking methods."
    )
    
    parser.add_argument(
        "--pdf-dir",
        default=PDF_DIR,
        help="Directory containing PDF files to ingest."
    )
    
    parser.add_argument(
        "--db-dir",
        default=DB_DIR,
        help="Directory to store the ChromaDB database."
    )
    
    parser.add_argument(
        "--chunkers",
        nargs="+",
        choices=ChunkerFactory.list_chunkers(),
        default=ChunkerFactory.list_chunkers(),
        help="List of chunkers to use for creating collections."
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of collections even if they already exist."
    )
    
    return parser.parse_args()


def create_collection(chunker_name: str, db_dir: str, pdf_dir: str, force: bool = False) -> bool:
    """
    Create a ChromaDB collection using the specified chunker.
    
    Args:
        chunker_name (str): Name of the chunker to use.
        db_dir (str): Directory to store the ChromaDB database.
        pdf_dir (str): Directory containing PDF files to ingest.
        force (bool): Force recreation of collection even if it already exists.
        
    Returns:
        bool: True if collection was created successfully, False otherwise.
    """
    collection_name = f"rag_collection_{chunker_name}"
    
    # Check if collection already exists
    collection_path = os.path.join(db_dir, "chroma", collection_name)
    if os.path.exists(collection_path) and not force:
        print(f"Collection {collection_name} already exists. Use --force to recreate.")
        return False
    
    # Set up environment variables for the subprocess
    env = os.environ.copy()
    env["DB_COLLECTION"] = collection_name
    env["PDF_DIR"] = pdf_dir
    env["DB_DIR"] = db_dir
    
    # Build the command
    cmd = [sys.executable, "-m", "backend.indexing"]
    
    # Add chunker-specific parameters
    if chunker_name != "recursive_character":
        cmd.extend(["--chunker_name", chunker_name])
    
    # Run the ingestion process
    print(f"Creating collection {collection_name} with chunker {chunker_name}...")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully created collection {collection_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating collection {collection_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """
    Main function to create multiple ChromaDB collections.
    """
    args = parse_args()
    
    # Create the DB directory if it doesn't exist
    os.makedirs(args.db_dir, exist_ok=True)
    
    # Check if PDF directory exists and contains files
    if not os.path.exists(args.pdf_dir):
        print(f"Error: PDF directory {args.pdf_dir} does not exist.")
        sys.exit(1)
    
    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"Error: No PDF files found in {args.pdf_dir}.")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files in {args.pdf_dir}")
    
    # Create collections for each chunker
    successful = 0
    for chunker_name in args.chunkers:
        if create_collection(chunker_name, args.db_dir, args.pdf_dir, args.force):
            successful += 1
    
    print(f"\nSummary: Created {successful} out of {len(args.chunkers)} collections.")
    print("\nYou can now use these collections in the frontend app:")
    print("streamlit run frontend/app.py")


if __name__ == "__main__":
    main()