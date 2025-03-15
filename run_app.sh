#!/bin/bash

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

# Set the PYTHONPATH to include the project root directory
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
fi

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run frontend/app.py