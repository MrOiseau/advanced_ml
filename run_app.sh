#!/bin/bash

# More comprehensive approach to exclude problematic modules from Streamlit's file watcher
# Get paths to problematic modules
TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
TORCH_CLASSES_PATH=$(python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), '_classes.py'))")

# Set Streamlit server options
export STREAMLIT_SERVER_EXCLUDE_DIRS="$TORCH_PATH"
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

# Disable Streamlit's file watcher for external modules
export STREAMLIT_SERVER_WATCH_MODULES=false

# Run the Streamlit app with additional flags
# Use the Python interpreter from the virtual environment
python3 -m streamlit run frontend/app.py --server.fileWatcherType none