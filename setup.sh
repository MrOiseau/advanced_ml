#!/bin/bash
# Setup script for the Advanced RAG System

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/run_experiments.py
chmod +x scripts/generate_report.py
chmod +x scripts/visualize_results.py

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/pdfs
mkdir -p data/db
mkdir -p data/evaluation/results
mkdir -p data/thesis
mkdir -p data/visualizations

echo "Setup completed successfully!"
echo "Please make sure to create a .env file based on .env_example and add your API keys."
echo "You can then run experiments using: python scripts/run_experiments.py default"