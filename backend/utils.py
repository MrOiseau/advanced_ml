import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple, Set

def setup_logging(name: Optional[str] = None, log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name (Optional[str]): Name of the logger.
        log_level (int): Logging level (default: logging.INFO).
        log_file (Optional[str]): Optional file path to log messages.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(log_level)

    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Create handlers
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(stream_handler)

        # Optionally add a file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def remove_duplicates(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate entries from the dataset based on the question, context, and answer.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing 'question', 'context', and 'answer'.
    
    Returns:
        List[Dict[str, Any]]: A list with only unique entries.
    """
    unique_entries: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Tuple[str, ...], str]] = set()

    for entry in data:
        # Create a unique key by combining question, context (converted to tuple), and answer
        entry_key = (entry['question'], tuple(entry['context']), entry['answer'])

        # If the entry hasn't been seen before, add it to the result
        if entry_key not in seen:
            seen.add(entry_key)
            unique_entries.append(entry)

    return unique_entries

if __name__ == "__main__":
    # Setup logger
    logger = setup_logging()

    # Define the input and output file paths
    input_file = "data/evaluation/evaluation_dataset_chatgpt.json"
    output_file = "data/evaluation/evaluation_dataset_chatgpt_unique.json"

    try:
        # Load the dataset from the JSON file
        logger.info(f"Loading dataset from {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Remove duplicates from the dataset
        logger.info("Removing duplicates from the dataset")
        unique_data = remove_duplicates(data)

        # Save the filtered dataset back to a new JSON file
        logger.info(f"Saving unique dataset to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(unique_data, f, indent=4)

        logger.info("Process completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
