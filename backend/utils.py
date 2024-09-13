import logging
import os

def setup_logging(name=None, log_level=logging.INFO, log_file=None):
    """Set up logging configuration."""
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
