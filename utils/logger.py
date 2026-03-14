# utils/logger.py
import logging
import sys # Import sys for stdout

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """ Creates and configures a logger. """
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers if called again
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Use StreamHandler to output to console (stdout)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Optional: Prevent propagation to root logger if it has handlers
        # logger.propagate = False
    return logger

# Example Usage:
# In other files: from utils.logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")