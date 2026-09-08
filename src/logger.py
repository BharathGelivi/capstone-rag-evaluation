import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a centralized logger.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers are present to avoid duplication
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Output to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
