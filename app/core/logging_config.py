import logging
import os
from logging.handlers import RotatingFileHandler

# Define constants but don't execute logic yet
LOG_DIR = "logs"
MAX_BYTES = 5 * 1024 * 1024  
BACKUP_COUNT = 5       

# Define Loggers (Lazy initialization)
app_logger = logging.getLogger("app_logger")
security_logger = logging.getLogger("security_logger")
error_logger = logging.getLogger("error_logger")
access_logger = logging.getLogger("uvicorn.access")

def _create_handler(filename: str, level: int = logging.INFO) -> RotatingFileHandler:
    file_path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(
        file_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.setLevel(level)
    return handler

def configure_logging():
    """Called at startup to initialize logging rules."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    app_logger.setLevel(logging.DEBUG)
    app_logger.addHandler(_create_handler("app.log", logging.DEBUG))
    app_logger.addHandler(logging.StreamHandler()) 

    security_logger.setLevel(logging.INFO)
    security_logger.addHandler(_create_handler("security.log", logging.INFO))

    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(_create_handler("error.log", logging.ERROR))


    access_logger.handlers = [] 
    access_file_handler = _create_handler("access.log", logging.INFO)
    access_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    access_logger.addHandler(access_file_handler)
    access_logger.addHandler(logging.StreamHandler())