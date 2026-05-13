import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3


root_dir = os.path.dirname(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(log_dir_path, LOG_FILE)


def get_logger(name="root"):
    logger = logging.getLogger(name)

    # IMPORTANT FIX: prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger instance (safe import anywhere)
logger = get_logger()