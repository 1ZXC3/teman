import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging():
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
