import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FILENAME = 'output/agent_run.log'


def setup_logging(log_level=logging.INFO):
    """
    Configures logging for the application.

    Logs will be sent to both a rotating file and the console.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Root logger setup (adjust level as needed)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)  # Set root level

    # File Handler (Rotating)
    # Rotates when file reaches 2MB, keeps 5 backup files
    file_handler = RotatingFileHandler(
        LOG_FILENAME, maxBytes=2 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    # Set file handler level - maybe log more details to file?
    file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to file
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)  # Log INFO and above to console (or as passed)
    root_logger.addHandler(console_handler)

    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging configured: Level=%s, File=%s", logging.getLevelName(log_level), LOG_FILENAME)


# Example usage (optional)
if __name__ == '__main__':
    setup_logging(log_level=logging.DEBUG)
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
    # Check the 'output/agent_run.log' file