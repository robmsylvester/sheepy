import logging

from rich.logging import RichHandler


def get_std_out_logger(log_level=logging.INFO):
    handler = RichHandler(rich_tracebacks=True, log_time_format="%Y-%m-%d %H:%M:%S")
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.Logger("sheepy")
    logger.addHandler(handler)
    return logger
