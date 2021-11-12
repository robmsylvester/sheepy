import sys
import logging

def get_std_out_logger():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(filename)s:%(lineno)d]'
    ))
    logger = logging.Logger('stdout_logger')
    logger.addHandler(handler)
    return logger