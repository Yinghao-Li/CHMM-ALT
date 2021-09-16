import os
import sys
import tqdm
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """
    Don't let logger print interfere with tqdm progress bar
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# noinspection PyArgumentList
def set_logging(log_dir: Optional[str] = None):
    """
    setup logging
    Last modified: 07/20/21

    Parameters
    ----------
    log_dir: where to save logging file. Leave None to save no log files

    Returns
    -------

    """
    if log_dir:
        log_dir = os.path.abspath(log_dir)
        if not os.path.isdir(os.path.split(log_dir)[0]):
            os.makedirs(os.path.abspath(os.path.normpath(os.path.split(log_dir)[0])))
        if os.path.isfile(log_dir):
            os.remove(log_dir)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                # logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir),
                TqdmLoggingHandler(),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                # logging.StreamHandler(sys.stdout),
                TqdmLoggingHandler()
            ],
        )


def logging_args(args):
    """
    Logging model arguments into logs
    Last modified: 08/19/21

    Parameters
    ----------
    args: arguments

    Returns
    -------
    None
    """
    arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                    and not attr.startswith("__") and not attr.startswith("_")}
    logger.info(f"Parameters: ({type(args)})")
    for arg_element, value in arg_elements.items():
        logger.info(f"\t{arg_element}: {value}")
