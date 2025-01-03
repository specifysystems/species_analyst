"""Module containing standardized logging class for lmpy."""
from io import StringIO
import logging
import os
from pprint import pp
import sys

from spanalyst.common.util import get_today_str


# .............................................................................
DIR = "log"
INTERVAL = 1000000
FORMAT = " ".join(["%(asctime)s", "%(levelname)-8s", "%(message)s"])
DATE_FORMAT = '%d %b %Y %H:%M'
FILE_MAX_BYTES = 52000000
FILE_BACKUP_COUNT = 5


# ......................................................
def prettify_object(print_obj):
    """Format an object for output.

    Args:
        print_obj (obj): Object to pretty print in output

    Returns:
        formatted string representation of object
    """
    strm = StringIO()
    pp(print_obj, stream=strm)
    obj_str = strm.getvalue()
    return obj_str


# ...............................................
def logit(msg, logger=None, refname=None, print_obj=None, log_level=logging.INFO):
    """Method to log a message to a logger/file/stream or print to console.

    Args:
        msg: message to print.
        logger (bison.common.log.Logger): logger instance or None
        refname: calling function name.
        print_obj: object to pretty print for increased readability.
        log_level: logging constant error level (logging.INFO, logging.DEBUG,
            logging.WARNING, logging.ERROR)
    """
    if print_obj is not None:
        obj_str = prettify_object(print_obj)
        msg = f"{msg}\n{obj_str}"
    if refname is not None:
        msg = f"{refname}: {msg}"
    if logger is not None:
        logger.log(msg, log_level=log_level)
    else:
        print(msg)


# .....................................................................................
class Logger:
    """Class containing a logger for consistent logging."""

    # .......................
    def __init__(
            self, log_name, log_path=None, log_console=True, log_level=logging.DEBUG):
        """Constructor.

        Args:
            log_name (str): A name for the logger.
            log_path (str): Path for logfile.
            log_console (bool): Flag indicating logs be written to the console.
            log_level (int): What level of logs should be retained.
        """
        self.logger = None
        todaystr = get_today_str()
        self.name = f"{log_name}_{todaystr}"
        if log_path is None:
            log_path = os.getcwd()
        self.log_directory = log_path
        os.makedirs(self.log_directory, exist_ok=True)
        self.filename = os.path.join(self.log_directory, f"{self.name}.log")
        self.log_console = log_console
        self.log_level = log_level

        handlers = []
        handlers.append(logging.FileHandler(self.filename, mode="w"))
        if self.log_console:
            handlers.append(logging.StreamHandler(stream=sys.stdout))

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(FORMAT, DATE_FORMAT)
        for handler in handlers:
            handler.setLevel(self.log_level)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False

    # ........................
    def log(self, msg, refname=None, log_level=logging.INFO):
        """Log a message.

        Args:
            msg (str): A message to write to the logger.
            refname (str): Class or function name to use in logging message.
            log_level (int): A level to use when logging the message.
        """
        if self.logger is not None:
            if refname is not None:
                msg = f"{refname}: {msg}"
            self.logger.log(log_level, msg)


# .....................................................................................
__all__ = ["Logger"]
