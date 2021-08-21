import sys
import os
import logging
from logging.config import dictConfig

# Init logging
console = logging.StreamHandler(stream=sys.stdout)
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)
fmt = logging.Formatter("%(asctime)s [%(levelname)s]:%(filename)s.%(name)s, in line %(lineno)s >> %(message)s")
console.setFormatter(fmt)
console.setLevel(logging.INFO)
root_logger.addHandler(console)

class NotConfigured(Exception):pass

class LoggerAdaptor(logging.LoggerAdapter):
    """
    Log Message Adaptor : Adapt each log message with a specific message prefix

    Key function to implemnt : process(msg : Message, kwargs : Map<Key, Object>)
    """

    """
    :param logger: Logger
    :param prefix: str, message prefix, usually names of class, modules and packages
    :param kwargs: Map<Key, Object>, keywords used by base logging.LoggerAdaptor
    """
    def __init__(self, prefix, logger):
        # super(self, App_LoggerAdaptor).__init__(logger, {})
        logging.LoggerAdapter.__init__(self, logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return "[%s] %s" % (self.prefix, msg), kwargs

# if we have a configuration file something defined like that of django
def configure_logging(config):
    # mkdir
    if not os.path.isdir(config):
        os.makedirs(config)

    if config:
        dictConfig(config)
    else:
        raise NotConfigured(details="passing null config!")


def init_logging():
    logging.basicConfig(level=logging.DEBUG)
    console = logging.StreamHandler(stream=sys.stdout)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s]:%(filename)s.%(name)s, in line %(lineno)s >> %(message)s")
    console.setFormatter(fmt)
    console.setLevel(logging.DEBUG)
    root_logger.addHandler(console)


