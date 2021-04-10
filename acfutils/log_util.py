import os
import sys
import logging
from datetime import datetime


class ACFLogger(object):
    """docstring for ACFLogger"""

    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"

    def __init__(self, name, filepath=".", suffix=None, level=logging.DEBUG):
        log_name = "{name}_{date}".format(name=name, date=datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        if suffix is not None:
            log_name += "_{suff}".format(suff=suffix)

        log_name += ".log"

        if filepath is not None:
            logging.basicConfig(filename=os.path.join(filepath, log_name),
                                filemode='a',
                                format=self.FORMAT)
        else:
            logging.basicConfig(format=self.FORMAT)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def get_logger(self):
        return self.logger


def create_logger(name, filepath=".", suffix=None, level=logging.DEBUG):
    logger = ACFLogger(name, filepath=filepath, suffix=suffix, level=level)
    return logger.get_logger()
