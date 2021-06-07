import os
from datetime import datetime
import logging
import logging.handlers


def setFilename():
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")

    return filename


class TraceLog(object):

    def __init__(self):
        self.logger = logging.getLogger('crumbs')
        self.logger.setLevel(logging.DEBUG)

        self.filename = setFilename()
        self.file_max_bytes = 10 * 1024 * 1024

        self.file_handler = logging.handlers.RotatingFileHandler(filename='.{}.txt'.format(self.filename),
                                                                 maxBytes=self.file_max_bytes,
                                                                 backupCount=10)
        self.stream_handler = logging.StreamHandler()

        self.fomatter = logging.Formatter('%(asctime)s - %(message)s')
        self.file_handler.setFormatter(self.fomatter)
        self.stream_handler.setFormatter(self.fomatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)

        self.logger.info('Start')

    def __del__(self):
        self.logger.info('End')

    def write(self, msg):
        self.logger.info(msg)
