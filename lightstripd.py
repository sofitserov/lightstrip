import time
import httpserver
import logging
import logging.handlers

from config import *

# Named global logger from config
logger = logging.getLogger("lightstrip")

daemonHandler = logging.handlers.RotatingFileHandler(
    LOGFILE, maxBytes=100000, backupCount=5)
daemonHandler.setFormatter(formatter)
logger.addHandler(daemonHandler)
httpserver.runserver()

