# This is a sample Python script.
import visualization
import sys
import time

import logging

logger = logging.getLogger("lightstrip")

if __name__ == '__main__':
    if sys.argv[1] == "spectrum":
        visualization.start_spectrum()
    elif sys.argv[1] == "energy":
        visualization.start_energy()
    elif sys.argv[1] == "scroll":
        visualization.start_scroll()
    else:
        logger.error("Unknown visualization effect:", sys.argv[1])
        sys.exit(-1)

    # Start listening to live audio stream
    try:
        time.sleep(84600)
    except KeyboardInterrupt as e:
        logger.critical("got keyboard interrupt, exiting.")
        pass
    visualization.stop_everything()
    pass
