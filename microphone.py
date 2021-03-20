import numpy as np
import pyaudio
import threading
import queue
import logging
import time

import config

logger = logging.getLogger("lightstrip")


class AudioReader(threading.Thread):
    def __init__(self, audio_queue, rate, fps):
        threading.Thread.__init__(self)
        logger.info("initializing audio reader...")
        self.daemon = True
        self.audio_queue = audio_queue
        self.audio = pyaudio.PyAudio()
        self.frames_per_buffer = int(rate / fps)
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer)
        self.running = True
        return

    def shutdown(self):
        self.running = False
        return

    def run(self):
        logger.info("starting audio streamer...")
        while self.running:
            y = np.fromstring(
                self.stream.read(
                    self.frames_per_buffer,
                    exception_on_overflow=False),
                dtype=np.int16)
            y = y.astype(np.float32)
            self.audio_queue.put_nowait(y)
            pass
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        logger.info("shutting audio streamer...")
        return

