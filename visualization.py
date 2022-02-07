#!/usr/bin/python3
import time
import threading
import queue
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import sys

import config
import microphone
import dsp
import led

import logging

logger = logging.getLogger("lightstrip")

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""


def frames_per_second():
    """Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                            alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.N_PIXELS // 2))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y ** 2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Update the LED strip
    return np.concatenate((p[:, ::-1], p), axis=1)

def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= 2.0 * float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3] ** scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3] ** scale))
    b = int(np.mean(y[2 * len(y) // 3:] ** scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)


_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)

def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))
    # Mirror the color channels for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    output = np.array([r, g, b]) * 255
    return output


fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                                alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                              alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()

# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

visualization_effect = visualize_energy
"""Visualization effect to display on the LED strip"""


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    # Normalize samples between 0 and 1
    y = audio_samples / 2.0 ** 15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        logger.info('No audio input. Volume below threshold. Volume: %f' % vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
        return
    # Transform audio input into the frequency domain
    N = len(y_data)
    N_zeros = 2 ** int(np.ceil(np.log2(N))) - N
    # Pad with zeros until the next power of two
    y_data *= fft_window
    y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
    YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
    # Construct a Mel filterbank from the FFT data
    mel = np.atleast_2d(YS).T * dsp.mel_y.T
    # Scale data to values more suitable for visualization
    # mel = np.sum(mel, axis=0)
    mel = np.sum(mel, axis=0)
    mel = mel ** 2.0
    # Gain normalization
    mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
    mel /= mel_gain.value
    mel = mel_smoothing.update(mel)
    # Map filterbank output onto LED strip
    output = visualization_effect(mel)
    led.pixels = output
    led.update()

    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            logger.info('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))
    return


class LightStreamer(threading.Thread):
    def __init__(self, audio_queue):
        threading.Thread.__init__(self)
        logger.info("initializing light streamer...")
        self.daemon = True
        self.audio_queue = audio_queue
        self.running = True
        return

    def shutdown(self):
        self.running = False
        return

    def run(self):
        logger.info("starting light streamer...")
        t = time.time()
        dropped = 0
        while self.running:
            y = self.audio_queue.get()
            while self.audio_queue.qsize() > 0:
                y = self.audio_queue.get()
                dropped += 1
                pass
            if time.time() - t > 10:
                t = time.time()
                logger.info("audio frames dropped=%d" % dropped)
                dropped = 0
                pass
            # update visualization based on microphone input
            microphone_update(y)
            pass
        logger.info("shutting light streamer...")
        return

_audio_reader = None
_light_streamer = None


def stop_everything():
    global _audio_reader, _light_streamer
    if _audio_reader and _audio_reader.is_alive():
        _audio_reader.shutdown()
        _audio_reader.join(1)
        _audio_reader = None
        pass
    if _light_streamer and _light_streamer.is_alive():
        _light_streamer.shutdown()
        _light_streamer.join(1)
        _light_streamer = None
        pass

    # Initialize LEDs to off
    led.pixels = np.tile(1, (3, config.N_PIXELS))
    led.update()
    return


def start_everything():
    global _audio_reader, _light_streamer
    # Initialize LEDs to off
    led.pixels = np.tile(1, (3, config.N_PIXELS))
    led.update()
    # Create audio queue
    audio_queue = queue.Queue()
    _audio_reader = microphone.AudioReader(audio_queue, config.MIC_RATE, config.FPS)
    _light_streamer = LightStreamer(audio_queue)
    _audio_reader.start()
    _light_streamer.start()
    return


def start_energy():
    global visualization_effect
    stop_everything()
    visualization_effect = visualize_energy
    start_everything()
    return


def start_spectrum():
    global visualization_effect
    stop_everything()
    visualization_effect = visualize_spectrum
    start_everything()
    return


def start_scroll():
    global visualization_effect
    stop_everything()
    visualization_effect = visualize_scroll
    start_everything()
    return

