"""Threaded RTSP readers: each stream keeps only its latest frame, so slow processing never lags."""
import os
import socket
import threading
import time
from urllib.parse import urlparse

import cv2

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


def reachable(url, timeout=2.0):
    """Does the stream's host accept a connection? OpenCV opens streams one at a time, and opening an
    unreachable address blocks for ~30 s, so every other camera would wait behind it."""
    u = urlparse(url)
    if not u.hostname or u.scheme not in ("rtsp", "rtsps", "http", "https"):
        return True                       # a file, or something we cannot check
    port = u.port or {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}[u.scheme]
    try:
        socket.create_connection((u.hostname, port), timeout).close()
        return True
    except OSError:
        return False


class Stream(threading.Thread):
    def __init__(self, cam, url):
        super().__init__(daemon=True)
        self.cam, self.url = cam, url
        self.frame, self.t, self.ok, self.n = None, 0.0, False, 0
        self._lock = threading.Lock()
        self._stop = False
        self.start()

    def run(self):
        while not self._stop:
            if not reachable(self.url):
                self.ok = False
                time.sleep(3)
                continue
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                                                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
            if not cap.isOpened():
                self.ok = False
                time.sleep(2)
                continue
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break
                with self._lock:
                    self.frame, self.t, self.ok, self.n = frame, time.time(), True, self.n + 1
            cap.release()
            self.ok = False
            time.sleep(1)

    def latest(self):
        with self._lock:
            return (None, 0.0) if self.frame is None else (self.frame.copy(), self.t)

    def stop(self):
        self._stop = True


def open_streams(urls):
    return {cam: Stream(cam, url) for cam, url in urls.items()}


def tile(frames, cols, cell_w=480):
    """Grid of resized frames for display. frames: list of BGR images (or None)."""
    import numpy as np
    if not frames:
        return np.zeros((10, 10, 3), np.uint8)
    h0, w0 = 1080, 1920
    for f in frames:
        if f is not None:
            h0, w0 = f.shape[:2]
            break
    cell_h = int(cell_w * h0 / w0)
    rows = (len(frames) + cols - 1) // cols
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for i, f in enumerate(frames):
        if f is None:
            continue
        r, c = divmod(i, cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cv2.resize(f, (cell_w, cell_h))
    return canvas
