# -*- coding: utf-8 -*-
"""A top-down PNG of the map, so a human can point at the chairs.

A table of seeds asks you to recognise furniture from four numbers. A picture of
the floor asks you to recognise furniture from its shape, which is what eyes are
for. This writes that picture with nothing but numpy and the standard library —
no matplotlib, no PIL, because the machine holding the map is the robot's and
installing a plotting stack on it to look at ten blobs is not a fair trade.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

# A 3x5 bitmap font. Only digits and a few letters: the labels on this image are
# proposal numbers, and a number is all a reader needs to find the row it names.
FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    " ": ("000", "000", "000", "000", "000"),
}


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an (h, w, 3) uint8 array as a PNG."""
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    height, width, _ = rgb.shape
    # Filter byte 0 (None) in front of every scanline: no prediction, so the
    # rows go out as they are and zlib does all the work.
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        handle.write(chunk(b"IEND", b""))


class Canvas(object):
    """An RGB image addressed in map metres, so callers never convert by hand."""

    def __init__(self, x0: float, y0: float, x1: float, y1: float,
                 cell: float, scale: int = 4, background=(255, 255, 255)):
        self.x0, self.y0, self.cell, self.scale = x0, y0, cell, scale
        self.cols = max(1, int(round((x1 - x0) / cell)) * scale)
        self.rows = max(1, int(round((y1 - y0) / cell)) * scale)
        self.img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        self.img[:, :] = background

    # y grows upward in the map and downward in an image, so the row flips.
    def px(self, x: float, y: float):
        col = int((x - self.x0) / self.cell * self.scale)
        row = self.rows - 1 - int((y - self.y0) / self.cell * self.scale)
        return col, row

    def blit_cells(self, values: np.ndarray, colours: np.ndarray) -> None:
        """Paint a (rows, cols) cell grid, upsampled by `scale`, where mask is set."""
        big = np.repeat(np.repeat(colours, self.scale, axis=0), self.scale, axis=1)
        mask = np.repeat(np.repeat(values, self.scale, axis=0), self.scale, axis=1)
        rows = min(self.rows, big.shape[0])
        cols = min(self.cols, big.shape[1])
        target = self.img[:rows, :cols]
        target[mask[:rows, :cols]] = big[:rows, :cols][mask[:rows, :cols]]

    def dot(self, x: float, y: float, colour, size: int = 1) -> None:
        col, row = self.px(x, y)
        r0, r1 = max(0, row - size), min(self.rows, row + size + 1)
        c0, c1 = max(0, col - size), min(self.cols, col + size + 1)
        if r0 < r1 and c0 < c1:
            self.img[r0:r1, c0:c1] = colour

    def box(self, x0: float, y0: float, x1: float, y1: float, colour) -> None:
        c0, r1 = self.px(x0, y0)
        c1, r0 = self.px(x1, y1)
        c0, c1 = sorted((max(0, c0), min(self.cols - 1, c1)))
        r0, r1 = sorted((max(0, r0), min(self.rows - 1, r1)))
        self.img[r0, c0:c1 + 1] = colour
        self.img[r1, c0:c1 + 1] = colour
        self.img[r0:r1 + 1, c0] = colour
        self.img[r0:r1 + 1, c1] = colour

    def line(self, x0: float, y0: float, x1: float, y1: float, colour) -> None:
        c0, r0 = self.px(x0, y0)
        c1, r1 = self.px(x1, y1)
        steps = max(abs(c1 - c0), abs(r1 - r0), 1) + 1
        cols = np.linspace(c0, c1, steps).round().astype(int)
        rows = np.linspace(r0, r1, steps).round().astype(int)
        inside = ((cols >= 0) & (cols < self.cols) & (rows >= 0) & (rows < self.rows))
        self.img[rows[inside], cols[inside]] = colour

    def polygon(self, corners, colour) -> None:
        corners = np.asarray(corners, dtype=np.float64)
        for i in range(len(corners)):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            self.line(a[0], a[1], b[0], b[1], colour)

    def text(self, x: float, y: float, label: str, colour, size: int = 2) -> None:
        col, row = self.px(x, y)
        for char in str(label):
            glyph = FONT.get(char, FONT[" "])
            for gy, line in enumerate(glyph):
                for gx, bit in enumerate(line):
                    if bit != "1":
                        continue
                    r0, c0 = row + gy * size, col + gx * size
                    r1, c1 = min(self.rows, r0 + size), min(self.cols, c0 + size)
                    if 0 <= r0 < self.rows and 0 <= c0 < self.cols:
                        self.img[r0:r1, c0:c1] = colour
            col += 4 * size

    def save(self, path: str) -> None:
        write_png(path, self.img)


def height_ramp(heights: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Blue (low) through green to orange (high): height readable without a key."""
    t = np.clip((heights - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    red = np.clip(-0.4 + 2.0 * t, 0, 1)
    green = np.clip(1.6 * np.minimum(t, 1.0 - 0.55 * t), 0, 1)
    blue = np.clip(1.0 - 1.8 * t, 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
