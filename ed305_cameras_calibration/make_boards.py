#!/usr/bin/env python3
"""Make the printable boards and floor tags (PNG at 300 dpi, true size).

  python make_boards.py --out print

Boards A, B, C go on A1 landscape at 100% scale; floor tags on A4. After printing, measure one
square of each board and the black edge of a tag and write the real sizes into config/boards.yaml.

  python make_boards.py --page A3     # smaller printer: squares are shrunk to fit, the script prints
                                      # the square size you must then write into config/boards.yaml
"""
import argparse
import os

import cv2
import numpy as np

from calib.common import FloorTags, load_boards, load_yaml

DPI = 300
MM = DPI / 25.4
PAGES = {"A1": (841, 594), "A2": (594, 420), "A3": (420, 297)}   # landscape, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="print")
    ap.add_argument("--tags", type=int, default=12, help="how many floor tags (ids 0..n-1)")
    ap.add_argument("--page", default="A1", choices=sorted(PAGES), help="paper for the boards")
    args = ap.parse_args()
    page_w, page_h = PAGES[args.page]
    os.makedirs(args.out, exist_ok=True)

    spec = load_yaml("boards.yaml")["boards"]
    for name, b in load_boards().items():
        s = spec[name]
        # square size: as configured, shrunk if the page is too small (10 mm side margins, 27 mm below)
        sq_mm = min(s["square_m"] * 1000, (page_w - 20) / s["squares_x"], (page_h - 27) / s["squares_y"])
        sq_mm = np.floor(sq_mm * 2) / 2
        w = int(s["squares_x"] * sq_mm * MM)
        h = int(s["squares_y"] * sq_mm * MM)
        img = b.board.generateImage((w, h), marginSize=0, borderBits=1)
        page = np.full((h + int(27 * MM), w + int(20 * MM)), 255, np.uint8)
        page[int(5 * MM):int(5 * MM) + h, int(10 * MM):int(10 * MM) + w] = img
        cv2.putText(page, "ED305 board %s  %s  %dx%d squares of %.1f mm" % (
            name, s["dictionary"], s["squares_x"], s["squares_y"], sq_mm),
            (int(10 * MM), h + int(21 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 2.5 * sq_mm / 70, 0, 6)
        cv2.imwrite(os.path.join(args.out, "board_%s_%s.png" % (name, args.page)), page)
        if abs(sq_mm - s["square_m"] * 1000) > 0.01:
            print("board %s shrunk to fit %s: square_m: %.4f  marker_m: %.4f  <- put these in config/boards.yaml"
                  % (name, args.page, sq_mm / 1000, 0.75 * sq_mm / 1000))

    tags = FloorTags()
    px = int(tags.size * 1000 * MM)
    for i in range(args.tags):
        tag = cv2.aruco.generateImageMarker(tags.dictionary, i, px)
        page = np.full((px + int(60 * MM), px + int(40 * MM)), 255, np.uint8)
        page[int(20 * MM):int(20 * MM) + px, int(20 * MM):int(20 * MM) + px] = tag
        cv2.putText(page, "floor tag %d  %.0f mm  this edge = +x" % (i, tags.size * 1000),
                    (int(20 * MM), px + int(45 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 4)
        cv2.imwrite(os.path.join(args.out, "floor_tag_%02d_A4.png" % i), page)
    print("written to", args.out)


if __name__ == "__main__":
    main()
