#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a PCD to a PLY, keeping every field as a scalar CloudCompare reads.

Picking a point only means something if the pick is taken in the same frame
the thing being marked is drawn in, so the cloud a viewer opens has to be the
cloud the pipeline used -- not a lookalike exported some other way. This
converts in place, field for field, rather than round-tripping through a
loader that would quietly drop intensity and leave nothing to aim at.

    python3 scripts/pcd_to_ply.py <in.pcd> [out.ply] [--ascii]

ascii, binary and binary_compressed PCDs all read; a packed ``rgb`` field
becomes real red/green/blue bytes, and anything else keeps its own name.
"""

import argparse
import os
import re
import sys

import numpy as np

# PCD type/size pairs to numpy, the whole of what PCL writes.
NUMPY_OF = {("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
            ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
            ("F", 4): "<f4", ("F", 8): "<f8"}
PLY_OF = {"i1": "char", "i2": "short", "i4": "int", "i8": "double",
          "u1": "uchar", "u2": "ushort", "u4": "uint", "u8": "double",
          "f4": "float", "f8": "double"}


def lzf_decompress(source: bytes, expected: int) -> bytes:
    """The LZF variant PCL uses for DATA binary_compressed.

    Spelled out rather than pulled in: it is twenty lines, it is the only
    reason a perfectly good map cloud would refuse to open, and the
    alternative is a dependency that exists to decompress one header.
    """
    out = bytearray()
    i = 0
    while i < len(source):
        control = source[i]
        i += 1
        if control < 32:                       # a literal run
            out += source[i:i + control + 1]
            i += control + 1
        else:                                  # a back reference
            length = control >> 5
            if length == 7:
                length += source[i]
                i += 1
            reference = len(out) - ((control & 0x1F) << 8) - source[i] - 1
            i += 1
            if reference < 0:
                raise ValueError("corrupt binary_compressed payload")
            for _ in range(length + 2):
                out.append(out[reference])
                reference += 1
    if len(out) != expected:
        raise ValueError(f"decompressed {len(out)} bytes, header said {expected}")
    return bytes(out)


def read_pcd_any(path: str):
    """(structured array, field names) for any PCD PCL writes."""
    with open(path, "rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            text = line.decode("ascii", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break
        payload = handle.read()

    fields = header["FIELDS"].split()
    sizes = [int(v) for v in header["SIZE"].split()]
    types = header["TYPE"].split()
    counts = [int(v) for v in header.get("COUNT", " ".join("1" * len(fields))).split()]
    if any(c != 1 for c in counts):
        raise ValueError(f"{path}: COUNT > 1 fields are not handled: {counts}")
    n = int(header.get("POINTS") or int(header["WIDTH"]) * int(header["HEIGHT"]))

    dtype = np.dtype([(name, NUMPY_OF[(kind, size)])
                      for name, kind, size in zip(fields, types, sizes)])
    data = header["DATA"].lower()

    if data == "ascii":
        text = payload.decode("ascii", "replace")
        rows = np.array([row.split() for row in text.split("\n") if row.strip()],
                        dtype=np.float64)
        points = np.empty(n, dtype=dtype)
        for index, name in enumerate(fields):
            points[name] = rows[:n, index].astype(dtype[name])
        return points, fields

    if data == "binary":
        return np.frombuffer(payload, dtype=dtype, count=n).copy(), fields

    if data == "binary_compressed":
        packed, plain = np.frombuffer(payload[:8], dtype="<u4")
        raw = lzf_decompress(payload[8:8 + packed], int(plain))
        # Compressed PCDs are stored field by field, not point by point.
        points, offset = np.empty(n, dtype=dtype), 0
        for name in fields:
            width = dtype[name].itemsize * n
            points[name] = np.frombuffer(raw[offset:offset + width],
                                         dtype=dtype[name], count=n)
            offset += width
        return points, fields

    raise ValueError(f"{path}: unknown DATA {header['DATA']!r}")


def write_ply(path: str, points, fields, ascii_out: bool) -> None:
    columns, lines = [], []
    for name in fields:
        values = points[name]
        if name == "rgb":
            # Packed into a float by PCL; CloudCompare wants three bytes.
            packed = values.copy().view("<u4") if values.dtype.kind == "f" else \
                values.astype("<u4")
            for channel, shift in (("red", 16), ("green", 8), ("blue", 0)):
                columns.append((channel, "u1", ((packed >> shift) & 0xFF).astype("u1")))
                lines.append(f"property uchar {channel}")
            continue
        kind = values.dtype.str.lstrip("<>|")
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        columns.append((safe, kind, values))
        lines.append(f"property {PLY_OF[kind]} {safe}")

    head = ["ply",
            "format " + ("ascii 1.0" if ascii_out else "binary_little_endian 1.0"),
            f"element vertex {len(points)}", *lines, "end_header"]
    with open(path, "wb") as handle:
        handle.write(("\n".join(head) + "\n").encode("ascii"))
        if ascii_out:
            stacked = np.column_stack([c[2].astype(np.float64) for c in columns])
            np.savetxt(handle, stacked, fmt="%.6g")
        else:
            out = np.empty(len(points), dtype=[(c[0], c[1]) for c in columns])
            for name, _, values in columns:
                out[name] = values
            handle.write(out.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("out", nargs="?", default=None)
    parser.add_argument("--ascii", action="store_true",
                        help="write a text PLY (bigger, but readable)")
    args = parser.parse_args()

    source = os.path.expanduser(args.source)
    out = os.path.expanduser(args.out) if args.out else os.path.splitext(source)[0] + ".ply"
    points, fields = read_pcd_any(source)
    write_ply(out, points, fields, args.ascii)

    xyz = np.column_stack([points[a] for a in ("x", "y", "z")]).astype(np.float64)
    finite = xyz[np.isfinite(xyz).all(axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    print(f"{len(points):,} points, fields: {', '.join(fields)}")
    print("extent  x {:.2f}..{:.2f}   y {:.2f}..{:.2f}   z {:.2f}..{:.2f}".format(
        low[0], high[0], low[1], high[1], low[2], high[2]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
