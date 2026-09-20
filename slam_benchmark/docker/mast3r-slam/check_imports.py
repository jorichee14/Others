#!/usr/bin/env python3
"""Import everything main.py imports, at BUILD time.

Two runs in a row died on an import after unrolling the bag: libGL for the
full opencv-python, then libusb for pyrealsense2, which the dataloader imports
even for a folder of images. Each cost a run to find. This walks main.py's
import statements and imports each module, so a missing shared library fails
the build instead.
"""
import ast
import importlib
import sys

path = sys.argv[1]
tree = ast.parse(open(path).read())
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        mods.update(a.name for a in n.names)
    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
        mods.add(n.module)
failed = []
for m in sorted(mods):
    try:
        importlib.import_module(m)
    except Exception as e:                                   # noqa: BLE001
        failed.append(f"{m}: {type(e).__name__}: {e}")
if failed:
    sys.exit("main.py imports that fail in this image:\n  " + "\n  ".join(failed))
print(f"main.py: {len(mods)} imports ok")
