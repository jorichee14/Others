#!/usr/bin/env python3
"""Namespaced CSI topics must not collapse onto one key.

'/mobile_1_sniffer/infra_1/csi' and '/mobile_1_sniffer/mobile_2/csi' share a
first path segment, so keying on it (as node_of_topic does) silently keeps only
the last file read -- a lost channel with no error anywhere.
"""
import sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import link_of_topic          # noqa: E402
from csi_analysis import load_csi         # noqa: E402

F = []
def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:46s} {got!r}")
    if not ok: F.append(name)

print("link_of_topic")
check("unnamespaced", link_of_topic("/infra_1/csi"), ("", "infra_1"))
check("namespaced", link_of_topic("/mobile_1_sniffer/infra_1/csi"), ("mobile_1_sniffer", "infra_1"))
check("alias applied", link_of_topic("/mobile2/csi"), ("", "mobile_2"))
check("second sniffer", link_of_topic("/infra_1_sniffer/mobile_1/csi"), ("infra_1_sniffer", "mobile_1"))

def fake(d, topics):
    for t in topics:
        pd.DataFrame({"log_time_ns": [1, 2], "rssi": [-50, -51]}).to_parquet(
            Path(d) / (t.strip("/").replace("/", "__") + ".parquet"), index=False)

print("\nload_csi keys on the transmitter")
with tempfile.TemporaryDirectory() as d:
    fake(d, ["/mobile_1_sniffer/infra_1/csi", "/mobile_1_sniffer/mobile_2/csi"])
    out = load_csi(Path(d))
    check("both streams survive", sorted(out), ["infra_1", "mobile_2"])
    check("receiver recorded", out["infra_1"]["rx"].iloc[0], "mobile_1_sniffer")

with tempfile.TemporaryDirectory() as d:
    fake(d, ["/infra_1/csi", "/mobile_2/csi"])
    check("unnamespaced unchanged", sorted(load_csi(Path(d))), ["infra_1", "mobile_2"])

with tempfile.TemporaryDirectory() as d:
    # two receivers watching the same transmitter -- must not overwrite
    fake(d, ["/mobile_1_sniffer/infra_1/csi", "/infra_1_sniffer/infra_1/csi"])
    check("two receivers, one transmitter",
          sorted(load_csi(Path(d))), ["infra_1_sniffer/infra_1", "mobile_1_sniffer/infra_1"])

print()
if F: print(f"{len(F)} failed: {', '.join(F)}"); sys.exit(1)
print("all checks passed")
