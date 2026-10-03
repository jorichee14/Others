#!/usr/bin/env python3
"""What a bag's topics hold: type, count, message definition, first messages.

    python scoop_pipeline/analysis/bags/show_topics.py <bag dir> [PATTERN ...] [-n 2] [--full]

PATTERNs pick topics as in a shell ('/*/wifi/*', '/*/ntp/*'); none = all.
The definition is the one the bag carries (MCAP stores it), so the robots'
own message types (wifi, ntp, iperf, CSI) show without their packages; the
types it uses are named, --full prints them too. Long arrays are shortened
to their length and first values. Reads only the first -n messages of each
topic.

    python scoop_pipeline/analysis/bags/show_topics.py <merged> '/*/wifi/*' '/*/ntp/*' '/*/csi*' '/*/csi_publisher/*'
"""
import argparse
import fnmatch
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag                                               # noqa: E402
from scoop.bag import BagError                                      # noqa: E402

LONG = 12


class Short(str):
    """A shortened array, printed as is."""
    def __repr__(self):
        return str(self)


def as_plain(m, depth=0):
    """A decoded message -> dicts / lists / scalars, long sequences shortened."""
    if depth > 8:
        return "..."
    if isinstance(m, (bytes, bytearray, memoryview)):
        b = bytes(m)
        return b if len(b) <= LONG else Short("<%d bytes: %s ...>" % (len(b), b[:LONG].hex()))
    if hasattr(m, "tolist") and not isinstance(m, (int, float, str)):
        m = m.tolist()
    if isinstance(m, (list, tuple)):
        if len(m) > LONG:
            return Short("<%d items: %s ...>" % (len(m), [as_plain(x, depth + 1) for x in m[:4]]))
        return [as_plain(x, depth + 1) for x in m]
    names = getattr(m, "__slots__", None) or getattr(m, "__dataclass_fields__", None) \
        or (vars(m) if hasattr(m, "__dict__") else None)
    if names is not None and not isinstance(m, (int, float, str, bool)):
        return {k: as_plain(getattr(m, k), depth + 1) for k in names if not k.startswith("_")}
    return m


def show(v, indent="    ", out=print):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, dict):
                out(f"{indent}{k}:")
                show(x, indent + "  ", out)
            else:
                out(f"{indent}{k}: {x!r}" if isinstance(x, str) and not isinstance(x, Short)
                    else f"{indent}{k}: {x}")
    else:
        out(f"{indent}{v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("-n", type=int, default=2, help="messages to print per topic")
    ap.add_argument("--full", action="store_true", help="also the definitions of the types used")
    a = ap.parse_args()
    try:
        r = bag.open_bag(a.bag)
        info = r.topics()
        schemas = bag.topic_schemas(r)
        topics = [t for t in sorted(info)
                  if not a.patterns or any(fnmatch.fnmatchcase(t, p) for p in a.patterns)]
        if not topics:
            sys.exit("no topic matches %s" % " ".join(a.patterns))
        for t in topics:
            ti = info[t]
            print("=" * 78)
            print(f"{t}   {ti.msgtype}   {ti.count} messages")
            sc = schemas.get(t)
            if sc is not None:
                text = sc.data.decode("utf-8", "replace")
                parts = text.split("\n" + "=" * 80 + "\n")
                print("  definition:")
                for ln in parts[0].strip().splitlines():
                    print("    " + ln)
                deps = [p.strip().splitlines()[0].replace("MSG: ", "") for p in parts[1:] if p.strip()]
                if deps and not a.full:
                    print("  uses: " + ", ".join(deps) + "   (--full to print them)")
                elif deps:
                    for p in parts[1:]:
                        for ln in p.strip().splitlines():
                            print("    " + ln)
            n = 0
            for _, log_ns, payload, decode in r.iter_raw([t]):
                print(f"  message {n + 1} (log time {log_ns / 1e9:.6f}):")
                try:
                    show(as_plain(decode(payload)))
                except Exception as e:                            # noqa: BLE001
                    print(f"    (cannot decode: {e}; {len(payload)} bytes)")
                n += 1
                if n >= a.n:
                    break
    except BagError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
