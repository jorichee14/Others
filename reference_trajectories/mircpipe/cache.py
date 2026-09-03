"""Bag -> parquet cache, shared by every analysis.

The NTP, WiFi, CSI and radar analyses all want the same thing: the scalar
fields of a set of topics as a table. Extracting once into parquet means a
second analysis costs no bag decode, and the analyses themselves never touch
rosbag2.

    from mircpipe import cache
    d  = cache.ensure(bag, out_dir)          # extract everything flattenable
    df = cache.load(d, "/mobile_1/global_pose")

Message fields are flattened to columns with dotted names (pose.position.x),
arrays of scalars are kept as list columns, and every row carries `t` (header
stamp, seconds) and `t_bag`. A topic whose message has no flattenable fields
is skipped.
"""
import json
import os

SKIP_FIELDS = ("data",)          # raw image/pointcloud payloads
MAX_LIST = 64                    # keep short arrays, drop long payloads


def topic_filename(topic):
    return topic.strip("/").replace("/", "__") + ".parquet"


def _flatten(msg, prefix="", out=None, depth=0):
    out = {} if out is None else out
    if depth > 6:
        return out
    for name in getattr(msg, "get_fields_and_field_types", lambda: {})():
        if name in SKIP_FIELDS and prefix == "":
            continue
        v = getattr(msg, name)
        key = prefix + name
        if hasattr(v, "get_fields_and_field_types"):
            _flatten(v, key + ".", out, depth + 1)
        elif isinstance(v, (bytes, bytearray, memoryview)):
            continue
        elif isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
            seq = list(v) if not hasattr(v, "tolist") else v.tolist()
            if not seq:
                out[key] = None
            elif hasattr(seq[0], "get_fields_and_field_types"):
                continue                              # nested message arrays
            elif len(seq) <= MAX_LIST:
                out[key] = seq
        else:
            out[key] = v
    return out


def ensure(bag_path, cache_root, topics=None, refresh=False, printer=print):
    """Extract `topics` (default: everything flattenable) into
    <cache_root>/<bag name>/ as one parquet per topic. Returns the directory.
    Already-extracted topics are skipped unless refresh."""
    import pandas as pd
    from .bag import iter_topic, topic_types

    name = os.path.basename(os.path.normpath(bag_path)).replace(".mcap", "")
    d = os.path.join(cache_root, name)
    os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(d, "cache_meta.json")
    meta = {}
    if os.path.exists(meta_path) and not refresh:
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
    types = topic_types(bag_path)
    want = list(topics) if topics else sorted(types)
    printer("  parquet cache %s: %d topics in bag, %d requested" % (d, len(types), len(want)))
    for tp in want:
        if tp not in types:
            continue
        fn = os.path.join(d, topic_filename(tp))
        if os.path.exists(fn) and not refresh and meta.get(tp, {}).get("rows"):
            continue
        rows = []
        try:
            for t, m in iter_topic(bag_path, tp):
                r = _flatten(m)
                if not r:
                    break
                r["t"] = t
                rows.append(r)
        except Exception as e:
            printer("    ! %s: %s" % (tp, e))
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_parquet(fn, index=False)
        meta[tp] = dict(rows=len(df), type=types[tp], file=os.path.basename(fn))
        printer("    %-58s %6d rows -> %s" % (tp, len(df), os.path.basename(fn)))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return d


def load(cache_dir, topic):
    """The table for one topic; raises if it was never extracted."""
    import pandas as pd
    fn = os.path.join(cache_dir, topic_filename(topic))
    if not os.path.exists(fn):
        raise SystemExit("%s not in the cache at %s - extract it first"
                         % (topic, cache_dir))
    return pd.read_parquet(fn)


def available(cache_dir):
    """{topic: metadata} of what the cache holds."""
    p = os.path.join(cache_dir, "cache_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}
