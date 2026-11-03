#!/usr/bin/env python3
"""A rosbag2 MCAP folder without metadata.yaml -> its metadata.yaml, without ROS.

    python3 scoop_pipeline/processing/reindex_bag.py <bag folder> [<bag folder> ...]
        [--force] [--no-recover]

`ros2 bag record` writes metadata.yaml when it stops cleanly; a recorder that
was killed (or a copy of the .mcap files alone) leaves the split files and no
metadata.yaml, and every step here (process_recording.py, merge_session.py,
run_pass.py) finds a bag by its metadata.yaml. This writes it from the files
themselves, as `ros2 bag reindex` would: the files in the recorder's order
(_0, _1, ..., _10), each file's start / duration / count, every topic with its
type, serialization format, QoS profiles (the channel's own, as rosbag2 stored
them) and message count.

A file whose summary cannot be read (the last one, when the recorder was
killed while writing it) is recovered first, as `mcap recover` does: every
record up to where the file stops is written to a new <name>.mcap, and the
original is kept beside it as <name>.mcap.unindexed (delete it once the bag
reads). --no-recover lists such files and stops instead.

--force   write metadata.yaml again where the folder has one already
"""
import argparse
import re
import sys
from pathlib import Path

import yaml


def natural(p):
    """bag_2.mcap before bag_10.mcap."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def summary_of(path):
    """The file's MCAP summary, or None when it has none / cannot be read."""
    from mcap.reader import make_reader
    try:
        with open(path, "rb") as fh:
            return make_reader(fh).get_summary()
    except Exception:                                  # truncated / no footer
        return None


def recover(path, log=print):
    """Every complete record of an MCAP file whose end is missing, into a new
    file in its place; the original is kept as <name>.mcap.unindexed.
    -> the number of messages kept."""
    from mcap.records import Channel, Message, Schema
    from mcap.stream_reader import StreamReader
    from mcap.writer import CompressionType, Writer
    path = Path(path)
    tmp = path.with_name(path.name + ".recovering")
    schemas, channels, n = {}, {}, 0
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        w = Writer(dst, compression=CompressionType.ZSTD)
        w.start(profile="ros2", library="scoop reindex_bag.py")
        try:
            for rec in StreamReader(src, skip_magic=False, emit_chunks=False).records:
                if isinstance(rec, Schema):
                    if rec.id not in schemas:
                        schemas[rec.id] = w.register_schema(rec.name, rec.encoding, rec.data)
                elif isinstance(rec, Channel):
                    if rec.id not in channels:
                        channels[rec.id] = w.register_channel(
                            rec.topic, rec.message_encoding, schemas.get(rec.schema_id, 0),
                            dict(rec.metadata or {}))
                elif isinstance(rec, Message):
                    if rec.channel_id in channels:
                        w.add_message(channels[rec.channel_id], rec.log_time, rec.data,
                                      rec.publish_time, rec.sequence)
                        n += 1
        except Exception as e:                         # where the file stops
            log("  %s: stops at a broken record (%s); kept %d messages before it"
                % (path.name, type(e).__name__, n))
        w.finish()
    path.rename(path.with_name(path.name + ".unindexed"))
    tmp.rename(path)
    return n


def file_entry(path, summary):
    """(file entry, {topic: [type, format, qos, count]}) from a file's summary."""
    st = summary.statistics
    topics = {}
    counts = dict(getattr(st, "channel_message_counts", None) or {}) if st else {}
    if st is None or not counts and st.message_count:
        counts, t0, t1 = count_messages(path)
    else:
        t0, t1 = st.message_start_time, st.message_end_time
    for cid, ch in summary.channels.items():
        sc = summary.schemas.get(ch.schema_id)
        e = topics.setdefault(ch.topic, [sc.name if sc else "", ch.message_encoding,
                                         (ch.metadata or {}).get("offered_qos_profiles", ""), 0])
        e[3] += counts.get(cid, 0)
    n = sum(counts.values())
    if n == 0:
        t0 = t1 = 0
    return ({"path": path.name, "starting_time": {"nanoseconds_since_epoch": int(t0)},
             "duration": {"nanoseconds": int(t1 - t0)}, "message_count": int(n)}, topics)


def count_messages(path):
    """{channel id: count}, first and last log time, by reading every message
    (a summary without statistics)."""
    from mcap.reader import make_reader
    counts, t0, t1 = {}, None, None
    with open(path, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages():
            counts[ch.id] = counts.get(ch.id, 0) + 1
            t0 = msg.log_time if t0 is None else min(t0, msg.log_time)
            t1 = msg.log_time if t1 is None else max(t1, msg.log_time)
    return counts, t0 or 0, t1 or 0


def reindex(folder, force=False, recover_broken=True, log=print):
    """Write <folder>/metadata.yaml from its .mcap files. -> its path."""
    folder = Path(folder).expanduser().resolve()
    meta = folder / "metadata.yaml"
    if meta.exists() and not force:
        raise SystemExit("%s has a metadata.yaml already (--force writes it again)" % folder)
    files = sorted(folder.glob("*.mcap"), key=natural)
    if not files:
        raise SystemExit("no .mcap files in %s" % folder)
    log("%s: %d file(s), %s ... %s" % (folder, len(files), files[0].name, files[-1].name))
    nums = [int(m.group(1)) for m in (re.search(r"_(\d+)\.mcap$", f.name) for f in files) if m]
    missing = sorted(set(range(max(nums) + 1)) - set(nums)) if nums else []
    if missing:
        log("  ! split file(s) missing: %s (the bag has a hole there)"
            % ", ".join("_%d" % i for i in missing))
    summaries, broken = {}, []
    for f in files:
        s = summary_of(f)
        if s is None:
            broken.append(f)
        summaries[f] = s
    if broken:
        log("  no readable summary (not closed cleanly): %s" % ", ".join(f.name for f in broken))
        if not recover_broken:
            raise SystemExit("recover them first (without --no-recover), or with "
                             "`mcap recover <file> -o <file>.recovered`")
        for f in broken:
            log("  recovering %s ..." % f.name)
            recover(f, log)
            summaries[f] = summary_of(f)
            if summaries[f] is None:
                raise SystemExit("%s: still no summary after recovery" % f)
    entries, topics = [], {}
    for f in files:
        e, t = file_entry(f, summaries[f])
        entries.append(e)
        for name, (typ, fmt, qos, n) in t.items():
            cur = topics.setdefault(name, [typ, fmt, qos, 0])
            cur[3] += n
            if not cur[2] and qos:
                cur[2] = qos
    timed = [e for e in entries if e["message_count"]]
    t0 = min(e["starting_time"]["nanoseconds_since_epoch"] for e in timed) if timed else 0
    t1 = max(e["starting_time"]["nanoseconds_since_epoch"] + e["duration"]["nanoseconds"]
             for e in timed) if timed else 0
    total = sum(e["message_count"] for e in entries)
    doc = {"rosbag2_bagfile_information": {
        "version": 5,
        "storage_identifier": "mcap",
        "duration": {"nanoseconds": int(t1 - t0)},
        "starting_time": {"nanoseconds_since_epoch": int(t0)},
        "message_count": int(total),
        "topics_with_message_count": [
            {"topic_metadata": {"name": name, "type": typ, "serialization_format": fmt or "cdr",
                                "offered_qos_profiles": qos},
             "message_count": int(n)} for name, (typ, fmt, qos, n) in topics.items()],
        "compression_format": "",
        "compression_mode": "",
        "relative_file_paths": [f.name for f in files],
        "files": entries}}
    with open(meta, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    import datetime
    utc = lambda ns: datetime.datetime.fromtimestamp(ns * 1e-9, datetime.timezone.utc)  # noqa: E731
    log("  %s UTC, %.1f s, %d messages" % (utc(t0).strftime("%Y-%m-%d %H:%M:%S") if t0 else "-",
                                          (t1 - t0) * 1e-9, total))
    gaps = [(a["path"], b["path"], (b["starting_time"]["nanoseconds_since_epoch"]
                                    - a["starting_time"]["nanoseconds_since_epoch"]
                                    - a["duration"]["nanoseconds"]) * 1e-9)
            for a, b in zip(timed, timed[1:])]
    for a, b, g in gaps:
        if g > 1.0:
            log("  ! %.1f s with no message between %s and %s" % (g, a, b))
    w = max(len(n) for n in topics)
    for name, (typ, _, qos, n) in sorted(topics.items()):
        log("    %-*s %8d  %s%s" % (w, name, n, typ, "" if qos else "  (no QoS stored)"))
    log("wrote %s" % meta)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("bags", nargs="+", help="rosbag2 folder(s) of .mcap files")
    ap.add_argument("--force", action="store_true", help="replace an existing metadata.yaml")
    ap.add_argument("--no-recover", action="store_true",
                    help="stop at a file without a summary instead of recovering it")
    a = ap.parse_args(argv)
    for b in a.bags:
        reindex(b, a.force, not a.no_recover)


if __name__ == "__main__":
    sys.exit(main())
