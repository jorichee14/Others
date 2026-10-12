#!/usr/bin/env python3
"""How far an IMU's stamps run behind the reference IMU's in a processed bag,
and (--out) a copy of the bag with them moved back.

    python3 processing/imu_sync.py <bag> [--topic /mobile_1/zed/imu/data]
        [--ref /mobile_1/ouster/imu] [--max-shift 5] [--out <new bag>] [--shift S]
        [--compression zstd|none]

For bags processed before the retimed step did this itself (configs/
recording.yaml: ouster.imu_sync): a retimed or a merged bag, where the
reference (the Ouster's IMU) is already on the capture clock. Without --out
it only measures (scoop/imu_sync.py: |w| of both lined up). With --out every
message is copied byte for byte, except --topic's header stamps, less the
measured shift (or --shift S); refused when the two do not correlate (0.8)
unless --shift is given. Log times are kept. The result is written to
<new bag>/imu_sync.json.
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, imu_sync as IS                              # noqa: E402
from scoop.bag import BagError, TopicSchema                         # noqa: E402
from scoop.bagwrite import BagWriter, write_in_order                # noqa: E402
from scoop.retime import header_stamp_ns, with_header_stamp         # noqa: E402


def restamp(src, out, topic, shift_ns, compression="zstd", log=print):
    reader = bag.open_bag(src)
    qos = bag.topic_qos(reader)

    def records():
        for _, sc, ch, msg in bag.iter_mcap_records(reader):
            data = msg.data
            if ch.topic == topic:
                data = with_header_stamp(data, header_stamp_ns(data) - shift_ns)
            yield (msg.log_time, ch.topic, TopicSchema(sc.name, sc.encoding, bytes(sc.data), ch.message_encoding),
                   msg.publish_time, msg.sequence, dict(ch.metadata), data)

    with BagWriter(out, library="scoop.imu_sync", compression=compression) as w:
        for t, q in qos.items():
            w.set_qos(t, q)
        write_in_order(w, records(), 0.0)
    log(f"wrote {w.n} messages -> {w.dir}; {topic} moved back {shift_ns * 1e-9:.3f} s")
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--topic", default="/mobile_1/zed/imu/data")
    ap.add_argument("--ref", default="/mobile_1/ouster/imu")
    ap.add_argument("--max-shift", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="write a corrected copy here (a new folder)")
    ap.add_argument("--shift", type=float, default=None, help="use this shift (s) instead of the measured one")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "none"])
    a = ap.parse_args()
    try:
        r = IS.measure(bag.open_bag(a.bag), a.topic, a.ref, max_shift=a.max_shift)
        print(f"{a.topic}: {r.shift_s:+.3f} s behind {a.ref} (correlation {r.correlation:.2f}, "
              f"scale {r.scale:.2f}, turning {r.turn_rms:.3f} rad/s, {r.samples} samples)"
              + ("" if r.ok else " -- NOT sure: too little turning or correlation"))
        if not a.out:
            return
        if a.shift is None and not r.ok:
            sys.exit("not writing: the measurement is not sure; give --shift to force one")
        if os.path.exists(a.out):
            sys.exit(f"{a.out} exists; pick a new folder")
        shift_ns = int(round((a.shift if a.shift is not None else r.shift_s) * 1e9))
        restamp(a.bag, a.out, a.topic, shift_ns, a.compression)
        with open(os.path.join(a.out, "imu_sync.json"), "w") as fh:
            json.dump({**r.as_dict(), "applied_shift_s": shift_ns * 1e-9, "source": os.path.abspath(a.bag)},
                      fh, indent=1)
    except BagError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
