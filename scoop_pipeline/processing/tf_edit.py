#!/usr/bin/env python3
"""Show or change the TF (/tf and /tf_static) of a bag.

    python scoop_pipeline/processing/tf_edit.py <bag> --show
    python scoop_pipeline/processing/tf_edit.py <bag> -o <out bag>
        [--drop PARENT->CHILD ...]
        [--add PARENT CHILD x y z qx qy qz qw]... [--add-file configs/static_tf.yaml]
        [--odom-tum FILE | --odom-topic TOPIC] [--odom-frame FRAME] [--odom-parent FRAME]
        [--compression none|zstd]

--show         print the TF tree (/tf_static, and the first 30 s of /tf) and stop
--drop         remove edges from /tf and /tf_static (shell patterns), e.g. the
               ZED's odometry: 'map_zed->odom_zed' 'odom_zed->*'
--add          a static transform: pose of CHILD in PARENT (m, quaternion xyzw),
               what `tf2_echo PARENT CHILD` prints; repeat for more
--add-file     static transforms from a yaml (configs/static_tf.yaml format)
--odom-tum     a new odometry: a TUM trajectory (GLIM's traj_lidar.txt) ...
--odom-topic   ... or an Odometry/PoseStamped topic of the bag
--odom-frame   the frame those poses are of (os_lidar for traj_lidar.txt;
               an Odometry's child_frame_id by default)
--odom-parent  the world frame name (the topic's header.frame_id by default)

Edits apply in that order: drop, add, odometry. A frame has one parent: a
static transform whose child already has one is added the other way round
(child -> root of the parent's tree), and the odometry is published as
<parent> -> <root of the frame's tree>; looking up PARENT -> CHILD (or
<parent> -> <frame>) always gives exactly what you gave. All static
transforms end up in one latched /tf_static message.

The output is a new bag (-o): every other topic copied byte for byte. The
work is in scoop/tftree.py and scoop/merge.py.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import merge, tftree                                     # noqa: E402
from scoop.bag import BagError                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--out", help="the new bag (must not exist)")
    ap.add_argument("--show", action="store_true", help="print the TF tree and stop")
    ap.add_argument("--drop", nargs="+", default=[], metavar="PARENT->CHILD")
    ap.add_argument("--add", nargs=9, action="append", default=[],
                    metavar=("PARENT", "CHILD", "X", "Y", "Z", "QX", "QY", "QZ", "QW"))
    ap.add_argument("--add-file", help="static transforms from a yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--odom-tum", help="new odometry: TUM trajectory file")
    g.add_argument("--odom-topic", help="new odometry: Odometry/PoseStamped topic")
    ap.add_argument("--odom-frame", help="the frame the odometry poses are of")
    ap.add_argument("--odom-parent", help="the odometry's world frame")
    ap.add_argument("--compression", default="none", choices=["none", "zstd"],
                    help="zstd: smaller, slower to write and read (default none)")
    a = ap.parse_args()
    try:
        if a.show:
            print(tftree.format_tree(tftree.tf_edges([a.bag])))
            return
        if not a.out:
            sys.exit("error: -o <out bag> is needed (or --show)")
        if os.path.exists(a.out):
            sys.exit(f"error: {a.out} exists")
        cals = [tftree.Calibration(p, c, [float(v) for v in r[:3]], [float(v) for v in r[3:]])
                for p, c, *r in a.add]
        if a.add_file:
            import yaml
            with open(os.path.expanduser(a.add_file)) as fh:
                cals += tftree.load_calibrations(yaml.safe_load(fh))
        odom = None
        if a.odom_tum:
            if not a.odom_frame or not a.odom_parent:
                raise BagError("--odom-tum needs --odom-frame and --odom-parent")
            odom = (tftree.read_tum(a.odom_tum), a.odom_frame, a.odom_parent)
        elif a.odom_topic:
            poses, parent, body = tftree.read_pose_topic([a.bag], a.odom_topic)
            body, parent = a.odom_frame or body, a.odom_parent or parent
            if not body:
                raise BagError(f"{a.odom_topic} has no child_frame_id: pass --odom-frame")
            odom = (poses, body, parent)
        if not (a.drop or cals or odom):
            sys.exit("error: nothing to change (--drop, --add, --add-file, --odom-*)")
        ed = tftree.plan_edits([a.bag], a.drop, cals, odom)
        for n in ed.notes:
            print(f"    {n}")
        merge.merge_bags([a.bag], a.out, compression=a.compression, static_tf=ed.static_tf,
                         drop_tf=ed.drop, extra_tf=ed.extra_tf)
        print("\nTF tree of the new bag:")
        print(tftree.format_tree(tftree.tf_edges([a.out])))
    except (BagError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
