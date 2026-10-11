#!/usr/bin/env python
"""Self-test for datasets/hydra.py on a small two-robot datasets/swarm_slam.py
export (colour, depth, camera_info and odometry under /r0 and /r1).

Each robot must get a bag of its own with the images copied byte for byte
under /camera/..., and /tf r<i>/odom -> r<i>/base -> r<i>/camera that
composes to the export's odometry, the body frame x forward and z up (the
camera's z forward is the body's x, its y down the body's -z); a hydra_ros
config with the robot's id, frames and depth range and the scene-graph
stream with its mesh (without it under --no-mesh); the launch file; ground
truth copied. An existing bag is never overwritten.

    python scoop_pipeline/tests/test_hydra_export.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import hydra as H                                                   # noqa: E402
import mcap_convert as mc                                           # noqa: E402
import swarm_slam as SS                                             # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

N = 12
T0 = 1_790_000_000_000_000_000


def cam_pose(i, k):
    """The colour camera (optical) in odom: moving along x, yawing slowly."""
    a = 0.05 * k + 0.3 * i
    R_yaw = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R_yaw @ H.R_BODY_OPTICAL                  # camera looking along the yawed x
    T[:3, 3] = [0.1 * k, -1.0 + 2.0 * i, 0.4 + 0.1 * i]
    return T


def gt_pose(i, k):
    """The ground truth: the odometry's camera, 1 m further along x and a little higher."""
    T = cam_pose(i, k).copy()
    T[:3, 3] += [1.0, 0.0, 0.05]
    return T


def make_export(d):
    os.makedirs(d)
    tgt = mc.Intrinsics(50.0, 50.0, 31.5, 23.5, 64, 48, [0.0] * 4, None)
    w = BagWriter(os.path.join(d, "bag"))
    for k in range(N):
        for i in (0, 1):
            t = T0 + k * 100_000_000 + i * 1_000_000
            cam = "r%d/camera" % i
            bgr = np.full((48, 64, 3), 10 * k + i, np.uint8)
            dep = np.full((48, 64), 3000 + k, np.uint16)
            w.write("/r%d/color/image_raw" % i, SS.image(bgr, "bgr8", cam, t), t)
            w.write("/r%d/color/camera_info" % i, SS.camera_info(tgt, cam, t), t)
            w.write("/r%d/aligned_depth_to_color/image_raw" % i, SS.image(dep, "16UC1", cam, t), t)
            w.write("/r%d/odom" % i, SS.odometry(cam_pose(i, k), "r%d/odom" % i, cam, t), t)
    w.close()
    open(os.path.join(d, "report.txt"), "w").write(
        "bag                 : x\nrobots              : r0 = mobile_1, r1 = mobile_2\n")
    for i in (0, 1):
        open(os.path.join(d, "gt_r%d.tum" % i), "w").write("%d 0 0 0 0 0 0 1\n" % i)


def records(path):
    from scoop import bag as B
    r = B.open_bag(path)
    return [(ch.topic, msg) for _, _, ch, msg in B.iter_mcap_records(r)]


def main():
    f = []
    tmp = tempfile.mkdtemp()
    try:
        exp = os.path.join(tmp, "datasets", "swarm_slam", "pass_x")
        make_export(exp)
        res = H.convert(exp)
        out = res["out"]
        if out != os.path.join(tmp, "datasets", "hydra", "pass_x"):
            f.append("default out %s" % out)
        src = records(os.path.join(exp, "bag"))
        from scoop import rosmsg
        ts = rosmsg.typestore()
        for i, machine in ((0, "mobile_1"), (1, "mobile_2")):
            rec = records(os.path.join(out, "bag_r%d" % i))
            topics = sorted({t for t, _ in rec})
            want = sorted(list(H.TOPICS.values()) + ["/tf"])
            if topics != want:
                f.append("r%d topics %s, want %s" % (i, topics, want))
            for s, d in H.TOPICS.items():
                a = [m.data for t, m in src if t == "/r%d/%s" % (i, s)]
                b = [m.data for t, m in rec if t == d]
                if a != b:
                    f.append("r%d %s: not copied byte for byte (%d vs %d)" % (i, d, len(a), len(b)))
            err, k = 0.0, 0
            for t, m in rec:
                if t != "/tf":
                    continue
                tf = ts.deserialize_cdr(m.data, "tf2_msgs/msg/TFMessage").transforms
                got = {(x.header.frame_id, x.child_frame_id): x for x in tf}
                key_ob, key_bc = ("r%d/odom" % i, "r%d/base" % i), ("r%d/base" % i, "r%d/camera" % i)
                if set(got) != {key_ob, key_bc}:
                    f.append("r%d tf frames %s" % (i, sorted(got)))
                    break

                def T_of(x):
                    tr, q = x.transform.translation, x.transform.rotation
                    return mc.xyzq_to_T([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
                T_ob, T_bc = T_of(got[key_ob]), T_of(got[key_bc])
                err = max(err, np.abs(T_ob @ T_bc - cam_pose(i, k)).max())
                # the body frame: z up, x along the camera's view
                err = max(err, np.abs(T_ob[:3, 2] - [0, 0, 1]).max())
                err = max(err, np.abs(T_ob[:3, 0] - cam_pose(i, k)[:3, 2]).max())
                if mc.stamp_ns(got[key_ob]) != T0 + k * 100_000_000 + i * 1_000_000:
                    f.append("r%d tf %d stamped %d" % (i, k, mc.stamp_ns(got[key_ob])))
                k += 1
            if k != N or not err < 1e-6:
                f.append("r%d tf: %d of %d, off by %.2e" % (i, k, N, err))
            y = open(os.path.join(out, "hydra_r%d.yaml" % i)).read()
            lo, hi = H.RANGE[machine]
            for s in ("robot_id: %d" % i, "odom_frame: r%d/odom" % i, "robot_frame: r%d/base" % i,
                      "sensor_frame: r%d/camera" % i, "map_frame: map", "serialize_dsg_mesh: true",
                      "min_dsg_separation_s: 1", "max_range: %g" % hi, "min_range: %g" % lo):
                if s not in y:
                    f.append("hydra_r%d.yaml lacks %r" % (i, s))
            if open(os.path.join(out, "gt_r%d.tum" % i)).read() != "%d 0 0 0 0 0 0 1\n" % i:
                f.append("gt_r%d.tum not copied" % i)
        L = open(os.path.join(out, "scoop_hydra.launch.yaml")).read()
        for s in ("hydra/input/frontleft/rgb/image_raw, to: /camera/color/image_raw",
                  "hydra/input/frontleft/depth_registered/image_rect, to: /camera/depth/image_rect",
                  "closed_set_node", "use_sim_time, value: true",
                  "exit_after_clock: $(var exit_after_clock)", "{enable_lcd: $(var enable_lcd)}",
                  "--config-utilities-file $(var lcd_config)",
                  "set_remap: {from: /hydra/backend/dsg, to: $(var dsg_topic)}"):
            if s not in L:
                f.append("launch lacks %r" % s)
        try:
            H.convert(exp)
            f.append("an existing bag_r0 was overwritten")
        except mc.ConvertError:
            pass
        o2 = os.path.join(tmp, "only1")
        H.convert(exp, o2, robots=[1], mesh=False)
        if os.path.exists(os.path.join(o2, "bag_r0")) or not os.path.exists(os.path.join(o2, "bag_r1")):
            f.append("--robots 1 wrote %s" % sorted(os.listdir(o2)))
        if "serialize_dsg_mesh: false" not in open(os.path.join(o2, "hydra_r1.yaml")).read():
            f.append("--no-mesh still sends the mesh")
        # --configs-only: configs and launch rewritten, bags untouched
        import yaml
        bag0 = os.path.join(out, "bag_r0")
        before = sorted((n, os.path.getsize(os.path.join(bag0, n))) for n in os.listdir(bag0))
        os.remove(os.path.join(out, "hydra_r0.yaml"))
        H.convert(exp, configs_only=True)
        if not os.path.exists(os.path.join(out, "hydra_r0.yaml")):
            f.append("--configs-only did not rewrite hydra_r0.yaml")
        if sorted((n, os.path.getsize(os.path.join(bag0, n))) for n in os.listdir(bag0)) != before:
            f.append("--configs-only touched bag_r0")
        try:
            H.convert(exp, os.path.join(tmp, "nobags"), configs_only=True)
            f.append("--configs-only without bags did not refuse")
        except mc.ConvertError:
            pass
        doc = yaml.safe_load(open(os.path.join(out, "scoop_hydra.launch.yaml")))
        acts = doc["launch"]
        for a_ in acts:
            if "arg" in a_ and set(a_["arg"]) - {"name", "default", "description"}:
                f.append("launch arg %s has unexpected keys %s" % (a_["arg"].get("name"),
                                                                  set(a_["arg"]) - {"name", "default", "description"}))
        if not any(a_.get("arg", {}).get("name") == "visualize" for a_ in acts):
            f.append("launch has no visualize arg")
        g = [a_["group"] for a_ in acts if "group" in a_]
        if not g or g[0].get("if") != "$(var visualize)" or \
                {"from": "hydra_visualizer/dsg", "to": "$(var dsg_topic)"} not in [
                    c.get("set_remap") for c in g[0]["children"]]:
            f.append("launch: visualizer group wrong: %s" % g)
        elif {"name": "use_sim_time", "value": "true"} not in [c.get("set_parameter") for c in g[0]["children"]]:
            f.append("launch: visualizer and rviz not on sim time")
        # --poses gt: /tf from gt_r<i>.tum (offset from the odometry), a frame without one skipped
        exg = os.path.join(tmp, "datasets", "swarm_slam", "pass_g")
        make_export(exg)
        for i in (0, 1):
            with open(os.path.join(exg, "gt_r%d.tum" % i), "w") as fh:
                for k in range(N):
                    if i == 1 and k == 5:
                        continue
                    fh.write(mc.tum_line(T0 + k * 100_000_000 + i * 1_000_000, gt_pose(i, k)) + "\n")
        rg = H.convert(exg, poses="gt")
        if rg["out"] != os.path.join(tmp, "datasets", "hydra", "pass_g_gt"):
            f.append("--poses gt out %s" % rg["out"])
        for i in (0, 1):
            k, err = 0, 0.0
            for t, m in records(os.path.join(rg["out"], "bag_r%d" % i)):
                if t != "/tf":
                    continue
                tf = {(x.header.frame_id, x.child_frame_id): x
                      for x in ts.deserialize_cdr(m.data, "tf2_msgs/msg/TFMessage").transforms}
                x_ob = tf[("r%d/odom" % i, "r%d/base" % i)]
                x_bc = tf[("r%d/base" % i, "r%d/camera" % i)]
                kk = (mc.stamp_ns(x_ob) - T0 - i * 1_000_000) // 100_000_000

                def T_of(x):
                    tr, q = x.transform.translation, x.transform.rotation
                    return mc.xyzq_to_T([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
                err = max(err, np.abs(T_of(x_ob) @ T_of(x_bc) - gt_pose(i, kk)).max())
                k += 1
            want = N - (1 if i == 1 else 0)
            if k != want or not err < 1e-6:
                f.append("--poses gt r%d: %d tf of %d, off by %.2e" % (i, k, want, err))
        rep = open(os.path.join(rg["out"], "report.txt")).read()
        if "ground truth" not in rep or "1 frames without a pose" not in rep:
            f.append("--poses gt report: %s" % rep)
    except Exception:                                              # noqa: BLE001
        import traceback
        f.append(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "hydra export ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
