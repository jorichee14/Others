#!/usr/bin/env python
"""Self-tests for the Phase 0 foundations: traj, config, layout, stamps and
the runner. No bag, no data.

The runner tests drive real subprocesses through dummy stages in a temp
data_root, so skip / re-run / failure behaviour is checked exactly as a stage
script will see it.

    python scripts/test_pipeline.py
"""
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoop import bag, runner, traj                                  # noqa: E402
from scoop.config import deep_merge, digest, load_config, stage_view  # noqa: E402
from scoop.layout import PASS, SITE, Layout, LayoutError, parse_unit  # noqa: E402
from scoop.stages import Stage                                        # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok    %s" % name)
    except Exception:
        FAILED.append(name)
        print("  FAIL  %s" % name)
        traceback.print_exc()


def random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    R = traj.quat_to_rot(q)
    # the near-180-degree rotations are where the non-trace branches run
    for k, axis in enumerate(np.eye(3)):
        x, y, z = axis * np.sin(np.pi / 2 - 5e-4)
        R[k] = traj.quat_to_rot(np.array([x, y, z, np.cos(np.pi / 2 - 5e-4)]))
    return R


# --------------------------------------------------------------------------- #
# traj
# --------------------------------------------------------------------------- #
def test_quat_roundtrip_all_branches():
    R = random_rotations(500)
    q = traj.rot_to_quat(R)
    assert np.all(q[:, 3] >= 0)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(traj.quat_to_rot(q), R, atol=1e-10)
    tr = np.trace(R, axis1=1, axis2=2)
    assert (tr < 0).sum() >= 3, "test must exercise the non-trace branches"


def test_tum_roundtrip_and_cleanup():
    tmp = Path(tempfile.mkdtemp())
    try:
        R = random_rotations(6, seed=1)
        T = np.tile(np.eye(4), (6, 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = np.arange(18).reshape(6, 3) * 0.1
        t = 1_700_000_000.0 + np.arange(6) * 0.1
        p = traj.save_tum(tmp / "a.tum", traj.Traj(t, T), comment="frame=os_lidar")
        back = traj.load_tum(p)
        np.testing.assert_array_equal(back.t, t)
        np.testing.assert_allclose(back.T, T, atol=2e-6)
        # unsorted, duplicated and lost rows, as other tools write them
        lines = p.read_text().splitlines()
        body = [l for l in lines if not l.startswith("#")]
        lost = "1700000000.550000000 0 0 0 0 0 0 0"
        (tmp / "b.tum").write_text("\n".join(["# x", body[3], body[0], body[0], lost]
                                             + body[1:3] + body[4:]) + "\n")
        b = traj.load_tum(tmp / "b.tum")
        assert b.lost == 1 and len(b) == 6
        np.testing.assert_array_equal(b.t, t)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_nearest_matches_bag_selector():
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.05, 0.15, 200)) + 100.0
    tr = traj.Traj(t, np.tile(np.eye(4), (200, 1, 1)))
    q = np.concatenate([rng.uniform(t[0] - 1, t[-1] + 1, 2000),
                        0.5 * (t[:-1] + t[1:]), t])          # exact ties + hits
    got = tr.nearest(q, 0.03)
    sel = bag.nearest_pose(t, 0.03)
    ref = np.array([-1 if (k := sel(int(round(s * 1e9)))) is None else k for s in q])
    # ns rounding can move a tie by 1 ns; allow disagreement only at exact ties
    diff = got != ref
    ties = np.isin(q, 0.5 * (t[:-1] + t[1:]))
    assert not np.any(diff & ~ties), q[diff & ~ties][:5]


def test_interp_hits_samples_and_clamps():
    R = random_rotations(5, seed=4)
    T = np.tile(np.eye(4), (5, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.arange(15).reshape(5, 3)
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    tr = traj.Traj(t, T)
    Rs, ps, gap = tr.interp(t)
    np.testing.assert_allclose(Rs, R, atol=1e-10)
    np.testing.assert_allclose(ps, T[:, :3, 3])
    np.testing.assert_allclose(gap, 0.0)
    # midpoint: half the relative rotation, halfway in translation
    Rm, pm, gm = tr.interp([0.05])
    rel = R[0].T @ R[1]
    half = R[0].T @ Rm[0]
    ang = lambda M: np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))  # noqa: E731
    np.testing.assert_allclose(ang(half), ang(rel) / 2, atol=1e-9)
    np.testing.assert_allclose(pm[0], 0.5 * (T[0, :3, 3] + T[1, :3, 3]))
    np.testing.assert_allclose(gm, 0.05)
    # outside the span: clamped to the end pose, gap reports the distance
    Ro, po, go = tr.interp([-1.0, 1.4])
    np.testing.assert_allclose(Ro[0], R[0], atol=1e-10)
    np.testing.assert_allclose(po[1], T[4, :3, 3])
    np.testing.assert_allclose(go, [1.0, 1.0])


def test_inv_se3():
    R = random_rotations(4, seed=5)
    T = np.tile(np.eye(4), (4, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = [[1, 2, 3], [-4, 5, 0], [0, 0, 9], [7, 7, 7]]
    np.testing.assert_allclose(traj.inv_se3(T) @ T, np.tile(np.eye(4), (4, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(traj.inv_se3(T[0]), np.linalg.inv(T[0]), atol=1e-12)


def test_traj_rejects_unsorted():
    try:
        traj.Traj([0.0, 0.0], np.tile(np.eye(4), (2, 1, 1)))
    except ValueError:
        return
    raise AssertionError("duplicate stamps accepted")


# --------------------------------------------------------------------------- #
# config / layout
# --------------------------------------------------------------------------- #
def test_deep_merge_and_digest():
    a = {"x": 1, "d": {"p": 1, "q": [1, 2]}}
    b = {"d": {"q": [3], "r": 2}}
    m = deep_merge(a, b)
    assert m == {"x": 1, "d": {"p": 1, "q": [3], "r": 2}} and a["d"]["q"] == [1, 2]
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    cfg = {"stages": {"s": {"k": 1}}, "sensor": {"t": "/x"}, "other": 5}
    assert stage_view(cfg, "s", ("sensor",)) == {"stage": {"k": 1}, "sensor": {"t": "/x"}}


def test_repo_configs_load():
    old = os.environ.pop("SCOOP_DATA", None)
    try:
        cfg = load_config("site_1")
        assert cfg["site"] == "site_1" and set(cfg["passes"]) == {"A", "B"}
        assert cfg["stages"]["s02_refine"]["max_corr"] == [0.4, 0.2, 0.1]
        os.environ["SCOOP_DATA"] = "/data/x"
        assert load_config("site_1")["data_root"] == "/data/x"
    finally:
        os.environ.pop("SCOOP_DATA", None)
        if old is not None:
            os.environ["SCOOP_DATA"] = old


def test_units_and_paths():
    assert parse_unit("site_1").level == SITE
    u = parse_unit("site_1/mapping_B")
    assert u.level == PASS and u.pass_id == "B"
    s = parse_unit("site_2/session_02/seq_05/")
    assert s.rel == "site_2/session_02/seq_05"
    for bad in ("site1", "site_1/session_02", "site_1/foo/seq_1", ""):
        try:
            parse_unit(bad)
        except LayoutError:
            continue
        raise AssertionError(bad)
    L = Layout({"data_root": "/d", "passes": {"A": {"bag": "raw/x", "traj": "/abs/t.txt"}}})
    assert L.work(parse_unit("site_1"), "s06") == Path("/d/work/site_1/_site/s06")
    assert L.release(s) == Path("/d/release/site_2/session_02/seq_05")
    assert L.release(u) == Path("/d/release/site_1")
    a = parse_unit("site_1/mapping_A")
    assert L.pass_input(a, "bag") == Path("/d/raw/x")
    assert L.pass_input(a, "traj") == Path("/abs/t.txt")
    assert L.reference_pass() == "A"


# --------------------------------------------------------------------------- #
# runner, with dummy stages in real subprocesses
# --------------------------------------------------------------------------- #
STAGE_A = '''
def run(ctx):
    src = ctx.inputs["traj"].read_text()
    ctx.outputs["out"].write_text(src.upper() + str(ctx.params.get("k", 0)))
    print("stage a ran", ctx.unit)
'''
STAGE_B = '''
def run(ctx):
    ctx.outputs["out"].write_text(ctx.inputs["a"].read_text() + "|b")
'''
STAGE_C = '''
def run(ctx):
    parts = [p.read_text() for p in ctx.inputs["all"]]
    ctx.outputs["out"].write_text(" + ".join(parts))
'''
STAGE_FAIL = '''
import os
def run(ctx):
    if os.environ.get("DUMMY_FAIL"):
        raise RuntimeError("boom")
    ctx.outputs["out"].write_text("fine")
'''


class Sandbox:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp())
        sd = self.dir / "scripts"
        sd.mkdir()
        for n, body in (("a", STAGE_A), ("b", STAGE_B), ("c", STAGE_C), ("f", STAGE_FAIL)):
            (sd / f"{n}.py").write_text(textwrap.dedent(body))
        self.pass_stages = [
            Stage("a", PASS, {"traj": "@traj"}, {"out": "a.txt"}, path=sd / "a.py"),
            Stage("b", PASS, {"a": "a/a.txt"}, {"out": "b.txt"}, path=sd / "b.py"),
        ]
        self.site_stages = [
            Stage("c", SITE, {"all": "each:b/b.txt"}, {"out": "release:ref/c.txt"},
                  path=sd / "c.py"),
            Stage("f", SITE, {"c": "ref:a/a.txt"}, {"out": "f.txt"}, path=sd / "f.py"),
            Stage("z", SITE, {}, {"out": "z.txt"}, phase=9, path=sd / "missing.py"),
        ]
        for p in ("A", "B"):
            d = self.dir / "raw" / "site_1" / f"mapping_{p}"
            d.mkdir(parents=True)
            (d / "traj.txt").write_text(f"traj {p}")
        self.cfg = {"data_root": str(self.dir), "reference_pass": "A",
                    "passes": {p: {"traj": f"raw/site_1/mapping_{p}/traj.txt"} for p in "AB"},
                    "stages": {"a": {"k": 1}}}
        self._orig = (runner.stages_for, runner.ORDER)
        levels = {PASS: self.pass_stages, SITE: self.site_stages}
        runner.stages_for = lambda lvl: levels[lvl]
        runner.ORDER = [s.name for s in self.pass_stages + self.site_stages]
        self.log = []

    def run(self, **kw):
        self.log = []
        items = runner.plan(parse_unit("site_1"), self.cfg, Layout(self.cfg))
        items = runner.select(items, only=kw.pop("only", ()), start=kw.pop("start", None),
                              until=kw.pop("until", None))
        return runner.execute(items, self.cfg, out=self.log.append, **kw)

    def ran(self):
        return [l.split()[2].rstrip(":") for l in self.log if l.startswith("[run ]")]

    def close(self):
        runner.stages_for, runner.ORDER = self._orig
        shutil.rmtree(self.dir, ignore_errors=True)


def _sandbox(fn):
    def wrapped():
        sb = Sandbox()
        try:
            fn(sb)
        finally:
            sb.close()
    wrapped.__name__ = fn.__name__
    return wrapped


@_sandbox
def test_runner_first_run_then_skip(sb):
    code = sb.run()
    assert code == 3, (code, sb.log)                 # stops at the unimplemented z
    assert sb.ran() == ["site_1/mapping_A:a", "site_1/mapping_A:b", "site_1/mapping_B:a",
                        "site_1/mapping_B:b", "site_1:c", "site_1:f"], sb.log
    assert (sb.dir / "release/site_1/ref/c.txt").read_text() == "TRAJ A1|b + TRAJ B1|b"
    assert (sb.dir / "work/site_1/mapping_A/a/stamp.json").exists()
    assert "stage a ran site_1/mapping_A" in (sb.dir / "work/site_1/mapping_A/a/log.txt").read_text()
    assert sb.run(until="f") == 0 and sb.ran() == [], sb.log


@_sandbox
def test_runner_config_change_cascades(sb):
    sb.run(until="f")
    sb.cfg["stages"]["a"]["k"] = 2
    sb.run(until="f")
    # a re-ran on both passes, its outputs changed, so b, c and f follow
    assert sb.ran() == ["site_1/mapping_A:a", "site_1/mapping_A:b", "site_1/mapping_B:a",
                        "site_1/mapping_B:b", "site_1:c", "site_1:f"], sb.log


@_sandbox
def test_runner_identical_output_does_not_cascade(sb):
    sb.run(until="f")
    sb.run(until="f", force=["a"])
    assert sb.ran() == ["site_1/mapping_A:a", "site_1/mapping_B:a"], sb.log


@_sandbox
def test_runner_input_edit_and_output_loss(sb):
    sb.run(until="f")
    (sb.dir / "raw/site_1/mapping_B/traj.txt").write_text("traj B edited")
    sb.run(until="f")
    assert sb.ran() == ["site_1/mapping_B:a", "site_1/mapping_B:b", "site_1:c"], sb.log
    (sb.dir / "release/site_1/ref/c.txt").unlink()
    sb.run(until="f")
    assert sb.ran() == ["site_1:c"], sb.log


@_sandbox
def test_runner_failure_leaves_no_stamp(sb):
    sb.run(until="f")
    os.environ["DUMMY_FAIL"] = "1"
    try:
        assert sb.run(only=["f"], force=["f"]) == 1
    finally:
        del os.environ["DUMMY_FAIL"]
    assert not (sb.dir / "work/site_1/_site/f/stamp.json").exists()
    assert "boom" in (sb.dir / "work/site_1/_site/f/log.txt").read_text()
    sb.run(until="f")
    assert sb.ran() == ["site_1:f"], sb.log


@_sandbox
def test_runner_select_and_dry_run(sb):
    assert sb.run(until="a", dry_run=True) == 0
    assert [l for l in sb.log if l.startswith("[would run]")] == [
        "[would run] site_1/mapping_A:a: never run", "[would run] site_1/mapping_B:a: never run"]
    assert not (sb.dir / "work").exists()
    sb.run(until="b")
    sb.run(start="b", until="b", force=["b"])
    assert sb.ran() == ["site_1/mapping_A:b", "site_1/mapping_B:b"], sb.log


@_sandbox
def test_runner_missing_input(sb):
    (sb.dir / "raw/site_1/mapping_A/traj.txt").unlink()
    assert sb.run() == 2
    assert any("input missing" in l for l in sb.log), sb.log


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print("\n%d/%d passed" % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
