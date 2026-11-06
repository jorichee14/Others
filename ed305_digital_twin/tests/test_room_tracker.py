import numpy as np
import pytest

from ed305.tracking import Measurement, SensorModel, TrackerConfig, id_metrics, load_config, track
from ed305.tracking.adapters import polar_measurement, read_measurements_csv, write_measurements_csv
from ed305.tracking.metrics import tracks_to_dict
from ed305.tracking.sim import simulate


def score(kind, seed=0, link=True, drop_sensor=None):
    meas, sensors, zones, gt = simulate(kind, seed=seed)
    if drop_sensor:
        meas = [m for m in meas if m.sensor != drop_sensor]
        sensors = {k: v for k, v in sensors.items() if k != drop_sensor}
    tracks, owner = track(meas, sensors, zones, TrackerConfig(), link=link)
    return id_metrics(gt, tracks_to_dict(tracks)), tracks, meas, owner


def test_dense_cameras_keep_identities():
    m, tracks, _, _ = score("dense")
    base, _, _, _ = score("dense", link=False)
    assert m["IDF1"] >= 0.93
    assert m["pred_ids"] <= m["gt_ids"] + 2
    assert m["IDF1"] > base["IDF1"] + 0.15              # stage 2 is what joins the pieces


def test_sparse_cameras_bridge_the_blind_band():
    m, _, _, _ = score("sparse")
    base, _, _, _ = score("sparse", link=False)
    assert m["IDF1"] >= 0.8
    assert m["IDF1"] > base["IDF1"] + 0.3


def test_adding_radar_does_not_hurt():
    fused, _, _, _ = score("fused")
    cams_only, _, _, _ = score("fused", drop_sensor="radar")
    assert fused["IDF1"] >= cams_only["IDF1"] - 0.02
    assert fused["IDF1"] >= 0.93


def test_robot_identity_from_its_own_pose():
    _, tracks, meas, owner = score("fused")
    robots = [tk for tk in tracks if tk.id == "robot_1"]
    assert len(robots) == 1
    own = [meas[k] for k in robots[0].measurements]
    assert sum(m.sensor == "robot_1" for m in own) > 0.95 * sum(m.sensor == "robot_1" for m in meas)
    assert all(tk.ident is None for tk in tracks if tk.id != "robot_1")


def test_self_reporting_sensor_sees_only_its_target():
    sm = SensorModel("robot_1", p_detect=0.99, ident="robot_1")
    pd = sm.pd_for([None, "robot_1", "robot_2"], np.zeros((3, 2)))
    assert pd.tolist() == [0.0, pytest.approx(0.99), 0.0]


def test_static_people_fade_from_radar_averaged_over_speed_doubt():
    sm = SensorModel("radar", p_detect=0.9, p_detect_static=0.05, min_speed=0.2)
    xy = np.zeros((3, 2))
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.15, 0.0]])
    exact = sm.pd(xy, v)
    assert exact[0] == pytest.approx(0.05) and exact[1] == pytest.approx(0.9)
    sure = sm.pd(xy, v, np.repeat(np.eye(2)[None] * 1e-6, 3, 0))
    unsure = sm.pd(xy, v, np.repeat(np.eye(2)[None] * 0.04, 3, 0))
    assert sure[2] == pytest.approx(0.05, abs=0.01)
    assert 0.2 < unsure[2] < 0.8                           # might be walking: in between


def test_id_metrics_perfect_and_swapped():
    t = np.arange(100) / 10
    a = np.c_[t * 0, t * 0]
    b = np.c_[t * 0 + 5, t * 0]
    gt = {"A": (t, a), "B": (t, b)}
    assert id_metrics(gt, {"x": (t, a), "y": (t, b)})["IDF1"] == pytest.approx(1.0)
    half = len(t) // 2
    swapped = {"x": (t, np.r_[a[:half], b[half:]]), "y": (t, np.r_[b[:half], a[half:]])}
    m = id_metrics(gt, swapped)
    assert m["IDF1"] == pytest.approx(0.5, abs=0.02)
    assert m["switches"] == 2


def test_config_and_csv_roundtrip(tmp_path):
    cfg = tmp_path / "s.yaml"
    cfg.write_text("""
sensors:
  radar: {p_detect: 0.9, clutter: 0.02, coverage: {polygon: [[0, 0], [2, 0], [2, 2], [0, 2]]}}
  robot_1: {ident: robot_1, p_detect: 0.99}
zones:
  entry: [{circle: [0, 0, 1]}]
tracker: {max_gap: 30}
""")
    sensors, zones, over = load_config(cfg)
    assert sensors["radar"].pd(np.array([[1, 1], [3, 3]])).tolist() == [pytest.approx(0.9), 0.0]
    assert sensors["robot_1"].ident == "robot_1" and over == {"max_gap": 30}
    assert zones.log_p(np.array([[0.5, 0], [5, 5]]))[0] > zones.log_p(np.array([[5, 5]]))[0]
    meas = [polar_measurement(1.0, "radar", (0, 0), 0.0, 2.0, 0.1, feat=np.ones(4)),
            Measurement(1.1, "robot_1", [1, 2], np.eye(2) * 0.01, ident="robot_1")]
    write_measurements_csv(tmp_path / "m.csv", meas)
    back = read_measurements_csv(tmp_path / "m.csv")
    assert np.allclose(back[0].xy, meas[0].xy, atol=1e-3) and np.allclose(back[0].cov, meas[0].cov, atol=1e-5)
    assert back[1].ident == "robot_1" and back[0].feat is not None and back[1].feat is None


def test_trust_values_follow_the_evidence():
    from collections import Counter
    from ed305.tracking import build_tracklets, link_tracklets
    meas, sensors, zones, gt = simulate("sparse", seed=0)
    cfg = TrackerConfig()
    tracks = link_tracklets(build_tracklets(meas, sensors, cfg), sensors, zones, cfg, (0, 120))
    right, wrong, sig_seen, sig_unseen = [], [], [], []
    for tk in tracks:
        own = [Counter(meas[k].meta["truth"] for k in tr.meas if meas[k].meta["truth"]).most_common(1)[0][0]
               for tr in tk.tracklets]
        for o0, o1, tr in zip(own, own[1:], tk.tracklets[1:]):
            (right if o0 == o1 else wrong).append(tr.link_conf)
        S = tk.state_at(np.arange(tk.t0, tk.t1, 0.1), cfg)
        sd = np.sqrt(np.linalg.eigvalsh(S["cov"]).max(1))
        sig_seen += list(sd[S["seen"]])
        sig_unseen += list(sd[~S["seen"]])
    assert np.median(sig_seen) < 0.1 < np.median(sig_unseen)        # unseen people: less sure where they are
    if wrong:
        assert np.median(wrong) < np.median(right)                    # wrong identity links: less sure
