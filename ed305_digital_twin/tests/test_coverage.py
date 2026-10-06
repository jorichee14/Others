import numpy as np
import yaml

from ed305.camera import Camera, load_cameras, look_at
from ed305.coverage import Room, covered, greedy_subset, segment_hits_box, visibility

K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])


def cam(name, pos, target):
    return Camera(name, K, 1920, 1080, look_at(pos, target))


def room(obstacles=()):
    return Room(x=(-5, 5), y=(-5, 5), height=3, obstacles=list(obstacles), z=np.array([0.0]),
                xy_step=0.5, min_px_per_m=0, margin_px=0, k=2)


def test_look_at_puts_target_at_image_centre_and_up_is_up():
    c = cam("a", [-3, 0, 2.5], [0, 0, 0])
    uv, z, _ = c.project(np.array([[0, 0, 0], [0, 0, 1.0]]))
    assert np.allclose(uv[0], [960, 540])
    assert uv[1, 1] < uv[0, 1]          # higher in the room = higher in the image
    assert z[0] > 0


def test_straight_down_camera():
    c = cam("d", [0, 0, 3], [0, 0, 0])
    ok, _ = c.in_view(np.array([[0, 0, 0], [0, 0, 4.0], [10, 0, 0]]))
    assert ok.tolist() == [True, False, False]


def test_box_hides_floor_but_not_its_own_top():
    c = cam("d", [0, 0, 3], [0, 0, 0])
    box = ("table", np.array([-1, -1, 0.0]), np.array([1, 1, 0.75]))
    pts = np.array([[0, 0, 0.75], [2, 0, 0.0], [0.5, 0, 0.0]])
    vis, blocked = visibility([c], room([box]), pts)
    assert vis[0].tolist() == [True, True, False]
    assert not blocked.any()            # on the box surface is not inside it
    assert segment_hits_box(np.zeros(3) + [0, 0, 3], np.array([[0.5, 0, 0.0]]), box[1], box[2])[0]


def test_rvec_tvec_matches_T(tmp_path):
    T = look_at([2, -1, 2.6], [0, 0, 0])
    R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]      # room -> camera
    th = np.arccos((np.trace(R) - 1) / 2)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    doc = {"defaults": {"width": 1920, "height": 1080, "fx": 1000},
           "cameras": {"a": {"T_world_cam": T.tolist()},
                       "b": {"rvec": (w * th).tolist(), "tvec": t.tolist()},
                       "c": {"position": [2, -1, 2.6], "look_at": [0, 0, 0]},
                       "off": {"position": [0, 0, 1], "look_at": [1, 0, 0], "enabled": False}}}
    p = tmp_path / "cams.yaml"
    p.write_text(yaml.safe_dump(doc))
    a, b, c = load_cameras(p)
    assert np.allclose(a.T_world_cam, b.T_world_cam) and np.allclose(a.T_world_cam, c.T_world_cam)


def test_greedy_subset_keeps_full_coverage():
    cams = [cam(f"c{i}", [4 * np.cos(a), 4 * np.sin(a), 2.6], [0, 0, 0])
            for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False))]
    r = room()
    _, _, pts = r.grid()
    vis, blocked = visibility(cams, r, pts)
    sub = greedy_subset(vis, 2, ~blocked)
    assert len(sub) < len(cams)
    assert np.array_equal(covered(vis, 2) & ~blocked, covered(vis, 2, sub) & ~blocked)
