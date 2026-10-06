import numpy as np
import pytest

from ed305.camera import Camera, look_at
from ed305.coverage import Room, visibility
from ed305.placement import (Mount, aim_from_pose, optimise, pose_from_aim,
                             room_aim_point, score)

K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])


@pytest.mark.parametrize("h,t", [(0, 0), (90, 30), (-135, 60), (170, 89.5), (45, 90)])
def test_aim_round_trip_and_roll_zero(h, t):
    T = pose_from_aim([1, 2, 2.6], h, t)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    h2, t2, roll = aim_from_pose(T)
    assert abs((h2 - h + 180) % 360 - 180) < 1e-6 and abs(t2 - t) < 1e-6 and abs(roll) < 1e-6


def test_pose_from_aim_matches_look_at():
    T = look_at([-3, 0, 2.5], [0, 0, 0])
    h, t, _ = aim_from_pose(T)
    assert np.allclose(pose_from_aim([-3, 0, 2.5], h, t), T)


def test_room_aim_point_floor_and_wall():
    room = Room(x=(-4, 4), y=(-5, 5), height=3, obstacles=[], z=np.array([0.0]),
                xy_step=0.5, min_px_per_m=0, margin_px=0, k=2)
    name, p = room_aim_point(pose_from_aim([0, 0, 2.5], 0, 45), room)
    assert name == "floor" and np.allclose(p, [2.5, 0, 0])
    name, p = room_aim_point(pose_from_aim([0, 0, 2.5], 0, 10), room)
    assert name == "right wall" and np.isclose(p[0], 4)


def test_optimise_stays_in_range_and_never_worse():
    room = Room(x=(-4, 4), y=(-4, 4), height=3, obstacles=[], z=np.array([0.0, 1.0]),
                xy_step=0.5, min_px_per_m=0, margin_px=0, k=2)
    _, _, pts = room.grid()
    # two cameras staring at one corner: most of the room is blind
    cams = [Camera("a", K, 1920, 1080, look_at([-3.9, -3.9, 2.6], [-3, -3, 0])),
            Camera("b", K, 1920, 1080, look_at([3.9, -3.9, 2.6], [-3, -3, 0])),
            Camera("c", K, 1920, 1080, look_at([0, 3.9, 2.6], [0, 0, 0]))]
    mounts = [Mount("a", aim_from_pose(cams[0].T_world_cam)[0], (-30, 30), (20, 70)),
              Mount("b", aim_from_pose(cams[1].T_world_cam)[0], (-90, 90), (20, 70)),
              Mount("c", 0, (0, 0), (0, 0), fixed=True)]
    vis0, blocked = visibility(cams, room, pts)
    new, settings, moved = optimise(cams, mounts, room, pts, blocked, 2, 10, 0.5, log=lambda *_: None)
    vis1, _ = visibility(new, room, pts)
    assert score(vis1.sum(0), 2, ~blocked) > score(vis0.sum(0), 2, ~blocked)
    assert 2 not in moved and np.allclose(new[2].T_world_cam, cams[2].T_world_cam)
    for m, c in zip(mounts[:2], new[:2]):
        h, t, _ = aim_from_pose(c.T_world_cam)
        assert m.pan[0] - 1e-6 <= (h - m.heading0 + 180) % 360 - 180 <= m.pan[1] + 1e-6
        assert m.tilt[0] - 1e-6 <= t <= m.tilt[1] + 1e-6
        assert np.allclose(c.center, cams[mounts.index(m)].center)
