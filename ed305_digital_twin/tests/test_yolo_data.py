import cv2
import numpy as np

from ed305.yolo_data import (camera_of, class_map, read_labels, sample_video, split_rows, yolo_line)


def test_camera_of():
    assert camera_of("rec/cam_7_2026-11-03.mp4") == "cam_7"
    assert camera_of("CAM12.mkv") == "cam_12"
    assert camera_of("cam-03_flash.mp4") == "cam_3"
    assert camera_of("lobby.mp4") == "lobby"


def test_yolo_line_roundtrip_and_clip(tmp_path):
    line = yolo_line(1, (-10, 20, 110, 80), 200, 100)      # x0 clipped to 0
    p = tmp_path / "a.txt"
    p.write_text(line + "\n")
    (c, cx, cy, w, h), = read_labels(p)
    assert c == 1
    assert np.allclose([cx, cy, w, h], [55 / 200, 0.5, 110 / 200, 0.6])
    assert read_labels(tmp_path / "missing.txt") == []


def test_class_map_by_name():
    coco = {0: "person", 1: "bicycle", 56: "chair"}
    assert class_map(coco, ["person", "robot"]) == {0: 0}
    assert class_map(coco, ["robot", "Person"]) == {0: 1}
    assert class_map(["person", "robot"], ["person", "robot"]) == {0: 0, 1: 1}


def rows(video, n, cam="cam_1"):
    return [{"image": f"{video}_{i}.jpg", "camera": cam, "video": video, "t": float(i)} for i in range(n)]


def test_split_by_time_keeps_a_gap():
    r = rows("a.mp4", 100) + rows("b.mp4", 50, "cam_2")
    s = split_rows(r, val_frac=0.2, gap=10)
    a, b = s[:100], s[100:]
    assert a[80:] == ["val"] * 20 and a[:71] == ["train"] * 71 and a[71:80] == ["skip"] * 9
    assert b.count("val") == 10 and b.count("skip") == 9
    # no train frame within the gap of a val frame of the same video
    for x, sx in zip(r, s):
        if sx == "train":
            assert all(abs(x["t"] - y["t"]) > 9 for y, sy in zip(r, s) if sy == "val" and y["video"] == x["video"])


def test_split_by_camera():
    r = rows("a.mp4", 5, "cam_1") + rows("b.mp4", 5, "cam_2")
    assert split_rows(r, val_cameras=["cam_2"]) == ["train"] * 5 + ["val"] * 5


def test_sample_video_skips_still_frames(tmp_path):
    path = tmp_path / "cam_1.avi"
    w, h, fps = 320, 180, 10
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (h, w, 3), np.uint8)
    for i in range(100):                     # 10 s: still for 5 s, then a square moves
        f = bg.copy()
        if i >= 50:
            x = 20 + (i - 50) * 5
            cv2.rectangle(f, (x, 60), (x + 40, 120), (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    got = list(sample_video(path, every=1.0, min_change=0.005))
    ts = [round(t) for t, _ in got]
    assert ts[0] == 0
    assert not any(1 <= t <= 4 for t in ts)          # the still part gives one frame
    assert {6, 7, 8, 9} <= set(ts)                    # the moving part is kept
    assert len(list(sample_video(path, every=1.0, min_change=0))) == 10
    assert len(list(sample_video(path, every=1.0, min_change=0, max_frames=3))) == 3
