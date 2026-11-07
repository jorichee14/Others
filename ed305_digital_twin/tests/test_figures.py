import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_figures_and_schematics(tmp_path):
    run = tmp_path / "run"
    subprocess.run([sys.executable, str(HERE / "room_track.py"), "--sim", "dense", "--out", str(run)],
                   check=True, cwd=HERE, capture_output=True)
    subprocess.run([sys.executable, str(HERE / "make_figures.py"), str(run), "--persons", "3"],
                   check=True, cwd=HERE, capture_output=True)
    for name in ("floor_paths", "trust_timeline"):
        for ext in ("png", "pdf"):
            assert (run / "figures" / f"{name}.{ext}").stat().st_size > 5000
    subprocess.run([sys.executable, str(HERE / "make_schematics.py"), "--out", str(tmp_path / "s")],
                   check=True, cwd=HERE, capture_output=True)
    assert "Belief + trust" in (tmp_path / "s" / "framework.svg").read_text()
