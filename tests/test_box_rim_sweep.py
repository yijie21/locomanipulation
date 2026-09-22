import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_plan_is_seeded_round_robin_and_jitter_bounded(tmp_path):
    template = tmp_path / "t.request.json"
    template.write_text(
        json.dumps(
            {
                "runtime": {
                    "robot_dxy": [0.0, 0.225],
                    "robot_yaw": 1.5707963,
                    "scene_dir": str(tmp_path),
                    "backend": {},
                },
                "box": {},
            }
        )
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts/box_rim_sweep.py"),
        "plan",
        str(tmp_path / "run"),
        "--template",
        str(template),
        "--levels",
        "0.1",
        "0.2",
        "--per-level",
        "3",
        "--seed",
        "7",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    sweep = json.loads((tmp_path / "run/sweep.json").read_text())
    names = [r["name"] for r in sweep["rounds"]]
    assert names == ["dy0.100_r00", "dy0.200_r00", "dy0.100_r01", "dy0.200_r01", "dy0.100_r02", "dy0.200_r02"]
    for r in sweep["rounds"]:
        req = json.loads((tmp_path / "run/requests" / f"{r['name']}.request.json").read_text())["runtime"]
        assert req["robot_dxy"][1] == r["level"] and abs(req["robot_dxy"][0]) <= 0.02
        assert abs(req["robot_yaw"] - 1.5707963) <= math.radians(3.0) + 1e-9
        assert req["video_label"] == r["name"]
    subprocess.run(cmd, check=True, capture_output=True)  # same seed, same files
    assert json.loads((tmp_path / "run/sweep.json").read_text()) == sweep
