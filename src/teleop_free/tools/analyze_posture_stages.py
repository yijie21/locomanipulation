"""Measure posture-stage displacement from captures; optionally fail a drift limit."""

import argparse
import json
from pathlib import Path

import numpy as np


def change(before, after):
    a = before["pelvis"]
    b = after["pelvis"]
    delta = np.asarray(b["pos"]) - a["pos"]
    yaw = b["yaw"] - a["yaw"]
    return dict(
        xy=float(np.linalg.norm(delta[:2])),
        dz=float(delta[2]),
        yaw=float(np.arctan2(np.sin(yaw), np.cos(yaw))),
        final_speed=float(np.linalg.norm(b["vel"][:2])),
        fallen=after["fallen"],
    )


def analyze(folder):
    events = [json.loads(l) for l in (folder / "events.jsonl").read_text().splitlines()]
    boundaries = [e for e in events if e["event"] == "posture_boundary"]
    expected = ["before_arm_start", "after_arm_start", "after_squat", "after_hold"]
    if boundaries and boundaries[0]["stage"] == "before_settle":
        expected.insert(0, "before_settle")
    if not boundaries:
        boundaries = [
            dict(e, stage=e["event"]) for e in events if e["event"] in ("standing_delivered", "crouch_delivered")
        ]
        expected = ["standing_delivered", "crouch_delivered"]
    rows = []
    max_xy = 0.0
    for a, b in zip(boundaries, boundaries[1:]):
        row = dict(
            stage=a["stage"] + " -> " + b["stage"], duration=b["wall"] - a["wall"], **change(a["state"], b["state"])
        )
        row["initial_speed"] = float(np.linalg.norm(a["state"]["pelvis"]["vel"][:2]))
        if a.get("reference") is not None and b.get("reference") is not None:
            row["reference_change_max"] = float(np.max(np.abs(np.asarray(b["reference"]) - a["reference"])))
        states = []
        with (folder / "privileged.jsonl").open() as stream:
            for line in stream:
                e = json.loads(line)
                if a["wall"] <= e["wall"] <= b["wall"]:
                    states.append(e["state"])
        row["max_xy"] = max([row["xy"]] + [change(a["state"], s)["xy"] for s in states])
        max_xy = max(
            [max_xy, change(boundaries[0]["state"], b["state"])["xy"]]
            + [change(boundaries[0]["state"], s)["xy"] for s in states]
        )
        row["fallen"] = row["fallen"] or any(s["fallen"] for s in states)
        row["samples"] = len(states)
        rows.append(row)
    status_path = folder / "capture_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    complete = bool(
        status is not None
        and not status["dropped"]
        and not status["errors"]
        and not status["writer_alive"]
        and events
        and events[-1]["event"] == "experiment_end"
        and [e["stage"] for e in boundaries] == expected
    )
    summary = dict(
        capture=str(folder),
        complete=complete,
        stages=rows,
        max_xy=max_xy if rows else None,
        total=change(boundaries[0]["state"], boundaries[-1]["state"]) if len(boundaries) > 1 else None,
    )
    (folder / "posture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument(
        "--max-xy", type=float, help="Fail if any stage or cumulative endpoint XY exceeds this diagnostic limit (m)"
    )
    a = ap.parse_args()
    s = analyze(a.capture)
    print(json.dumps(s, indent=2))
    if a.max_xy is not None:
        if not s["complete"]:
            raise SystemExit("FAIL: missing posture stages or incomplete capture")
        if s["max_xy"] > a.max_xy or any(r["max_xy"] > a.max_xy or r["fallen"] for r in s["stages"]):
            raise SystemExit("FAIL: posture drift/fall criterion")
