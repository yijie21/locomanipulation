"""What is the bimanual arm chasing during `reach`? Per-round anatomy of CoordinatedReach from observations.jsonl.

Per reach frame: the time blend u = elapsed/planned, the hand's distance to its capture point (capture_error),
the IK residual (reference_hand_error: commanded hand vs what the whole-body solve delivered), the tracking
feedback magnitude, how far the grip targets moved since reach began, and the arm reference speed.

    python3 scripts/experiments/reach_anatomy_20260923.py outputs/runs/box_room_20260922
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARM = slice(15, 29)


def main() -> int:
    run = Path(sys.argv[1])
    rows = []
    for d in sorted(p for p in (run / "observations").iterdir() if p.is_dir()):
        obs = [json.loads(line) for line in (d / "observations.jsonl").read_text().splitlines()]
        why = json.loads((d / "result.json").read_text())["why"]
        reach = [o for o in obs if o["phase"] == "reach" and o.get("low_reach")]
        if len(reach) < 5:
            print(f"{d.name}: {why}, {len(reach)} reach frames")
            continue
        lr = [o["low_reach"] for o in reach]
        ref = np.array([o["reference_joints"] for o in reach])[:, ARM]
        speed = np.r_[0, np.linalg.norm(np.diff(ref, axis=0), axis=1)] * 25  # rad/s at 25 fps
        u = np.array([min(x["reference_elapsed"] / x["planned_reach_seconds"], 9) for x in lr])

        # capture_error is per-hand distances (reach_only) or per-hand (across, along, up) region errors
        def capture(x):
            e = np.asarray(x.get("capture_error", [np.nan, np.nan]), float)
            return float(np.max(np.abs(e) if e.ndim == 1 else np.linalg.norm(e, axis=1)))

        cap = np.array([capture(x) for x in lr])
        last = np.asarray(lr[-1].get("capture_error", [np.nan]), float)
        worst = np.abs(last).max(0).round(3).tolist() if last.ndim == 2 else None
        ikres = np.array([max(x.get("reference_hand_error", [np.nan])) for x in lr])
        fb = np.array([np.linalg.norm(x.get("tracking_correction", np.zeros((2, 3))), axis=1).max() for x in lr])
        grips = np.array([x["observed_grips"] for x in lr])
        drift = np.linalg.norm(grips - grips[0], axis=2).max(1)
        planned = lr[0]["planned_reach_seconds"]
        after = u >= 1
        print(
            f"{d.name}: {why:28s} frames {len(reach):3d} planned {planned:.1f}s | "
            f"capture {cap[0] * 100:5.1f}->{np.nanmin(cap) * 100:5.1f}->{cap[-1] * 100:5.1f} cm | "
            f"IK residual med {np.nanmedian(ikres) * 100:4.1f} end {ikres[-1] * 100:4.1f} cm | "
            f"feedback max {fb.max() * 100:4.1f} cm | grip drift {drift.max() * 100:4.1f} cm | "
            f"end |err| across/along/up {worst} | arm speed before/after plan {np.median(speed[~after]):.2f}/{np.median(speed[after]) if after.any() else float('nan'):.2f} rad/s, "
            f"{after.mean() * 100:3.0f}% frames past plan"
        )
        rows.append(
            dict(
                round=d.name,
                why=why,
                u=u.tolist(),
                capture=cap.tolist(),
                ik=ikres.tolist(),
                feedback=fb.tolist(),
                drift=drift.tolist(),
                speed=speed.tolist(),
            )
        )
    out = Path("outputs/diagnostics/reach_anatomy_20260923")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.json").write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
