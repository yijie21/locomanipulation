"""TODO 25.9.1 handoff records (docs/handoff/scalebfm-warehouse-rsi-20261014.md §6.1): one fixed-format record per round
at each stage boundary, the same extraction for SONIC's and ScaleBFM's round files. These continuous quantities, not
pass / fail postconditions, are what the Reviewer reads: a walk that passes its own stance check but leaves the reach
out of range shows up here as a stance far from where SONIC's successes stood.

Per round (sections absent when the round never got there):
* stance (W1 of warehouse_measure.py, at the reach's start): standoff_cm, lateral_cm, yaw_err_deg, low_gap_cm,
  low_gap_min_cm, still_s (brake to still);
* ready (at `handoff`): wrist_err_cm (max over the two wrists, commanded vs measured), pelvis_z_cm, torso_pitch_deg;
* reach (`handoff` to `close`, or to the end when the reach never closed): seconds, the waist joints' commanded and
  delivered p90 (|rad|, where the reference has joints), the commanded and delivered pelvis height minimum (ScaleBFM
  only: SONIC's stream carries no pelvis reference), wrist tracking error p90 (cm), and ScaleBFM's capture error
  best / last (cm, the worse hand) and the last per-hand [across, along, up];
* bend / hands / lift: W2, W3, W4 of warehouse_measure.py (torso pitch, pelvis low, waist demand; hand separation,
  drift, pinch; lifted, tilt).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/handoff.py \\
        DIR [DIR ...] --out records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from bottle_measure import WAIST, window  # noqa: E402
from warehouse_measure import Round, Scene  # noqa: E402

WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def p90(x) -> float | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return round(float(np.percentile(x, 90)), 4) if len(x) else None


def r(x, k=2):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def ready(rd: Round) -> dict:
    if "handoff" not in rd.ev:
        return {}
    i = min(int(np.searchsorted(rd.t, rd.ev["handoff"])), len(rd.t) - 1)
    cmd = rd.z["cmd_wrist"][i, :, :3]
    meas = np.stack([rd.body(w)[i, :3] for w in WRISTS])
    err = np.linalg.norm(cmd - meas, axis=1)
    return dict(
        wrist_err_cm=r(100 * np.max(err)) if np.all(np.isfinite(err)) else None,
        pelvis_z_cm=r(100 * rd.z["pelvis"][i, 2]),
        torso_pitch_deg=r(rd.torso_pitch([i])[0], 1),
    )


def reach(rd: Round) -> dict:
    if "handoff" not in rd.ev:
        return {}
    end = rd.ev.get("close", rd.ev.get("end", rd.t[-1]))
    w = window(rd.t, rd.ev["handoff"], end)
    if len(w) < 2:
        return {}
    out = dict(seconds=r(end - rd.ev["handoff"]))
    if "cmd_joints" in rd.z:
        cj, mj = rd.z["cmd_joints"][w], rd.z["joints"][w]
        have = np.all(np.isfinite(cj), axis=1)
        for n, j in WAIST.items():
            out[f"waist_{n}_cmd_p90"] = p90(np.abs(cj[have, j]))
            out[f"waist_{n}_meas_p90"] = p90(np.abs(mj[have, j]))
    cp = rd.z["cmd_pelvis"][w, 2]
    if np.any(np.isfinite(cp)):
        out["pelvis_z_cmd_min_cm"] = r(100 * np.nanmin(cp))
    out["pelvis_z_meas_min_cm"] = r(100 * rd.z["pelvis"][w, 2].min())
    cmd = rd.z["cmd_wrist"][w, :, :3]
    meas = np.stack([rd.body(x)[w, :3] for x in WRISTS], 1)
    out["wrist_track_err_p90_cm"] = (
        r(100 * p90(np.linalg.norm(cmd - meas, axis=2).max(1))) if np.any(np.isfinite(cmd)) else None
    )
    if "capture_err" in rd.z:
        e = np.linalg.norm(rd.z["capture_err"][w], axis=2).max(1)  # the worse hand's distance from its grasp point
        e = e[np.isfinite(e)]
        if len(e):
            out.update(capture_err_best_cm=r(100 * e.min()), capture_err_last_cm=r(100 * e[-1]))
            last = rd.z["capture_err"][w][np.isfinite(rd.z["capture_err"][w][:, 0, 0])][-1]
            out["capture_last_cm"] = np.round(100 * last, 1).tolist()  # per hand [across, along, up]
    return out


def record(npz: Path, sc: Scene) -> dict:
    rd = Round(npz, sc)
    f = rd.f
    rec = dict(
        label=npz.stem,
        start=f.get("label", npz.stem).replace("bfm", "sonic", 1),
        source=str(npz.parent),
        controller=f.get("controller"),
        spec_hash=f.get("spec_hash"),
        ok=bool(f.get("ok")),
        why=f.get("why") or f.get("expert_reason") or "",
        events={k: r(v) for k, v in f["events"].items()},
    )
    for name, fn in (("stance", rd.w1), ("ready", lambda: ready(rd)), ("reach", lambda: reach(rd)),
                     ("bend", rd.w2), ("hands", rd.w3), ("lift", rd.w4)):  # fmt: skip
        try:
            sec = fn()
        except (KeyError, IndexError, ValueError) as e:
            sec = {"error": f"{type(e).__name__}: {e}"}
        rec[name] = {k: (r(v, 3) if isinstance(v, float) else v) for k, v in (sec or {}).items()}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sc = Scene()
    n = 0
    with a.out.open("w") as fh:
        for d in a.dirs:
            for npz in sorted(d.glob("*.npz")):
                fh.write(json.dumps(record(npz, sc)) + "\n")
                n += 1
    print(f"{n} records -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
