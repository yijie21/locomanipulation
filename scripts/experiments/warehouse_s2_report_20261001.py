"""TODO section 2 (2026-10-01): one report over single-edge warehouse runs, from their manifests and per-tick observations.

    python3 scripts/experiments/warehouse_s2_report_20261001.py s2_base_20261001 s2_rewalk050_20261001 ... [--json out.json]

Per run: outcome counts; the accepted stance depth (and how many walk-ins it took); falls; seconds per round. From
`observations/<label>/observations.jsonl`, per round:
  * arm shake (TODO 2.4.10.20): over align + descend, the commanded elbow's sd and its correlation with the commanded
    wrist pitch (the 2026-09-24 finding was sd 9-15 deg at -0.95..-0.98), and its per-tick step's sd and sign flips;
  * across-wall delivery in descend (TODO 2.4.10.19): per hand, the tool point's offset across the wall (along the
    line between the two grasp points) as COMMANDED (forward kinematics of `reference_joints`) and as MEASURED (of
    `joints`), both in the measured pelvis frame, and the across component of the tracking correction, at the start
    and the end of descend (mm). Commanded minus measured is what SONIC did not deliver of the joint reference; the
    root attitude is left out on both sides. `frame_check_mm` is the median distance between the measured FK point
    and the recorded `measured_tools` in reach (should be ~0: same frame, same model);
  * camera visibility by phase (TODO 2.4.10.13): fraction of ticks in which each stream sees >= 4 box keypoints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RUNS = Path("outputs/runs")
ARM = {"left": (18, 20), "right": (25, 27)}  # MuJoCo order: elbow, wrist pitch


def rows_of(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for m in sorted(run.glob("manifest_*.jsonl"))
        for line in m.read_text().splitlines()
        if line.strip()
    ]


def observations(run: Path, label: str) -> list[dict]:
    path = run / "observations" / label / "observations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def shake(ticks: list[dict]) -> dict:
    """Over align + descend, per hand: the commanded elbow's sd (deg) and its correlation with the wrist pitch -- the
    2026-09-24 numbers (sd 9-15 deg, corr -0.95..-0.98) are these -- and how often its per-tick change flips sign (a
    tick-rate oscillation flips about every tick, a slow null-space drift almost never)."""
    sel = [t for t in ticks if t.get("phase") in ("align", "descend")]
    if len(sel) < 10:
        return {}
    ref = np.degrees(np.array([t["reference_joints"] for t in sel], dtype=float))
    out = {}
    for side, (elbow, wrist) in ARM.items():
        step = np.diff(ref[:, elbow])
        out[side] = dict(
            elbow_sd_deg=round(float(ref[:, elbow].std()), 2),
            corr=round(float(np.corrcoef(ref[:, elbow], ref[:, wrist])[0, 1]), 3),
            tick_step_sd_deg=round(float(step.std()), 3),
            sign_flips=round(float(np.mean(np.sign(step[1:]) != np.sign(step[:-1]))), 3),
        )
    return out


_SOLVERS = {}


def solver_for(run: Path):
    """The expert's own tool-point model (BimanualIK on the run's scene copy); None outside .venv_sim."""
    if run not in _SOLVERS:
        try:
            from locomani.primitives.arm_ik import ArmIK
            from locomani.tasks.crate.bimanual_ik import BimanualIK

            scene = next(run.glob("*/scene_locomani.xml"))
            _SOLVERS[run] = BimanualIK(ArmIK(str(scene)), waist_pitch_max=0.52, waist_dofs="all")  # tool_poses only
        except (ImportError, StopIteration):
            _SOLVERS[run] = None
    return _SOLVERS[run]


def pelvis_frame(t: dict, world) -> np.ndarray:
    w, x, y, z = t["base_quaternion"]
    return Rotation.from_quat([x, y, z, w]).inv().apply(np.asarray(world, dtype=float) - np.asarray(t["base_position"]))


def across(ticks: list[dict], solver) -> dict:
    lr = [t for t in ticks if t.get("low_reach") and "observed_grips" in t["low_reach"]]
    if solver is None or not lr:
        return {}
    check = [
        float(
            np.median(
                np.linalg.norm(
                    solver.tool_poses(np.asarray(t["joints"]))[0] - pelvis_frame(t, t["low_reach"]["measured_tools"]),
                    axis=1,
                )
            )
        )
        for t in lr
        if t.get("phase") == "reach"
    ]
    desc = [t for t in lr if t.get("phase") == "descend"]
    if not desc:
        return dict(frame_check_mm=round(1000 * float(np.median(check)), 1) if check else None)

    def at(t):
        grips_world = np.asarray(t["low_reach"]["observed_grips"], dtype=float)
        grips = pelvis_frame(t, grips_world)
        u = pelvis_frame(t, grips_world[1]) - pelvis_frame(t, grips_world[0])
        u[2] = 0.0
        u /= max(np.linalg.norm(u), 1e-9)
        measured = solver.tool_poses(np.asarray(t["joints"]))[0]
        commanded = solver.tool_poses(np.asarray(t["reference_joints"]))[0]
        correction = pelvis_frame(
            t, np.asarray(t["low_reach"].get("tracking_correction") or np.zeros((2, 3)))
        ) - pelvis_frame(t, np.zeros(3))
        return {
            side: dict(
                measured_mm=round(1000 * float((measured[i] - grips[i]) @ u), 1),
                commanded_mm=round(1000 * float((commanded[i] - grips[i]) @ u), 1),
                correction_mm=round(1000 * float(correction[i] @ u), 1),
            )
            for i, side in enumerate(("left", "right"))
        }

    return dict(
        frame_check_mm=round(1000 * float(np.median(check)), 1) if check else None,
        descend_ticks=len(desc),
        start=at(desc[0]),
        end=at(desc[-1]),
    )


def visibility(ticks: list[dict]) -> dict:
    seen = defaultdict(lambda: defaultdict(list))
    for t in ticks:
        visible = (t.get("guidance") or {}).get("visible") or {}
        for stream in ("ego_view", "left_wrist", "right_wrist"):
            seen[t.get("phase")][stream].append(len(visible.get(stream) or []) >= 4)
    return {phase: {s: round(float(np.mean(v)), 2) for s, v in streams.items()} for phase, streams in seen.items()}


def report(name: str) -> dict:
    run = RUNS / name
    rows = rows_of(run)
    per_round, vis = [], defaultdict(lambda: defaultdict(list))
    for r in rows:
        ticks = observations(run, r["label"])
        attempts = r.get("stand_attempts") or []
        accepted = [a for a in attempts if a.get("accepted")]
        entry = dict(
            label=r["label"],
            ok=bool(r["ok"]),
            why=r.get("why"),
            seconds=round(float(r.get("seconds") or 0), 1),
            walk_ins=len(attempts),
            depth_first=round(attempts[0]["depth_m"], 3) if attempts else None,
            depth=round(accepted[-1]["depth_m"], 3) if accepted else None,
            shake=shake(ticks),
            across=across(ticks, solver_for(run)),
        )
        per_round.append(entry)
        for phase, streams in visibility(ticks).items():
            for s, frac in streams.items():
                vis[phase][s].append(frac)
    depths = [e["depth"] for e in per_round if e["depth"] is not None]
    elbow = [e["shake"][s]["elbow_sd_deg"] for e in per_round for s in ("left", "right") if e["shake"]]
    corr = [e["shake"][s]["corr"] for e in per_round for s in ("left", "right") if e["shake"]]
    return dict(
        run=name,
        rounds=len(rows),
        ok=sum(e["ok"] for e in per_round),
        why=dict(Counter(e["why"] for e in per_round)),
        falls=sum(1 for e in per_round if "fall" in str(e["why"]) or "fell" in str(e["why"])),
        seconds_median=float(np.median([e["seconds"] for e in per_round])) if per_round else None,
        depth_median=float(np.median(depths)) if depths else None,
        depth_le_050=sum(d <= 0.50 for d in depths),
        walk_ins=dict(Counter(e["walk_ins"] for e in per_round)),
        elbow_sd_deg_median=float(np.median(elbow)) if elbow else None,
        elbow_wrist_corr_median=float(np.median(corr)) if corr else None,
        visibility={p: {s: round(float(np.mean(v)), 2) for s, v in st.items()} for p, st in vis.items()},
        per_round=per_round,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = [report(n) for n in a.runs]
    for r in out:
        print(
            f"{r['run']:32s} ok {r['ok']:2d}/{r['rounds']:2d}  falls {r['falls']}  depth med {r['depth_median']}  "
            f"<=0.50 {r['depth_le_050']}  walk-ins {r['walk_ins']}  s {r['seconds_median']}  "
            f"elbow sd {r['elbow_sd_deg_median']} corr {r['elbow_wrist_corr_median']}"
        )
        print(f"{'':32s} why {r['why']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
