"""TODO 26.3.2 (user 2026-10-15): bottle_states.py's round files from an unrecorded SONIC production run -- the sim's state
trace (`<run>/collect/state_trace/gym_<wall>_state_trace.npz`: seq, qpos at 50 Hz, from just before the walk) plus the
manifest, instead of the recording's pose sidecar (recording pulled the real-time factor to 0.835-0.864 and voided half
the rounds).

* arrays: t, pelvis, joints (MuJoCo order), bottle and jaw from qpos, and qpos itself; cmd_wrist / cmd_pelvis are NaN
  (the streamed targets live only in the sidecar), so SONIC's videos from these files have no ghost spheres;
* the clock: the grasp samples in the manifest pair trace `seq` with simulation time (50 Hz); a round with none falls
  back to the approach-stop request (also seq + time);
* events: the simple expert's phases and the guard's brake are simulation times; the walk, the hand-back and the
  verification stages are wall-clock `elapsed_s` in the execution journal, mapped by a straight-line fit of the phases'
  journal times on their simulation times (its slope is the real-time factor, plan §6: < 0.85 voids a round);
* facts: as bottle_states.sonic_round (the walk plan, the outcome), program and label prefix as its CLI.
Trace files pair with manifest rows in order (one per round; checked by count).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_bottle_rsi_20261015/trace_states.py \\
        outputs/runs/<run> <out states dir> [--program pool_accept] [--label-prefix sonicb]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from bottle_states import SCENE, _find, save  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

DT = 0.02
LATE = ("navigate", "simple.handback", "carry_and_verify", "verify_terminal_hold", "complete")


def clock(row: dict) -> tuple[int, float]:
    """A (trace seq, simulation time) pair from the manifest."""
    samples = (_find(row, "simple") or {}).get("samples") if isinstance(_find(row, "simple"), dict) else None
    if samples:
        return int(samples[0]["seq"]), float(samples[0]["t"])
    for a in row["execution"]["metrics"].get("approach_stop_requests") or []:
        req = a["request"]
        return int(req["observation_sequence"]), float(req["observation_physical_time_s"])
    return None


def convert(trace: Path, row: dict, m: mujoco.MjModel) -> tuple[dict, dict]:
    z = np.load(trace)
    seq, q = z["seq"], z["qpos"]
    anchor = clock(row)
    t = anchor[1] + (seq - anchor[0]) * DT if anchor else (seq - seq[0]) * DT
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES]
    ev = {p["phase"]: float(p["t"]) for p in (_find(row, "phases") or [])}
    guard = _find(row, "guard") or {}
    if guard.get("brake"):
        ev["brake"] = float(guard["brake"]["t"])
    journal = {e["stage"]: float(e["elapsed_s"]) for e in row["execution"]["events"]}
    pairs = [(journal[f"simple.{k}"], v) for k, v in ev.items() if f"simple.{k}" in journal]
    rtf = None
    if len(pairs) >= 2:
        w, s = np.array(pairs).T
        slope, icpt = np.polyfit(w, s, 1)  # simulation seconds per wall second
        rtf = round(float(slope), 3)
        for stage in LATE:
            if stage in journal:
                name = {"navigate": "walk", "simple.handback": "handback"}.get(stage, stage)
                ev[name] = float(slope * journal[stage] + icpt)
    else:  # an early failure: the walk starts with the trace, the rest is unknown
        ev["walk"] = float(t[0])
        span = journal.get("complete", max(journal.values())) - journal.get("navigate", 0.0)
        rtf = round(float((t[-1] - t[0]) / span), 3) if span > 0 else None
    facts = dict(
        controller="sonic",
        program="production",
        scene=str(SCENE.relative_to(REPO)),
        events=ev,
        stages=[e["stage"] for e in row["execution"]["events"]],
        edge=dict(axis=guard.get("axis"), line=guard.get("line")),
        band=guard.get("band"),
        guard=guard,
        psi=row["init"]["psi"],
        spawn_yaw=row["init"]["yaw"],
        stand_xy=row["init"]["stand_xy"],
        robot_xy=row["init"]["robot_xy"],
        table_top=row["table_top"],
        ok=row["ok"],
        why=row["why"],
        case_id=row["case_id"],
        seconds=row.get("seconds"),
        stand=_find(row, "stand"),
        rtf=rtf,
        source="state_trace",
    )
    T = len(t)
    arrays = dict(
        t=t, qpos=q, pelvis=q[:, :7], joints=q[:, qadr], bottle=q[:, 40:47], jaw=q[:, m.jnt_qposadr[m.joint("right_hand_gripper_1_joint").id]],
        cmd_pelvis=np.full((T, 7), np.nan), cmd_wrist=np.full((T, 2, 7), np.nan),
    )  # fmt: skip
    return arrays, facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--program", default="production")
    ap.add_argument("--label-prefix", default="sonic")
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(str(SCENE / "scene_locomani.xml"))
    if m.body("water_bottle_object").jntadr[0] != m.njnt - 1 or m.jnt_qposadr[m.njnt - 1] != 40:
        raise SystemExit("the bottle's free joint is not at qpos[40:47] in this scene")
    rows = [json.loads(line) for p in sorted((a.run / "collect").glob("manifest_*.jsonl")) for line in p.open()]
    traces = sorted(
        (a.run / "collect" / "state_trace").glob("*_state_trace.npz"), key=lambda p: int(p.stem.split("_")[1])
    )
    if len(traces) < len(rows):
        raise SystemExit(f"{len(traces)} traces for {len(rows)} manifest rows")
    for row, tr in zip(rows, traces):  # a round that hung leaves a trace without a row: it is the last one, dropped
        arrays, facts = convert(tr, row, m)
        facts["program"] = a.program
        label = f"{a.label_prefix}_r{row['round']:02d}a{row['attempt']}"
        save(a.out, label, arrays, facts)
        print(label, "ok" if row["ok"] else row["why"][:40], len(arrays["t"]), "ticks rtf", facts["rtf"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
