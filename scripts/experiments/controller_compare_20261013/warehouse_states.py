"""Warehouse L1 (TODO 25.3, docs/handoff/controller-compare-20261013.md §15): the per-round state file both controllers
are measured from, and the SONIC side's converters into it.

A round file `<label>.npz` holds, at 50 Hz on the simulation clock:
    t (T,), qpos (T,nq) the whole scene state (robot, jaws, box), pelvis (T,7) pos + wxyz, joints (T,29) MuJoCo order,
    box (T,7), cmd_pelvis (T,7) and cmd_wrist (T,2,7) [left, right] commanded poses, cmd_jaw (T,2) commanded jaw
    targets (NaN where the controller had none)
and `<label>.json` the round's facts: controller, program, scene, events (name -> t), edge {name, axis, line},
platform_top, box_top, ok, why, rtf. warehouse_measure.py reads only these, so both controllers go through the same
arithmetic -- the contact counts included, since `qpos` is the full scene state of the same mars model.

SONIC production rounds (crate expert): the sim's state trace (`<run>/trace/<label>_state_trace.npz`, qpos per state
tick), the expert's command log (`observations/<label>/commands.jsonl`: phase starts, the streamed pose's current
frame, the jaw targets) and `result.json`. Commanded wrists = FK of the current streamed frame on the measured pelvis
(bottle_states' rule). SONIC plan rounds (`--plan`, plan_episode): the state trace and the skills manifest; the
events come from the measured state (jaws, box height), there is no command log.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/warehouse_states.py \\
        outputs/runs/controller_compare_warehouse_sonic_20261013 outputs/controller_compare_20261013/warehouse/sonic/states
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from bottle_states import Fk, save  # noqa: E402

from locomani.primitives.g1_joints import isaac_to_mujoco  # noqa: E402

SCENE = REPO / "outputs" / "scene" / "warehouse_box_45x32x33_mars"
BOX = "nor_cal_box_object"
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
JAWS = ("left_hand_gripper_1_joint", "right_hand_gripper_1_joint")
PHASES = ("walk", "settle", "handoff", "reach", "align", "descend", "close", "lift", "stand", "hold")
JAW_OPEN, JAW_CLOSED = 0.0245, -0.06  # the crate expert's jaw targets (visual.close_target)


def square_yaw(heading: float, box_yaw: float) -> float:
    """World yaw that faces the box square-on from the walk-in `heading`: the box's walls are a quarter turn apart and
    the scene lays its long side along the edge (box yaw = heading - pi/2), so the box's misalignment is its yaw
    relative to that, folded into +-pi/4."""
    off = (box_yaw - heading + np.pi / 2 + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(heading + off)


def scene_facts(edge: str, line: float | None = None) -> dict:
    """The walk-in edge of the platform (scene_meta surfaces.A.edges): world axis (unit, toward the platform), the
    edge line along it (the guard's own `line` when given: the nearer of the edge and the box's front face), and the
    platform and box top heights."""
    meta = json.loads((SCENE / "scene_meta.json").read_text())
    surf = meta["surfaces"]["A"]
    h = np.radians(surf["edges"][edge]["heading_deg"])
    axis = np.array([np.cos(h), np.sin(h)])
    reach = abs(axis[0]) * surf["flat_half_extent"][0] + abs(axis[1]) * surf["flat_half_extent"][1]
    edge_xy = np.asarray(surf["xy"][:2], float) - axis * reach
    box = meta["objects"]["nor_cal_box"]
    return dict(
        edge=dict(name=edge, axis=axis.tolist(), line=float(axis @ edge_xy) if line is None else float(line)),
        heading=float(h),
        platform_top=float(surf["top"]),
        box_top=float(surf["top"]) + float(box["height"]),
        box_half=list(box["box"]["half_extent"]),
    )


def _trace(trace: Path, fk: Fk):
    m = fk.m
    z = np.load(trace)
    q = z["qpos"]
    badr = m.jnt_qposadr[m.body(BOX).jntadr[0]]
    return z["seq"], q, q[:, :7], q[:, fk.qadr], q[:, badr : badr + 7]


def sonic_round(run: Path, label: str, fk: Fk) -> tuple[dict, dict]:
    res = json.loads((run / "observations" / label / "result.json").read_text())
    cmds = [json.loads(line) for line in (run / "observations" / label / "commands.jsonl").open()]
    seq, q, pelvis, joints, box = _trace(run / "trace" / f"{label}_state_trace.npz", fk)
    # the command log's clock: simulation_time = 0.02 * state_seq + offset, constant within a round
    off = float(np.median([c["simulation_time"] - 0.02 * c["state_seq"] for c in cmds]))
    t = 0.02 * seq + off
    T = len(t)
    cmd_wrist = np.full((T, 2, 7), np.nan)
    cmd_jaw = np.full((T, 2), np.nan)
    ev = {}
    last_q = last_jaw = None
    k = 0
    for i, s in enumerate(seq):
        while k < len(cmds) and cmds[k]["state_seq"] <= s:
            c = cmds[k]
            ev.setdefault(c["phase"], float(0.02 * c["state_seq"] + off))
            p = c["payload"]
            if c["channel"] == "pose":
                last_q = isaac_to_mujoco(np.asarray(p["diagnostic_current"]["joint_pos"], float))
                last_jaw = (p["left_hand_joints"][0], p["right_hand_joints"][0])
            elif "left_hand_position" in p:
                last_q = None  # the planner owns the body: no streamed frame
                last_jaw = (p["left_hand_position"][0], p["right_hand_position"][0])
            k += 1
        if last_q is not None:
            fk.put(pelvis[i], last_q)
            cmd_wrist[i] = [fk.pose(w) for w in WRISTS]
        if last_jaw is not None:
            cmd_jaw[i] = last_jaw
    guard = res.get("guard") or {}
    if (guard.get("brake") or {}).get("t") is not None:
        ev["brake"] = float(guard["brake"]["t"])
    ev["end"] = float(0.02 * res["evaluation_state_seq"] + off) if res.get("evaluation_state_seq") else float(t[-1])
    pl = res.get("placement") or {}
    facts = dict(
        controller="sonic",
        program="production",
        scene=str(SCENE.relative_to(REPO)),
        events=ev,
        **scene_facts(pl.get("edge", "ym"), guard.get("line")),
        guard=guard,
        stand_attempts=res.get("stand_attempts"),
        placement=pl,
        rise=res.get("rise"),
        tilt_deg=res.get("tilt_deg"),
        rtf=res.get("real_time_factor"),
        seed=res.get("seed"),
        ok=bool(res["ok"]),
        why=res.get("why", ""),
        label=label,
    )
    arrays = dict(
        t=t,
        qpos=q,
        pelvis=pelvis,
        joints=joints,
        box=box,
        cmd_pelvis=np.full((T, 7), np.nan),
        cmd_wrist=cmd_wrist,
        cmd_jaw=cmd_jaw,
    )
    return arrays, facts


def measured_events(t, q, box, fk: Fk) -> dict:
    """Events from the state alone (plan rounds): `close` = both jaws past zero (closing), `lift` = the box 5 cm above its
    start, `release` = after the lift, a jaw back above zero; `carry` = `lift` (warehouse_measure moves it to the base's
    motion onset)."""
    m = fk.m
    jaw = np.stack([q[:, m.jnt_qposadr[m.joint(j).id]] for j in JAWS], 1)
    ev = {}
    shut = np.flatnonzero(np.all(jaw < 0.0, 1))  # an open jaw wobbles 4 mm while walking; a closed one sits near -0.017
    if len(shut):
        ev["close"] = float(t[shut[0]])
    up = np.flatnonzero(box[:, 2] - box[0, 2] > 0.05)
    if len(up):
        ev["lift"] = ev["carry"] = float(t[up[0]])
        after = np.flatnonzero((t > t[up[0]]) & np.any(jaw > 0.0, 1))
        if len(after):
            ev["release"] = float(t[after[0]])
    return ev


def plan_round(trace: Path, row: dict, fk: Fk, program: str, edge: str) -> tuple[dict, dict]:
    seq, q, pelvis, joints, box = _trace(trace, fk)
    t = (seq - seq[0]) * 0.02
    T = len(t)
    ev = measured_events(t, q, box, fk)
    # no command log: the jaws count as commanded closed from the measured closure to the measured release
    cmd_jaw = np.full((T, 2), JAW_OPEN)
    if "close" in ev:
        cmd_jaw[(t >= ev["close"]) & (t < ev.get("release", np.inf))] = JAW_CLOSED
    facts = dict(
        controller="sonic",
        program=program,
        scene=str(SCENE.relative_to(REPO)),
        events=ev,
        **scene_facts(edge),
        ok=row.get("ok"),
        why=row.get("why", ""),
        steps=row.get("completed"),
        steps_attempted=row.get("steps"),
        round=row.get("round"),
        trace=trace.name,
    )
    arrays = dict(
        t=t,
        qpos=q,
        pelvis=pelvis,
        joints=joints,
        box=box,
        cmd_pelvis=np.full((T, 7), np.nan),
        cmd_wrist=np.full((T, 2, 7), np.nan),
        cmd_jaw=cmd_jaw,
    )
    return arrays, facts


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    fk = Fk(SCENE)
    if argv and argv[0] == "--plan":
        # warehouse_states.py --plan <plan_episode run> <out> <program>
        run, out, program = Path(argv[1]), Path(argv[2]), argv[3]
        rows = [json.loads(line) for line in (run / "skills" / "manifest.jsonl").open()]
        traces = sorted((run / "trace").glob("*_state_trace.npz"), key=lambda f: f.stat().st_mtime)
        if len(traces) != len(rows):
            print(f"{len(traces)} traces for {len(rows)} rounds: pairing by order is unsafe, stopping")
            return 1
        for row, tr in zip(rows, traces):
            arrays, facts = plan_round(tr, row, fk, program, "ym")
            label = f"sonic_{program}_r{row['round']:02d}"
            save(out, label, arrays, facts)
            print(label, "ok" if row.get("ok") else row.get("why"), len(arrays["t"]), "ticks", facts["events"])
        return 0
    program = "production"
    if "--program" in argv:  # a production-expert variant (the extrapolation configs): its name in the tables
        k = argv.index("--program")
        program = argv[k + 1]
        del argv[k : k + 2]
    run, out = (Path(a) for a in argv)
    for res in sorted((run / "observations").glob("*/result.json")):
        label = res.parent.name
        if not (run / "trace" / f"{label}_state_trace.npz").exists():
            print(label, "no state trace")
            continue
        arrays, facts = sonic_round(run, label, fk)
        facts["program"] = program
        name = "sonic_" + label.removeprefix("visual_crate_")
        save(out, name, arrays, facts)
        print(
            name,
            "ok" if facts["ok"] else facts["why"],
            len(arrays["t"]),
            "ticks",
            f"rtf {facts['rtf']}",
            sorted(facts["events"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
