"""Bottle L1 (TODO 25.2, docs/handoff/controller-compare-20261013.md §10): the one per-round state file both controllers
are measured from, and the SONIC side's converter into it.

A round file `<label>.npz` holds, at 50 Hz on the simulation clock:
    t (T,), pelvis (T,7) pos + wxyz, joints (T,29) MuJoCo order, bottle (T,7),
    cmd_pelvis (T,7) and cmd_wrist (T,2,7) [left, right] commanded poses (NaN where the controller had none),
    jaw (T,) right jaw joint (NaN when not recorded)
and `<label>.json` the round's facts: controller, program, scene, events (name -> t), edge {axis, line}, band,
psi, stand_xy, ok, why. bottle_measure.py reads only these, so both controllers go through the same arithmetic.

SONIC: the pose sidecar (pelvis, bottle, joints, the streamed joint targets) of a recorded production run plus the
manifest's guard record and phase times. Its commanded wrists are FK of the streamed joint targets on the measured
pelvis -- the way the simple expert built them.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/bottle_states.py \\
        outputs/runs/controller_compare_bottle_sonic_20261013 outputs/controller_compare_20261013/bottle/sonic/states
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES, isaac_to_mujoco  # noqa: E402

SCENE = REPO / "outputs" / "scene" / "sim_lowcom"
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


class Fk:
    """Body poses of the stock G1 (the bottle scene's robot) from pelvis pose + 29 joints in MuJoCo order."""

    def __init__(self, scene=SCENE):
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene) / "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])

    def put(self, pelvis, joints):
        self.d.qpos[:7] = pelvis
        self.d.qpos[self.qadr] = joints
        mujoco.mj_kinematics(self.m, self.d)

    def pose(self, body):
        b = self.m.body(body).id
        return np.concatenate([self.d.xpos[b], self.d.xquat[b]])


def save(out: Path, label: str, arrays: dict, facts: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{label}.npz", **arrays)
    (out / f"{label}.json").write_text(json.dumps(facts, indent=1))


def sonic_round(sidecar: Path, row: dict, fk: Fk) -> tuple[dict, dict]:
    recs = [json.loads(line) for line in sidecar.open()]
    # the sidecar starts a few ticks before the round's reset (the previous pose, then the clock restarts at 0)
    back = [i for i in range(1, len(recs)) if recs[i]["t"] <= recs[i - 1]["t"]]
    recs = recs[back[-1] :] if back else recs
    T = len(recs)
    t = np.array([r["t"] for r in recs])
    pelvis = np.array([r["pelvis"] for r in recs])
    joints = np.array([r["joints"] for r in recs])
    bottle = np.array([r["objects"]["water_bottle"] for r in recs])
    cmd_wrist = np.full((T, 2, 7), np.nan)
    for i, r in enumerate(recs):
        send = (r.get("reference_command_state") or {}).get("pose_send")
        if r.get("reference_command_state", {}).get("control_owner_requested") != "streamed" or not send:
            continue
        fk.put(pelvis[i], isaac_to_mujoco(np.asarray(send["joint_pos"], float)))
        cmd_wrist[i] = [fk.pose(w) for w in WRISTS]
    stages = [e["stage"] for e in row["execution"]["events"]]
    walls = np.array([r["wall"] for r in recs])
    simple = _find(row, "phases") or []
    ev = {p["phase"]: float(p["t"]) for p in simple}
    guard = _find(row, "guard") or {}
    if guard.get("brake"):
        ev["brake"] = float(guard["brake"]["t"])
    # the walk starts on the first tick the planner is asked to move
    moving = [i for i, r in enumerate(recs) if (r.get("cmd") or {}).get("mode", 0) != 0]
    if moving:
        ev["walk"] = float(t[moving[0]])
    # the late stages exist only as wall times in the execution journal: the sidecar tick nearest in wall time
    for e in row["execution"]["events"]:
        if e["stage"] in ("simple.handback", "carry_and_verify", "verify_terminal_hold", "complete"):
            ev[e["stage"].replace("simple.", "")] = float(t[int(np.argmin(np.abs(walls - e["wall_time"])))])
    facts = dict(
        controller="sonic",
        program="production",
        scene=str(SCENE.relative_to(REPO)),
        events=ev,
        stages=stages,
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
    )
    arrays = dict(
        t=t,
        pelvis=pelvis,
        joints=joints,
        bottle=bottle,
        cmd_pelvis=np.full((T, 7), np.nan),
        cmd_wrist=cmd_wrist,
        jaw=np.full(T, np.nan),
    )
    return arrays, facts


def _find(d, key):
    """First value under `key` anywhere in a nested manifest row (the guard / phases records sit a few levels down)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def plan_round(trace: Path, row: dict, fk: Fk, program: str) -> tuple[dict, dict]:
    """A plan_episode round (no sidecar, no expert journal): the sim's own state trace (qpos at 50 Hz, one row per
    privileged publish) read by joint name; `lift` = the first tick the bottle is 6 cm above its start, which is
    also where `carry` starts (bottle_measure moves it to the base's motion onset)."""
    m = fk.m
    z = np.load(trace)
    q = z["qpos"]
    t = (z["seq"] - z["seq"][0]) * 0.02
    badr = m.jnt_qposadr[m.body("water_bottle_object").jntadr[0]]
    jaw = m.jnt_qposadr[m.joint("right_hand_gripper_1_joint").id]
    T = len(t)
    bottle = q[:, badr : badr + 7]
    up = np.flatnonzero(bottle[:, 2] - bottle[0, 2] > 0.06)
    ev = {} if not len(up) else {"lift": float(t[up[0]]), "carry": float(t[up[0]])}
    arrays = dict(
        t=t,
        pelvis=q[:, :7],
        joints=q[:, fk.qadr],
        bottle=bottle,
        cmd_pelvis=np.full((T, 7), np.nan),
        cmd_wrist=np.full((T, 2, 7), np.nan),
        jaw=q[:, jaw],
    )
    facts = dict(
        controller="sonic",
        program=program,
        scene=str(SCENE.relative_to(REPO)),
        events=ev,
        edge=dict(axis=None, line=None),
        band=None,
        ok=row.get("ok"),
        why=row.get("why", ""),
        steps=row.get("completed"),
        steps_attempted=row.get("steps"),
        round=row.get("round"),
        trace=trace.name,
    )
    return arrays, facts


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "--plan":
        # bottle_states.py --plan <plan_episode run> <out> <program>
        run, out, program = Path(argv[1]), Path(argv[2]), argv[3]
        fk = Fk()
        rows = [json.loads(line) for line in (run / "skills" / "manifest.jsonl").open()]
        traces = sorted((run / "trace").glob("*_state_trace.npz"), key=lambda f: f.stat().st_mtime)
        if len(traces) != len(rows):
            print(f"{len(traces)} traces for {len(rows)} rounds: pairing by order is unsafe, stopping")
            return 1
        for row, tr in zip(rows, traces):
            arrays, facts = plan_round(tr, row, fk, program)
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
    fk = Fk()
    rows = [json.loads(line) for m in sorted((run / "collect").glob("manifest_*.jsonl")) for line in m.open()]
    sidecars = sorted((run / "collect" / "pose_sidecar").glob("*.jsonl"))
    by_round = {}
    for s in sidecars:
        parts = s.stem.split("_")
        by_round[(int(parts[parts.index("round") + 1]), int(parts[parts.index("attempt") + 1]))] = s
    for row in rows:
        s = by_round.get((row["round"], row["attempt"]))
        if s is None:
            print("no sidecar for round", row["round"], row["attempt"])
            continue
        arrays, facts = sonic_round(s, row, fk)
        facts["program"] = program
        label = f"sonic_r{row['round']:02d}a{row['attempt']}"
        save(out, label, arrays, facts)
        print(label, "ok" if row["ok"] else row["why"], len(arrays["t"]), "ticks", sorted(facts["events"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
