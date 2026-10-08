"""mink warehouse study: track the expert's own align + descend targets with mink, with and without collision limits.

For every recorded round (cases.json of the CuRobo study) the expert's pre-feedback palm targets (ghost trace)
from the first `align` tick to the last `descend` tick are tracked by mink, tick by tick:
- the body moves as it was measured: pelvis, legs, jaws and the box are written from the state trace every tick
  (`--world static_box` holds the box at its first tracked pose, `--world static` holds body and box there);
- mink moves only the waist and both arms (17 dofs; every other dof is frozen in the QP), from the measured
  joints at the first align tick, under the expert's joint limits and joint speed (carry.joint_speed);
- `free`: palm FrameTasks + a weak posture task; `guarded`: the same plus a collision avoidance limit between
  the hand proxies (mink_model) and the carton walls + platform (mink's limit with its bound fixed to the QP's
  displacement units, mink_model.displacement_collision_limit).
Each tick is judged with the true hand hulls (cases.Surfaces) and the palm position error to the target.

    PYTHONPATH=src:scripts/experiments/mink_warehouse_20261008:scripts/experiments/curobo_warehouse_20261008:outputs/mink_warehouse_20261008/pylib \
        "$VENV_SIM_PY" scripts/experiments/mink_warehouse_20261008/mink_descend.py \
        outputs/curobo_warehouse_20261008/cases.json outputs/mink_warehouse_20261008/descend.json
"""

from __future__ import annotations

import argparse
import json
import time

import mink
import mujoco
import numpy as np
import yaml
from cases import DIST_MAX, Surfaces
from mink_model import HAND_BODIES, SIDES, build, displacement_collision_limit, install_robust_distance

TRACKED = ("align", "descend")
SEQ_DT = 0.02  # s per state sequence number (50 Hz publisher)
SUBSTEPS = 4
POSITION_COST, ORIENTATION_COST, POSTURE_COST = 1.0, 0.6, 1e-2


class Tracker:
    def __init__(self, model, proxies, obstacles, active_dofs, joint_speed, guarded, min_distance, detect):
        self.model, self.active = model, np.asarray(active_dofs)
        frozen = [i for i in range(model.nv) if i not in set(active_dofs)]
        self.freeze = mink.DofFreezingTask(model, dof_indices=frozen)
        self.frames = {
            s: mink.FrameTask(
                f"{s}_wrist_yaw_link", "body", position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST
            )
            for s in SIDES
        }
        self.posture = mink.PostureTask(model, cost=POSTURE_COST)
        hinge_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_dofadr[j] in set(active_dofs)
        ]
        self.limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, {n: joint_speed for n in hinge_names}),
        ]
        if guarded:
            pairs = [(proxies[s], obstacles) for s in SIDES]
            self.limits.append(
                displacement_collision_limit()(
                    model,
                    pairs,
                    minimum_distance_from_collisions=min_distance,
                    collision_detection_distance=detect,
                )
            )
        self.config = mink.Configuration(model)

    def start(self, qpos):
        """Start from the measured posture, the planned joints clipped into the expert's limits (1 mrad inside)."""
        q = np.asarray(qpos, dtype=float).copy()
        m = self.model
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_dofadr[j] in set(self.active.tolist()):
                a = m.jnt_qposadr[j]
                q[a] = np.clip(q[a], m.jnt_range[j, 0] + 1e-3, m.jnt_range[j, 1] - 1e-3)
        self.config.update(q)
        self.posture.set_target_from_configuration(self.config)

    def step(self, measured_qpos, targets, dt):
        """Write the measured body, keep our active dofs, solve toward the palm targets, integrate dt."""
        q = np.asarray(measured_qpos, dtype=float).copy()
        mine = self.config.q.copy()
        qadr = self._active_qadr()
        q[qadr] = mine[qadr]
        self.config.update(q)
        for s, (pos, quat) in zip(SIDES, targets):
            self.frames[s].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(quat)), np.asarray(pos))
            )
        for _ in range(SUBSTEPS):
            v = mink.solve_ik(
                self.config,
                [*self.frames.values(), self.posture],
                dt / SUBSTEPS,
                solver="daqp",
                damping=1e-3,
                limits=self.limits,
                constraints=[self.freeze],
            )
            self.config.integrate_inplace(v, dt / SUBSTEPS)
        return self.config.q.copy()

    def _active_qadr(self):
        if not hasattr(self, "_qadr"):
            m = self.model
            dof_to_qpos = {}
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    dof_to_qpos[m.jnt_dofadr[j]] = m.jnt_qposadr[j]
            self._qadr = np.array([dof_to_qpos[i] for i in self.active])
        return self._qadr


def judge_tick(model, data, surfaces, hand_geoms, obstacles, qpos, ghost_pos):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    worst = (DIST_MAX, None, None)
    for s in SIDES:
        for g in hand_geoms[s]:
            for o in obstacles:
                dist = surfaces.to_box(data, g, o)
                if dist < worst[0]:
                    worst = (dist, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", s)
    err = [
        float(
            np.linalg.norm(
                data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link")] - ghost_pos[h]
            )
        )
        for h, s in enumerate(SIDES)
    ]
    return worst, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--min-distance", type=float, default=0.003, help="minimum_distance_from_collisions, m")
    ap.add_argument("--detect", type=float, default=0.08, help="collision_detection_distance, m")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--world",
        choices=("measured", "static_box", "static"),
        default="measured",
        help="measured: body and box from the trace every tick; static_box: the box held at its first tracked pose; "
        "static: body and box both held there (pure arm + waist motion)",
    )
    args = ap.parse_args()

    install_robust_distance()
    data = json.load(open(args.cases))
    meta, cases = data["meta"], [c for c in data["cases"] if "phases" in c and "descend" in c["phases"]]
    run = meta["run"]
    run_config = yaml.safe_load(open(f"{run}/config.yaml"))
    carry, visual = run_config["task"]["carry"], run_config["task"]["visual"]
    model, proxies, obstacles, active, _ = build(
        meta["scene"], float(carry["waist_pitch_max"]), float(visual["elbow_min_deg"])
    )
    jdata = mujoco.MjData(model)
    hand_geoms = {
        s: [
            g
            for g in range(model.ngeom)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            in {f"{s}_{p}" for p in HAND_BODIES}
            and (model.geom_contype[g] or model.geom_conaffinity[g])
            and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_proxy")
        ]
        for s in SIDES
    }
    surfaces = Surfaces(model, [g for s in SIDES for g in hand_geoms[s]])
    box_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "nor_cal_box_free")]
    hinge_qadr = [model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    body_qadr = np.r_[np.arange(7), hinge_qadr[:12]]  # free base + legs
    trackers = {
        name: Tracker(
            model,
            proxies,
            obstacles,
            active,
            float(carry["joint_speed"]),
            name == "guarded",
            args.min_distance,
            args.detect,
        )
        for name in ("free", "guarded")
    }

    rows = []
    for k, c in enumerate(cases[: args.limit or None]):
        ghost = np.load(f"{run}/trace/{c['label']}_ghost.npz")
        trace = np.load(f"{run}/trace/{c['label']}_state_trace.npz")
        lo, hi = c["phases"][TRACKED[0]][0], c["phases"]["descend"][1]
        ticks = np.flatnonzero((ghost["seq"] >= lo) & (ghost["seq"] <= hi))
        if len(ticks) < 2:
            continue
        raw = lambda seq: trace["qpos"][int(np.clip(np.searchsorted(trace["seq"], seq), 0, len(trace["seq"]) - 1))]  # noqa: E731
        first = raw(ghost["seq"][ticks[0]]).copy()

        def qat(seq):
            q = raw(seq).copy()
            if args.world in ("static_box", "static"):
                q[box_qadr : box_qadr + 7] = first[box_qadr : box_qadr + 7]
            if args.world == "static":
                q[body_qadr] = first[body_qadr]
            return q

        descend_from = c["phases"]["descend"][0]
        row = dict(label=c["label"], ok=c["ok"], why=c["why"], ticks=int(len(ticks)), methods={})
        for name, tracker in trackers.items():
            tracker.start(qat(ghost["seq"][ticks[0]]))
            worst = dict(dist=DIST_MAX)
            t0, prev = time.perf_counter(), ghost["seq"][ticks[0]]
            for i in ticks:
                seq = ghost["seq"][i]
                dt = max(SEQ_DT, SEQ_DT * float(seq - prev))
                prev = seq
                targets = [(ghost["pos"][i, h], ghost["quat"][i, h]) for h in range(2)]
                q = tracker.step(qat(seq), targets, dt)
                (dist, geom, side), err = judge_tick(model, jdata, surfaces, hand_geoms, obstacles, q, ghost["pos"][i])
                if dist < worst["dist"]:
                    worst = dict(dist=dist, geom=geom, side=side, seq=int(seq), in_descend=bool(seq >= descend_from))
            solve_ms = 1000 * (time.perf_counter() - t0) / len(ticks)
            row["methods"][name] = dict(worst=worst, final_err=err, solve_ms_per_tick=solve_ms)
        rows.append(row)
        f, g = row["methods"]["free"], row["methods"]["guarded"]
        print(
            f"[{k + 1}/{len(cases)}] {c['label']} {c['why']:<16} free {1000 * f['worst']['dist']:+6.1f}mm "
            f"end {1000 * max(f['final_err']):5.1f}mm | guarded {1000 * g['worst']['dist']:+6.1f}mm "
            f"end {1000 * max(g['final_err']):5.1f}mm ({g['solve_ms_per_tick']:.1f} ms/tick)",
            flush=True,
        )
    with open(args.out, "w") as fh:
        json.dump(dict(min_distance=args.min_distance, detect=args.detect, world=args.world, rows=rows), fh)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
