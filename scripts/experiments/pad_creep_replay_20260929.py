#!/usr/bin/env python3
"""Offline grip-creep bench: replay a live hold's robot motion, simulate only the load (TODO 18.3.5).

    $VENV_SIM_PY scripts/experiments/pad_creep_replay_20260929.py outputs/runs/<run>/skills/think_00_1.npz \\
        --scene outputs/scene/pick_place --variant base --variant solimp=0.999,0.999

`plan_episode.py --think-s` stores every privileged state's sim time + qpos while a single-arm load waits
between two calls. This bench loads the scene, drives every robot DOF kinematically along that stream
(interpolated to the sim step, velocities by finite difference, a 1e6 armature so contact and gravity do not
move them within a step) except the four Dex1 finger joints, which
run the deploy's PD (kp 1.5, kd 0.1 on the closed setpoint -0.02, gear 250, 20 N clamp) exactly as the sim
computes it; the can and the basket are free. Each `--variant` rewrites the finger-pad geoms'
solimp / solref / condim in the compiled model and reports the load's palm-frame drift over the replay,
next to the live drift the npz recorded. `--still` replays the first frame only (no body tremor).

This is a screen, not evidence: the stream is the privileged publish rate, so tremor above its Nyquist is
missing, and the WBC does not react to the load. Candidates go back to the live stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.sim.build_sim_scene import PAD_MESHES  # noqa: E402

SIM_DT = 0.005  # gear_sonic SIMULATE_DT: the XML timestep is overridden to 1/sim_frequency (200 Hz)
FINGER_KP, FINGER_KD, FINGER_CLOSED = 1.5, 0.1, -0.02  # deploy's Dex1 PD and HAND_CLOSED


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    ap.add_argument("--scene", type=Path, default=REPO / "outputs/scene/pick_place")
    ap.add_argument("--target", default="soup_can")
    ap.add_argument("--side", default="right")
    ap.add_argument(
        "--variant",
        action="append",
        default=None,
        help="`base` (the XML as built) or comma-free `key=v1,v2` items joined by `;`: solimp=..., solref=..., "
        "condim=N, friction=...",
    )
    ap.add_argument("--still", action="store_true", help="hold the first frame instead of replaying the stream")
    ap.add_argument("--json", type=Path, default=None, help="write the rows here")
    return ap.parse_args(argv)


def pad_geoms(model) -> list[int]:
    mesh_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, m) for m in PAD_MESHES}
    return [
        g
        for g in range(model.ngeom)
        if model.geom_dataid[g] in mesh_ids
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def apply_variant(model, spec: str) -> dict:
    pads = pad_geoms(model)
    assert len(pads) == 4, f"expected 4 pad geoms, found {len(pads)}"
    applied = {}
    if spec != "base":
        for item in spec.split(";"):
            key, _, val = item.partition("=")
            nums = [float(v) for v in val.split(",")]
            for g in pads:
                if key == "solimp":
                    model.geom_solimp[g, : len(nums)] = nums
                elif key == "solref":
                    model.geom_solref[g, : len(nums)] = nums
                elif key == "condim":
                    model.geom_condim[g] = int(nums[0])
                elif key == "friction":
                    model.geom_friction[g, : len(nums)] = nums
                else:
                    raise SystemExit(f"unknown variant key {key!r}")
            applied[key] = nums
    g = pads[0]
    applied["effective"] = dict(
        solref=model.geom_solref[g].tolist(),
        solimp=model.geom_solimp[g].tolist(),
        condim=int(model.geom_condim[g]),
        priority=int(model.geom_priority[g]),
        friction=model.geom_friction[g].tolist(),
    )
    return applied


def robot_dofs(model, meta) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """qpos / qvel indices driven kinematically (robot, minus fingers) and the finger joints' (qadr, dadr)."""
    object_bodies = {info["body_id"] for info in meta["objects"].values()}
    kin_q, kin_v, fq, fv = [], [], [], []
    for j in range(model.njnt):
        b = model.jnt_bodyid[j]
        while model.body_parentid[b] != 0:
            b = model.body_parentid[b]
        if b in object_bodies:
            continue
        name = model.joint(j).name
        nq = {0: 7, 1: 4, 2: 1, 3: 1}[model.jnt_type[j]]
        nv = {0: 6, 1: 3, 2: 1, 3: 1}[model.jnt_type[j]]
        qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
        if "hand_gripper" in name:
            fq.append(qa)
            fv.append(da)
            continue
        kin_q.extend(range(qa, qa + nq))
        kin_v.extend(range(da, da + nv))
    return np.asarray(kin_q), np.asarray(kin_v), fq, fv


def finger_actuators(model, fq) -> list[tuple[int, int]]:
    """(actuator id, qpos adr) for the four finger motors."""
    out = []
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        if model.jnt_qposadr[j] in fq:
            out.append((a, int(model.jnt_qposadr[j])))
    assert len(out) == 4, out
    return out


def palm_offset(model, data, palm_body: int, obj_body: int) -> np.ndarray:
    R = data.xmat[palm_body].reshape(3, 3)
    return R.T @ (data.xpos[obj_body] - data.xpos[palm_body])


def replay(scene: Path, npz: Path, variant: str, *, target: str, side: str, still: bool) -> dict:
    meta = json.loads((scene / "scene_meta.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(scene / "scene_locomani.xml"))
    model.opt.timestep = SIM_DT
    applied = apply_variant(model, variant)
    data = mujoco.MjData(model)
    rec = np.load(npz)
    t_rec, q_rec = rec["qpos"][:, 0], rec["qpos"][:, 1:]
    assert q_rec.shape[1] == model.nq, f"npz qpos has {q_rec.shape[1]} values, scene nq {model.nq}"
    kin_q, kin_v, fq, fv = robot_dofs(model, meta)
    # Immovable within a step: without this every link falls g*dt^2 per step under gravity and the load it
    # carries, the reset below puts the robot back but not the can, and that leak reads as ~5 cm/s of slip.
    model.dof_armature[kin_v] = 1e6
    fingers = finger_actuators(model, fq)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    obj = meta["objects"][target]["body_id"]

    data.qpos[:] = q_rec[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ref = palm_offset(model, data, palm, obj)
    z0 = float(data.xpos[obj][2])
    duration = float(t_rec[-1] - t_rec[0])
    steps = int(duration / SIM_DT)
    prev = q_rec[0].copy()
    samples, max_drift = [], 0.0
    for k in range(1, steps + 1):
        t = t_rec[0] + k * SIM_DT
        if still:
            target_q = q_rec[0]
        else:
            i = min(int(np.searchsorted(t_rec, t)), len(t_rec) - 1)
            i0 = max(i - 1, 0)
            w = 0.0 if t_rec[i] == t_rec[i0] else (t - t_rec[i0]) / (t_rec[i] - t_rec[i0])
            target_q = (1 - w) * q_rec[i0] + w * q_rec[i]
        # kinematic robot: position and the velocity that got it there
        full = data.qpos.copy()
        full[kin_q] = target_q[kin_q]
        vel = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, vel, SIM_DT, prev, full)
        data.qpos[kin_q] = full[kin_q]
        data.qvel[kin_v] = vel[kin_v]
        prev = full
        for a, qa in fingers:
            da = model.jnt_dofadr[model.actuator_trnid[a, 0]]
            data.ctrl[a] = np.clip(FINGER_KP * (FINGER_CLOSED - data.qpos[qa]) - FINGER_KD * data.qvel[da], -20, 20)
        mujoco.mj_step(model, data)
        if k % 20 == 0:
            d = float(np.linalg.norm(palm_offset(model, data, palm, obj) - ref))
            max_drift = max(max_drift, d)
            samples.append([round(k * SIM_DT, 2), round(d, 4), round(float(data.xpos[obj][2]) - z0, 4)])
    live = rec["drift"] if "drift" in rec.files else None
    return dict(
        npz=str(npz),
        variant=variant,
        still=still,
        applied=applied,
        seconds=round(duration, 2),
        drift_m=samples[-1][1] if samples else None,
        max_drift_m=round(max_drift, 4),
        dz_m=samples[-1][2] if samples else None,
        rate_cm_s=round(100 * samples[-1][1] / max(duration, 1e-3), 3) if samples else None,
        live_drift_m=None if live is None else round(float(live[-1, 1]), 4),
        live_rate_cm_s=None if live is None else round(100 * float(live[-1, 1]) / max(duration, 1e-3), 3),
        samples=samples[:: max(1, len(samples) // 20)],
    )


def main(argv=None) -> int:
    args = parse(argv)
    rows = []
    for npz in args.npz:
        for variant in args.variant or ["base"]:
            row = replay(args.scene, npz, variant, target=args.target, side=args.side, still=args.still)
            rows.append(row)
            print(
                f"{npz.name} {variant:40s} {'still' if args.still else 'replay'}: drift {row['drift_m']} m "
                f"({row['rate_cm_s']} cm/s, dz {row['dz_m']}) | live {row['live_drift_m']} m ({row['live_rate_cm_s']} cm/s)",
                flush=True,
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
