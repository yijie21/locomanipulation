"""C1-C7 and the clip tracking tests, SONIC side (TODO 25.4, plan §4.3): the same clips on the streamed channel.

Needs the stack up on the clip's mars floor scene (sonic_clip.sh brings it up and down). A run: the diag_brake reset
(IDLE, reset, start_control, 3 s IDLE), StreamedArm takes the pose stream from the measured joints, then the clip
anchored at the measured pelvis (xy, yaw): BLEND_S of smoothstep from the measured joints to the clip's first frame,
the clip's 29 joints with their finite-difference velocities, the pelvis attitude of each frame, heading-anchored
(the deploy's anchor orientation uses the full base quaternion; SONIC is not given the pelvis position), the hands from the clip's grip column (Dex1 open / closed). Every pushed frame the privileged
state is read: rows of (sim t, qpos, clip time of the frame pushed). Output <out>/<clip>/streamed_r<rep>.npz + .json in
bfm_clip.py's format, plus the round's real-time factor (plan §6: < 0.85 voids the round).

    bash scripts/experiments/controller_compare_20261013/sonic_clip.sh walk_mars 0 0 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import clips as CL  # noqa: E402

from locomani.backends.sonic.agent_server import HAND_CLOSED, HAND_OPEN  # noqa: E402
from locomani.backends.sonic.primitives import LocomotionMode, Primitives  # noqa: E402
from locomani.backends.sonic.streamed_arm import StreamedArm  # noqa: E402
from locomani.backends.sonic.streamed_arm import reanchor as reanchor_planner  # noqa: E402
from locomani.sim.privileged_pub import EpisodeControlClient  # noqa: E402

BLEND_S = 1.5  # bfm_clip.BLEND_S


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def run_one(p, reset, z, f, rep, out: Path) -> dict:
    label = f"{f['name']}_r{rep:02d}"
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    facts = dict(controller="sonic", clip=f["name"], item=f["item"], mode="streamed", rep=rep, scene=f["scene"])
    if s["fallen"]:
        facts.update(anchor_xy=None, anchor_yaw=None, fell_clip_t=-9.0, why="fallen after the reset")
        (out / f["name"]).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f["name"] / f"streamed_r{rep:02d}.npz", t=np.zeros(0), qpos=np.zeros((0, 1)), clip_t=np.zeros(0)
        )
        (out / f["name"] / f"streamed_r{rep:02d}.json").write_text(json.dumps(facts, indent=1))
        return facts
    if abs(float(p.pelvis_yaw(s))) > np.radians(5):
        reanchor_planner(p)
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
    arm = StreamedArm(p, "right", left_guard=False)
    arm.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
    p.cmd.set_hands(left=HAND_OPEN.copy(), right=HAND_OPEN.copy())
    arm.start()
    s = p.obs()
    xy, yaw = np.asarray(p.pelvis_xy(s), float), float(p.pelvis_yaw(s))
    q_anch, _ = CL.anchored(z, xy, yaw)
    joints = q_anch[:, 7:]
    vel = np.gradient(joints, 1.0 / CL.HZ, axis=0)
    q0 = np.asarray(arm.q, float).copy()
    nb = int(BLEND_S * CL.HZ)
    frames, vels, roots, clip_t, grips = [], [], [], [], []
    for i in range(nb):
        u = CL.smooth((i + 1) / nb)
        du = 6 * ((i + 1) / nb) * (1 - (i + 1) / nb) / BLEND_S
        frames.append(q0 + u * (joints[0] - q0))
        vels.append(du * (joints[0] - q0))
        roots.append(yaw_quat(yaw))
        clip_t.append((i - nb) / CL.HZ)
        grips.append(np.zeros(2))
    for i in range(len(joints)):
        frames.append(joints[i])
        vels.append(vel[i])
        roots.append(q_anch[i, 3:7].copy())  # the full pelvis attitude: the deploy anchors on it (motion_anchor_ori_b)
        clip_t.append(i / CL.HZ)
        grips.append(z["grip"][i])
    rows = dict(t=[], qpos=[], clip_t=[])
    fell = None
    t0_sim, t0_wall = float(p.obs()["t"]), time.time()
    for q, v, rq, ct, g in zip(frames, vels, roots, clip_t, grips):
        st = p.obs()
        rows["t"].append(float(st["t"]))
        rows["qpos"].append(np.asarray(st["qpos"], float))
        rows["clip_t"].append(ct)
        if st["fallen"]:
            fell = round(ct, 2)
            break
        arm.root_quat = rq
        arm.hands = {s_: (HAND_CLOSED if g[k] > 0.5 else HAND_OPEN).copy() for k, s_ in enumerate(("left", "right"))}
        arm._push(q, reference_velocity=v)
    rtf = (float(p.obs()["t"]) - t0_sim) / max(time.time() - t0_wall, 1e-6)
    try:
        arm.stop()  # back to the planner either way: the next reset's IDLE frames need it
    except Exception as e:  # noqa: BLE001 -- a fallen robot's stop is best effort
        print(f"[sonic_clip] stop after {label}: {e}", flush=True)
    label = f"streamed_r{rep:02d}"
    (out / f["name"]).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f["name"] / f"{label}.npz", **{k: np.asarray(v) for k, v in rows.items()})
    facts.update(
        anchor_xy=[float(v) for v in xy],
        anchor_yaw=yaw,
        blend_s=BLEND_S,
        fell_clip_t=fell,
        rtf=round(rtf, 3),
        switch=arm.handoff_diagnostics.get("switch"),
    )
    (out / f["name"] / f"{label}.json").write_text(json.dumps(facts, indent=1, default=float))
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=REPO / "outputs" / "controller_compare_20261013" / "clips")
    ap.add_argument("--scene", required=True, help="only the clips of this scene (the stack's)")
    ap.add_argument("--names", default="all")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    names = sorted(q.stem for q in a.clips.glob("*.npz")) if a.names == "all" else a.names.split(",")
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    falls_in_row = 0
    for name in names:
        z, f = CL.load(a.clips, name)
        if f["scene"] != a.scene:
            continue
        for rep in range(a.repeats):
            if (a.out / name / f"streamed_r{rep:02d}.json").exists():
                continue
            if (a.out / "STOP").exists() or falls_in_row >= 3:
                print(
                    "stopping (STOP file, or 3 rounds in a row down before the clip: the stack is not standing the robot up)",
                    flush=True,
                )
                return 1
            r = run_one(p, reset, z, f, rep, a.out)
            # a fall during a clip is a result; only a robot already down after the reset says the stack is broken
            falls_in_row = falls_in_row + 1 if r.get("anchor_xy") is None else 0
            print(f"{name} rep {rep}: fell {r.get('fell_clip_t')} rtf {r.get('rtf')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
