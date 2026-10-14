"""S1, ScaleBFM side (TODO 25.4, plan §4.4): the transient of a mode switch.

A trial on the mars floor scene: stand (jittered start, bfm_clip's), raise both jaws over 2 s to a pre-grasp pose in
front (0.35 ahead, +-0.20, 0.95 m, approach 30 deg down) under mode A, hold, switch to mode B at T_SWITCH and hold on.
The keypoints B adds are set, at the switch tick, from FK of the measured pose (plan §4.4) and then held; the ones A
had keep their values. A `--walk` trial switches while the pelvis reference moves forward at 0.3 m/s. Read over
+-0.5 s around the switch, against the same window of a no-switch trial (A throughout) with the same jitter: the
pelvis displacement and its jump, the wrists' jump, the feet's slide, falls.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_switch.py --pairs 2-6,6-2,2-7,7-2,0-2 --repeats 10 \\
        --out outputs/controller_compare_20261013/switch/bfm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import clips as CL  # noqa: E402
from w0_admission import FootWatch, seat  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

SCENE = REPO / "outputs" / "scene" / "walk_mars"
T_RAISE, T_SWITCH, T_END = 2.0, 5.0, 7.0
WIN = 0.5


def trial(pol, sim, a, mode_a, mode_b, rep, walk, pair_b):
    rng = np.random.default_rng([a.seed, mode_a, pair_b, rep, int(walk)])  # the twin shares the jitter
    yaw0 = float(np.radians(rng.uniform(-3, 3)))
    sim.reset(rng.uniform(-0.02, 0.02, 2), yaw0, 0.0)
    seat(sim, rng, a.jitter_q)
    bids = {b: sim.m.body(b).id for b in CL.KEYPOINTS}
    mujoco.mj_kinematics(sim.m, sim.d)
    stand = {b: (sim.d.xpos[i].copy(), sim.d.xquat[i].copy()) for b, i in bids.items()}
    pol.mode = torch.tensor([mode_a], dtype=torch.long, device=a.device)
    pol.reset(sim.state())
    refs = R.Refs(pol, sim)
    st8 = dict(anchor=None, frozen=None)
    pre = {}

    def ref(t):
        if st8["anchor"] is None:
            return dict(stand)
        xy, yaw = st8["anchor"]
        Rz = CL.rz(yaw)
        adv = 0.3 * max(0.0, t - T_RAISE - 1.0) if walk else 0.0
        out = dict(st8["frozen"] or stand)
        pel = np.array([xy[0], xy[1], stand["pelvis"][0][2]]) + Rz @ np.array([adv, 0, 0])
        out["pelvis"] = (pel, CL.quat_of(Rz))
        u = CL.smooth((t - R.SETTLE_S) / T_RAISE)
        for s, sgn in (("left", 1), ("right", -1)):
            w0p, w0q = stand[CL.WRIST[s]]
            Rt = Rz @ CL.jaw_attitude(30)
            tgt = pel + Rz @ np.array([0.35, sgn * 0.20, 0.95 - pel[2]]) - Rt @ CL.OFF
            out[CL.WRIST[s]] = (w0p + u * (tgt - w0p), CL.slerp(w0q, CL.quat_of(Rt), u))
        return out

    rows = dict(t=[], qpos=[])
    fw = FootWatch(sim)
    fell = None
    k = 0
    while k * R.DT < T_END:
        t = k * R.DT
        if st8["anchor"] is None and t >= R.SETTLE_S:
            st8["anchor"] = (sim.d.qpos[:2].copy(), R.yaw_of(sim.d.qpos[3:7]))
        if abs(t - T_SWITCH) < 1e-9 and mode_b != mode_a:
            # the keypoints B adds: FK of the measured pose, held from here; A's keep following their references
            mujoco.mj_kinematics(sim.m, sim.d)
            held = {b: (sim.d.xpos[i].copy(), sim.d.xquat[i].copy()) for b, i in bids.items()}
            st8["frozen"] = held
            pol.mode = torch.tensor([mode_b], dtype=torch.long, device=a.device)
        if abs(t - (T_SWITCH - WIN)) < 1e-9:
            fw.start()
        if fw.p0 is not None and t <= T_SWITCH + WIN:
            fw.tick()
        rows["t"].append(t)
        rows["qpos"].append(sim.d.qpos.copy())
        if sim.fallen():
            fell = round(t, 2)
            break
        sim.step(pol(sim.state(), refs.futures(ref, t)))
        k += 1
    q = np.asarray(rows["qpos"])
    out = dict(mode_a=mode_a, mode_b=mode_b, rep=rep, walk=walk, fell=fell, **fw.result())
    if fell is None:
        i0, i1 = int(round((T_SWITCH - WIN) / R.DT)), int(round((T_SWITCH + WIN) / R.DT))
        i_s = int(round(T_SWITCH / R.DT))
        wr = []
        for i in (i0, i_s, i1):
            sim.d.qpos[:] = q[i]
            mujoco.mj_kinematics(sim.m, sim.d)
            wr.append([sim.d.xpos[bids[CL.WRIST[s]]].copy() for s in ("left", "right")])
        vel = np.linalg.norm(np.diff(q[i0 : i1 + 1, :3], axis=0), axis=1) / R.DT
        out.update(
            pelvis_move_cm=100 * float(np.linalg.norm(q[i1, :3] - q[i_s, :3])),
            pelvis_speed_max=float(vel.max()),
            wrist_move_cm=100 * float(max(np.linalg.norm(wr[2][j] - wr[1][j]) for j in range(2))),
        )
    return out, q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="2-6,6-2,2-7,7-2,0-2,2-0")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--jitter-q", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2541)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pol = R.Policy(device=a.device, backend=ScaleBfmBackendConfig())
    sim = R.Sim(pol.meta, backend=ScaleBfmBackendConfig(scene_dir=str(SCENE)))
    rows = []
    for pair in a.pairs.split(","):
        ma, mb = (int(v) for v in pair.split("-"))
        for rep in range(a.repeats):
            for b in (mb, ma):  # the switch, then its no-switch twin (same jitter seed)
                r, _ = trial(pol, sim, a, ma, b, rep, a.walk, mb)
                r["baseline"] = b == ma
                rows.append(r)
            print(pair, rep, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-2].items()}, flush=True)
        (a.out / f"switch{'_walk' if a.walk else ''}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
