"""C1-C7 and the clip tracking tests, ScaleBFM side (TODO 25.4, plan §4.3 / §5): every clip of clips.py, every mode
that can express it, `--repeats` times each.

A run: reset on the clip's mars floor scene with a jittered start (xy +-2 cm, yaw +-3 deg, joints +-`--jitter-q`; the
sim is deterministic), stand SETTLE_S on the seated pose's own keypoints, then the clip anchored at the settled pelvis
(xy, yaw): BLEND_S of smoothstep from the standing keypoints to the clip's first frame, the clip, its last frame held.
The policy is given all 14 keypoints every tick (FK of the clip on the mars model); the mode decides which it sees.
The jaws follow the clip's grip column. Per run: <out>/<clip>/m<mode>_r<rep>.npz (t, qpos at 50 Hz, clip_t =
clip time, negative before the clip) + .json (anchor, fall). clip_measure.py reads them and SONIC's runs alike.

Modes (`--modes auto`): a clip that walks needs the pelvis or the ankles (0, 2, 3, 4, 6, 7); one with hand keyframes
needs the wrists (1-7); a walking reach needs both (2, 3, 4, 6, 7).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_clip.py --names C2_high_reach --modes 2 --repeats 1 \\
        --out outputs/controller_compare_20261013/complex/bfm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import clips as CL  # noqa: E402
from w0_admission import seat  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

CLIPS = REPO / "outputs" / "controller_compare_20261013" / "clips"
BLEND_S = 1.5
WALK_MODES, HAND_MODES, BOTH_MODES = (0, 2, 3, 4, 6, 7), (1, 2, 3, 4, 5, 6, 7), (2, 3, 4, 6, 7)


def walks(z) -> bool:
    q = z["qpos"]
    travel = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    yaw = np.unwrap([CL.yaw_of(v) for v in q[:, 3:7]])
    return travel > 0.3 or abs(yaw[-1] - yaw[0]) > np.radians(45) or float(np.ptp(q[:, 0])) > 0.3


def modes_for(z, f) -> tuple[int, ...]:
    if walks(z):
        return BOTH_MODES if f["keyframes"] else WALK_MODES
    return HAND_MODES


def body_keypoints(sim, bids):
    return np.array([np.concatenate([sim.d.xpos[b], sim.d.xquat[b]]) for b in bids])


def run_one(pol, sim, z, f, mode, rep, a, out: Path) -> dict:
    rng = np.random.default_rng([a.seed, sum(map(ord, f["name"])), mode, rep])
    yaw0 = float(np.radians(rng.uniform(-3, 3)))
    xy0 = rng.uniform(-0.02, 0.02, 2)
    sim.reset(xy0, yaw0, 0.0)
    seat(sim, rng, a.jitter_q)
    bids = [sim.m.body(b).id for b in CL.KEYPOINTS]
    stand = body_keypoints(sim, bids)
    pol.mode = torch.tensor([mode], dtype=torch.long, device=a.device)
    pol.reset(sim.state())
    refs = R.Refs(pol, sim)
    T = len(z["qpos"])
    st8 = dict(anchor=None, kp=None, start=None)

    def frame_at(t):
        if t < R.SETTLE_S or st8["kp"] is None:
            return stand
        u = t - R.SETTLE_S
        if u < BLEND_S:
            s = CL.smooth(u / BLEND_S)
            k0 = st8["start"]
            k1 = st8["kp"][0]
            pos = k0[:, :3] + s * (k1[:, :3] - k0[:, :3])
            quat = np.array([CL.slerp(k0[i, 3:], k1[i, 3:], s) for i in range(len(k0))])
            return np.concatenate([pos, quat], 1)
        i = min(T - 1, int(round((u - BLEND_S) * CL.HZ)))
        return st8["kp"][i]

    def ref(t):
        kp = frame_at(t)
        return {b: (kp[i, :3], kp[i, 3:]) for i, b in enumerate(CL.KEYPOINTS)}

    rows = dict(t=[], qpos=[], clip_t=[])
    end = R.SETTLE_S + BLEND_S + T / CL.HZ
    fell = None
    k = 0
    t_wall = time.time()
    while True:
        t = k * R.DT
        if t >= end:
            break
        if st8["kp"] is None and t >= R.SETTLE_S:
            yaw = R.yaw_of(sim.d.qpos[3:7])
            st8["anchor"] = (sim.d.qpos[:2].copy(), yaw)
            _, st8["kp"] = CL.anchored(z, sim.d.qpos[:2].copy(), yaw)
            st8["start"] = stand
        u = t - R.SETTLE_S - BLEND_S
        g = z["grip"][int(np.clip(round(u * CL.HZ), 0, T - 1))] if u >= 0 else np.zeros(2)
        sim.grip_target[:2] = R.GRIP_CLOSED if g[0] > 0.5 else R.GRIP_OPEN
        sim.grip_target[2:] = R.GRIP_CLOSED if g[1] > 0.5 else R.GRIP_OPEN
        rows["t"].append(t)
        rows["qpos"].append(sim.d.qpos.copy())
        rows["clip_t"].append(u)
        if sim.fallen():
            fell = round(u, 2)
            break
        sim.step(pol(sim.state(), refs.futures(ref, t)))
        k += 1
    label = f"m{mode}_r{rep:02d}"
    (out / f["name"]).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f["name"] / f"{label}.npz", **{key: np.asarray(v) for key, v in rows.items()})
    facts = dict(
        controller="scalebfm",
        clip=f["name"],
        item=f["item"],
        mode=mode,
        rep=rep,
        scene=f["scene"],
        anchor_xy=[float(v) for v in st8["anchor"][0]] if st8["anchor"] else None,
        anchor_yaw=float(st8["anchor"][1]) if st8["anchor"] else None,
        settle_s=R.SETTLE_S,
        blend_s=BLEND_S,
        fell_clip_t=fell,
        jitter=dict(xy=xy0.tolist(), yaw_deg=float(np.degrees(yaw0)), q=a.jitter_q),
        perturb=a.perturb or None,
        wall_s=round(time.time() - t_wall, 1),
    )
    (out / f["name"] / f"{label}.json").write_text(json.dumps(facts, indent=1))
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=CLIPS)
    ap.add_argument("--names", default="all", help="'all' or comma-separated clip names")
    ap.add_argument("--modes", default="auto", help="'auto' (by clip) or comma-separated modes")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jitter-q", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th (clip, mode) job")
    ap.add_argument("--perturb", default="", help="sim/perturb.py spec (JSON), TODO 25.5")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    names = sorted(p.stem for p in a.clips.glob("*.npz")) if a.names == "all" else a.names.split(",")
    jobs = []
    for name in names:
        z, f = CL.load(a.clips, name)
        modes = modes_for(z, f) if a.modes == "auto" else tuple(int(v) for v in a.modes.split(","))
        jobs += [(name, m) for m in modes]
    k, n = (int(v) for v in a.shard.split("/"))
    jobs = jobs[k::n]
    pol = R.Policy(device=a.device, backend=ScaleBfmBackendConfig())
    sims = {}
    for name, mode in jobs:
        z, f = CL.load(a.clips, name)
        if f["scene"] not in sims:
            cfg = ScaleBfmBackendConfig(scene_dir=str(REPO / "outputs" / "scene" / f["scene"]))
            sims[f["scene"]] = R.Sim(pol.meta, backend=cfg, perturb=json.loads(a.perturb) if a.perturb else None)
        for rep in range(a.repeats):
            if (a.out / f["name"] / f"m{mode}_r{rep:02d}.json").exists():
                continue
            r = run_one(pol, sims[f["scene"]], z, f, mode, rep, a, a.out)
            print(f"{name} mode {mode} rep {rep}: fell {r['fell_clip_t']} ({r['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
