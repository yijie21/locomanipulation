"""Protocol 2 (each reach planned from the measured state, lml_selfmodel.drive2) inside LocoManiLab's benchmark host
(humanoid_benchmark: robosuite G1FloatingBody, the G1_BRIDGE composite with the scalebfm backend), 2026-11-18.

Why a second driver: the benchmark tasks run in this host, not in g1_backend's IntegratedSim2Sim, and the host's robot
is not the same body: another robot.xml (md5 differs) and no per-controller armature / damping / frictionloss (PD
torques only). A self model must describe the body it is used on, so its data are collected here, in the host's empty
scene (EmptyEnv), with the same commands, planning and segment layout as drive2.

How the host takes a reference: `set_vla_action({ref_root_pos, ref_root_quat, ref_joint_pos})` once per 50 Hz tick
(the backend's RefQposAdapter FKs it on g1_backend's robot_fk.xml -- the model our Planner plans on -- and pushes it to a
RealtimeMotionBuffer that the policy samples `delay_s` (0.14 s) behind the newest frame, on sim time). The episode's
first push carries MOTION_FIRST_FRAME and the buffer fixes an xy + yaw transform T for the episode (our frame -> world).
Here everything is planned and recorded in world coordinates (as drive2) and pushed through T^-1, read from the buffer.

Two host conventions handled: the host's pelvis body frame sits at the floor (a constant offset c from robot_fk's pelvis
frame, calibrated by FK at start), so root positions are reported in robot_fk's convention (pelvis origin + R c); the
robot spawns where the scene puts it, so the stance reference is placed at the measured spawn xy / yaw.

    PYTHONPATH=$G1B:/workspace/code/loco/src python lml_host.py drive --out DIR --resets 30000 --shard K --tag scalebfm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lml_selfmodel as L  # noqa: E402

HB = Path(os.environ.get("LML_HB", "/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/humanoid_benchmark"))


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def mat_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


class HostDriver:
    """One robosuite env (EmptyEnv or a task env) with G1_BRIDGE scalebfm; sample() as lml_selfmodel.Driver.sample."""

    def __init__(self, env_name: str = "EmptyEnv", backend: str = "scalebfm", seed: int = 0, **env_kw):
        sys.path[:0] = [str(HB), str(L.G1B)]
        import mujoco
        import robocasa  # noqa: F401 -- registers the envs
        import robosuite
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
        from robocasa.g1_host.bridge import load_sim2real_controller_config

        cfg = load_sim2real_controller_config(backend=backend, pico_source="mock", inference_backend="onnx-cpu")
        self.env = robosuite.make(env_name=env_name, robots="G1FloatingBody", controller_configs=cfg, has_renderer=False,
                                  has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, control_freq=50,
                                  hard_reset=False, seed=seed, **env_kw)  # fmt: skip
        self.mj = mujoco
        self.env.reset()
        self.m, self.d = m, d = self.env.sim.model._model, self.env.sim.data._data
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_" + n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_" + n)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.vadr = np.array([m.jnt_dofadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.qadr_mj = np.array([m.jnt_qposadr[jid(n)] for n in MUJOCO_JOINT_NAMES])
        fj = jid("floating_base_joint")
        assert m.jnt_type[fj] == mujoco.mjtJoint.mjJNT_FREE
        self.rq, self.rv = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.pelvis = bid("pelvis")
        self.wrist = {s: bid(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.torso = bid("torso_link")
        self.to_mj = np.array([L.SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])
        self.comp = self.env.robots[0].composite_controller
        self.plan_s = 0.0
        self.pl = L.Planner()
        self.c = self._calibrate()
        self.buf = None

    # -- conventions --------------------------------------------------------------------------------------------------
    def _calibrate(self) -> np.ndarray:
        """c = host-body-relative-to-host-pelvis minus robot_fk-body-relative-to-robot_fk-pelvis (pelvis frame), the same
        for every body when the two kinematic trees agree; checked on both wrists and the torso at the current pose."""
        m, d = self.m, self.d
        self.mj.mj_kinematics(m, d)
        Rp, pp = d.xmat[self.pelvis].reshape(3, 3), d.xpos[self.pelvis]
        q_mj = d.qpos[self.qadr_mj].copy()
        out = self.pl.fk_world(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), q_mj)
        cs = []
        for s, b in self.wrist.items():
            fk_rel = out[s][0] - out[s][1] @ L.HAND_OFF
            cs.append(Rp.T @ (d.xpos[b] - pp) - fk_rel)
        c = np.mean(cs, axis=0)
        spread = max(float(np.linalg.norm(x - c)) for x in cs)
        assert spread < 2e-3, f"host and robot_fk kinematics disagree beyond a constant pelvis offset ({spread:.4f} m)"
        return c

    def pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The pelvis in robot_fk's convention: world position and rotation."""
        d = self.d
        R = d.xmat[self.pelvis].reshape(3, 3).copy()
        return d.xpos[self.pelvis] + R @ self.c, R

    def sample(self) -> dict:
        d = self.d
        p, R = self.pelvis_pose()
        return dict(root=np.r_[p, mat_quat(R)], q=d.qpos[self.qadr].copy(), dq=d.qvel[self.vadr].copy(),
                    w=d.qvel[self.rv + 3 : self.rv + 6].copy(),
                    wr={s: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for s, b in self.wrist.items()},
                    torso=(d.xpos[self.torso].copy(), d.xmat[self.torso].reshape(3, 3).copy()))  # fmt: skip

    # -- the reference ------------------------------------------------------------------------------------------------
    def _find_buffer(self):
        seen, stack = set(), [(self.comp, 0)]
        while stack:
            o, depth = stack.pop()
            if id(o) in seen or depth > 5:
                continue
            seen.add(id(o))
            if hasattr(o, "push_payload") and hasattr(o, "_segment_yaw_offset_quat"):
                return o
            for v in getattr(o, "__dict__", {}).values():
                if hasattr(v, "__dict__") and not isinstance(v, (np.ndarray, str, bytes, int, float)):
                    stack.append((v, depth + 1))
        raise RuntimeError("no RealtimeMotionBuffer under the composite controller")

    def transform(self) -> tuple[float, np.ndarray, np.ndarray]:
        """The buffer's episode transform (our frame -> world): yaw offset, source and destination root xy."""
        b = self.buf
        q = np.asarray(b._segment_yaw_offset_quat, float)
        return 2 * math.atan2(q[3], q[0]), np.asarray(b._segment_src_root_pos_no_z, float)[:2], np.asarray(b._segment_dst_root_pos_no_z, float)[:2]

    def to_frame(self, fw: np.ndarray) -> np.ndarray:
        """A world qpos frame (root xyz, wxyz, joints) -> the frame to push (T^-1: xy and yaw; z and joints unchanged)."""
        dyaw, src, dst = self.transform()
        R = L.rot_z(-dyaw)
        out = fw.astype(np.float64).copy()
        out[:2] = (R @ np.r_[fw[:2] - dst, 0.0])[:2] + src
        out[3:7] = mat_quat(R @ L._qmat(fw[3:7]))
        return out

    def push(self, fw: np.ndarray) -> None:
        f = self.to_frame(fw) if self.buf is not None else fw.astype(np.float64)
        self.comp.set_vla_action(dict(ref_root_pos=f[:3].astype(np.float32), ref_root_quat=f[3:7].astype(np.float32),
                                      ref_joint_pos=f[7:].astype(np.float32)))  # fmt: skip

    def step(self) -> None:
        self.env.step(np.zeros(self.env.action_dim))

    # -- one reset ----------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.env.reset()
        # robosuite rebuilds the robot's controllers on reset: the composite (and its motion buffer) must be fetched
        # again, or the actions go to a stale object and the live one stays in operator mode, settling forever
        self.comp = self.env.robots[0].composite_controller
        self.buf = None

    def start_frames(self) -> np.ndarray:
        """The stance reference at the measured spawn xy and heading (robot_fk's default pose and standing height)."""
        x = self.sample()
        yaw = L.yaw_of(L._qmat(x["root"][3:]))
        start = (np.r_[x["root"][:2], self.pl.start_pose()[0][2]], quat_yaw(yaw), L.DEFAULT_Q.copy())
        return L.hold(start, L.START_HOLD_S + L.MARGIN_S + L.PRE_S)

    def plan(self, x: dict, ref_w: np.ndarray, c: dict):
        """lml_selfmodel.OnlineDriver._plan in world coordinates (ref_w = the current reference frame, world)."""
        return L.OnlineDriver._plan(self, self.pl, x, ref_w, c)

    def run_reset(self, rng, cmds: list[dict] | None = None) -> tuple[dict, dict, np.ndarray, list[dict]]:
        """As lml_selfmodel.OnlineDriver.run_reset_online: stance, then CHAIN reaches planned at their command rows."""
        self.reset()
        cmds = cmds or [L.sample_command(rng) for _ in range(L.CHAIN)]
        frames = self.start_frames()
        reaches, rows, fell, lost, why, f = [], {}, None, None, "", 0
        next_cmd, end_frame, k, anchor = int(round(L.START_HOLD_S * L.FPS)), None, 0, None
        while True:
            self.push(frames[min(f, len(frames) - 1)])
            if self.buf is None:
                self.buf = self._find_buffer()
            self.step()
            if f == 1:
                assert not self.comp._settling, "the bridge is still settling: our reference is not being tracked"
            x = rows[f] = self.sample()
            if float(x["root"][2]) < L.FALL_Z:
                fell, why = f, "height"
                break
            if anchor is not None:
                dev = (x["root"][:2] - anchor[0]) - (frames[min(f, len(frames) - 1), :2] - anchor[1])
                if float(np.linalg.norm(dev)) > L.LOST_M:
                    fell, lost, why = f, f, "lost"
                    break
            if k < len(cmds) and f == next_cmd:
                try:
                    blend, last, rec = self.plan(x, frames[f], cmds[k])
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"ik {type(e).__name__}: {e}"
                    break
                rec.update(chain=k, f_cmd=f, f_end=f + len(blend))
                tail = L.POST_S + (L.PRE_S if k < len(cmds) - 1 else 0.2) + L.MARGIN_S
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, tail)])
                reaches.append(rec)
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                post = int(round(L.POST_S * L.FPS))
                next_cmd = rec["f_end"] + post + int(round(L.PRE_S * L.FPS))
                end_frame = rec["f_end"] + post
                k += 1
            if k == len(cmds) and end_frame is not None and f >= end_frame:
                break
            if f >= len(frames) - 1:
                why = "motion ended"
                break
            f += 1
        dyaw, src, dst = self.transform()
        info = dict(fell=fell, lost=lost, why=why, steps=len(rows), planned=len(reaches),
                    transform=dict(dyaw=dyaw, src=src.tolist(), dst=dst.tolist()))  # fmt: skip
        return rows, info, frames, reaches


def drive(out: Path, shard: int, tag: str, resets: int, seed: int, id0: int, shards: int, limit: int | None) -> None:
    """lml_selfmodel.drive2's chunked, resumable loop on the host."""
    ids = list(range(shard, resets, shards))
    n = min(limit or len(ids), len(ids))
    od = out / f"data_{tag}"
    od.mkdir(parents=True, exist_ok=True)
    chunks = [range(c, min(c + L.CHUNK, n)) for c in range(0, n, L.CHUNK)]
    done = lambda c: (od / f"w{shard:03d}_{c:03d}_log.json").exists()  # noqa: E731
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    print(f"host {tag} shard {shard}: {before}/{len(ids)} resets already driven", flush=True)
    if before == n:
        return
    dr = HostDriver(seed=seed + shard)
    print(f"host {tag} shard {shard}: pelvis offset c {np.round(dr.c, 4).tolist()}", flush=True)
    t0, nres, nseg, nfall, nlost = time.time(), 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        segs, log = [], []
        for j in rng_ids:
            i = ids[j]
            rows, info, frames, reaches = dr.run_reset(np.random.default_rng(seed + i))
            s = L.segments2(rows, info["fell"], info["lost"], frames, reaches, id0 + i, dr.pl)
            segs += s
            log.append(dict(reset=id0 + i, n_segs=len(s), **info))
            nres, nseg = nres + 1, nseg + len(s)
            nfall, nlost = nfall + (info["fell"] is not None and info["lost"] is None), nlost + (info["lost"] is not None)
            if nres % 10 == 0:
                print(f"host {tag} shard {shard}: {before + nres}/{len(ids)} resets, {nseg} segments, falls {nfall}, "
                      f"lost {nlost}, {time.time() - t0:.0f} s (planning {dr.plan_s:.0f} s)", flush=True)  # fmt: skip
        if segs:
            tmp = od / f".w{shard:03d}_{c:03d}.tmp.npz"
            L.save(segs, tmp)
            os.replace(tmp, od / f"w{shard:03d}_{c:03d}.npz")
        (od / f"w{shard:03d}_{c:03d}_log.json").write_text(json.dumps(log))
    print(f"host {tag} shard {shard} done: {before + nres} resets ({nres} this run, {nseg} segments, falls {nfall}, "
          f"lost {nlost}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["drive", "merge"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resets", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=41000000)
    ap.add_argument("--id0", type=int, default=4000000)
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--tag", default="scalebfm")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.phase == "drive":
        drive(a.out, a.shard, a.tag, a.resets, a.seed, a.id0, a.shards, a.limit)
    else:
        L.merge(a.out, a.tag)
