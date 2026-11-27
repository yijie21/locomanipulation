"""selfmodel_simpledata_20261126, evaluation 3: SIMPLE's real pick episodes with a data3 recorder on the Motor.

The episodes are the main metric's A arm (simple_bench.py <task> --method direct, the default combination: perception
column, single RGB-D look, ScaleBFM, leg odometry; bench_via.sh's A arguments), run by simple_bench's own main() with
this file's hooks installed first:

  every tick   Motor.tick (beneath simple_bench's, gen_reach.Recorder's and leg_odo.TickFeed's wrappers) records the
               row as sx_collect's host does (SimpleHost.sample, SRig.record: pelvis in robot_fk's convention, joints,
               velocities, wrists, spheres, fingers, keypoints, the settle speed) and the Motor's reference pose at that
               tick as the frame; Motor.reset clears both.
  each reach   Motor.reach (the pregrasp, the entry; the lift after the close is not recorded) is run as simple_bench
               calls it; Motor._blend is wrapped for the call so its poses and durations are kept (the plan: RT, RQ, RP,
               ref_q, ref_pel, reach_s, as sc_collect.plan_native_b records them). The command row is the last row
               before the call. Kinds: pregrasp "pre" (chain 0), entry "entry" (chain 1), as data3b's kinds.
  the walk     gen_reach.open_walk (simple_bench's approach, one or more legs): the start row and the asked
               displacement (the legs summed, heading frame at the start); walk_real measured at the first command.

Conventions: the record is in the training data's convention, the hand point 13 cm ahead of wrist_yaw_link
(lml_selfmodel.HAND_OFF; the collection's REACH_HAND_OFF), while simple_bench's Motor plans the Dex3 palm (REACH_HAND_OFF
0.0415 in the wrist frame, runtime.OFF). A reach's command is a full wrist pose (palm point T, wrist attitude R), so the
recorded command target is the 13 cm point of that same pose: T + R (HAND_OFF - OFF); ik_err is the plan's last pose's
13 cm point (the IK model's FK, sx_collect.NativeFK) to it; the label is the measured 13 cm point.

Labels (--label):
  task   the episode as it is (the main metric's rhythm): a reach's end label is the last 0.1 s row (counted from its
         command) at or before the next command: the entry's command for the pregrasp, the fingers' first close
         command for the entry. settled_end: d3's settle rule held at that row (pelvis and both hand points under
         2 cm/s for 0.5 s). The supplementary label.
  main   a measurement episode: before each command (the pregrasp, the entry) and before the close, the robot waits
         by d3's rule -- from the reference's end (the walk's: when walk_to returned) the first 0.1 s row (from the
         reach's command) with the rule held, at most 3 s after that end -- and that row is the reach's end label (and
         the next command's row). Only when a command goes changes; what is commanded is simple_bench's own (the
         perception's goals). The primary label.
Out (--out DIR): DIR/data/<task>.npz (+ arm/<task>.npz), data3b's layout (data.py reads it; kind 1 pre, 2 entry),
DIR/rows/<task>.jsonl (simple_bench's rows), DIR/log/<task>.jsonl (per episode: the walk, the reaches' command / label
rows, waits, settled, fell). Reset ids 90_000_000 (main) / 91_000_000 (task) + 1000 * task + 100 * level + index.

    cd <asrun repo>; export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 PYTHONPATH=src MUJOCO_GL=egl \
        SIMPLE_EVAL_OUT=/workspace/code/loco/outputs/simple_eval_20261112 SX_REPO=<repo> CUDA_VISIBLE_DEVICES=...
    $PY .../sd_real.py TabletopGraspMP --label main --episodes 0-29 --out DIR [--device cpu|cuda]
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = Path(os.environ.get("SX_REPO", EXP.parents[1]))
sys.path[:0] = [str(EXP / "selfmodel_simplecollect_20261126"), str(EXP / "simple_eval_20261112")]
import sc_collect as SC  # noqa: E402

SX, D3, L = SC.SX, SC.D3, SC.L
FPS, TICKS = L.FPS, L.TICKS
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
ID0 = {"main": 90_000_000, "task": 91_000_000}
SCALARS = D3.SCALARS_B + ("pert_rms_deg", "walk_task")


class EpHost(SX.SimpleHost):
    """SimpleHost's record interface (sample, pelvis_pose, c, nfk, ...) on an episode's existing env.Env."""

    def __init__(self, env):  # SimpleHost.__init__ without building an Env, resetting it or forcing REACH_HAND_OFF
        import mujoco

        from locomani.backends.scalebfm import motor as MO
        from locomani.backends.scalebfm import runtime as RT
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.controller, self.stance, self.mj, self.MO, self.RT = "scalebfm", "episode", mujoco, MO, RT
        self.env, self.mot = env, env.m
        sim = env.sim
        self.m, self.d = m, d = sim.m, sim.d  # noqa: F841
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.vadr = np.array([m.jnt_dofadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.qadr_mj = np.array([m.jnt_qposadr[jid(n)] for n in MUJOCO_JOINT_NAMES])
        fj = jid("floating_base_joint")
        self.rq, self.rv = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.pelvis = bid("pelvis")
        assert int(m.jnt_bodyid[fj]) == self.pelvis
        self.wrist = {s: bid(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.torso = bid("torso_link")
        self.to_mj = np.array([L.SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])
        self.pl = L.Planner()
        self.nfk = SX.NativeFK(self.mot.ik)
        self.ref_delay_f = 0
        self.frames = None
        self.c = self._calibrate()
        self.off = np.asarray(RT.OFF, float).copy()  # the Motor's hand point (the Dex3 palm)


class EpRec:
    """One episode's record (see the module docstring)."""

    def __init__(self, env, mode: str, a):
        self.env, self.mot, self.mode, self.a = env, env.m, mode, a
        self.h = EpHost(env)
        self.rig = SX.SRig(self.h)
        self.side = a.side
        self.clear()
        m = self.mot
        tick0, reach0, reset0 = m.tick, m.reach, m.reset

        def tick():
            k = m.k
            self.frames[k] = SC.pose_frame(self.h, m.ref(m.t))
            ok = tick0()
            if ok:
                x = self.rows[k] = self.h.sample()
                self.rig.record(x)
                if self.close_f is None and self.entry_ret is not None:
                    fc = self.env.sim.fingers_cmd.get(self.side)
                    if fc is not None and not np.array_equal(np.asarray(fc), self.f_pre):
                        self.close_f = k  # the close's first command went with this tick: the row before is its moment
            return ok

        def reset(*args, **kw):
            out = reset0(*args, **kw)
            self.clear()
            return out

        def reach(side, pos, rot=None, **kw):
            n = self.n_reach
            self.n_reach += 1
            if n >= 2 or side != self.side:  # the lift after the close: not recorded
                return reach0(side, pos, rot, **kw)
            return self.reach(n, reach0, side, pos, rot, kw)

        m.tick, m.reach, m.reset = tick, reach, reset

    def clear(self) -> None:
        self.rows, self.frames, self.reaches = {}, {}, []
        self.rig.clear()
        self.n_reach, self.walk0, self.walk_asked, self.walk_end_f = 0, None, None, None
        self.entry_ret, self.close_f, self.f_pre, self.waits = None, None, None, []

    # -- the walk -------------------------------------------------------------------------------------------------------
    def walk_leg(self, fwd: float, left: float) -> None:
        if self.walk0 is None:
            self.walk0 = self.last()
            self.walk_asked = np.zeros(2)
        self.walk_asked += (fwd, left)

    def last(self) -> int:
        return self.mot.k - 1

    # -- the settle rule --------------------------------------------------------------------------------------------------
    def quiet_at(self, f: int) -> bool:
        """d3's rule held at row f: the settle speed under SETTLE_V on the SETTLE_F rows ending at f."""
        fs = range(f - D3.SETTLE_F + 1, f + 1)
        return all(g in self.rows for g in fs) and max(self.rows[g]["v"] for g in fs) < D3.SETTLE_V

    def wait(self, open_f: int, f_cmd: int | None) -> tuple[int, bool]:
        """--label main: tick until d3's rule holds on a 0.1 s row (from f_cmd), at most until open_f + 3 s; returns
        the row and whether it settled."""
        m = self.mot
        t0 = self.last()
        while True:
            f = self.last()
            grid = f_cmd is None or (f - f_cmd) % TICKS == 0
            if grid and f >= open_f + D3.SETTLE_F and self.quiet_at(f):
                ok = True
                break
            if grid and f >= open_f + D3.SETTLE_MAX_F:
                ok = False
                break
            if m.fell_at is not None or not m.tick():  # one tick on the current reference (Motor.hold's, without its clock test)
                return f, False
        self.waits.append(dict(open_f=open_f, from_f=t0, at_f=f, waited_s=(f - t0) / FPS, settled=ok))
        return f, ok

    def label_prev(self, f: int, settled: bool) -> None:
        rc = self.reaches[-1]
        rc.update(f_label=f, settled_end=bool(settled), settle_s=(f - rc["f_end"]) / FPS)

    def grid_before(self, f_cmd: int, f: int) -> int:
        return f_cmd + TICKS * ((f - f_cmd) // TICKS)

    # -- a reach ----------------------------------------------------------------------------------------------------------
    def reach(self, n: int, reach0, side, pos, rot, kw):
        m, h = self.mot, self.h
        kind = "pre" if n == 0 else "entry"
        if n == 0:
            open_f = self.walk_end_f if self.walk_end_f is not None else self.last()
        else:
            open_f = self.reaches[-1]["f_end"]
        if self.mode == "main":
            f, ok = self.wait(open_f, None if n == 0 else self.reaches[-1]["f_cmd"])
            if n == 1:
                self.label_prev(f, ok)
        elif n == 1:  # task: the pregrasp's label at the entry's command
            f = self.grid_before(self.reaches[-1]["f_cmd"], self.last())
            self.label_prev(f, self.quiet_at(f))
        f_cmd = self.last()
        x = self.rows[f_cmd]
        if n == 1:
            self.f_pre = np.asarray(self.env.sim.fingers_cmd.get(side), float).copy()
        cap = {}
        saved = {k: m.__dict__[k] for k in ("_blend",) if k in m.__dict__}
        blend0 = m._blend

        def blend(poses, durations):
            cap["poses"], cap["durs"] = list(poses), list(durations)
            return blend0(poses, durations)

        m._blend = blend
        t0 = time.time()
        try:
            out = reach0(side, pos, rot, **kw)
        finally:
            m.__dict__.pop("_blend", None)
            m.__dict__.update(saved)
        if "poses" in cap:
            self.reaches.append(self.rec(kind, n, x, f_cmd, open_f, pos, rot, kw, cap, time.time() - t0))
        if n == 1:
            self.entry_ret = self.last()
            if self.mode == "main" and "poses" in cap and m.fell_at is None:  # the entry's label before the close
                f, ok = self.wait(self.reaches[-1]["f_end"], self.reaches[-1]["f_cmd"])
                self.label_prev(f, ok)
        return out

    def rec(self, kind, n, x, f_cmd, open_f, pos, rot, kw, cap, plan_s) -> dict:
        h = self.h
        side = self.side
        p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
        Rh = L.rot_z(L.yaw_of(L._qmat(quat)))
        rw = np.asarray(rot, float)
        tw = np.asarray(pos, float) + rw @ (L.HAND_OFF - h.off)  # the 13 cm point of the commanded wrist pose
        poses, durs = cap["poses"], cap["durs"]
        tup = [(np.asarray(q.pos, float) + L._qmat(q.quat) @ h.c, np.asarray(q.quat, float), np.asarray(q.joints, float)) for q in poses]
        last = tup[-1]
        hk = h.nfk.fk_world(last[0] - L._qmat(last[1]) @ h.c, last[1], last[2])
        first = np.r_[tup[0][0], tup[0][1], tup[0][2]]
        gap = float(np.abs(first - self.frames[f_cmd]).max()) if f_cmd in self.frames else float("nan")
        dyaw, pitch, roll = D3.euler_zyx(Rh.T @ rw)
        pw, Rw = x["wr"][side]
        hp = pw + Rw @ L.HAND_OFF
        hands = {side: dict(target_w=tw, rot_w=rw, dyaw=dyaw, pitch=pitch, roll=roll, core=False,
                            ik_err=float(np.linalg.norm(hk[side][0] - tw)),
                            cmd_target=np.r_[(Rh.T @ np.r_[tw[:2] - p[:2], 0.0])[:2], tw[2]],
                            d_start_cm=100 * float(np.linalg.norm(tw - hp)), pre_back_cm=100 * float(self.a.pre),
                            att_change_deg=SX._ang(Rw, rw), via=[], palm_target=np.asarray(pos, float))}  # fmt: skip
        pz = kw.get("pelvis_z")
        speed = float(kw.get("speed") or self.mot.cfg.reach_speed)
        return dict(mode=side, kind=kind, speed=speed, pair_att=0, hands=hands, pelvis_z=None if pz is None else float(pz),
                    reach_s=float(sum(durs)), plan_s=plan_s, ref_q=last[2].copy(), ref_pel=np.r_[last[0], last[1]],
                    delta=np.zeros(3), RT=np.r_[0.0, np.cumsum(durs)], RQ=np.array([q[2] for q in tup]),
                    RP=np.array([np.r_[q[0], q[1]] for q in tup]), start_gap=gap, slot=n, chain=n, f_cmd=f_cmd,
                    f_end=f_cmd + int(round(float(sum(durs)) * FPS)), pre_wait_s=(f_cmd - open_f) / FPS, d_prev=np.zeros(3),
                    f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip

    # -- the episode's segments ------------------------------------------------------------------------------------------
    def finish(self, reset_id: int) -> tuple[list[dict], dict]:
        m = self.mot
        rows, reaches = self.rows, self.reaches
        if self.mode == "task" and reaches and reaches[-1]["kind"] == "entry" and reaches[-1]["f_label"] is None:
            if self.close_f is not None:  # the entry's label at the close's first command
                f = self.grid_before(reaches[-1]["f_cmd"], self.close_f - 1)
                self.label_prev(f, self.quiet_at(f))
        fell = max(rows) if (m.fell_at is not None and rows) else None
        walked = self.walk0 is not None
        winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
        if walked:
            winfo["asked"] = [float(self.walk_asked[0]), float(self.walk_asked[1]), 0.0]
            if reaches:
                x0, x1 = rows[self.walk0], rows[reaches[0]["f_cmd"]]
                p0, y0 = x0["root"][:2].astype(float), L.yaw_of(L._qmat(x0["root"][3:]))
                dp = L.rot_z(y0).T @ np.r_[x1["root"][:2] - p0, 0.0]
                winfo["real"] = [float(dp[0]), float(dp[1]), D3.wrap(L.yaw_of(L._qmat(x1["root"][3:])) - y0)]
        info = dict(fell=fell, lost=None, why="", steps=len(rows), walked=walked, walk=winfo, planned=len(reaches), delay_f=0)
        segs = []
        if rows and reaches:
            K = max(self.frames) + 1
            F = np.array([self.frames[k] if k in self.frames else self.frames[max(j for j in self.frames if j < k)] for k in range(K)])
            segs = D3.segments(rows, info, F, reaches, reset_id, self.h.nfk, self.rig)
            SX.add_fields(segs, reaches)
            D3.augment_b(segs, reaches)
            for s in segs:
                s["pert_rms_deg"], s["walk_task"] = 0.0, int(walked)
        log = dict(reset=reset_id, label=self.mode, walked=walked, walk=winfo, walk0_f=self.walk0, walk_end_f=self.walk_end_f,
                   fell=fell, close_f=self.close_f, waits=self.waits, n_segs=len(segs),
                   reaches=[{k: rc[k] for k in ("kind", "chain", "f_cmd", "f_end", "f_label", "settled_end", "settle_s", "pre_wait_s",
                                                "reach_s", "start_gap")} | dict(ik_err_cm=100 * rc["hands"][self.side]["ik_err"],
                                                d_start_cm=rc["hands"][self.side]["d_start_cm"]) for rc in reaches],
                   calib_c=np.round(self.h.c, 5).tolist(), off=self.h.off.tolist())  # fmt: skip
        return segs, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--label", required=True, choices=["main", "task"])
    ap.add_argument("--episodes", default="0-29")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None, help="REACH_BFM_DEVICE (the A arm ran ScaleBFM on its GPU; unset: env.py's default)")
    a = ap.parse_args()
    if a.device:
        os.environ["REACH_BFM_DEVICE"] = a.device
    for sub in ("data", "rows", "log"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    import simple_bench as SB

    E, GR, EP = SB.E, SB.GR, SB.EP
    ti = TASKS.index(a.task)
    state: dict = {}
    env_init0 = E.Env.__init__

    def env_init(self, *args, **kw):  # every episode's Env gets a recorder
        env_init0(self, *args, **kw)
        state["rec"] = EpRec(self, a.label, state["sb_args"])

    E.Env.__init__ = env_init
    open_walk0 = GR.open_walk

    def open_walk(m, forward, left, turn):
        r = state.get("rec")
        r.walk_leg(float(forward), float(left))
        assert turn == 0.0, "simple_bench walks without a turn"
        out = open_walk0(m, forward, left, turn)
        r.walk_end_f = r.last()
        return out

    GR.open_walk = open_walk
    build0 = EP.build
    EP.build = lambda ep, out=None: build0(ep, out or a.out / "scene" / f"{ep['task']}_L{ep['level']}_e{ep['index']}")
    episode0 = SB.episode
    segs_all, rows_f, log_f = [], a.out / "rows" / f"{a.task}.jsonl", a.out / "log" / f"{a.task}.jsonl"

    def episode(ep, method, sb_a, models, rep=None):
        state["sb_args"] = sb_a
        assert not sb_a.clear and not sb_a.avoid, "the A arm: no waypoint, no planned path"
        row = episode0(ep, method, sb_a, models, rep)
        rec = state.pop("rec")
        rid = ID0[a.label] + 1000 * ti + 100 * int(ep["level"]) + int(ep["index"])
        segs, log = rec.finish(rid)
        log.update(task=a.task, level=ep["level"], index=ep["index"], success_45=row.get("success_45"),
                   success_official=row.get("success_official"), why=row.get("why"), palm_cm=row.get("palm_cm"),
                   stages=[{k: s.get(k) for k in ("stage", "err_cm", "err_deg", "ik_cm")} for s in row.get("stages", [])])  # fmt: skip
        segs_all.extend(segs)
        with open(log_f, "a") as f:
            f.write(json.dumps(log, default=D3._json) + "\n")
        SC.save_chunk(segs_all, a.out / "data", a.task, rec.rig, SCALARS)
        print(f"sd_real {a.task} {a.label} L{ep['level']} e{ep['index']}: segments {len(segs)}, reaches "
              f"{[(r['kind'], r['f_cmd'], r['f_label'], r['settled_end']) for r in log['reaches']]}, waits "
              f"{[round(w['waited_s'], 2) for w in log['waits']]}, success_45 {row.get('success_45')}", flush=True)  # fmt: skip
        return row

    SB.episode = episode
    sys.argv = [str(Path(SB.__file__)), a.task, "--method", "direct", "--bench-set", "controller=scalebfm", "--bench-set", "sensor=rgbd",
                "--trace", "--episodes", a.episodes, "--out", str(rows_f)]  # fmt: skip
    SB.main()
    print(f"sd_real {a.task} {a.label} done: {len(segs_all)} segments", flush=True)


if __name__ == "__main__":
    main()
