"""The minimal proof's test, in the real simulator: grasp the probe box under structured perception errors.

  A  world, open loop: the probe's own pre-grasp -> entry (0.08 m/s) -> grip at the BELIEVED box
  B  relative servo, untrained: integrate (believed grasp - seen jaw) into the jaw command, no advance while off
  C  policy "rel" trained in the self world model (policy.py)
  D  policy "kin": the same without the seen jaw (grasp - kinematic jaw)

Success = lifted 5 cm and held, with the box moved at most KNOCK_CM before the grip (a pushed box that still gets
lifted counts as knocked, as in the scene RSI's failure classes).

Every method starts from the same IK pre-grasp 13 cm short of the believed grasp point. Conditions: CONDS below.
Box: 9 placements in +-2 cm.

    CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/selfwm_20261029/eval.py run --procs 6
    $PY scripts/experiments/selfwm_20261029/eval.py report
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
import zlib

import env as E
import numpy as np
import perception as P

EV = E.OUT / "eval"
METHODS = ("A", "B", "C", "D")
# inside the policies' training range (3 deg, 8 %) but past the grasp's tolerance at the box (lateral about 1.75 cm):
# at 0.55 m, 3 deg about the camera's up axis moves the believed box 2.8 cm sideways, 3 deg of pitch 2.5 cm forward,
# +8 % depth 2.0 cm forward and 3.8 cm down (outputs/selfwm_20261029/eval/conditions.md)
CONDS = {
    "none": dict(dR=(0, 0, 0), scale=0.0),
    "yaw3": dict(dR=(0, 3, 0), scale=0.0),
    "yaw-3": dict(dR=(0, -3, 0), scale=0.0),
    "pitch3": dict(dR=(3, 0, 0), scale=0.0),
    "depth8": dict(dR=(0, 0, 0), scale=0.08),
    "yaw3_pitch3": dict(dR=(3, 3, 0), scale=0.0),
}
PLACES = [(dx, dy) for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)]
GRASP_OFF = np.array([0.01, 0.0, -0.001])
PRE_BACK = 0.13
STEPS = 30  # 3 s of hand program
TOL = np.array([0.012, 0.010, 0.015])
KNOCK_CM = 2.0  # scripts/rsi/failure_class.MOVED_CM: moved more than this before the grip = knocked


def cells():
    return [
        dict(id=f"{m}_{c}_{i}", method=m, cond=c, place=i)
        for m, c, i in itertools.product(METHODS, CONDS, range(len(PLACES)))
    ]


class Runner:
    def __init__(self):
        import policy as POLM
        import torch
        import wm as WMM

        self.torch, self.POLM = torch, POLM
        self.env = E.Env()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, _ = WMM.load(self.dev)  # only its state normaliser is used here
        self.pis = {}

    def pi(self, kind: str):
        if kind not in self.pis:
            pi = self.POLM.Pi(kind).to(self.dev)
            pi.load_state_dict(self.torch.load(self.POLM.POL / f"pi_{kind}.pt", map_location=self.dev))
            pi.eval()
            self.pis[kind] = pi
        return self.pis[kind]

    def read(self, P_pel, err, rng):
        k = self.env.kinematics()
        Rc = np.stack([k[6:9], k[9:12], np.cross(k[6:9], k[9:12])], -1)
        return P.read_np(P_pel, k[3:6], Rc, err, rng)

    def truth_pel(self, P_w):
        p, Rp = self.env.pelvis()
        return Rp.T @ (P_w - p)

    def run(self, c: dict) -> dict:
        env, m = self.env, self.env.m
        rng = np.random.default_rng(zlib.crc32(c["id"].encode()))
        cond = CONDS[c["cond"]]
        err = dict(dR=P.rot_xyz(*[math.radians(a) for a in cond["dR"]]), scale=cond["scale"])
        env.reset(box_xy=PLACES[c["place"]])
        box0 = env.sim.d.xpos[env.box].copy()
        reads = [self.read(self.truth_pel(box0), err, rng)[0] for _ in range(5)]
        g_bel = np.mean(reads, 0) + GRASP_OFF  # believed grasp point, pelvis frame at the reading
        p, Rp = env.pelvis()
        g_world = p + Rp @ g_bel  # stored in the world (odometry) frame, as a world-referenced pipeline does
        self.last_g_world = g_world  # videos.py draws it
        out = env.pregrasp(g_world - Rp @ np.array([PRE_BACK, 0.0, 0.0]))
        res = dict(c, pre_ok=bool(out is not None and out.ok))
        if out is None:
            return dict(res, fell=True, lifted=False)
        if c["method"] == "A":
            m.reach(E.SIDE, g_world, env.wrist_rot(), speed=0.08)
            res["steps"] = None
        else:
            res["steps"] = self.hand_program(c["method"], err, rng)
        jaw = env.jaw_world()
        before = env.sim.d.xpos[env.box].copy()
        grasp_now = before + GRASP_OFF  # the grasp point of the box where it is at the close
        res["close_err_cm"] = round(100 * float(np.linalg.norm(jaw - grasp_now)), 2)
        res["close_err_xyz_cm"] = [round(100 * float(v), 2) for v in jaw - grasp_now]
        res["knocked_cm"] = round(100 * float(np.linalg.norm(before[:2] - box0[:2])), 2)
        m.grip(E.SIDE)
        if E.SIDE in m.hand_override:
            m.adopt_hand(E.SIDE)
        res["gripped"] = E.SIDE in m.gripping(env.box)
        m.reach(E.SIDE, env.jaw_world() + np.array([0.0, 0.0, 0.10]), env.wrist_rot(), speed=0.08)
        m.hold(1.0)
        z = float(env.sim.d.xpos[env.box][2])
        res["lifted"] = bool(E.SIDE in m.gripping(env.box) and z > box0[2] + 0.05)
        res["fell"] = m.fell_at is not None
        return res

    def hand_program(self, method: str, err: dict, rng) -> int:
        """10 Hz hand program from the pre-grasp: every step reads the box (where it is now) and the jaw through the
        camera, moves the jaw target (world, inside the commanded region gen_data covered) and closes once the believed
        jaw-to-grasp error has been inside TOL for two steps."""
        env, torch = self.env, self.torch
        target_w = env.commanded_world()  # start where the motor's command is (no jump)
        u_prev = np.zeros(3)
        inside = 0
        for t in range(STEPS):
            p, Rp = env.pelvis()
            k = env.kinematics()
            jaw_kin = k[:3]
            b_read, _ = self.read(self.truth_pel(env.sim.d.xpos[env.box].copy()), err, rng)
            j_read, j_vis = self.read(jaw_kin, err, rng)
            g_read = b_read + GRASP_OFF
            rel = g_read - (jaw_kin if method == "D" else j_read)
            inside = inside + 1 if np.all(np.abs(rel) < TOL) else 0
            if inside >= 2:
                env.command(p + Rp @ jaw_kin, np.zeros(3))
                env.step()
                return t
            if method == "B":
                lateral_ok = abs(rel[1]) < TOL[1] and abs(rel[2]) < TOL[2]
                u = np.clip(2.0 * rel * np.array([1.0 if lateral_ok else 0.0, 1.0, 1.0]), -0.15, 0.15)
            else:
                kind = "rel" if method == "C" else "kin"
                s = torch.tensor(env.self_state()[None], device=self.dev)
                f = lambda x: torch.tensor(np.asarray(x, np.float32)[None], device=self.dev)  # noqa: E731
                with torch.no_grad():
                    o = self.POLM.observe(kind, s, f(jaw_kin), f(g_read), f(j_read), f([float(j_vis)])[0], self.model.ns,
                                          f(Rp.T @ (target_w - p)), f(u_prev))  # fmt: skip
                    u = self.pi(kind)(o)[0].cpu().numpy()
            target_w = np.clip(target_w + Rp @ u * E.STEP_S, E.ROAM_LO, E.ROAM_HI)
            env.command(target_w, Rp @ u)
            u_prev = u
            if not env.step():
                return t
        return STEPS


def worker(ids: list[str]) -> None:
    r = Runner()
    by = {c["id"]: c for c in cells()}
    for i in ids:
        f = EV / f"{i}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = r.run(by[i])
        res["wall_s"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res))
        print(i, res.get("lifted"), res.get("close_err_cm"), res.get("knocked_cm"), flush=True)


def run(procs: int) -> None:
    EV.mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in cells()]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    jobs = []
    for w in range(procs):
        log = open(EV / f"w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--ids", ",".join(ids[w::procs])]
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))
    print("exit codes", [j.wait() for j in jobs])


def report() -> None:
    rows = [json.loads(f.read_text()) for f in sorted(EV.glob("*.json")) if f.name != "report.json"]
    out = {}
    for cond in CONDS:
        out[cond] = {}
        for m in METHODS:
            R = [r for r in rows if r["cond"] == cond and r["method"] == m]
            if not R:
                continue
            ok = [r["lifted"] and r["knocked_cm"] <= KNOCK_CM for r in R]
            out[cond][m] = dict(
                success=f"{sum(ok)}/{len(R)}",
                lifted=f"{sum(r['lifted'] for r in R)}/{len(R)}",
                knocked=sum(r["knocked_cm"] > KNOCK_CM for r in R if r.get("knocked_cm") is not None),
                close_err_cm=round(float(np.median([r["close_err_cm"] for r in R if "close_err_cm" in r])), 2),
                fell=sum(bool(r.get("fell")) for r in R),
            )
    (EV / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "report"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs)
    elif a.what == "worker":
        worker([x for x in a.ids.split(",") if x])
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
