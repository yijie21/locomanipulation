"""Step 2 test: the grasp-frame policy (policy2.py) on grip-probe variants, in the real simulator.

Cells: side grasps of boxes 4 / 6 / 8 cm wide on tables 0.65 / 0.75 / 0.90 m (closing-axis tolerance 2.0 / 1.45 /
0.5 cm), and straight-down pinches of 5 cm boxes 5 / 8 cm tall on tables 0.60 / 0.75 m (the probe's own top-down
geometry: fingertips 4 mm over the table). Three placements, conditions none / yaw3 / depth8, methods:

  A  world, open loop: the probe's IK reach to the believed grasp (entry 8 cm/s side, 5 cm/s down)
  B  relative servo in the grasp frame: u = 2/s x (believed grasp - seen jaw), no advance while laterally off
  C  policy2 "rel"; D  policy2 "kin" (no seen hand); R  policy2 "res" (B plus a residual trained in the world model)

Every method starts from the same IK pre-grasp 13 cm back along the approach axis (10 cm above for the pinch).
Success = lifted 5 cm with the box moved at most 2 cm before the grip.

    $PY scripts/experiments/selfwm_20261029/eval2.py run --procs 8 --gpus 0,1
    $PY scripts/experiments/selfwm_20261029/eval2.py report
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
import eval as EV
import gen_data2 as G2
import numpy as np
import perception as P

EV2 = E.OUT / "eval2"
METHODS = ("A", "B", "C", "D", "R")
CONDS = {k: EV.CONDS[k] for k in ("none", "yaw3", "depth8")}
PLACES = [(0.0, 0.0), (0.015, -0.015), (-0.015, 0.015)]
SIDE_BOXES = [dict(width=w, top=t, height=0.15) for w in (0.04, 0.06, 0.08) for t in (0.65, 0.75, 0.90)]
DOWN_BOXES = [dict(width=0.05, top=t, height=h) for t in (0.60, 0.75) for h in (0.05, 0.08)]
JAW = 0.095
STEPS = 35


def cells():
    out = []
    for kind, boxes in (("side", SIDE_BOXES), ("down", DOWN_BOXES)):
        for bi, m, c, pi in itertools.product(range(len(boxes)), METHODS, CONDS, range(len(PLACES))):
            out.append(dict(id=f"{kind}{bi}_{m}_{c}_{pi}", kind=kind, box=bi, method=m, cond=c, place=pi))
    return out


class Runner2:
    def __init__(self):
        import policy2 as P2
        import torch
        import wm2 as W2

        self.torch, self.P2 = torch, P2
        self.env = E.Env("grip")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = W2.load(self.dev)
        self.pis = {}
        self.cap = self.env.cap

    def pi(self, kind):
        if kind not in self.pis:
            pi = self.P2.Pi2(kind).to(self.dev)
            pi.load_state_dict(self.torch.load(self.P2.POL2 / f"pi2_{kind}.pt", map_location=self.dev))
            pi.eval()
            self.pis[kind] = pi
        return self.pis[kind]

    def read(self, P_pel, err, rng):
        k = self.env.kinematics()
        Rc = np.stack([k[6:9], k[9:12], np.cross(k[6:9], k[9:12])], -1)
        return P.read_np(P_pel, k[3:6], Rc, err, rng)

    def pel(self, x_w):
        p, Rp = self.env.pelvis()
        return Rp.T @ (np.asarray(x_w) - p)

    def geometry(self, c, box0):
        """Grasp point (world), wrist rotation, tolerances (depth, closing, third), support plane z, pre-grasp, via."""
        cap = self.cap
        yaw = E.yaw_of(self.env.pelvis()[1])
        if c["kind"] == "side":
            b = dict(SIDE_BOXES[c["box"]])
            if "top" in c:
                b["top"] = c["top"]
            rot = G2.wrist_rot(yaw, "level")
            grasp = np.r_[box0[0] + 0.01, box0[1], b["top"] + max(cap.HOUSING_BELOW_M, min(b["height"] / 2, 0.10))]
            pre = grasp - rot[:, 0] * 0.13
            tol = np.array([0.012, min(0.02, max(0.005, (JAW - b["width"]) / 2 - 0.003)), 0.02])
        else:
            b = DOWN_BOXES[c["box"]]
            rot = G2.wrist_rot(yaw, "down")
            grasp = np.r_[box0[:2], b["top"] + cap.FINGERTIP_M + cap.TIP_CLEAR_M]
            pre = grasp + np.array([0.0, 0.0, 0.10])
            tol = np.array([0.008, min(0.02, (JAW - b["width"]) / 2 - 0.003), 0.02])
        x_via = cap.EDGE_M - cap.VIA_OUT_M
        via = [np.array([x_via, pre[1], pre[2]])] if pre[0] > x_via else []
        return grasp, rot, tol, b["top"], pre, via, b

    def run(self, c):
        env, m = self.env, self.env.m
        rng = np.random.default_rng(zlib.crc32(c["id"].encode()))
        cd = CONDS[c["cond"]]
        err = dict(dR=P.rot_xyz(*[math.radians(a) for a in cd["dR"]]), scale=cd["scale"])
        b = dict((SIDE_BOXES if c["kind"] == "side" else DOWN_BOXES)[c["box"]])
        if "top" in c:  # posture.py's wide test: the same box on another table
            b["top"] = c["top"]
        cell = dict(
            E.BOX, width=b["width"], depth=b["width"] if c["kind"] == "down" else 0.06, height=b["height"], top=b["top"]
        )
        env.probe._size(cell)
        dx, dy = PLACES[c["place"]]
        m.reset((0.0, 0.0), 0.0)
        centre = np.array(
            [self.cap.EDGE_M + cell["inboard"] + dx, -self.cap.LATERAL_M + dy, cell["top"] + cell["height"] / 2 + 0.001]
        )
        m.place("probe_object", centre, np.array([1.0, 0.0, 0.0, 0.0]))
        m.hold(1.0)
        box0 = env.sim.d.xpos[env.box].copy()
        g_true, rot, tol, plane_z, _, _, _ = self.geometry(c, box0)
        off = g_true - box0  # grasp point relative to the box centre (from the known object model)
        reads = [self.read(self.pel(box0), err, rng)[0] for _ in range(5)]
        p, Rp = env.pelvis()
        box_bel_w = p + Rp @ np.mean(reads, 0)
        g_bel_w = box_bel_w + off
        self.last = dict(g_bel_w=g_bel_w.copy(), off=off.copy(), rot=rot.copy())  # videos2.py draws them
        # the pre-grasp relative to the BELIEVED grasp: 13 cm back along the approach (side), 10 cm above (pinch)
        pre = g_bel_w - rot[:, 0] * 0.13 if c["kind"] == "side" else g_bel_w + np.array([0.0, 0.0, 0.10])
        x_via = self.cap.EDGE_M - self.cap.VIA_OUT_M
        via = [np.array([x_via, pre[1], pre[2]])] if pre[0] > x_via else []
        pel_z = None if g_true[2] >= 0.60 else max(0.42, round(g_true[2] + 0.05, 2))
        if "pel" in c:  # posture.py: a chosen pelvis height (None = standing)
            pel_z = c["pel"]
        out = m.reach(E.SIDE, pre, rot, via=via, pelvis_z=pel_z)
        res = dict(c, pre_ok=bool(out.ok))
        if m.fell_at is not None:
            return dict(res, fell=True, lifted=False, knocked_cm=0.0)
        if c["method"] == "A":
            m.reach(E.SIDE, g_bel_w, rot, speed=0.08 if c["kind"] == "side" else 0.05, pelvis_z=pel_z)
            res["steps"] = None
        else:
            res["steps"] = self.program(c["method"], err, rng, off, rot, tol, plane_z)
        jaw = env.jaw_world()
        now = env.sim.d.xpos[env.box].copy()
        e = rot.T @ (jaw - (now + off))
        res["close_err_cm"] = round(100 * float(np.linalg.norm(e)), 2)
        res["close_err_grasp_frame_cm"] = [round(100 * float(v), 2) for v in e]
        res["knocked_cm"] = round(100 * float(np.linalg.norm(now[:2] - box0[:2])), 2)
        m.grip(E.SIDE)
        if E.SIDE in m.hand_override:
            m.adopt_hand(E.SIDE)
        res["gripped"] = E.SIDE in m.gripping(env.box)
        m.reach(E.SIDE, env.jaw_world() + np.array([0.0, 0.0, 0.10]), rot, speed=0.08, pelvis_z=pel_z)
        m.hold(1.0)
        res["lifted"] = bool(E.SIDE in m.gripping(env.box) and env.sim.d.xpos[env.box][2] > box0[2] + 0.05)
        res["fell"] = m.fell_at is not None
        return res

    def program(self, method, err, rng, off, rot_w, tol, plane_z) -> int:
        env, torch = self.env, self.torch
        target_w = env.commanded_world()
        env.command_side(E.SIDE, target_w, np.zeros(3), rot_w)
        u_prev = np.zeros(3)
        inside = 0
        f = lambda x: torch.tensor(np.asarray(x, np.float32)[None], device=self.dev)  # noqa: E731
        for t in range(STEPS):
            p, Rp = env.pelvis()
            k = env.kinematics()
            jaw_kin = k[:3]
            b_read, _ = self.read(self.pel(env.sim.d.xpos[env.box].copy()), err, rng)
            g_read = b_read + Rp.T @ off
            j_read, j_vis = self.read(jaw_kin, err, rng)
            Rw = Rp.T @ rot_w
            rel = Rw.T @ (g_read - (jaw_kin if method == "D" else j_read))
            inside = inside + 1 if np.all(np.abs(rel) < tol) else 0
            if inside >= 2:
                env.command_side(E.SIDE, p + Rp @ jaw_kin, np.zeros(3), rot_w)
                env.step()
                return t
            if method == "B":
                ok = abs(rel[1]) < tol[1] and abs(rel[2]) < tol[2]
                u = np.clip(2.0 * rel * np.array([1.0 if ok else 0.0, 1.0, 1.0]), -0.15, 0.15)
            else:
                kind = {"C": "rel", "D": "kin", "R": "res"}[method]
                s = torch.tensor(env.self_state()[None], device=self.dev)
                jaw_w = env.jaw_world()
                with torch.no_grad():
                    o = self.P2.observe(kind, s, self.model.ns, f(Rw), f(g_read), f(jaw_kin), f(j_read), f([float(j_vis)])[0],
                                        f(Rp.T @ (target_w - p)), f(u_prev), f(tol), f([jaw_w[2] - plane_z])[0],
                                        f(np.zeros(3)), f([0.0])[0], torch.tensor([0 if rot_w[2, 0] > -0.5 else 2], device=self.dev),
                                        f([1.0])[0])  # fmt: skip
                    u = self.pi(kind)(o)[0].cpu().numpy()
            target_w = target_w + rot_w @ u * E.STEP_S
            env.command_side(E.SIDE, target_w, rot_w @ u, rot_w)
            u_prev = u
            if not env.step():
                return t
        return STEPS


def worker(ids):
    r = Runner2()
    by = {c["id"]: c for c in cells()}
    for i in ids:
        f = EV2 / f"{i}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = r.run(by[i])
        res["wall_s"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res))
        print(i, res.get("lifted"), res.get("close_err_cm"), res.get("knocked_cm"), flush=True)


def run(procs, gpus, methods):
    EV2.mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in cells() if c["method"] in methods]
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(EV2 / f"w{w}_{methods}.log", "w")
        jobs.append(subprocess.Popen([str(E.SIM_PY), __file__, "worker", "--ids", ",".join(ids[w::procs])], stdout=log,
                                     stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))  # fmt: skip
    print("exit codes", [j.wait() for j in jobs])


def report():
    rows = [json.loads(f.read_text()) for f in sorted(EV2.glob("*.json")) if f.name != "report.json"]
    out = {}
    for kind in ("side", "down"):
        for cond in CONDS:
            for m in METHODS:
                R = [r for r in rows if r["kind"] == kind and r["cond"] == cond and r["method"] == m]
                if not R:
                    continue
                ok = [r["lifted"] and r["knocked_cm"] <= EV.KNOCK_CM for r in R]
                out.setdefault(kind, {}).setdefault(cond, {})[m] = dict(
                    success=f"{sum(ok)}/{len(R)}", lifted=sum(r["lifted"] for r in R),
                    knocked=sum(r["knocked_cm"] > EV.KNOCK_CM for r in R),
                    close_err_cm=round(float(np.median([r["close_err_cm"] for r in R if "close_err_cm" in r])), 2),
                    steps_median=None if m == "A" else float(np.median([r["steps"] for r in R if r.get("steps") is not None])))  # fmt: skip
    (EV2 / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "report"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--ids", default="")
    ap.add_argument("--methods", default="".join(METHODS))
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs, a.gpus, a.methods)
    elif a.what == "worker":
        worker([x for x in a.ids.split(",") if x])
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
