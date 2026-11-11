"""Item 1: choose the pelvis height with the self world model, without trying anything in the simulator.

For one grasp (believed grasp point, wrist rotation, tolerances, table height, box size, hand) and each candidate pelvis
height, the chooser
  1. retrieves the K recorded start states (gen_data2.py, standing episodes only) whose IK reach target, pelvis
     height target, hand and wrist attitude are nearest to the candidate's pre-grasp: real post-reach states, IK
     errors included;
  2. places the same grasp relative to each start's intended pre-grasp (13 cm ahead along the approach) and imagines
     the approach in the world model with the method's own controller (B servo or R), nominal perception;
  3. predicts success: closed inside tolerance (two steps in a row, as the executor does), wrist within ATT_DEG of the
     grasp rotation at the close, and no finger point inside the box (+3 mm) before the close;
  4. scores the candidate by the predicted success rate over the K starts; ties go to the higher pelvis.

    $PY scripts/experiments/selfwm_20261029/chooser.py run --procs 8 --gpus 0,1     # choose, then execute B and R
    $PY scripts/experiments/selfwm_20261029/chooser.py report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import env as E
import eval2 as EV2
import numpy as np
import perception as P
import policy as POLM
import policy2 as P2
import posture as PO
import torch
import wm2 as W2

OUT = E.OUT / "posture"
K = 16
STEPS = 35
ATT_DEG = 20.0
FINGER_Y = 0.0475  # open fingers on the wrist's +-y
# the four finger points (wrist frame): tips and roots of the two open fingers. The jaw centre and the point between the
# tips sit inside the box at a correct grasp, so they cannot be part of the knock test.
PTS = np.array([[0.033, FINGER_Y, 0.0], [0.033, -FINGER_Y, 0.0], [0.0, FINGER_Y, 0.0], [0.0, -FINGER_Y, 0.0]])


DATA_P = (
    E.OUT / "data2" / "all_ps.npz"
)  # envelope data + posture-workspace rows + the servo's own behaviour (--probe, --servo)
WM_P = E.OUT / "wm2ps"  # the world model trained on it (wm2p: without the servo episodes); wm2/ stays untouched


class _Norm:
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd

    def fwd(self, x):
        return (x - self.mu) / self.sd


DATA_WIDE = E.OUT / "data2" / "all_wide.npz"  # + the wide posture range (gen_data2.py --wide, with and without --servo)
WM_WIDE = E.OUT / "wm2wide"


class Chooser:
    def __init__(self, dev="cuda", data=DATA_P, wm=WM_P, pels=None):
        self.dev = dev
        self.pels = list(PO.PELS if pels is None else pels)
        self.model, self.D = W2.load(dev, data, wm)
        old = torch.load(W2.WM2 / "model.pt", map_location=dev)  # R was trained on wm2's state normaliser
        self.ns_pol = _Norm(old["ns.mu"], old["ns.sd"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        z, D = self.D["z"], self.D
        W = z["W"]
        so = D["so"]
        keep = (~z["walked"].astype(bool)) & (D["na"] >= 30) & (D["split"] == 0)
        self.eps = np.where(keep)[0]
        w0 = W[so[self.eps]]
        anchor = z["anchor"][self.eps]
        # these episodes reset at the origin facing +x and did not walk, so the IK target in the world IS its place in
        # the stance frame (the recorded pelvis is the post-reach one, which a lowered posture pushes forward)
        self.feat = dict(rel_xy=anchor[:, :2], z=anchor[:, 2], pel=z["pel_z"][self.eps], side=z["side"][self.eps],
                         att=z["att"][self.eps], w0=w0, anchor=anchor)  # fmt: skip
        self.res = None

    def neighbours(self, pre_rel_xy, pre_z, pel, side, att):
        f = self.feat
        m = (f["side"] == side) & (f["att"] == att)
        m &= (f["pel"] < 0) if pel is None else (f["pel"] > 0)
        idx = np.where(m)[0]
        d = np.linalg.norm(f["rel_xy"][idx] - pre_rel_xy, axis=1) + np.abs(f["z"][idx] - pre_z)
        if pel is not None:
            d = d + 2.0 * np.abs(f["pel"][idx] - pel)
        order = np.argsort(d)[:K]
        return idx[order], d[order]

    def imagine(self, ids, task, method):
        """Roll the method out from the K starts; returns predicted success per start and diagnostics."""
        D, z = self.D, self.D["z"]
        so, ao = D["so"][self.eps[ids]], D["ao"][self.eps[ids]]
        # start at each episode's SECOND recorded step with its true history (state and command of step 0), as in
        # training: right after the IK hand-over there is no true history, and starting there was 1 cm off in one step
        return self.imagine_from(D["S"][so], D["A"][ao], D["S"][so + 1], z["W"][so], z["W"][so + 1], self.feat["w0"][ids],
                                 self.feat["anchor"][ids], task, method)  # fmt: skip

    def imagine_from(self, S0, A0, S1, w_hist, w1, w0, anchor, task, method):
        """The same from explicit starts: states S0 -> S1 under command A0, world rows w_hist / w1, the start pelvis w0
        (for the frame the task is carried into) and the IK target each start reached for."""
        dev, model = self.dev, self.model
        B = len(S1)
        t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev)  # noqa: E731
        sp, ap, s = t(S0), t(A0), t(S1)
        p, yaw = t(w1[:, :3]), t(w1[:, 3])
        # task geometry relative to the intended pre-grasp, carried into each start's frame (standing starts face +x)
        dyaw = w0[:, 3] - task["yaw"]
        rot = np.array([E.rot_z(float(a)) for a in dyaw])
        grasp_w = anchor + np.einsum("bij,j->bi", rot, task["grasp"] - task["pre"])
        box_w = anchor + np.einsum("bij,j->bi", rot, task["box"] - task["pre"])
        Rw_w = np.einsum("bij,jk->bik", rot, task["rot"])
        grasp_w, box_w, Rw_w, R_box = t(grasp_w), t(box_w), t(Rw_w), t(rot)  # the box keeps the world's axes
        half = t(task["half"])
        tol = t(np.tile(task["tol"], (B, 1)))
        plane = t(np.full(B, task["plane"]))
        side = t(np.full(B, task["side"]))
        att = torch.full((B,), task["att"], device=dev, dtype=torch.long)
        p_h, yaw_h = t(w_hist[:, :3]), t(w_hist[:, 3])  # the command in force was given in step 0's pelvis frame
        target_w = p_h + torch.einsum("bij,bj->bi", POLM.rz(yaw_h) @ POLM.tilt(sp[:, 58:61]), ap[:, 1:4])
        u_prev = torch.zeros(B, 3, device=dev)
        zero = torch.zeros(B, 3, device=dev)
        eye = torch.eye(3, device=dev).expand(B, 3, 3)
        sc = torch.zeros(B, 1, device=dev)
        pts = t(PTS)
        inside = torch.zeros(B, device=dev)
        closed = torch.full((B,), -1, device=dev, dtype=torch.long)
        hit = torch.zeros(B, dtype=torch.bool, device=dev)
        att_err = torch.zeros(B, device=dev)
        fin_err = torch.zeros(B, 3, device=dev)
        pi = None if method == "B" else self.policy()
        with torch.no_grad():
            out = model.fk(s[:, :29])
            for k in range(STEPS):
                Rp = POLM.rz(yaw) @ POLM.tilt(s[:, 58:61])
                jaw_pel, Rj_pel = out["jaw_right"] if task["side"] > 0 else out["jaw_left"]
                cam_pel, Rc = out["head_camera"]
                Rw = torch.einsum("bji,bjk->bik", Rp, Rw_w)
                to_pel = lambda x: torch.einsum("bji,bj->bi", Rp, x - p)  # noqa: E731
                g_read, _ = P.read_torch(to_pel(grasp_w), cam_pel, Rc, sc, eye, zero)
                j_read, j_vis = P.read_torch(jaw_pel, cam_pel, Rc, sc, eye, zero)
                jaw_w = p + torch.einsum("bij,bj->bi", Rp, jaw_pel)
                Rj_w = Rp @ Rj_pel
                # the hand's points (wrist frame offsets) in the world: any inside the box before the close is a knock
                hp = jaw_w[:, None] + torch.einsum("bij,nj->bni", Rj_w, pts)
                inbox = (torch.einsum("bji,bnj->bni", R_box, hp - box_w[:, None]).abs() <= half + 0.003).all(-1).any(-1)
                hit |= inbox & (closed < 0)
                rel = P2.gframe(Rw, g_read - j_read)
                now_in = (rel.abs() < tol).all(-1) & (closed < 0)
                inside = torch.where(now_in, inside + 1, torch.zeros_like(inside))
                newly = (inside >= 2) & (closed < 0)
                closed = torch.where(newly, torch.full_like(closed, k), closed)
                cosang = ((torch.einsum("bij,bij->b", Rj_w, Rw_w) - 1) / 2).clamp(-1, 1)
                att_err = torch.where(newly, torch.rad2deg(torch.acos(cosang)), att_err)
                fin_err = torch.where(newly[:, None], P2.gframe(Rw_w, jaw_w - grasp_w), fin_err)
                if bool((closed >= 0).all()):
                    break
                o = P2.observe("rel" if method == "B" else "res", s, self.ns_pol, Rw, g_read, jaw_pel, j_read, j_vis,
                               to_pel(target_w), u_prev, tol, jaw_w[:, 2] - plane, zero, torch.zeros(B, device=dev), att, side)  # fmt: skip
                u = P2.servo(o) if method == "B" else pi(o)
                target_w = target_w + torch.einsum("bij,bj->bi", Rw_w, u) * E.STEP_S
                a = torch.cat([side[:, None], to_pel(target_w), torch.einsum("bji,bj->bi", Rp, torch.einsum("bij,bj->bi", Rw_w, u)),
                               Rw[:, :, 0], Rw[:, :, 1]], -1)  # fmt: skip
                s_next, odo = model.step(sp, ap, s, a)
                p = p + torch.einsum("bij,bj->bi", POLM.rz(yaw), odo[:, :3])
                yaw = yaw + odo[:, 3]
                sp, ap, s = s, a, s_next
                out = model.fk(s[:, :29])
        ok = (closed >= 0) & ~hit & (att_err < ATT_DEG)
        return dict(ok=ok.cpu().numpy(), closed=closed.cpu().numpy(), hit=hit.cpu().numpy(), att=att_err.cpu().numpy())

    def policy(self):
        if self.res is None:
            self.res = P2.Pi2("res").to(self.dev)
            self.res.load_state_dict(torch.load(P2.POL2 / "pi2_res.pt", map_location=self.dev))
            self.res.eval()
        return self.res

    def choose(self, task, method):
        rows = []
        for pel in self.pels:
            ids, dist = self.neighbours(task["pre_rel_xy"], task["pre"][2], pel, task["side"], task["att"])
            r = self.imagine(ids, task, method)
            rows.append(dict(pel=pel, p_success=float(r["ok"].mean()), knock=float(r["hit"].mean()),
                             att_deg=float(np.median(r["att"][r["closed"] >= 0])) if (r["closed"] >= 0).any() else None,
                             closed=float((r["closed"] >= 0).mean()), nn_cm=round(100 * float(dist.mean()), 1)))  # fmt: skip
        best = max(rows, key=lambda r: (round(r["p_success"], 3), -self.pels.index(r["pel"])))
        return best["pel"], rows


def task_of(r: EV2.Runner2, c, box0, g_bel_w, rot, tol, plane_z, box_dims):
    """The chooser's view of a grasp: believed grasp, pre-grasp, box (believed centre, half sizes), rotation, tolerances."""
    pre = g_bel_w - rot[:, 0] * 0.13 if c["kind"] == "side" else g_bel_w + np.array([0.0, 0.0, 0.10])
    yaw = E.yaw_of(r.env.pelvis()[1])  # the probe resets at the origin facing +x, like the chooser's data starts
    return dict(grasp=g_bel_w, pre=pre, pre_rel_xy=pre[:2], box=box0, half=np.array(box_dims) / 2, rot=rot, tol=tol,
                plane=plane_z, side=1.0, att=0 if c["kind"] == "side" else 2, yaw=yaw)  # fmt: skip


def cells():
    out = []
    for cell in PO.CELLS:
        for m in "BR":
            for cond in ("none", "yaw3"):
                for place in range(len(EV2.PLACES)):
                    base = next(x for x in EV2.cells() if x["id"] == f"{cell}_{m}_{cond}_{place}")
                    out.append(dict(base, id=f"{base['id']}_chosen"))
    return out


def wide_cells():
    return [dict(c, id=c["id"].replace("_px", "_chosen")) for c in PO.wide_cells(methods="B", with_pel=False)]


def worker(ids, wide=False):
    r = EV2.Runner2()
    ch = Chooser(r.dev, DATA_WIDE, WM_WIDE, PO.PELS_WIDE) if wide else Chooser(r.dev)
    by = {c["id"]: c for c in (wide_cells() if wide else cells())}
    d = OUT / ("chosen_wide" if wide else "chosen")
    for i in ids:
        f = d / f"{i}.json"
        if f.exists():
            continue
        c = dict(by[i])
        t0 = time.time()
        # the chooser runs on what the robot believes before it moves: one round of readings, no simulator trial
        env, m = r.env, r.env.m
        b = dict((EV2.SIDE_BOXES if c["kind"] == "side" else EV2.DOWN_BOXES)[c["box"]])
        if "top" in c:
            b["top"] = c["top"]
        cell = dict(
            E.BOX, width=b["width"], depth=b["width"] if c["kind"] == "down" else 0.06, height=b["height"], top=b["top"]
        )
        env.probe._size(cell)
        dx, dy = EV2.PLACES[c["place"]]
        m.reset((0.0, 0.0), 0.0)
        centre = np.array(
            [r.cap.EDGE_M + cell["inboard"] + dx, -r.cap.LATERAL_M + dy, cell["top"] + cell["height"] / 2 + 0.001]
        )
        m.place("probe_object", centre, np.array([1.0, 0.0, 0.0, 0.0]))
        m.hold(1.0)
        box0 = env.sim.d.xpos[env.box].copy()
        cd = EV2.CONDS[c["cond"]]
        err = dict(dR=P.rot_xyz(*[math.radians(a) for a in cd["dR"]]), scale=cd["scale"])
        rng = np.random.default_rng(12345)
        reads = [r.read(r.pel(box0), err, rng)[0] for _ in range(5)]
        p, Rp = env.pelvis()
        box_bel = p + Rp @ np.mean(reads, 0)
        g_true, rot, tol, plane_z, _, _, _ = r.geometry(c, box0)
        g_bel = box_bel + (g_true - box0)
        task = task_of(r, c, box_bel, g_bel, rot, tol, plane_z, (cell["depth"], cell["width"], cell["height"]))
        pel, rows = ch.choose(task, c["method"])
        c["pel"] = pel
        res = r.run(c)  # executes from a fresh reset with the chosen pelvis height
        res.update(chosen_pel=pel, choice=rows, choose_s=round(time.time() - t0 - res.get("wall_s", 0), 1))
        res["wall_s"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res, default=float))
        print(i, "pel", pel, res.get("lifted"), res.get("knocked_cm"), flush=True)


def run(procs, gpus, wide=False):
    (OUT / ("chosen_wide" if wide else "chosen")).mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in (wide_cells() if wide else cells())]
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / (f"chosen_wide_w{w}.log" if wide else f"chosen_w{w}.log"), "w")
        jobs.append(subprocess.Popen([str(E.SIM_PY), __file__, "worker", "--ids", ",".join(ids[w::procs])] + (["--wide"] if wide else []), stdout=log,
                                     stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))  # fmt: skip
    print("exit codes", [j.wait() for j in jobs])


def report():
    ok = lambda r: r["lifted"] and r["knocked_cm"] <= 2.0  # noqa: E731
    sweep = [json.loads(f.read_text()) for f in (OUT / "sweep").glob("*.json")]
    chosen = [json.loads(f.read_text()) for f in (OUT / "chosen").glob("*.json")]
    out = {}
    for cell in PO.CELLS:
        for m in "BR":
            row = {}
            for pel in PO.PELS:
                R = [r for r in sweep if r["id"].startswith(f"{cell}_{m}_") and r.get("pel") == pel]
                row["stand" if pel is None else f"{pel:.2f}"] = f"{sum(map(ok, R))}/{len(R)}"
            best = max(
                PO.PELS,
                key=lambda pel: sum(ok(r) for r in sweep if r["id"].startswith(f"{cell}_{m}_") and r.get("pel") == pel),
            )
            C = [r for r in chosen if r["id"].startswith(f"{cell}_{m}_")]
            row["chosen"] = f"{sum(map(ok, C))}/{len(C)}"
            row["chosen_pels"] = sorted({str(r["chosen_pel"]) for r in C})
            row["best_fixed"] = "stand" if best is None else f"{best:.2f}"
            out[f"{cell}_{m}"] = row
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def report_wide():
    """Per table: every fixed height (sweep_wide), the per-table best in hindsight, and the chooser."""
    ok = lambda r: r["lifted"] and r["knocked_cm"] <= 2.0  # noqa: E731
    sweep = [json.loads(f.read_text()) for f in (OUT / "sweep_wide").glob("*.json")]
    chosen = [json.loads(f.read_text()) for f in (OUT / "chosen_wide").glob("*.json")]
    out, tot = {}, {}
    for ti, top in enumerate(PO.TOPS_WIDE):
        for m in "BR":
            row = {}
            for pel in PO.PELS_WIDE:
                R = [r for r in sweep if r["id"].startswith(f"wide{ti}_{m}_") and r.get("pel") == pel]
                if R:
                    key = "stand" if pel is None else f"{pel:.2f}"
                    row[key] = sum(map(ok, R))
                    tot.setdefault(m, {}).setdefault(key, 0)
                    tot[m][key] += row[key]
            C = [r for r in chosen if r["id"].startswith(f"wide{ti}_{m}_")]
            if row:
                row["best"] = max(v for k, v in row.items())
                tot.setdefault(m, {}).setdefault("best", 0)
                tot[m]["best"] += row["best"]
            if C:
                row["chosen"] = sum(map(ok, C))
                row["n"] = len(C)
                row["picks"] = sorted({str(r["chosen_pel"]) for r in C})
                tot.setdefault(m, {}).setdefault("chosen", 0)
                tot[m]["chosen"] += row["chosen"]
            out[f"top{top:.2f}_{m}"] = row
    out["total"] = tot
    (OUT / "report_wide.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "report", "report_wide"])
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs, a.gpus, a.wide)
    elif a.what == "worker":
        worker([x for x in a.ids.split(",") if x], a.wide)
    elif a.what == "report_wide":
        report_wide()
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
