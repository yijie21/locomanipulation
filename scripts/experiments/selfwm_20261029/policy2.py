"""Step 2: one relative-reach policy for any grasp frame, trained only inside the generalised self world model (wm2.py).

Grasp frame = the wrist rotation R_w of the grasp: x the approach axis (the wrist x), y the closing axis (the fingers
sit on the wrist's +-y), z the third axis (the side grasp's housing hangs 3.1 cm along -z). Everything the policy reads
and outputs is in this frame, so one network serves level side grasps, tilt45 and straight-down pinches, either hand.

Imagined episode: a start right after a recorded IK pre-grasp (gen_data2.py, any hand / attitude / posture / post-walk);
grasp point 10-16 cm ahead along the approach axis, +-3 cm off it; tolerances drawn per episode (closing axis 0.5-2.0
cm, third axis 1-3 cm, depth 0.8-1.5 cm); a support plane 0-6 cm under the hand's lowest point at the grasp; half the episodes a neighbour box beside it.
Synthetic head-camera readings (perception.py) with extrinsic error 0-3 deg per axis and depth scale 0-8 %.
Action: target velocity u (grasp frame, +-0.15 m/s per axis), integrated into the jaw target like the data's commands.

    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/policy2.py train --obs rel
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/policy2.py train --obs kin
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import env as E
import numpy as np
import perception as P
import policy as POLM
import torch
import torch.nn as nn
import wm as WMM
import wm2 as W2

POL2 = E.OUT / "policy2"
H = 28
VMAX = 0.15
HOUSING = torch.tensor([0.0, 0.0, -0.031])  # wrist frame, the side grasp's housing under the jaw centre
FINGERTIP = torch.tensor([0.033, 0.0, 0.0])  # wrist frame, fingertips ahead of the jaw centre
CLEAR = 0.004  # m above the support plane
EXT_DEG, SCALE = 3.0, 0.08


CLIP_X = 10.0  # a sample's state gradient may be at most this many times the batch median at that step
REGION = torch.tensor([0.18, 0.12, 0.12])  # target clamp around the start jaw, world axes (the data's command box)


def clip_per_sample(x):
    """Leave forward values alone; in the backward pass shrink any sample whose gradient norm is more than CLIP_X times
    the batch median (a few world-model states have locally steep Jacobians: one sample in 256 carried a 57 vs 0.08
    median sensitivity and drowned the batch, outputs/selfwm_20261029/policy2/try1)."""
    if not x.requires_grad:
        return x

    def hook(g):
        n = g.norm(dim=-1, keepdim=True)
        cap = CLIP_X * n.median().clamp(min=1e-12)
        return g * torch.clamp(cap / n.clamp(min=1e-12), max=1.0)

    x.register_hook(hook)
    return x


def obs_dim(kind: str) -> int:
    return 58 + 3 + 1 + 3 + 3 + 3 + 1 + 3 + 1 + 3 + (3 + 1 if kind in ("rel", "res") else 0)


RES_MAX = 0.06  # m/s, the residual's reach around the servo (kind "res")


def servo(o):
    """Method B as a function of the observation: u = 2/s x (believed grasp - seen jaw) in the grasp frame, no advance
    along the approach while the closing or third axis is off by more than its tolerance."""
    rel = o[:, -4:-1] / 10.0
    tol = o[:, 68:71] / 100.0
    ok = ((rel[:, 1].abs() < tol[:, 1]) & (rel[:, 2].abs() < tol[:, 2])).float()
    return (2.0 * rel * torch.stack([ok, torch.ones_like(ok), torch.ones_like(ok)], -1)).clamp(-VMAX, VMAX)


class Pi2(nn.Module):
    """kind "rel" / "kin": u = VMAX tanh(net). kind "res": u = servo + RES_MAX tanh(net), the net's last layer starting
    at zero, so training starts from method B and the world model only has to improve on it (the free "rel" policy beat
    the servo on the median sample but ran away on about one sample in ten: p90 final error 24.6 cm)."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.net = WMM.mlp(obs_dim(kind), 3, width=512, depth=3)
        if kind == "res":
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, o):
        if self.kind == "res":
            return (servo(o) + RES_MAX * torch.tanh(self.net(o))).clamp(-VMAX, VMAX)
        return VMAX * torch.tanh(self.net(o))


def gframe(R_w, v):
    """World (or pelvis) vector -> grasp frame."""
    return torch.einsum("bji,bj->bi", R_w, v)


def observe(
    kind, s, ns, R_w, g_read, jaw_kin, j_read, j_vis, target, u_prev, tol, plane_dz, box_rel, box_on, att, side
):
    """All positions are pelvis-frame points; everything is re-expressed in the grasp frame (R_w, pelvis frame)."""
    parts = [ns.fwd(s)[:, :58],
             10 * gframe(R_w, g_read - jaw_kin), 10 * plane_dz[:, None],
             10 * gframe(R_w, target - jaw_kin), u_prev / VMAX, 100 * tol, box_on[:, None],
             10 * box_rel, side[:, None], torch.nn.functional.one_hot(att, 3).float()]  # fmt: skip
    if kind in ("rel", "res"):
        parts += [10 * gframe(R_w, g_read - j_read), j_vis[:, None]]
    return torch.cat(parts, -1)


def rot_from6(c):
    a, b = c[:, :3], c[:, 3:6]
    a = a / a.norm(dim=-1, keepdim=True)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def imagine(model, pi, st, B, dev, gen, noise=True):
    sp, ap, s, w, side, att = (x for x in st)
    i = torch.randint(0, len(s), (B,), device=dev, generator=gen)
    sp, ap, s, w, side, att = sp[i], ap[i], s[i], w[i], side[i], att[i]
    U = lambda *sh: torch.rand(*sh, device=dev, generator=gen)  # noqa: E731
    p, yaw = w[:, :3].clone(), w[:, 3].clone()
    Rp = POLM.rz(yaw) @ POLM.tilt(s[:, 58:61])
    Rw_pel0 = rot_from6(ap[:, 7:13])  # the episode's wrist rotation, pelvis frame at the start
    Rw_w = Rp @ Rw_pel0  # world; fixed for the episode
    jaw0, _ = model.hands(s[:, :29], side)
    jaw0_w = p + torch.einsum("bij,bj->bi", Rp, jaw0)
    n, c, t = Rw_w[:, :, 0], Rw_w[:, :, 1], Rw_w[:, :, 2]
    d = 0.10 + 0.06 * U(B)
    grasp_w = jaw0_w + n * d[:, None] + c * (0.06 * U(B) - 0.03)[:, None] + t * (0.06 * U(B) - 0.03)[:, None]
    tol = torch.stack([0.008 + 0.007 * U(B), 0.005 + 0.015 * U(B), 0.01 + 0.02 * U(B)], -1)  # depth, closing, third
    # support plane: under the lowest grasp-time point of the hand (housing or fingertips) by the clearance
    pts_g = torch.stack(
        [torch.einsum("bij,j->bi", Rw_w, HOUSING.to(dev)), torch.einsum("bij,j->bi", Rw_w, FINGERTIP.to(dev))], 1
    )
    # side grasps sit anywhere from just over the surface to half a tall object up; pinches just over it
    plane_z = grasp_w[:, 2] + pts_g[..., 2].min(1).values - CLEAR - 0.06 * U(B) ** 2
    box_on = (U(B) < 0.5).float()
    side_off = torch.where(U(B) < 0.5, 1.0, -1.0) * (0.07 + 0.05 * U(B))
    box_w = grasp_w + c * side_off[:, None] + n * (0.04 * U(B) - 0.02)[:, None]
    ang = (U(B, 3) * 2 - 1) * math.radians(EXT_DEG) * float(noise)
    dR, scale = POLM.small_rot(ang), (U(B, 1) * 2 - 1) * SCALE * float(noise)
    target_w = p + torch.einsum("bij,bj->bi", Rp, ap[:, 1:4])
    u_prev = torch.zeros(B, 3, device=dev)
    cost, last = 0.0, None
    terms = {k: 0.0 for k in ("track", "keep_out", "support", "obstacle", "smooth")}
    per_sample = torch.zeros(B, device=dev)
    out = model.fk(s[:, :29])
    for k in range(H):
        Rp = POLM.rz(yaw) @ POLM.tilt(s[:, 58:61])
        jaw_pel = torch.where(side[:, None] > 0, out["jaw_right"][0], out["jaw_left"][0])
        cam_pel, Rc = out["head_camera"]
        Rw = torch.einsum("bji,bjk->bik", Rp, Rw_w)  # wrist rotation in the pelvis frame now
        to_pel = lambda x: torch.einsum("bji,bj->bi", Rp, x - p)  # noqa: E731
        nz = (
            (lambda: torch.randn(B, 3, device=dev, generator=gen)) if noise else (lambda: torch.zeros(B, 3, device=dev))
        )
        g_read, _ = P.read_torch(to_pel(grasp_w), cam_pel, Rc, scale, dR, nz())
        j_read, j_vis = P.read_torch(jaw_pel, cam_pel, Rc, scale, dR, nz())
        b_read, _ = P.read_torch(to_pel(box_w), cam_pel, Rc, scale, dR, nz())
        jaw_w_now = p + torch.einsum("bij,bj->bi", Rp, jaw_pel)
        plane_dz = jaw_w_now[:, 2] - plane_z  # the robot knows its own height above a seen surface
        o = observe(pi.kind, s, model.ns, Rw, g_read, jaw_pel, j_read, j_vis, to_pel(target_w), u_prev, tol, plane_dz,
                    gframe(Rw, b_read - g_read) * box_on[:, None], box_on, att, side)  # fmt: skip
        u = pi(o)
        reg = REGION.to(dev)
        target_w = torch.maximum(
            torch.minimum(target_w + torch.einsum("bij,bj->bi", Rw_w, u) * E.STEP_S, jaw0_w + reg), jaw0_w - reg
        )
        a = torch.cat([side[:, None], to_pel(target_w), torch.einsum("bji,bj->bi", Rp, torch.einsum("bij,bj->bi", Rw_w, u)),
                       Rw[:, :, 0], Rw[:, :, 1]], -1)  # fmt: skip
        s_next, odo = model.step(sp, ap, s, a)
        s_next = clip_per_sample(s_next)
        p = p + torch.einsum("bij,bj->bi", POLM.rz(yaw), odo[:, :3])
        yaw = yaw + odo[:, 3]
        sp, ap, s = s, a, s_next
        Rp1 = POLM.rz(yaw) @ POLM.tilt(s[:, 58:61])
        out = model.fk(s[:, :29])  # also the next step's kinematics
        jaw1 = torch.where(side[:, None] > 0, out["jaw_right"][0], out["jaw_left"][0])
        jaw_w = p + torch.einsum("bij,bj->bi", Rp1, jaw1)
        e = gframe(Rw_w, jaw_w - grasp_w)  # x depth along the approach, y closing, z third
        frac = (k + 1) / H
        track = (e**2).sum(-1) * (0.2 + 2.0 * frac * frac)
        near = torch.relu(e[:, 0] + 0.10)
        lateral = torch.relu(e[:, 1].abs() - tol[:, 1]) + torch.relu(e[:, 2].abs() - tol[:, 2])
        keep_out = 50.0 * near * lateral + 20.0 * torch.relu(e[:, 0] - tol[:, 0])
        hand_pts = jaw_w[:, None] + pts_g  # housing, fingertips (world)
        support = 20.0 * torch.relu(plane_z[:, None] + CLEAR - hand_pts[..., 2]).sum(-1)
        dist_box = (torch.cat([hand_pts, jaw_w[:, None]], 1) - box_w[:, None]).norm(dim=-1).min(1).values
        obstacle = 20.0 * box_on * torch.relu(0.06 - dist_box)
        smooth = 0.02 * (((u - u_prev) / VMAX) ** 2).sum(-1)
        step_cost = track + keep_out + support + obstacle + smooth
        cost = cost + step_cost.mean()
        per_sample = per_sample + step_cost.detach()
        for name, v in (
            ("track", track),
            ("keep_out", keep_out),
            ("support", support),
            ("obstacle", obstacle),
            ("smooth", smooth),
        ):
            terms[name] += float(v.detach().mean()) / H
        u_prev = u
        last = e
    fin = last.norm(dim=-1).detach()
    ok = (
        ((last[:, 0].abs() < tol[:, 0]) & (last[:, 1].abs() < tol[:, 1]) & (last[:, 2].abs() < tol[:, 2]))
        .float()
        .mean()
    )
    stats = dict(final_cm=float(100 * fin.median()), final_p90_cm=float(100 * torch.quantile(fin, 0.9)), inside_tol=float(ok),
                 **{f"term_{k}": round(v, 5) for k, v in terms.items()},
                 worst_sample_cost=float(per_sample.max() / H), median_sample_cost=float(per_sample.median() / H))  # fmt: skip
    return cost / H, stats


def starts(D, dev, which=(0,)):
    """Each episode of `which` split right after its IK reach: (s_prev, a_prev, s, world row, side, attitude)."""
    eps = np.where(np.isin(D["split"], which))[0]
    eps = eps[D["na"][eps] >= 1]
    so, ao = D["so"][eps], D["ao"][eps]
    t = lambda x: torch.tensor(np.asarray(x), device=dev)  # noqa: E731
    z = D["z"]
    return (t(D["S"][so]), t(D["A"][ao]), t(D["S"][so]), t(z["W"][so]), t(z["side"][eps].astype(np.float32)),
            t(z["att"][eps].astype(np.int64)))  # fmt: skip


def train(kind: str, iters: int, B: int, dev: str, data=W2.DATA, wm=W2.WM2, out=POL2, lr: float = 3e-4) -> None:
    torch.manual_seed(1)
    gen = torch.Generator(device=dev).manual_seed(1)
    model, D = W2.load(dev, data, wm)
    for prm in model.parameters():
        prm.requires_grad_(False)
    pi = Pi2(kind).to(dev)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    st = starts(D, dev)
    st_val = starts(D, dev, (1, 2))
    log, t0 = [], time.time()
    for it in range(iters):
        c, stats = imagine(model, pi, st, B, dev, gen)
        opt.zero_grad()
        c.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 250 == 0 or it == iters - 1:
            with torch.no_grad():
                _, vstats = imagine(model, pi, st_val, 1024, dev, gen)
            row = dict(
                it=it,
                cost=float(c.detach()),
                **stats,
                **{f"val_{k}": v for k, v in vstats.items()},
                s=round(time.time() - t0, 1),
            )
            log.append(row)
            print(json.dumps(row), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(pi.state_dict(), out / f"pi2_{kind}.pt")
    json.dump(dict(kind=kind, iters=iters, batch=B, H=H, log=log), open(out / f"train_{kind}.json", "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train"])
    ap.add_argument("--obs", choices=["rel", "kin", "res"], default="rel")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--data", default=str(W2.DATA))
    ap.add_argument("--wm", default=str(W2.WM2))
    ap.add_argument("--out", default=str(POL2))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pathlib import Path

    train(a.obs, a.iters, a.batch, dev, Path(a.data), Path(a.wm), Path(a.out), a.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
