"""Relative grasp-entry policy trained by backpropagation through the self world model (wm.py), never in the simulator.

Imagined episode: start from a recorded pre-grasp state; the box is static in the stance frame at its probe place
+-3 cm; every 0.1 s the policy reads the box and its own jaw through the synthetic camera (perception.py) with a
sampled extrinsic error (0-3 deg) and depth scale (0-8 %), and outputs a jaw-target VELOCITY u: the target integrates
u and is sent with lead u, inside the region gen_data.py's commands covered -- the same kind of command the world
model was fit on (a first try that output saturated target offsets alternated sign every step and exploited the model).
Cost: the TRUE jaw-to-grasp error (rising weight), a keep-out around the box (no advance while laterally off, no
overshoot past the grasp point), the table (the housing hangs 3.1 cm under the jaw centre), changes of u.

  obs "rel"  proprio + kinematic jaw + believed grasp point + seen jaw + their difference    (method C)
  obs "kin"  proprio + kinematic jaw + believed grasp point + (grasp - kinematic jaw)       (method D, no seen hand)

    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/policy.py train --obs rel
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/policy.py train --obs kin
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
import torch
import torch.nn as nn
import wm as WMM

POL = E.OUT / "policy"
H = 25  # imagined steps (2.5 s)
VMAX = 0.15  # m/s, target velocity
TABLE_Z = 0.795  # m, lowest jaw centre over the 0.75 m table top (housing 3.1 cm under it + margin)
GRASP_OFF = np.array([0.01, 0.0, -0.001])  # probe: grasp point = box centre + 1 cm along x, z at the jaw height
BOX0 = np.array([0.38, -0.15, 0.826])  # box centre at the probe place (stance frame)
JITTER_M = 0.03
EXT_DEG, SCALE = 3.0, 0.08


def obs_dim(kind: str) -> int:
    return 58 + 3 + 3 + 3 + 3 + (3 + 3 + 1 if kind == "rel" else 3)


class Pi(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.net = WMM.mlp(obs_dim(kind), 3, width=256, depth=3)

    def forward(self, o):
        return VMAX * torch.tanh(self.net(o))


def clamp_region(x):
    lo = torch.tensor(E.ROAM_LO, device=x.device, dtype=x.dtype)
    hi = torch.tensor(E.ROAM_HI, device=x.device, dtype=x.dtype)
    return torch.maximum(torch.minimum(x, hi), lo)


def observe(kind: str, s, jaw_kin, g_read, j_read, j_vis, ns: WMM.Norm, target_pel, u_prev):
    q = ns.fwd(s)[:, :58]
    parts = [q, 10 * jaw_kin, 10 * g_read, 10 * (target_pel - jaw_kin), u_prev / VMAX]
    if kind == "rel":
        parts += [10 * j_read, 10 * (g_read - j_read), j_vis[:, None]]
    else:
        parts += [10 * (g_read - jaw_kin)]
    return torch.cat(parts, -1)


def rot6(k):
    """Camera rotation from the two stored columns (Gram-Schmidt)."""
    a, b = k[:, 6:9], k[:, 9:12]
    a = a / a.norm(dim=-1, keepdim=True)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def rz(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def tilt(g):
    """Batched yaw-free rotation with R^T (0,0,-1) = g (wm.tilt_rot)."""
    a = g / g.norm(dim=-1, keepdim=True)
    b = torch.tensor([0.0, 0.0, -1.0], device=g.device).expand_as(a)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    zero = torch.zeros_like(c)
    vx = torch.stack([torch.stack([zero, -v[:, 2], v[:, 1]], -1), torch.stack([v[:, 2], zero, -v[:, 0]], -1),
                      torch.stack([-v[:, 1], v[:, 0], zero], -1)], -2)  # fmt: skip
    eye = torch.eye(3, device=g.device).expand_as(vx)
    return eye + vx + vx @ vx / (1 + c)[:, None, None]


def small_rot(angles):
    """Batched rot_xyz from (B, 3) small angles."""
    cx, sx = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
    cy, sy = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
    cz, sz = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([torch.stack([o, z, z], -1), torch.stack([z, cx, -sx], -1), torch.stack([z, sx, cx], -1)], -2)
    Ry = torch.stack([torch.stack([cy, z, sy], -1), torch.stack([z, o, z], -1), torch.stack([-sy, z, cy], -1)], -2)
    Rz = torch.stack([torch.stack([cz, -sz, z], -1), torch.stack([sz, cz, z], -1), torch.stack([z, z, o], -1)], -2)
    return Rz @ Ry @ Rx


def sample_errors(B, dev, gen):
    ang = (torch.rand(B, 3, device=dev, generator=gen) * 2 - 1) * math.radians(EXT_DEG)
    scale = (torch.rand(B, 1, device=dev, generator=gen) * 2 - 1) * SCALE
    return small_rot(ang), scale


def imagine(model, pi, start, B, dev, gen, train=True):
    """One batch of imagined episodes; returns (cost, stats)."""
    sp, ap, s0, w0 = start
    i = torch.randint(0, len(s0), (B,), device=dev, generator=gen)
    sp, ap, s, w = sp[i], ap[i], s0[i], w0[i]
    p, yaw = w[:, :3].clone(), w[:, 3].clone()
    box = torch.tensor(BOX0, device=dev, dtype=torch.float32).expand(B, 3).clone()
    box[:, :2] += (torch.rand(B, 2, device=dev, generator=gen) * 2 - 1) * JITTER_M
    grasp_w = box + torch.tensor(GRASP_OFF, device=dev, dtype=torch.float32)
    dR, scale = sample_errors(B, dev, gen)
    off = torch.tensor(GRASP_OFF, device=dev, dtype=torch.float32)
    Rp = rz(yaw) @ tilt(s[:, 58:61])
    target_w = p + torch.einsum("bij,bj->bi", Rp, ap[:, :3])  # the command in force at the hand-over
    u_prev = torch.zeros(B, 3, device=dev)
    cost, errs = 0.0, []
    for t in range(H):
        k = model.kin(s)
        jaw_pel, cam_pel, Rc = k[:, :3], k[:, 3:6], rot6(k)
        Rp = rz(yaw) @ tilt(s[:, 58:61])
        box_pel = torch.einsum("bji,bj->bi", Rp, box - p)
        target_pel = torch.einsum("bji,bj->bi", Rp, target_w - p)
        nb = torch.randn(B, 3, device=dev, generator=gen)
        nj = torch.randn(B, 3, device=dev, generator=gen)
        b_read, _ = P.read_torch(box_pel, cam_pel, Rc, scale, dR, nb)
        j_read, j_vis = P.read_torch(jaw_pel, cam_pel, Rc, scale, dR, nj)
        g_read = b_read + off
        o = observe(pi.kind, s, jaw_pel, g_read, j_read, j_vis, model.ns, target_pel, u_prev)
        u = pi(o)  # pelvis frame
        u_w = torch.einsum("bij,bj->bi", Rp, u)
        target_w = clamp_region(target_w + u_w * E.STEP_S)
        a = torch.cat([torch.einsum("bji,bj->bi", Rp, target_w - p), u], -1)
        s_next, odo = model.step(sp, ap, s, a)
        p = p + torch.einsum("bij,bj->bi", rz(yaw), odo[:, :3])
        yaw = yaw + odo[:, 3]
        sp, ap, s = s, a, s_next
        k1 = model.kin(s)
        Rp1 = rz(yaw) @ tilt(s[:, 58:61])
        jaw_w = p + torch.einsum("bij,bj->bi", Rp1, k1[:, :3])
        e = jaw_w - grasp_w
        frac = (t + 1) / H
        track = (e**2).sum(-1) * (0.2 + 2.0 * frac * frac)
        near = torch.relu(e[:, 0] + 0.10)
        lateral = torch.relu(e[:, 1].abs() - 0.012) + torch.relu(e[:, 2].abs() - 0.02)
        keep_out = 50.0 * near * lateral + 20.0 * torch.relu(e[:, 0] - 0.01)
        table = 20.0 * torch.relu(TABLE_Z - jaw_w[:, 2])
        smooth = 0.02 * (((u - u_prev) / VMAX) ** 2).sum(-1)
        cost = cost + (track + keep_out + table + smooth).mean()
        u_prev = u
        errs.append(e.norm(dim=-1).detach())
    final = errs[-1]
    stats = dict(final_cm=float(100 * final.median()), final_p90_cm=float(100 * torch.quantile(final, 0.9)))
    return cost / H, stats


def starts(eps, dev):
    """Imagination starts: each training episode right after its pre-grasp; the previous command is that episode's
    first one, which starts where the motor's servo-led command was (as the hand program does in eval.py)."""
    s0 = np.array([e["S"][0] for e in eps])
    a0 = np.array([e["A"][0] for e in eps], dtype=np.float32)
    w0 = np.array([e["W"][0] for e in eps])
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)  # noqa: E731
    return t(s0), t(a0), t(s0), t(w0)


def train(kind: str, iters: int, B: int, dev: str) -> None:
    torch.manual_seed(1)
    gen = torch.Generator(device=dev).manual_seed(1)
    model, tr, te = WMM.load(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    pi = Pi(kind).to(dev)
    opt = torch.optim.Adam(pi.parameters(), lr=3e-4)
    st = starts(tr, dev)
    log, t0 = [], time.time()
    for it in range(iters):
        c, stats = imagine(model, pi, st, B, dev, gen)
        opt.zero_grad()
        c.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            row = dict(it=it, cost=float(c.detach()), **stats, s=round(time.time() - t0, 1))
            log.append(row)
            print(json.dumps(row), flush=True)
    POL.mkdir(parents=True, exist_ok=True)
    torch.save(pi.state_dict(), POL / f"pi_{kind}.pt")
    json.dump(dict(kind=kind, iters=iters, batch=B, H=H, log=log), open(POL / f"train_{kind}.json", "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train"])
    ap.add_argument("--obs", choices=["rel", "kin"], default="rel")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(a.obs, a.iters, a.batch, dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
