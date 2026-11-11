"""Reach world-model candidates (2026-10-30), one interface (SPEC.md). All differentiable in the command u except the
kNN baseline's neighbour search.

  predict(batch) -> {pos (B,3), rot6 (B,6)  heading frame, final jaw pose;  optional pos_std (B,3), p_settled, p_fell}

  baselines   B0_command (nothing learned), B1_knn (k=8 neighbours, inverse-distance mean of delivered - target)
  end state   E1_direct, E2_residual (zero-init: equals B0_command), E3_ensemble (5 x E2, bootstrap),
              E4_joints (predicts final q and pelvis pose, exact FK composes the jaw pose)
  step        S1_mlp (wm2.py design, 8-step rollout loss), S2_ensemble (5 x S1), S3_gru (hidden 512, 4 warm-up + 12)

A batch is a dict of tensors from tensors(D, dev) indexed by sub(). Pelvis pose in the heading frame: position, yaw
change and the gravity vector in the pelvis frame; R = Rz(yaw) Ry(pitch) Rx(roll) is exact for env.yaw_of's yaw.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/models.py check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import env as E  # noqa: E402
import fk as FKM  # noqa: E402
import wm as WMM  # noqa: E402

NAMES = [
    "B0_command",
    "B1_knn",
    "E1_direct",
    "E2_residual",
    "E3_ensemble",
    "E4_joints",
    "S1_mlp",
    "S2_ensemble",
    "S3_gru",
]
STEP_NAMES = {"S1_mlp", "S2_ensemble", "S3_gru"}
IDENT6 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ENS = 5

# -- kinematics and rotations -------------------------------------------------------------------------------------
_FK = {}


def get_fk(dev) -> FKM.FK:
    """Exact FK on the walk_mars scene, built as wm2.build does (R.Sim, then FKM.FK(sim.m, sim.qadr))."""
    key = str(dev)
    if key not in _FK:
        import json

        sys.path[:0] = [str(E.REPO / "src")]
        from locomani.backends.scalebfm import runtime as R
        from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

        cfg = ScaleBfmBackendConfig(scene_dir=str(E.REPO / "outputs" / "scene" / "walk_mars"), object_name="")
        sim = R.Sim(json.load(open(R.META)), backend=cfg)
        _FK[key] = FKM.FK(sim.m, sim.qadr, device=dev)
    return _FK[key]


def gs6(a: torch.Tensor) -> torch.Tensor:
    """(..., 6) two columns -> (..., 3, 3) rotation by Gram-Schmidt."""
    a1 = F.normalize(a[..., :3], dim=-1)
    a2 = a[..., 3:] - (a1 * a[..., 3:]).sum(-1, keepdim=True) * a1
    a2 = F.normalize(a2, dim=-1)
    return torch.stack([a1, a2, torch.cross(a1, a2, dim=-1)], -1)


def cols6(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def pelvis_R(yaw: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Pelvis rotation in the heading frame from its yaw change and the gravity vector in the pelvis frame:
    g = (sin p, -cos p sin r, -cos p cos r) for R = Rz(yaw) Ry(p) Rx(r)."""
    g = F.normalize(g, dim=-1)
    p = torch.asin(g[..., 0].clamp(-1 + 1e-6, 1 - 1e-6))
    r = torch.atan2(-g[..., 1], -g[..., 2])
    cy, sy, cp, sp, cr, sr = yaw.cos(), yaw.sin(), p.cos(), p.sin(), r.cos(), r.sin()
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], -1).reshape(*yaw.shape, 3, 3)
    Ryx = torch.stack([cp, sp * sr, sp * cr, z, cr, -sr, -sp, cp * sr, cp * cr], -1).reshape(*yaw.shape, 3, 3)
    return Rz @ Ryx


def jaw_fk(fk: FKM.FK, q: torch.Tensor, side: torch.Tensor):
    """Acting jaw centre (B, 3) and wrist rotation (B, 3, 3), pelvis frame."""
    out = fk(q)
    right = side.reshape(-1, 1) > 0
    pos = torch.where(right, out["jaw_right"][0], out["jaw_left"][0])
    rot = torch.where(right.reshape(-1, 1, 1), out["jaw_right"][1], out["jaw_left"][1])
    return pos, rot


def compose(fk, pel, yaw, g, q, side):
    """Jaw pose in the heading frame from the pelvis pose (position, yaw change, gravity) and joints q."""
    Rp = pelvis_R(yaw, g)
    jp, jR = jaw_fk(fk, q, side)
    return pel + torch.einsum("bij,bj->bi", Rp, jp), cols6(Rp @ jR)


# -- batches ------------------------------------------------------------------------------------------------------
END_KEYS = ("u", "x0", "target", "rot6", "side", "q_0", "q_f", "pel_f", "dyaw_f", "g_f", "pos_f", "rot_f", "fell", "settled",
            "k0", "n", "start_z")  # fmt: skip


def tensors(D: dict, dev, step: bool = False) -> dict:
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)  # noqa: E731
    GD = {k: t(D[k]) for k in ("u", "x0", "target", "rot6", "side", "q_0", "q_f", "pel_f", "dyaw_f", "g_f", "pos_f",
                               "rot_f", "start_z")}  # fmt: skip
    GD["fell"], GD["settled"] = t(D["fell"]), t(D["settled"])
    GD["k0"], GD["n"] = t(D["k0"], torch.long), t(D["n_steps"], torch.long)
    if step:
        GD.update({k: t(v) for k, v in DT.pad(D).items()})
    return GD


def sub(GD: dict, idx: torch.Tensor, step: bool = False) -> dict:
    return {k: v[idx] for k, v in GD.items() if step or k not in ("S", "O", "tc")}


def predict_batched(model, GD, idx: torch.Tensor, bs: int = 1024) -> dict:
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            outs.append(model.predict(sub(GD, idx[i : i + bs], model.step)))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def pos_error_cm(model, GD, idx) -> torch.Tensor:
    p = predict_batched(model, GD, idx)
    return 100 * (p["pos"] - GD["pos_f"][idx]).norm(dim=-1)


# -- baselines ----------------------------------------------------------------------------------------------------
class Model(nn.Module):
    step = False

    def prepare(self, GD, train_idx, seed=0):
        pass

    def sample_loss(self, GD, B):
        raise NotImplementedError

    def groups(self):
        return [list(self.parameters())]


class B0Command(Model):
    def predict(self, b):
        return dict(pos=b["target"], rot6=b["rot6"])


class B1Knn(Model):
    """k=8 nearest training segments in normalized [u, pelvis height at k0, walked, chain]."""

    def __init__(self, n: int = 1, k: int = 8):
        super().__init__()
        self.k = k
        self.register_buffer("mu", torch.zeros(DT.U_DIM + 3))
        self.register_buffer("sd", torch.ones(DT.U_DIM + 3))
        self.register_buffer("Fk", torch.zeros(n, DT.U_DIM + 3))
        self.register_buffer("off", torch.zeros(n, 3))

    @staticmethod
    def feat(b):
        return torch.cat([b["u"], b["x0"][:, [64, 204, 205]]], -1)  # height at k0, walked, chain

    def fit(self, GD, idx):
        ok = idx[GD["fell"][idx] == 0]
        f = self.feat(sub(GD, ok))
        self.mu.copy_(f.mean(0))
        self.sd.copy_(f.std(0).clamp_min(1e-3))
        self.Fk = (f - self.mu) / self.sd
        self.off = GD["pos_f"][ok] - GD["target"][ok]

    def predict(self, b):
        f = (self.feat(b) - self.mu) / self.sd
        d, j = torch.cdist(f, self.Fk).topk(self.k, largest=False)
        w = 1 / (d + 1e-6)
        w = w / w.sum(-1, keepdim=True)
        o = self.off[j]  # (B, k, 3)
        m = (w[..., None] * o).sum(1)
        sd = ((w[..., None] * (o - m[:, None]) ** 2).sum(1)).sqrt()
        return dict(pos=b["target"] + m, rot6=b["rot6"], pos_std=sd)


# -- end-state models ---------------------------------------------------------------------------------------------
def head_loss(o, b):
    ok = 1 - b["fell"]
    bce = lambda lg, y, m: (
        (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * m).sum() / m.sum().clamp_min(1)
    )  # noqa: E731
    return bce(o["fell_logit"], b["fell"], torch.ones_like(ok)) + bce(o["settled_logit"], b["settled"], ok)


def pose_loss(pos, rot6, b):
    """Huber (2 cm) on the position components in cm, 0.5 x scaled squared rot6 error; non-fallen segments only."""
    ok = 1 - b["fell"]
    e = (pos - b["pos_f"]) * 100
    lp = F.smooth_l1_loss(e, torch.zeros_like(e), beta=2.0, reduction="none").mean(-1)
    lr = 0.5 * (((rot6 - b["rot_f"]) / 0.1) ** 2).mean(-1)
    return ((lp + lr) * ok).sum() / ok.sum().clamp_min(1)


def zero_last(net: nn.Sequential):
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)


class EndNet(Model):
    """MLP on [x0, u] (normalized). kind: direct, residual, joints."""

    def __init__(self, kind: str, D: dict, idx, seed: int = 0, width: int = 512, depth: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        ok = idx[D["fell"][idx] == 0]
        self.nx, self.nu = WMM.Norm(D["x0"][idx]), WMM.Norm(D["u"][idx])
        if kind == "direct":
            self.ny = WMM.Norm(np.c_[D["pos_f"][ok], D["rot_f"][ok]])
            nout = 9
        elif kind == "residual":
            self.register_buffer(
                "sd_d", torch.tensor((D["pos_f"][ok] - D["target"][ok]).std(0), dtype=torch.float32).clamp_min(1e-3)
            )
            nout = 9
        else:
            dq = D["q_f"][ok] - D["q_0"][ok]
            self.ny = WMM.Norm(np.c_[dq, D["pel_f"][ok], D["dyaw_f"][ok], D["g_f"][ok]])
            nout = 36
        self.f = WMM.mlp(DT.X0_DIM + DT.U_DIM, nout + 2, width=width, depth=depth)
        if kind == "residual":
            zero_last(self.f)  # untrained = B0_command
        self.nout = nout

    def forward_(self, b):
        y = self.f(torch.cat([self.nx.fwd(b["x0"]), self.nu.fwd(b["u"])], -1))
        z = y[:, : self.nout]
        o = dict(fell_logit=y[:, -2], settled_logit=y[:, -1])
        if self.kind == "direct":
            v = self.ny.inv(z)
            o.update(pos=v[:, :3], rot6=v[:, 3:])
        elif self.kind == "residual":
            Rres = gs6(torch.tensor(IDENT6, device=z.device) + z[:, 3:9])
            o.update(pos=b["target"] + z[:, :3] * self.sd_d, rot6=cols6(gs6(b["rot6"]) @ Rres))
        else:
            v = self.ny.inv(z)
            q = b["q_0"] + v[:, :29]
            fk = get_fk(z.device)
            pos, rot6 = compose(fk, v[:, 29:32], v[:, 32], v[:, 33:36], q, b["side"])
            o.update(pos=pos, rot6=rot6, q=q, v=v)
        return o

    def predict(self, b):
        o = self.forward_(b)
        return dict(
            pos=o["pos"],
            rot6=o["rot6"],
            p_fell=torch.sigmoid(o["fell_logit"]),
            p_settled=torch.sigmoid(o["settled_logit"]),
        )

    def loss(self, b):
        o = self.forward_(b)
        l = pose_loss(o["pos"], o["rot6"], b) + head_loss(o, b)
        if self.kind == "joints":
            ok = 1 - b["fell"]
            t = self.ny.fwd(torch.cat([b["q_f"] - b["q_0"], b["pel_f"], b["dyaw_f"][:, None], b["g_f"]], -1))
            lj = (((self.ny.fwd(o["v"]) - t) ** 2).mean(-1) * ok).sum() / ok.sum().clamp_min(1)
            l = l + lj
        return l

    def prepare(self, GD, train_idx, seed=0):
        self.pool = train_idx

    def sample_loss(self, GD, B):
        j = self.pool[torch.randint(len(self.pool), (B,), device=self.pool.device)]
        return self.loss(sub(GD, j))


class EndEnsemble(Model):
    """5 members on bootstrap resamples; mean and std across members."""

    def __init__(self, members: list[Model]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.step = members[0].step

    def prepare(self, GD, train_idx, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for m in self.members:
            boot = torch.randint(len(train_idx), (len(train_idx),), generator=g).to(train_idx.device)
            m.prepare(GD, train_idx[boot], seed)

    def sample_loss(self, GD, B):
        return sum(m.sample_loss(GD, B) for m in self.members)

    def groups(self):
        return [list(m.parameters()) for m in self.members]

    def predict(self, b):
        os = [m.predict(b) for m in self.members]
        pos = torch.stack([o["pos"] for o in os])
        out = dict(
            pos=pos.mean(0), rot6=torch.stack([o["rot6"] for o in os]).mean(0), pos_std=pos.std(0, unbiased=False)
        )
        for k in ("p_fell", "p_settled"):
            if k in os[0]:
                out[k] = torch.stack([o[k] for o in os]).mean(0)
        return out


# -- step models --------------------------------------------------------------------------------------------------
def step_stats(D, idx):
    """Rows of the training segments: states S, actions [u, tc], state deltas, odometry (transitions only)."""
    S, A, Dl, O = [], [], [], []
    ok = idx[D["fell"][idx] == 0]
    for i in ok:
        a, n = D["so"][i], int(D["n_steps"][i])
        s = D["S"][a : a + n]
        S.append(s)
        tc = D["C"][a + 1 : a + n, 3]
        A.append(np.c_[np.repeat(D["u"][i][None], n - 1, 0), tc])
        Dl.append(s[1:] - s[:-1])
        O.append(D["O"][a : a + n - 1])
    return [np.concatenate(x) for x in (S, A, Dl, O)]


class StepNet(Model):
    """Autoregressive step model. kind "mlp": f([s_{t-1}, s_t, u, tc]); "gru": GRUCell([s_t, u, tc]). Outputs the
    state delta (65) and odometry (4). Action of the step t -> t+1 is (u, time since command at row t+1)."""

    step = True

    def __init__(self, kind: str, D: dict, idx, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        S, A, Dl, O = step_stats(D, idx)
        self.ns, self.na, self.nd, self.no = WMM.Norm(S), WMM.Norm(A), WMM.Norm(Dl), WMM.Norm(O)
        a = A.shape[1]
        if kind == "mlp":
            self.f = WMM.mlp(2 * 65 + a, 69, width=1024, depth=4)
            self.H, self.tf = 8, 1  # window transitions, teacher-forced inputs
        else:
            self.cell = nn.GRUCell(65 + a, 512)
            self.out = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 69))
            self.H, self.tf = 15, 4  # 4 warm-up inputs, loss on the 12 predictions after them
        self.hid = 512

    def core(self, prev, cur, act, h):
        a = self.na.fwd(act)
        if self.kind == "mlp":
            y = self.f(torch.cat([self.ns.fwd(prev), self.ns.fwd(cur), a], -1))
            return cur + self.nd.inv(y[:, :65]), self.no.inv(y[:, 65:]), h
        h = self.cell(torch.cat([self.ns.fwd(cur), a], -1), h)
        y = self.out(h)
        return cur + self.nd.inv(y[:, :65]), self.no.inv(y[:, 65:]), h

    def h0(self, B, dev):
        return torch.zeros(B, self.hid, device=dev) if self.kind == "gru" else None

    def predict(self, b):
        X, tc, u, k0, n = b["S"], b["tc"], b["u"], b["k0"], b["n"]
        B, L, _ = X.shape
        dev = X.device
        cur = prev = X[:, 0]
        h = self.h0(B, dev)
        pel = torch.stack([torch.zeros(B, device=dev), torch.zeros(B, device=dev), b["start_z"]], -1)
        yaw = torch.zeros(B, device=dev)
        fin = X[:, 0].clone()
        for j in range(int(n.max()) - 1):
            cur = torch.where((j <= k0)[:, None], X[:, j], cur)
            ds, o, h = self.core(prev, cur, torch.cat([u, tc[:, j : j + 1]], -1), h)
            nxt = ds
            live = ((j >= k0) & (j <= n - 2)).float()
            c, s = yaw.cos(), yaw.sin()
            step = torch.stack([c * o[:, 0] - s * o[:, 1], s * o[:, 0] + c * o[:, 1], o[:, 2]], -1)
            pel = pel + live[:, None] * step
            yaw = yaw + live * o[:, 3]
            fin = torch.where((j == n - 2)[:, None], nxt, fin)
            prev, cur = cur, nxt
        fk = get_fk(dev)
        pos, rot6 = compose(fk, pel, yaw, fin[:, 58:61], fin[:, :29], b["side"])
        return dict(pos=pos, rot6=rot6)

    def prepare(self, GD, train_idx, seed=0):
        n = GD["n"].cpu().numpy()
        ok = (GD["fell"][train_idx] == 0).cpu().numpy()
        seg, a0 = [], []
        lo = 1 if self.kind == "mlp" else 0
        for i, o in zip(train_idx.cpu().numpy(), ok):
            hi = n[i] - 1 - self.H
            if o and hi >= lo:
                seg.append(np.full(hi - lo + 1, i))
                a0.append(np.arange(lo, hi + 1))
        dev = GD["u"].device
        self.wseg, self.wa = torch.tensor(np.concatenate(seg), device=dev), torch.tensor(np.concatenate(a0), device=dev)

    def window_loss(self, GD, seg, a):
        X, O, tc, u = GD["S"][seg], GD["O"][seg], GD["tc"][seg], GD["u"][seg]
        ar = torch.arange(self.H + 1, device=X.device)
        B = len(seg)
        row = torch.arange(B, device=X.device)[:, None]
        Xw, Ow, Tw = X[row, a[:, None] + ar], O[row, a[:, None] + ar], tc[row, a[:, None] + ar]
        prev = X[torch.arange(B, device=X.device), (a - 1).clamp_min(0)]
        cur, h = Xw[:, 0], self.h0(B, X.device)
        ps, po = [], []
        for i in range(self.H):
            cur_in = Xw[:, i] if i < self.tf else cur
            ds, o, h = self.core(prev, cur_in, torch.cat([u, Tw[:, i : i + 1]], -1), h)
            prev, cur = cur_in, ds
            if i >= self.tf - 1:
                ps.append(ds)
                po.append(o)
        k = self.H - self.tf + 1
        ps, po = torch.stack(ps, 1), torch.stack(po, 1)
        ts, to = Xw[:, self.tf : self.tf + k], Ow[:, self.tf - 1 : self.tf - 1 + k]
        ls = ((self.ns.fwd(ps) - self.ns.fwd(ts)) ** 2).mean()
        lo = ((self.no.fwd(po) - self.no.fwd(to)) ** 2).mean()
        fk = get_fk(X.device)
        side = GD["side"][seg][:, None].expand(B, k).reshape(-1)
        pj = jaw_fk(fk, ps[..., :29].reshape(-1, 29), side)[0]
        tj = jaw_fk(fk, ts[..., :29].reshape(-1, 29), side)[0]
        lk = (((pj - tj) / 0.01) ** 2).mean()  # cm^2
        return ls + lo + 0.1 * lk

    def sample_loss(self, GD, B):
        j = torch.randint(len(self.wseg), (B,), device=self.wseg.device)
        return self.window_loss(GD, self.wseg[j], self.wa[j])


# -- construction -------------------------------------------------------------------------------------------------
def build(name: str, D: dict, idx: np.ndarray, dev, seed: int = 0, n_train: int | None = None) -> Model:
    """Model `name` with normalization from the training segments D[idx]. n_train sizes the kNN buffers when loading."""
    if name == "B0_command":
        m = B0Command()
    elif name == "B1_knn":
        m = B1Knn(n_train or 1)
    elif name == "E1_direct":
        m = EndNet("direct", D, idx, seed)
    elif name == "E2_residual":
        m = EndNet("residual", D, idx, seed)
    elif name == "E3_ensemble":
        m = EndEnsemble([EndNet("residual", D, idx, seed * 100 + i) for i in range(ENS)])
    elif name == "E4_joints":
        m = EndNet("joints", D, idx, seed)
    elif name == "S1_mlp":
        m = StepNet("mlp", D, idx, seed)
    elif name == "S2_ensemble":
        m = EndEnsemble([StepNet("mlp", D, idx, seed * 100 + i) for i in range(ENS)])
    elif name == "S3_gru":
        m = StepNet("gru", D, idx, seed)
    else:
        raise ValueError(name)
    return m.to(dev)


def check(spec=None) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(spec)
    sp = DT.splits(D)
    GD = tensors(D, dev, step=True)
    fk = get_fk(dev)
    # E2 at initialization equals B0_command
    tr = sp["train"]
    e2 = build("E2_residual", D, tr, dev)
    b0 = B0Command()
    idx = torch.tensor(sp["test_iid"][:512] if len(sp["test_iid"]) else tr[:512], device=dev)
    b = sub(GD, idx)
    with torch.no_grad():
        p2, p0 = e2.predict(b), b0.predict(b)
    print(f"E2_residual at init vs B0_command on {len(idx)} segments: max |pos diff| {float((p2['pos'] - p0['pos']).abs().max()):.2e} m, "
          f"max |rot6 diff| {float((p2['rot6'] - p0['rot6']).abs().max()):.2e}")  # fmt: skip
    # exact FK of the true final q plus the true final pelvis pose vs the recorded delivered jaw
    allv = torch.tensor(np.where(D["fell"] == 0)[0], device=dev)
    b = sub(GD, allv)
    with torch.no_grad():
        pos, rot6 = compose(fk, b["pel_f"], b["dyaw_f"], b["g_f"], b["q_f"], b["side"])
    err = 100 * (pos - b["pos_f"]).norm(dim=-1)
    ang = rot_angle_deg(gs6(rot6), gs6(b["rot_f"]))
    print(f"E4 composition (true q_f + true pelvis pose) on {len(allv)} non-fallen segments: jaw position error median "
          f"{float(err.median()):.3f} cm, p90 {float(err.quantile(0.9)):.3f}, p99 {float(err.quantile(0.99)):.3f}, max {float(err.max()):.3f}; "
          f"rotation error median {float(ang.median()):.2f} deg, p99 {float(ang.quantile(0.99)):.2f}")  # fmt: skip
    # S1 composition: pelvis from the TRUE odometry integrated from k0, true final q and gravity
    so = D["so"]
    pel = np.zeros((len(allv), 3), np.float32)
    yaw = np.zeros(len(allv), np.float32)
    ids = allv.cpu().numpy()
    for r, i in enumerate(ids):
        pel[r, 2] = D["start_z"][i]
        for t in range(int(D["k0"][i]), int(D["n_steps"][i]) - 1):
            o = D["O"][so[i] + t]
            c, s = np.cos(yaw[r]), np.sin(yaw[r])
            pel[r] += [c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]]
            yaw[r] += o[3]
    pos2, _ = compose(fk, torch.tensor(pel, device=dev), torch.tensor(yaw, device=dev), b["g_f"], b["q_f"], b["side"])
    err2 = 100 * (pos2 - b["pos_f"]).norm(dim=-1)
    print(f"S1 composition (true q_f + true odometry integrated from k0): jaw position error median {float(err2.median()):.3f} cm, "
          f"p90 {float(err2.quantile(0.9)):.3f}, p99 {float(err2.quantile(0.99)):.3f}, max {float(err2.max()):.3f}")  # fmt: skip
    print(f"within 0.5 cm: E4 {float((err < 0.5).float().mean()):.4f}, S1 {float((err2 < 0.5).float().mean()):.4f}")


def rot_angle_deg(Ra, Rb):
    tr = (Ra * Rb).sum((-1, -2))
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else None)
