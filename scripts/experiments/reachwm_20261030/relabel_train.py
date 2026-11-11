"""Fine-tune a copy of ScaleBFM on the relabelled reaches (2026-11-04, hindsight relabelling step 1; data: relabel_gen.py).

The student starts as the frozen controller (same checkpoint, every weight trainable). One sample = one 50 Hz tick of a
recorded reach: the 3-tick history and root position as recorded, the 6 future keypoint frames of the relabelled
reference (the plan for where the hand really landed), mode 4; the target = the action the frozen controller really
returned there. Loss = mean squared PD-target difference (raw action x action_scale, rad^2). Validation = the reaches of
every 20th reset (by hash), never trained on.

The policy's own forward (runtime.Policy.forward + network.HumanoidTransformerPolicyWrapperWithMode.forward) runs batch 1;
`forward` below is the same computation over a batch. Phase `check` asserts it reproduces the recorded actions from the
recorded (unrelabelled) inputs.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/relabel_train.py check
    CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/relabel_train.py train --steps 20000
Writes outputs/reachwm_20261030/relabel/student_<tag>.pt (the wrapper's parameters) and train_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_gen as RG  # noqa: E402

E = RG.E
OUT = RG.OUT
DATA = Path(os.environ.get("RELABEL_DATA", RG.DATA))
MODE = 4  # Motor.MODE_STAND: every recorded tick (reach and the hold after it)
STRIDE = int(os.environ.get("RELABEL_STRIDE", "2"))  # keep every STRIDE-th tick of a reach
VAL_EVERY = 20


def load_wrapper(device: str):
    """The controller (runtime.Policy's wrapper) with its buffers made ordinary tensors: load_policy builds it under
    inference_mode, and inference tensors cannot enter an autograd graph."""
    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.backends.scalebfm.network import load_policy

    w, meta, _, _ = load_policy(ScaleBfmBackendConfig().checkpoint_path(), R.MODE_TABLE, R.META, R.FK_XML, device)
    for (
        mod
    ) in w.modules():  # .to(device) ran inside inference_mode too, so the parameters are inference tensors as well
        for name, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[name] = b.clone()
        for name, p in list(mod._parameters.items()):
            if p is not None:
                mod._parameters[name] = torch.nn.Parameter(p.clone())
    return w, meta


def forward(w, rq, av, dp, dv, ac, rp, pw, qw, bodies):
    """Raw actions (B, 29) for a batch: Policy.forward's world-to-root transform, then the wrapper's forward with batch B.
    pw / qw: (B, F, len(bodies), 3 / 4) world frames of `bodies` (indices into selected_body_names); the other bodies get
    zeros and identity, which mode 4's mask multiplies away as it does the runtime's FK values."""
    from locomani.backends.scalebfm.network import (
        quat_apply,
        quat_apply_inverse,
        quat_mul,
        quat_mul_inverse_left,
        quat_mul_inverse_right,  # fmt: skip
    )

    B, H = rq.shape[:2]
    F = pw.shape[1]
    nsel = w.selected_link_indices.shape[-1]
    pw_full = torch.zeros(B, F, nsel, 3, device=rq.device)
    qw_full = torch.zeros(B, F, nsel, 4, device=rq.device)
    qw_full[..., 0] = 1
    pw_full[:, :, bodies], qw_full[:, :, bodies] = pw, qw
    root_q = rq[:, -1]
    tp = quat_apply_inverse(root_q[:, None, None].expand(-1, F, nsel, -1).reshape(-1, 4),
                            (pw_full - rp[:, None, None, :]).reshape(-1, 3)).view(B, F, nsel, 3)  # fmt: skip
    tq = quat_mul_inverse_left(root_q[:, None, None].expand(-1, F, nsel, -1).contiguous(), qw_full)
    # the wrapper, batched
    grav = quat_apply_inverse(rq.reshape(-1, 4), w.gravity_vec[:1, :1].expand(B, H, 3).reshape(-1, 3)).view(B, H, 3)
    prop = torch.cat([grav, av, dp - w.default_dof_pos, dv * 0.05], dim=-1)
    pt, at = w.prop_embedder(prop), w.action_embedder(ac)
    x = torch.empty(B, 2 * H, pt.shape[-1], device=rq.device)
    x[:, ::2] = pt
    x[:, 1:-1:2] = at[:, 1:]
    x[:, 2 * H - 1] = w.empty_embedding
    half = dp[:, -1][:, w.lab_to_xml_joint_indices].unsqueeze(-1) / 2
    jrot = torch.cat([torch.cos(half), w.joint_axis.unsqueeze(0).expand(B, -1, -1) * torch.sin(half)], dim=-1)
    pos = [torch.zeros(B, 3, device=rq.device)]
    quat = [torch.tensor([1.0, 0.0, 0.0, 0.0], device=rq.device).expand(B, 4)]
    for j in range(1, len(w.parent_indices)):
        par = int(w.parent_indices[j])
        lt = w.local_translation[j - 1].expand(B, 3).contiguous()
        lr = w.local_rotation[j - 1].expand(B, 4).contiguous()
        pos.append(pos[par] + quat_apply(quat[par], lt))
        quat.append(quat_mul(quat[par], quat_mul(lr, jrot[:, j - 1].contiguous())))
    bp = torch.stack(pos, 1)[:, w.selected_link_indices]
    bq = torch.stack(quat, 1)[:, w.selected_link_indices]
    tan = w.tan_vec.expand(B, -1, -1, -1)
    nrm = w.norm_vec.expand(B, -1, -1, -1)
    rel_pos = tp - bp[:, None]
    rot_tn = torch.cat([quat_apply(tq, tan), quat_apply(tq, nrm)], dim=-1)
    rel_rot = quat_mul_inverse_right(tq, bq[:, None].expand(-1, F, -1, -1).contiguous())
    rel_tn = torch.cat([quat_apply(rel_rot, tan), quat_apply(rel_rot, nrm)], dim=-1)
    toff = torch.arange(F, device=rq.device, dtype=torch.float32)[None, :, None].expand(B, -1, -1)
    task = torch.cat(
        [tp.flatten(2, 3), rel_pos.flatten(2, 3), rot_tn.flatten(2, 3), rel_tn.flatten(2, 3), toff], dim=-1
    )
    mode = torch.full((B,), MODE, dtype=torch.long, device=rq.device)
    task = task * w.mode_mappings[mode].unsqueeze(1)
    task = torch.cat([task, w.mode_vectors[mode].unsqueeze(1).expand(-1, F, -1)], dim=-1)
    tok = w.task_embedder(task)
    for blk in w.transformer_blocks:
        x = blk(x, tok, self_attn_mask=w.self_attn_mask)
    return w.projection_head(w.final_norm(x)[:, -1])


def load(device: str, keep_orig: bool = False, data: Path | None = None, actor: bool = False) -> dict:
    """Every recorded tick (every STRIDE-th per reach) as tensors on `device`, with a train / validation split by reset.
    actor=False: the teacher's ticks (relabel_gen.py's t_*), inputs pw1/qw1 (the relabelled frames), target its action.
    actor=True: the actor's ticks (RELABEL_ACT, a_*), inputs = the relabelled frames it was fed, target filled in by
    label() (pw/qw hold the original command's frames at the actor's states)."""
    files = sorted(p for p in (data or DATA).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    keys = ["rq", "av", "dp", "dv", "ac", "rp", "pw1", "qw1", "act"] + (["pw", "qw"] if keep_orig or actor else [])
    src = {k: (f"a_{dict(pw1='pw', qw1='qw', pw='pwc', qw='qwc').get(k, k)}" if actor else f"t_{k}") for k in keys}
    parts, val, nseg = {k: [] for k in keys}, [], 0
    for f in files:
        d = np.load(f)
        n = d["a_n_ticks" if actor else "n_ticks"]
        first = np.r_[0, np.cumsum(n)[:-1]]
        idx = np.concatenate([a + np.arange(0, c, STRIDE) for a, c in zip(first, n)])
        for k in keys:
            parts[k].append(d[src[k]][idx])
        is_val = ((d["reset"].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)) % VAL_EVERY == 0
        val.append(np.repeat(is_val, (n + STRIDE - 1) // STRIDE))
        nseg += len(n)
    out = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in parts.items()}
    out["val"] = torch.from_numpy(np.concatenate(val)).to(device)
    out["src"] = torch.full((len(out["val"]),), int(actor), device=device)
    out["files"], out["n_seg"] = len(files), nseg
    return out


def label(D: dict, w, bodies) -> None:
    """The actor's targets: the frozen controller's answer to the ORIGINAL command at the actor's own states (D["act"] holds
    the actor's own action until then; kept as D["own"])."""
    D["own"] = D["act"]
    tgt = torch.empty_like(D["act"])
    with torch.no_grad():
        for j in range(0, len(tgt), 32768):
            i = torch.arange(j, min(j + 32768, len(tgt)), device=tgt.device)
            tgt[i] = forward(w, *batch(D, i, orig=True), bodies)
    D["act"] = tgt


def merge(A: dict, B: dict) -> dict:
    keys = [k for k in A if isinstance(A[k], torch.Tensor) and k in B]
    out = {k: torch.cat([A[k], B[k]]) for k in keys}
    out["files"], out["n_seg"] = A["files"] + B["files"], A["n_seg"] + B["n_seg"]
    return out


def load_all(device: str, w, bodies, act_dirs: list[str], keep_orig: bool = False, teacher: list[str] = ()) -> dict:
    """teacher (2026-11-05, wrist-only relabel): the dirs whose teacher ticks train, default DATA; an actor dir also holds
    the teacher's ticks of its own reaches (relabel_gen.py records both), so the actor rounds can be the teacher set too."""
    D = None
    for t in teacher or [DATA]:
        T = load(device, keep_orig=keep_orig, data=Path(t))
        D = T if D is None else merge(D, T)
    for a in act_dirs:
        A = load(device, data=Path(a), actor=True)
        label(A, w, bodies)
        if not keep_orig:
            A.pop("pw"), A.pop("qw")
        D = merge(D, A)
    return D


def batch(D: dict, i, orig: bool = False):
    p, q = ("pw", "qw") if orig else ("pw1", "qw1")
    return D["rq"][i], D["av"][i], D["dp"][i], D["dv"][i], D["ac"][i], D["rp"][i], D[p][i], D[q][i]


def bodies_of(meta) -> list[int]:
    return [meta["selected_body_names"].index(n) for n in RG.BODIES]


def check(device: str) -> None:
    """The batched forward on the recorded inputs reproduces the recorded actions; and how far the relabelled inputs move
    the frozen controller's answer."""
    w, meta = load_wrapper(device)
    D = load(device, keep_orig=True)
    bodies = bodies_of(meta)
    scale = w.action_scale[0]
    n = len(D["act"])
    i = torch.randperm(n, device=device)[:20000]
    with torch.no_grad():
        a0 = forward(w, *batch(D, i, orig=True), bodies)
        a1 = forward(w, *batch(D, i), bodies)
    tgt = D["act"][i]
    d0 = ((a0 - tgt) * scale).abs()
    d1 = (a1 - tgt) * scale
    print(f"{D['files']} files, {D['n_seg']} reaches, {n} ticks (stride {STRIDE}), {int(D['val'].sum())} validation")
    print(
        f"recorded inputs: batched forward vs recorded action max {d0.max().item():.2e} rad, mean {d0.mean().item():.2e}"
    )
    print(f"relabelled inputs: frozen answer moves rms {d1.pow(2).mean().sqrt().item():.4f} rad "
          f"({np.degrees(d1.pow(2).mean().sqrt().item()):.2f} deg)")  # fmt: skip
    assert d0.max().item() < 1e-4, "the batched forward does not reproduce the policy"


def diag(device: str, tag: str, act_dirs: list[str], teacher: list[str] = ()) -> None:
    """Does the student read the reference? On validation ticks of each source (the teacher's ticks, the actor's): how much
    its answer moves between the relabelled and the original reference (the frozen controller moves 3.9 deg rms on the
    teacher's ticks), and how well "repeat the last action" alone fits."""
    w, meta = load_wrapper(device)
    bodies = bodies_of(meta)
    D = load_all(device, w, bodies, act_dirs, keep_orig=True, teacher=teacher)
    scale = w.action_scale[0]
    rms = lambda x: float(np.degrees(((x * scale) ** 2).mean().sqrt().item()))  # noqa: E731
    sd = torch.load(OUT / f"student_{tag}.pt", map_location=device, weights_only=False)["params"]
    frozen = {n: p.detach().clone() for n, p in w.named_parameters()}
    for src in sorted(set(D["src"].tolist())):
        i = torch.where(D["val"] & (D["src"] == src))[0][:30000]
        with torch.no_grad():
            for n, p in w.named_parameters():
                p.copy_(frozen[n])
            f_e, f_c = forward(w, *batch(D, i), bodies), forward(w, *batch(D, i, orig=True), bodies)
            for n, p in w.named_parameters():
                p.copy_(sd[n])
            s_e, s_c = forward(w, *batch(D, i), bodies), forward(w, *batch(D, i, orig=True), bodies)
        a, last = D["act"][i], D["ac"][i][:, -1]
        print(f"{'actor' if src else 'teacher'} ticks, rms deg per joint on {len(i)} validation ticks:")
        print(f"  target vs repeat the last action   {rms(a - last):.2f}")
        print(
            f"  target vs frozen on relabelled ref {rms(a - f_e):.2f}   (frozen on the original ref: {rms(a - f_c):.4f})"
        )
        print(f"  target vs student on relabelled    {rms(a - s_e):.2f}")
        print(
            f"  reference sensitivity, original vs relabelled ref: frozen {rms(f_c - f_e):.2f}, student {rms(s_c - s_e):.2f}"
        )


def task_path(w) -> list:
    """The parameters that read the reference and nothing else: the task embedder, and in every block the condition norm,
    the cross-attention's key and value projections (rows E..3E of in_proj; the query rows read the history and stay) and
    its output projection. Everything that reads the history is left as it is."""
    E = w.final_norm.scale.shape[0]
    out = list(w.task_embedder.parameters())
    for blk in w.transformer_blocks:
        ca = blk.cross_attention
        out += list(blk.cond_norm.parameters()) + list(ca.out_proj.parameters()) + [ca.in_proj_weight, ca.in_proj_bias]
        for p in (ca.in_proj_weight, ca.in_proj_bias):
            p.requires_grad_(True)
            p.register_hook(lambda g, E=E: torch.cat([torch.zeros_like(g[:E]), g[E:]]))
    return out


def train(
    device: str,
    steps: int,
    bs: int,
    lr: float,
    tag: str,
    part: str = "all",
    act_dirs: list[str] = (),
    act_frac: float = 0.5,
    teacher: list[str] = (),
) -> None:
    """part "all": every weight trains (r1: the student learned to continue its own last actions and stopped reading the
    reference, diag); "task": only task_path(w) (r2task: no better in closed loop). With act_dirs, act_frac of every batch
    comes from the actor's ticks; validation is reported per source and the best step is chosen on the actor's."""
    torch.manual_seed(0)
    w, meta = load_wrapper(device)
    bodies = bodies_of(meta)
    D = load_all(device, w, bodies, list(act_dirs), teacher=list(teacher))
    scale = w.action_scale[0]
    srcs = sorted(set(D["src"].tolist()))
    tr = {c: torch.where(~D["val"] & (D["src"] == c))[0] for c in srcs}
    va = {c: torch.where(D["val"] & (D["src"] == c))[0] for c in srcs}
    va = {c: v[torch.randperm(len(v), device=device)[:50000]] for c, v in va.items()}
    take = {c: (bs if len(srcs) == 1 else int(round(bs * (act_frac if c else 1 - act_frac)))) for c in srcs}
    for p in w.parameters():
        p.requires_grad_(part == "all")
    params = list(w.parameters()) if part == "all" else task_path(w)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * s / steps))
    )

    def val_loss() -> dict:
        out = {}
        with torch.no_grad():
            for c, v in va.items():
                tot = 0.0
                for j in range(0, len(v), 8192):
                    i = v[j : j + 8192]
                    tot += ((forward(w, *batch(D, i), bodies) - D["act"][i]) * scale).pow(2).mean(-1).sum().item()
                out["actor" if c else "teacher"] = tot / len(v)
        out["val"] = out.get("actor", out["teacher"])
        return out

    log = [dict(step=0, **val_loss())]
    print(f"train ticks {[len(t) for t in tr.values()]}, validation {[len(v) for v in va.values()]}, per batch {take}; "
          f"frozen on relabelled validation {log[0]}", flush=True)  # fmt: skip
    t0, run = time.time(), 0.0
    for s in range(steps):
        i = torch.cat([tr[c][torch.randint(len(tr[c]), (take[c],), device=device)] for c in srcs])
        loss = ((forward(w, *batch(D, i), bodies) - D["act"][i]) * scale).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if (s + 1) % 1000 == 0 or s + 1 == steps:
            log.append(dict(step=s + 1, train=run, **val_loss(), s=round(time.time() - t0)))
            print(json.dumps(log[-1]), flush=True)
            if log[-1]["val"] <= min(r["val"] for r in log[1:]):  # keep the best validation step
                best = dict(step=s + 1, params={k: v.detach().cpu().clone() for k, v in w.named_parameters()})
    torch.save(dict(params=best["params"], step=best["step"], steps=steps, bs=bs, lr=lr, stride=STRIDE, n_seg=D["n_seg"],
                    ticks={c: len(t) for c, t in tr.items()}, frozen_val=log[0], part=part, act=list(act_dirs),
                    act_frac=act_frac, teacher=list(teacher) or [str(DATA)]), OUT / f"student_{tag}.pt")  # fmt: skip
    json.dump(log, open(OUT / f"train_{tag}.json", "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "train", "diag"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--part", default="all", choices=["all", "task"])
    ap.add_argument("--act", default="", help="comma-separated RELABEL_ACT data dirs (actor ticks, DAgger labels)")
    ap.add_argument("--act-frac", type=float, default=0.5)
    ap.add_argument("--teacher", default="", help="comma-separated dirs of teacher ticks (default RELABEL_DATA)")
    a = ap.parse_args(argv)
    dev = "cuda"
    if a.what == "check":
        check(dev)
    elif a.what == "diag":
        diag(dev, a.tag, [x for x in a.act.split(",") if x], [x for x in a.teacher.split(",") if x])
    else:
        train(
            dev,
            a.steps,
            a.bs,
            a.lr,
            a.tag,
            a.part,
            [x for x in a.act.split(",") if x],
            a.act_frac,
            [x for x in a.teacher.split(",") if x],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
