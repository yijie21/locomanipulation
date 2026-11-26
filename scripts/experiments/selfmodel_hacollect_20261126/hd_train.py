"""selfmodel_hadata_20261126 revision 2b: one HA training run. train3b's frozen t3b_train.py (c7ecf581) does
data.load -> models.tensors(step=False) -> train.train(E3_ensemble, frac, seed, iters, lr 1e-3); this does the same, by
the same modules (imported from that file), with two changes and nothing else:
  1. val every 1000 iterations (the registration; train.EVAL_EVERY is 2000), set on the module before train.train;
  2. --init F (the fine-tune arm): the start state is F's weights with the normalisation rebuilt from this run's training
     split (the registration: both arms renormalised on HA's training set). models.build on this run's data and split
     (the model train.train builds), its buffers kept -- the normalisation: every member's nx.mu / nx.sd / nu.mu / nu.sd
     and sd_d, asserted to be all of its buffers --, every parameter F's (same names and shapes, asserted); written to
     <out-root>/init_ha_f<frac>.pt (.json: what was rebuilt) and passed as train.train's init, which logs its val as iteration 0 and keeps it as the state
     to beat.
Prints t3b_train's "<N> segments, splits {...}" line (the driver waits for it before starting the next run).

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES=<g> \
        $PY hd_train.py --t3b <train3b asrun>/t3b_train.py --frac F --seed S --data <spec> --out-root <dir> [--init F.pt] [--iters N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

EVAL_EVERY = 1000
T3B_MD5 = "c7ecf581"


def load_t3b(path: Path):
    h = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    assert h == T3B_MD5, f"{path}: md5 {h}, not train3b's frozen {T3B_MD5}"
    spec = importlib.util.spec_from_file_location("t3b_train", str(path))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


def init_ha(T, D, frac: float, seed: int, dev: str, f_path: Path, out: Path) -> tuple[Path, dict]:
    M, DT = T.M, T.DT
    ref = M.build("E3_ensemble", D, DT.splits(D, frac)["train"], dev, seed)
    sd = ref.state_dict()
    F = torch.load(f_path, map_location=dev)["state"]
    bufs = {n for n, _ in ref.named_buffers()}
    params = {n for n, _ in ref.named_parameters()}
    assert set(sd) == bufs | params and not (bufs & params), "state = buffers + parameters"
    assert all(n.endswith((".nx.mu", ".nx.sd", ".nu.mu", ".nu.sd", ".sd_d")) for n in bufs), f"buffers: {sorted(bufs)[:6]}"
    assert set(F) == set(sd), f"F's state names differ: {sorted(set(F) ^ set(sd))[:6]}"
    assert all(F[n].shape == sd[n].shape for n in sd), "F's shapes differ"
    mix = {n: (sd[n] if n in bufs else F[n]).clone() for n in sd}
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"init_ha_f{frac:g}.pt"
    torch.save(dict(state=mix, n_train=int(len(DT.splits(D, frac)["train"]))), p)
    info = dict(init_from=str(f_path), init_md5=hashlib.md5(f_path.read_bytes()).hexdigest()[:8], buffers_rebuilt=len(bufs),
                params_from_F=len(params), max_buffer_change=float(max((sd[n] - F[n]).abs().max() for n in bufs)))  # fmt: skip
    (out / f"init_ha_f{frac:g}.json").write_text(json.dumps(info, indent=1))
    print(f"init: {json.dumps(info)}", flush=True)
    return p, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3b", type=Path, required=True, help="train3b's frozen t3b_train.py")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--data", required=True, help="data.load spec (comma-separated files)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="the fine-tune arm: F's model.pt")
    a = ap.parse_args(argv)
    assert 0 < a.frac <= 1
    T = load_t3b(a.t3b)
    TR, DT, M = T.TR, T.DT, T.M
    assert DT.FDVEL and DT.GOAL_REF and DT.APPROACH and DT.QFIX and not DT.X0_REF, "the four training flags, X0_REF off"
    TR.EVAL_EVERY = EVAL_EVERY
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(a.data)
    GD = M.tensors(D, dev, step=False)
    print(f"{len(D['n_steps'])} segments, splits { {k: len(v) for k, v in DT.splits(D).items()} }", flush=True)
    init = None
    if a.init is not None:
        init, _ = init_ha(T, D, a.frac, a.seed, dev, a.init, a.out_root)
    TR.train("E3_ensemble", D, GD, a.frac, a.seed, a.iters, None, dev, a.out_root, init, 1e-3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
