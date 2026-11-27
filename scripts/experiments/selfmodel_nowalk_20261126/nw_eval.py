"""selfmodel_nowalk_20261126: arms D (simpledata's model, the reference), N, C, C2 scored on the same segments; read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY nw_eval.py --sd <simpledata dir> --arms D=<root>,N=<root>,C=<root>[,C2=<root>] --splits <splits.npz> \
        --out <dir> [--no-check]

Every model is E3_ensemble frac 1.0 (<root>/E3_ensemble_f1/model.pt), rebuilt on the scored data and loaded from its
state (normalisation included, as train.load_model). Arms whose splits.npz entry says so (N, C2) get x0's walk-flag
column 204 zeroed in every scored segment, as in their training. Error = the predicted final hand point to the real one
(cm, heading frame), as sd_eval / sx_eval; attitude (deg) beside it.
Segments (kept as simpledata kept them: not fallen / lost and settled at the end label, "ended" = not fallen only):
  primary    the real SIMPLE episodes, main label, settled (simpledata eval 3: 235 segments, 120 episodes). Layout =
             (task, episode index % 10) from the reset id (90_000_000 + 1000 task + 100 level + index): 40 layouts.
             Per arm the median [95 %: 4000 resamples of the layouts]; the paired ratios N/C (registered), N/C2, C2/C,
             N/D, C/D: median of one arm over the median of the other on the same resampled segments. Overall, first
             reach (pre-grasp, chain 0), second reach (entry, chain 1), per task, and the three tasks without
             TabletopGraspMP.
  secondary  the supplement label (task, ended and settled); the main label ended; the held-out splits of data3 /
             data3b (data.py's own buckets: crc32, so val holds the d3-walk resets, test_iid the task-walk ones), by
             start: in place (first, chained), task walk (test_iid; first, chained), d3 walk (val; first, chained),
             data3 val all (simpledata's denominator), data3b pre-grasp first + entry chain 1 (task walk, test_iid; in
             place, val + test_iid). Resamples: 2000 of the resets, paired.
Reproduction check (arm D, CPU): the medians simpledata recorded -- real main settled 0.7053, real task ended 0.8208,
data3 val all 1.8860, data3 test_iid all 1.5756 (|d| <= 2e-4 cm) -- else exit 1.
Writes <out>/eval.txt (this print), eval.json and per_segment.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is D's switch here"
M = TR.M
WALKED_COL = 204
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
ID0 = {"main": 90_000_000, "task": 91_000_000}
NB_REAL, NB_HELD = 4000, 2000
RATIOS = (("N", "C"), ("N", "C2"), ("C2", "C"), ("N", "D"), ("C", "D"), ("C2", "D"))
REPRO = {("real main", "settled"): 0.70526, ("real task", "ended"): 0.82076, ("data3 val", "all"): 1.88601, ("data3 test_iid", "all"): 1.57564}


def kept(D: dict, ended: bool = False) -> np.ndarray:
    ok = np.asarray(D["fell"]) == 0
    if not ended:
        ok &= np.asarray(D["settled_end"]) > 0
    return np.flatnonzero(ok)


def predict(arms: dict, D: dict, idx: np.ndarray) -> dict:
    GD = M.tensors(D, "cpu")
    ti = torch.tensor(idx)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos, att = {}, {}
    x0 = GD["x0"]
    for name, (root, zero) in arms.items():
        ck = torch.load(Path(root) / "E3_ensemble_f1" / "model.pt", map_location="cpu")
        m = M.build("E3_ensemble", D, np.arange(len(D["n_steps"])), "cpu", n_train=ck["n_train"])
        m.load_state_dict(ck["state"])
        m.eval()
        if zero:
            for mem in m.members:  # trained on a constant 0 column: Norm's mean 0, its floor 1e-3
                assert float(mem.nx.mu[WALKED_COL]) == 0.0 and abs(float(mem.nx.sd[WALKED_COL]) - 1e-3) < 1e-9, name
        x = x0.clone()
        if zero:
            x[:, WALKED_COL] = 0.0
        p = M.predict_batched(m, {**GD, "x0": x}, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).numpy().astype(np.float64)
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).numpy().astype(np.float64)
        del m
    return dict(pos=pos, att=att)


def groups(keys: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(keys, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(keys[o])) + 1)


def table(title: str, E: dict, A: dict, keys: np.ndarray, sel: np.ndarray, nb: int, rng, out: dict, unit: str) -> None:
    """Per arm median [CI], p90, attitude median; the paired ratios -- all on the same resamples of `keys` (clusters)."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        print(f"  {title}: no segments")
        return
    G = [idx[g] for g in groups(keys[idx])]
    names = list(E)
    rat = [(a, b) for a, b in RATIOS if a in E and b in E]

    def st(s):
        o = {n: np.median(E[n][s]) for n in names}
        o.update({f"p90_{n}": np.percentile(E[n][s], 90) for n in names})
        o.update({f"att_{n}": np.median(A[n][s]) for n in names})
        o.update({f"{a}/{b}": o[a] / o[b] for a, b in rat})
        return o

    pt = st(idx)
    bs = [st(np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])) for _ in range(nb)]
    ci = {k: (float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
    out[title] = dict(n=int(len(idx)), clusters=len(G), unit=unit, point={k: float(v) for k, v in pt.items()}, ci=ci)
    f = lambda k, d=3: f"{pt[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    print(f"  {title}: n {len(idx)} ({len(G)} {unit})")
    print("    median cm: " + " | ".join(f"{n} {f(n)}" for n in names))
    print("    p90 cm:    " + " | ".join(f"{n} {pt['p90_' + n]:.2f}" for n in names)
          + ";  attitude deg median: " + " | ".join(f"{n} {pt['att_' + n]:.2f}" for n in names))  # fmt: skip
    if rat:
        print("    ratio:     " + " | ".join(f"{a}/{b} {f(f'{a}/{b}')}" for a, b in rat))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=Path, required=True)
    ap.add_argument("--arms", required=True, help="name=root,... (root holds E3_ensemble_f1/model.pt)")
    ap.add_argument("--splits", type=Path, required=True, help="nw_split's splits.npz (which arms zero the walk flag)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-check", action="store_true", help="skip the reproduction check (smoke runs)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    Z = np.load(a.splits)
    arms = {}
    for item in a.arms.split(","):
        n, r = item.split("=", 1)
        arms[n] = (r, bool(Z[f"{n}_zero"]))
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"arms {arms}; splits {a.splits}; FDVEL {DT.FDVEL}; device cpu")
    res, seg, fails = {}, {}, []
    rng = np.random.default_rng(0)
    # real episodes
    for lab in ("main", "task"):
        D = DT.load(str(a.sd / "real" / lab / "data" / "all.npz"))
        for rule in ("settled", "ended"):
            idx = kept(D, ended=rule == "ended")
            P = predict(arms, D, idx)
            rid = np.asarray(D["reset"]).astype(np.int64)[idx] - ID0[lab]
            task, level, index = rid // 1000, rid % 1000 // 100, rid % 100
            layout = task * 100 + index
            ch = np.asarray(D["chain"])[idx]
            assert set(np.unique(task)) <= set(range(4)) and index.max() < 10 and level.max() < 3
            tag = f"real {lab}"
            prim = lab == "main" and rule == "settled"
            print(f"\n== {tag} label, kept {rule}{' (PRIMARY)' if prim else ''}: {len(idx)} segments, "
                  f"{len(np.unique(rid))} episodes, {len(np.unique(layout))} layouts; walked {int(np.asarray(D['walked'])[idx].sum())}")  # fmt: skip
            out = res.setdefault(f"{tag} {rule}", {})
            E, A = P["pos"], P["att"]
            table("all", E, A, layout, np.ones(len(idx), bool), NB_REAL, rng, out, "layouts")
            table("first reach (pre-grasp)", E, A, layout, ch == 0, NB_REAL, rng, out, "layouts")
            table("second reach (entry)", E, A, layout, ch == 1, NB_REAL, rng, out, "layouts")
            table("three tasks (no TabletopGraspMP)", E, A, layout, task != 3, NB_REAL, rng, out, "layouts")
            for t, nm in enumerate(TASKS):
                table(f"task {nm}", E, A, layout, task == t, NB_REAL, rng, out, "layouts")
            seg[f"{lab}_{rule}"] = dict(idx=idx, reset=rid + ID0[lab], task=task, layout=layout, chain=ch,
                                        **{f"err_{n}": E[n] for n in E}, **{f"att_{n}": A[n] for n in A})  # fmt: skip
            if "D" in E and (tag, rule) in REPRO:
                d = float(np.median(E["D"])) - REPRO[(tag, rule)]
                print(f"  reproduction: D median {np.median(E['D']):.5f} vs simpledata {REPRO[(tag, rule)]:.5f} ({d:+.5f})")
                if abs(d) > 2e-4:
                    fails.append(f"{tag} {rule} D median off by {d:+.5f}")
        del D
    # held-out splits
    for name in ("data3", "data3b"):
        D = DT.load(str(a.sd / name / "data" / "all.npz"))
        sp = DT.splits(D)
        ok = np.zeros(len(D["n_steps"]), bool)
        ok[kept(D)] = True
        sv, st_ = np.zeros(len(ok), bool), np.zeros(len(ok), bool)
        sv[sp["val"]], st_[sp["test_iid"]] = True, True
        idx = np.flatnonzero(ok & (sv | st_))
        P = predict(arms, D, idx)
        E, A = P["pos"], P["att"]
        g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
        rs, ch, wk, wt = g("reset").astype(np.int64), g("chain"), g("walked") > 0, g("walk_task") > 0
        val, test = sv[idx], st_[idx]
        print(f"\n== {name} held out (kept settled): val {int(val.sum())}, test_iid {int(test.sum())} segments")
        out = res.setdefault(f"{name} held out", {})
        cells = [("val all (simpledata's denominator)" if name == "data3" else "val all", val)]
        if name == "data3":
            cells += [("first, in place (val + test_iid)", (ch == 0) & ~wk), ("chained, in place (val + test_iid)", (ch >= 1) & ~wk),
                      ("first, task walk (test_iid)", test & (ch == 0) & wk & wt), ("chained, task walk (test_iid)", test & (ch >= 1) & wk & wt),
                      ("first, d3 walk (val)", val & (ch == 0) & wk & ~wt), ("chained, d3 walk (val)", val & (ch >= 1) & wk & ~wt),
                      ("test_iid all", test)]  # fmt: skip
        else:
            kd = g("kind")
            pe = ((kd == 1) & (ch == 0)) | ((kd == 2) & (ch == 1))
            cells += [("pre-grasp first + entry chain 1, task walk (test_iid)", test & pe & wk & wt),
                      ("pre-grasp first + entry chain 1, in place (val + test_iid)", pe & ~wk),
                      ("pre-grasp first + entry chain 1, d3 walk (val)", val & pe & wk & ~wt), ("test_iid all", test)]  # fmt: skip
        for lab_, sel in cells:
            table(lab_, E, A, rs, sel, NB_HELD, rng, out, "resets")
        seg[name] = dict(idx=idx, reset=rs, chain=ch, walked=wk, walk_task=wt, val=val, test=test,
                         **{f"err_{n}": E[n] for n in E}, **{f"att_{n}": A[n] for n in A})  # fmt: skip
        if "D" in E:
            for sname, s in (("val", val), ("test_iid", test)):
                key = (f"{name} {sname}", "all")
                if key in REPRO:
                    d = float(np.median(E["D"][s])) - REPRO[key]
                    print(f"  reproduction: {name} {sname} all D median {np.median(E['D'][s]):.5f} vs simpledata {REPRO[key]:.5f} ({d:+.5f})")
                    if abs(d) > 2e-4:
                        fails.append(f"{key} D median off by {d:+.5f}")
        del D
    np.savez(a.out / "per_segment.npz", **{f"{k}__{f}": v for k, d in seg.items() for f, v in d.items()})
    (a.out / "eval.json").write_text(json.dumps(dict(arms={k: list(v) for k, v in arms.items()}, results=res, fails=fails), indent=1))
    if a.no_check:
        print("\nreproduction check skipped (--no-check)")
        return
    print(f"\n{'REPRODUCTION PASS' if not fails else 'REPRODUCTION FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
