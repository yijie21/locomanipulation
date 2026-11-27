"""selfmodel_walkmix_20261126: several models scored on the same segments, paired ratios averaged over seeds; read-only.
nowalk's nw_eval.py (the same segments, cells, layouts, resamples and D check) with three changes: the models and the
pairs are given on the command line; a pair a/b is scored per seed (a_k / b_k, or a_k / b_0 when b has only seed 0) and
the registered statistic is the mean over seeds, all on the same resamples (as nowalk's nw2_eval); and nowalk's phase-1
models are checked against nowalk's own eval.json.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY wm_eval.py --sd <simpledata dir> --models Ma0=<root>,Ma1=<root>,...,N0=<root>:z,... --pairs Ma/Cm,Mb/Cm,... \
        --nowalk-eval <nowalk run_b dir> --out <dir> [--extra name=<all.npz>,...] [--no-check]

A model name is a base and a one-digit seed (Ma0, Cm1, D0); ":z" after the root zeroes x0's walk-flag column 204 in every
scored segment (nowalk's N, later T), checked against the model's own normalisation (mean 0, sd at the floor). Every
model is E3_ensemble frac 1.0 (<root>/E3_ensemble_f1/model.pt), rebuilt on the scored data and loaded from its state.
Error = the predicted final hand point to the real one (cm, heading frame); attitude (deg) beside it.
Segments: as nw_eval -- primary the real SIMPLE episodes, main label, settled (235 segments, 120 episodes, 40 layouts;
4000 resamples of the layouts): all, first reach (chain 0), second reach (chain 1), the three tasks without
TabletopGraspMP, each task; secondary the other labels / keeps and data3 / data3b held out (2000 resamples of the resets);
--extra files (T's new data, later): their val + test_iid buckets, first / chained.
Checks (else exit 1): D0 (simpledata's D) reproduces simpledata's four medians (|d| <= 2e-4 cm, nw_eval's REPRO); each
of nowalk's N0, N1, C0, C1 given reproduces its median in nowalk's eval_s<k>/eval.json, real main settled "all" and
"first reach" (|d| <= 2e-4 cm).
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


def seeds_of(E: dict, base: str) -> list[int]:
    return sorted(int(n[len(base):]) for n in E if n[: len(base)] == base and n[len(base):].isdigit() and len(n) == len(base) + 1)


def table(title: str, E: dict, A: dict, keys: np.ndarray, sel: np.ndarray, nb: int, rng, out: dict, unit: str, PAIRS) -> None:
    """Per model median [CI], p90, attitude median; each pair per seed and its mean over seeds -- all on the same resamples."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        print(f"  {title}: no segments")
        return
    G = [idx[g] for g in groups(keys[idx])]
    names = list(E)
    pairs = []
    for a, b in PAIRS:
        sa, sb = seeds_of(E, a), seeds_of(E, b)
        if sa and sb:
            pairs.append((a, b, [(k, k if k in sb else sb[0]) for k in sa if k in sb or sb == [0]]))

    def st(s):
        o = {n: np.median(E[n][s]) for n in names}
        o.update({f"p90_{n}": np.percentile(E[n][s], 90) for n in names})
        o.update({f"att_{n}": np.median(A[n][s]) for n in names})
        for a, b, ks in pairs:
            r = [o[f"{a}{k}"] / o[f"{b}{j}"] for k, j in ks]
            o.update({f"{a}{k}/{b}{j}": v for (k, j), v in zip(ks, r)})
            o[f"{a}/{b} mean"] = float(np.mean(r))
        return o

    pt = st(idx)
    bs = [st(np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])) for _ in range(nb)]
    ci = {k: (float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
    out[title] = dict(n=int(len(idx)), clusters=len(G), unit=unit, point={k: float(v) for k, v in pt.items()}, ci=ci)
    f = lambda k, d=3: f"{pt[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    print(f"  {title}: n {len(idx)} ({len(G)} {unit})")
    print("    median cm: " + " | ".join(f"{n} {f(n)}" for n in names))
    print("    p90 cm:    " + " | ".join(f"{n} {pt['p90_' + n]:.2f}" for n in names))
    print("    attitude deg median: " + " | ".join(f"{n} {pt['att_' + n]:.2f}" for n in names))
    for a, b, ks in pairs:
        print(f"    {a}/{b}: mean over seeds {f(f'{a}/{b} mean')}; per seed " + ", ".join(f"{a}{k}/{b}{j} {f(f'{a}{k}/{b}{j}')}" for k, j in ks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=Path, required=True)
    ap.add_argument("--models", required=True, help="name=root[:z],... (root holds E3_ensemble_f1/model.pt; :z zeroes the walk flag)")
    ap.add_argument("--pairs", required=True, help="a/b,... (model bases)")
    ap.add_argument("--nowalk-eval", type=Path, required=True, help="nowalk's run_b (eval_s<k>/eval.json)")
    ap.add_argument("--extra", default="", help="name=<all.npz>,...: extra held-out files (val + test_iid)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-check", action="store_true", help="skip the reproduction check (smoke runs)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    arms = {}
    for item in a.models.split(","):
        n, r = item.split("=", 1)
        assert n[-1].isdigit() and n not in arms, n
        z = r.endswith(":z")
        arms[n] = (r[:-2] if z else r, z)
    PAIRS = [tuple(p.split("/")) for p in a.pairs.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"models {arms}; pairs {PAIRS}; FDVEL {DT.FDVEL}; device cpu")
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
            table("all", E, A, layout, np.ones(len(idx), bool), NB_REAL, rng, out, "layouts", PAIRS)
            table("first reach (pre-grasp)", E, A, layout, ch == 0, NB_REAL, rng, out, "layouts", PAIRS)
            table("second reach (entry)", E, A, layout, ch == 1, NB_REAL, rng, out, "layouts", PAIRS)
            table("three tasks (no TabletopGraspMP)", E, A, layout, task != 3, NB_REAL, rng, out, "layouts", PAIRS)
            for t, nm in enumerate(TASKS):
                table(f"task {nm}", E, A, layout, task == t, NB_REAL, rng, out, "layouts", PAIRS)
            seg[f"{lab}_{rule}"] = dict(idx=idx, reset=rid + ID0[lab], task=task, layout=layout, chain=ch,
                                        **{f"err_{n}": E[n] for n in E}, **{f"att_{n}": A[n] for n in A})  # fmt: skip
            if "D0" in E and (tag, rule) in REPRO:
                d = float(np.median(E["D0"])) - REPRO[(tag, rule)]
                print(f"  reproduction: D0 median {np.median(E['D0']):.5f} vs simpledata {REPRO[(tag, rule)]:.5f} ({d:+.5f})")
                if abs(d) > 2e-4:
                    fails.append(f"{tag} {rule} D0 median off by {d:+.5f}")
            if prim:
                for k in (0, 1):
                    J = json.loads((a.nowalk_eval / f"eval_s{k}" / "eval.json").read_text())["results"]["real main settled"]
                    for base in ("N", "C"):
                        if f"{base}{k}" not in E:
                            continue
                        for cell, s_ in (("all", np.ones(len(idx), bool)), ("first reach (pre-grasp)", ch == 0)):
                            ref = J[cell]["point"][base]
                            d = float(np.median(E[f"{base}{k}"][s_])) - ref
                            print(f"  reproduction: {base}{k} {cell} median {np.median(E[f'{base}{k}'][s_]):.5f} vs nowalk eval_s{k} {ref:.5f} ({d:+.5f})")
                            if abs(d) > 2e-4:
                                fails.append(f"{base}{k} {cell} off by {d:+.5f}")
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
            table(lab_, E, A, rs, sel, NB_HELD, rng, out, "resets", PAIRS)
        seg[name] = dict(idx=idx, reset=rs, chain=ch, walked=wk, walk_task=wt, val=val, test=test,
                         **{f"err_{n}": E[n] for n in E}, **{f"att_{n}": A[n] for n in A})  # fmt: skip
        if "D0" in E:
            for sname, s in (("val", val), ("test_iid", test)):
                key = (f"{name} {sname}", "all")
                if key in REPRO:
                    d = float(np.median(E["D0"][s])) - REPRO[key]
                    print(f"  reproduction: {name} {sname} all D0 median {np.median(E['D0'][s]):.5f} vs simpledata {REPRO[key]:.5f} ({d:+.5f})")
                    if abs(d) > 2e-4:
                        fails.append(f"{key} D median off by {d:+.5f}")
        del D
    for item in [x for x in a.extra.split(",") if x]:
        name, p = item.split("=", 1)
        D = DT.load(p)
        sp = DT.splits(D)
        ok = np.zeros(len(D["n_steps"]), bool)
        ok[kept(D)] = True
        sv = np.zeros(len(ok), bool)
        sv[sp["val"]] = True
        sv[sp["test_iid"]] = True
        idx = np.flatnonzero(ok & sv)
        P = predict(arms, D, idx)
        E, A = P["pos"], P["att"]
        rs, ch, wk = (np.asarray(D[k])[idx] for k in ("reset", "chain", "walked"))
        rs = rs.astype(np.int64)
        print(f"\n== {name} held out (val + test_iid, kept settled): {len(idx)} segments, walked {int((wk > 0).sum())}")
        out = res.setdefault(f"{name} held out", {})
        for lab_, sel in (("first", ch == 0), ("chained", ch >= 1), ("all", np.ones(len(idx), bool))):
            table(lab_, E, A, rs, sel, NB_HELD, rng, out, "resets", PAIRS)
        seg[name] = dict(idx=idx, reset=rs, chain=ch, **{f"err_{n}": E[n] for n in E}, **{f"att_{n}": A[n] for n in A})
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
