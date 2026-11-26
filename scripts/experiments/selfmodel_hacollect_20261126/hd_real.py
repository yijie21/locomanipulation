"""selfmodel_hadata_20261126 revision 2b, evaluation 3 (section 3): the real HA episodes (hr_real.py, seeds 200-239) of one
controller, merged per task into the layout xfer's sx_eval reads, and scored by S_f1_s0 and T_f1_s0; read-only on the
episodes and the models.

    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo_b> CUDA_VISIBLE_DEVICES= \
        $PY hd_real.py --root <collection>/train/<ctrl> --real <real>/<ctrl>_ppbox,<real>/<ctrl>_doubledesk \
        --out <root>/eval/real --sd-range <frozen sd_range.py> --light <frozen hr_smoke_stats.py> \
        --touch <frozen hd_touch.py> [--seeds 200-239] [--json F]

Imported by path, md5 checked: simpledata's sd_range.py (1276163a: quantities, nn_dist, groups_of), hr_smoke_stats.py
(b4d65b92, asrun_real's: light_target, part), hd_touch.py (5526f05b, asrun_train's: touching, revision 1's rule). The
reader and the models: LOCO_REPO's data.py / train.py / models.py (repo_b: 174071cb / 8acdaea1 / 0ac27f6b, checked).

1 merge (sd_real_merge.py's way, its SIMPLE-only checks replaced by hr_real's): per --real directory the episodes'
  data/<task>_s<seed>_r<repeat>.npz (seeds in --seeds) concatenated key by key in seed order (the keys every episode has,
  data.load's rule) -> <out>/<dir name>/data/all.npz, with run.json (controller, planner, stance, delay_f: the
  collection's d3 run.json, the same host) and merge.txt. Checks (any failure: "REAL MERGE ... FAIL", exit 1): every
  log's segments = its npz's; every segment's reset = the log's, task_id / seed / repeat the log's; chains unique per
  episode and each a reach of the log with the same side and reach_no; (reset, chain) unique.
2 the segments: every merged segment; kept = t3b_eval.kept (not fallen / lost, end label settled); touching = revision 1's
  rule (touch_n > 1 N and touch_z >= 0.2 m); clean = kept and not touching. The light-touch group (revision 2a section
  10): kept, touching, and hr_smoke_stats.light_target(log, chain, side) True (None: no contact record, counted apart).
3 HA val: the same model's median on HA d3 val (all reaches, kept; <root>/data/d3.npz), the models built on that file
  (sx_eval's way: the normalisation comes from the state dict) and used for the real segments too.
4 the ratio: the clean reaches' median over HA val's median, per arm; 95 % from 1000 resamples of the episodes (val
  fixed), and beside it with val's resets resampled too; the arms' real medians and their difference T - S on the same
  resamples. Fewer than 10 clean reaches (revision 2a supplement item 12): "样本不足", no ratio, and the arm is chosen
  by HA val; otherwise the arm with the lower real median. By task, reach_no and task x reach_no as well.
5 light-touch group: per task its errors (each one listed when there are 20 or fewer) beside the clean reaches'.
6 range (sd_range's three checks): the 39 quantities inside the training set's 0.5-99.5 % (joints, pelvis, hands), the
  nearest training segment of another reset in the standardised x0 not farther than the training set's own 99th
  percentile (10 000 drawn, seed 0), the command target inside the standing reach table's box (d3_collect.in_table,
  D3.TABLE); in range = all three. The training set: D's spec (<root>/data/d3, d3b, mid_data3, mid_data3b), data.splits'
  train at frac 1.0. Reported on the clean reaches (and the light-touch group): the share inside [95 %, episodes], the
  errors inside / outside per arm; hr_real's cmd_in_range beside it; HA val's shares as the reference.
7 descriptive, every merged segment (revision 2a supplement items 10 and 12): per task x reach_no the contact counts
  (touching, by part: own hand / other hand / body, light-touch, peak force, the touched body's height) and the pelvis
  displacement from the command row to the end label row in the command's heading frame (forward / lateral, cm).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ["LOCO_REPO"])
EXP = REPO / "scripts" / "experiments"
sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029"), str(EXP / "selfmodel_data3_20261125")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is the HA models' switch"
assert os.environ.get("REACHWM_X0_REF", "0") == "0"
M = TR.M
for fn, mod, md in (("data.py", DT, "174071cb"), ("train.py", TR, "8acdaea1"), ("models.py", M, "0ac27f6b")):
    p = Path(mod.__file__).resolve()
    assert p.name == fn and hashlib.md5(p.read_bytes()).hexdigest().startswith(md), f"{p}: not repo_b's {fn} ({md})"
NB = 1000
MIN_CLEAN = 10
ARMS = ("S_f1_s0", "T_f1_s0")
TASK_ID = {"ppbox": 0, "doubledesk": 1, "opendoor": 2}


def load_by_path(name: str, path: Path, md5: str):
    assert hashlib.md5(path.read_bytes()).hexdigest().startswith(md5), f"{path}: md5 is not {md5}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def side_name(s) -> str:
    return "right" if float(s) > 0 else "left"


def merge(d: Path, out: Path, seeds: set, coll_run: dict, HS, TOUCH) -> tuple[Path, list[dict], bool, list[str]]:
    """One hr_real directory -> out/<d.name>/data/all.npz + run.json + merge.txt; per merged segment its log facts."""
    lines, ok, zs, info = [], True, [], []
    logs = []
    for p in sorted((d / "log").glob("*.json")):
        L = json.loads(p.read_text())
        if int(L["seed"]) in seeds:
            logs.append((p.stem, L))
    logs.sort(key=lambda x: (int(x[1]["seed"]), int(x[1]["repeat"])))
    tasks = {L["task"] for _, L in logs}
    if len(tasks) != 1:
        lines.append(f"tasks {sorted(tasks)}: want one")
        ok = False
    n_eps = len(logs)
    no_seg = 0
    for name, L in logs:
        zp = d / "data" / f"{name}.npz"
        if int(L.get("segments", 0)) == 0:
            no_seg += 1
            if zp.exists():
                lines.append(f"{name}: log has no segments but {zp.name} exists")
                ok = False
            continue
        if not zp.exists():
            lines.append(f"{name}: log has {L['segments']} segments, no {zp.name}")
            ok = False
            continue
        z = dict(np.load(zp))
        n = len(z["n_steps"])
        rc = L.get("reaches", [])
        bad = []
        if n != int(L["segments"]):
            bad.append(f"segments {n} != log {L['segments']}")
        if not (np.asarray(z["reset"]) == int(L["reset"])).all():
            bad.append("reset differs from the log's")
        for k, v in (("task_id", TASK_ID[L["task"]]), ("seed", int(L["seed"])), ("repeat", int(L["repeat"]))):
            if not (np.asarray(z[k]).astype(int) == v).all():
                bad.append(f"{k} differs from the log's")
        ch = np.asarray(z["chain"]).astype(int)
        if len(set(ch.tolist())) != n:
            bad.append(f"chains {ch.tolist()} not unique")
        hit = TOUCH.touching(z)
        for j in range(n):
            c = int(ch[j])
            if c >= len(rc):
                bad.append(f"chain {c} beyond the log's {len(rc)} reaches")
                continue
            sd = side_name(z["side"][j])
            if rc[c]["side"] != sd or int(rc[c]["reach_no"]) != int(z["reach_no"][j]):
                bad.append(f"chain {c}: side / reach_no {sd} {int(z['reach_no'][j])} != log {rc[c]['side']} {rc[c]['reach_no']}")
            tr = ((L.get("info") or {}).get("touch") or {}).get("reach") or []
            t = tr[c] if c < len(tr) else None
            info.append(dict(episode=name, task=L["task"], seed=int(L["seed"]), repeat=int(L["repeat"]), reset=int(L["reset"]), chain=c, side=sd,
                             reach_no=int(z["reach_no"][j]), touching=bool(hit[j]),
                             light=HS.light_target(L, c, sd) if hit[j] else False,
                             part=HS.part(t["body"], sd) if (hit[j] and t and t.get("body")) else "",
                             body=(t or {}).get("body")))  # fmt: skip
        if bad:
            lines.append(f"{name}: " + "; ".join(bad))
            ok = False
        zs.append(z)
    if not zs:
        lines.append("no segments")
        return out / d.name, info, False, lines
    keys = set.intersection(*(set(z) for z in zs))
    dropped = sorted(set().union(*(set(z) for z in zs)) - keys)
    allz = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k in keys}
    pairs = list(zip(allz["reset"].astype(int).tolist(), allz["chain"].astype(int).tolist()))
    if len(set(pairs)) != len(pairs):
        lines.append("duplicate (reset, chain)")
        ok = False
    od = out / d.name
    (od / "data").mkdir(parents=True, exist_ok=True)
    tmp = od / "data" / ".all.tmp.npz"
    np.savez_compressed(tmp, **allz)
    tmp.replace(od / "data" / "all.npz")
    task = sorted(tasks)[0] if tasks else None
    (od / "run.json").write_text(json.dumps(dict(controller=coll_run.get("controller"), planner="hr_real (d3_collect.plan_reach)",
                                                 stance=coll_run.get("stance"), delay_f=coll_run.get("delay_f"), task=task,
                                                 seeds=sorted({int(L["seed"]) for _, L in logs}), source=str(d),
                                                 note="stance / delay_f: the collection's d3 run.json (the same host)"), indent=1))  # fmt: skip
    sds = sorted({int(L["seed"]) for _, L in logs})
    head = (f"{d.name}: {n_eps} episodes (seeds {sds[0] if sds else '-'}-{sds[-1] if sds else '-'}, {len(sds)} seeds), {no_seg} without "
            f"segments, {len(pairs)} segments merged; keys left out (not in every episode): {dropped}")  # fmt: skip
    lines.insert(0, head)
    lines.append(f"REAL MERGE {d.name} " + ("PASS" if ok else "FAIL"))
    (od / "merge.txt").write_text("\n".join(lines) + "\n")
    return od, info, ok, lines


def groups_of(reset: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(reset, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(reset[o])) + 1)


def pick(G: list[np.ndarray], rng) -> np.ndarray:
    return np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])


def pct(v, q):
    v = [x for x in v if np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


def med(v: np.ndarray) -> float:
    return float(np.median(v)) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="<collection root>/train/<controller>")
    ap.add_argument("--real", required=True, help="hr_real --out directories, comma-separated (one per task)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="200-239")
    ap.add_argument("--sd-range", type=Path, required=True)
    ap.add_argument("--light", type=Path, required=True)
    ap.add_argument("--touch", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    SR = load_by_path("sd_range", a.sd_range, "1276163a")
    HS = load_by_path("hr_smoke_stats", a.light, "b4d65b92")
    TOUCH = load_by_path("hd_touch", a.touch, "5526f05b")
    import d3_collect as D3

    ctrl = a.root.resolve().name
    coll = a.root.resolve().parent.parent
    coll_run = json.loads((coll / ctrl / "d3" / "run.json").read_text())
    assert coll_run["controller"] == ctrl, (coll_run["controller"], ctrl)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    seeds = set(range(lo, hi + 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    print(f"{ctrl}: real episodes seeds {a.seeds}; device {dev}; reader {DT.__file__}; sd_range {a.sd_range}; "
          f"light_target {a.light}; touch rule {a.touch}")  # fmt: skip
    # 1 merge
    files, info, ok_all = [], [], True
    for d in a.real.split(","):
        od, inf, ok, lines = merge(Path(d), a.out, seeds, coll_run, HS, TOUCH)
        print("\n".join(lines))
        ok_all &= ok
        files.append(od / "data" / "all.npz")
        info += inf
    if not ok_all:
        print("REAL MERGE FAIL: stop")
        sys.exit(1)
    D = DT.load(",".join(str(f) for f in files))
    N = len(D["n_steps"])
    assert N == len(info), (N, len(info))
    assert [(int(r), int(c)) for r, c in zip(D["reset"], D["chain"])] == [(x["reset"], x["chain"]) for x in info], "segment order"
    assert [side_name(s) for s in D["side"]] == [x["side"] for x in info], "segment order (side)"
    task = np.array([x["task"] for x in info])
    rno = np.asarray(D["reach_no"]).astype(int)
    touching = np.array([x["touching"] for x in info])
    light = np.array([x["light"] is True for x in info])
    light_none = np.array([x["light"] is None for x in info])
    allidx = np.arange(N)
    kp = np.zeros(N, bool)
    kp[DT_kept(D, allidx)] = True
    clean = kp & ~touching
    lgrp = kp & touching & light
    rs = np.asarray(D["reset"]).astype(np.int64)
    print(f"\nsegments {N} from {len(np.unique(rs))} episodes; kept {int(kp.sum())} (fallen / lost {int((np.asarray(D['fell']) != 0).sum())}, "
          f"not settled {int(((np.asarray(D['fell']) == 0) & (np.asarray(D['settled_end']) == 0)).sum())}); touching {int(touching.sum())}; "
          f"clean {int(clean.sum())}; light-touch kept {int(lgrp.sum())} (touching kept without a contact record {int((kp & touching & light_none).sum())})")  # fmt: skip
    # 3 HA val and the models
    DV = DT.load(str(a.root / "data" / "d3.npz"))
    vi = DT_kept(DV, DT.splits(DV)["val"])
    nets = {n: TR.load_model("E3_ensemble", 1.0, DV, dev, root=a.root / f"models_{n[0]}_s0") for n in ARMS}

    def errs(Dx, idx):
        GD = M.tensors(Dx, dev)
        ti = torch.tensor(idx, device=dev)
        o = {}
        for n, m in nets.items():
            p = M.predict_batched(m, GD, ti)
            o[n] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        return o

    ev = errs(DV, vi)
    val_med = {n: float(np.median(ev[n])) for n in ARMS}
    Gv = groups_of(np.asarray(DV["reset"])[vi].astype(np.int64))
    er_all = errs(D, allidx)  # every merged segment; the sets below select
    print(f"HA val (d3 val, all reaches, kept): {len(vi)} segments from {len(Gv)} resets; models "
          + ", ".join(f"{n} {TR.model_dir('E3_ensemble', 1.0, a.root / f'models_{n[0]}_s0')}" for n in ARMS))  # fmt: skip
    rng = np.random.default_rng(0)
    out = dict(controller=ctrl, seeds=a.seeds, files=[str(f) for f in files], n_segments=int(N), n_kept=int(kp.sum()),
               n_touching=int(touching.sum()), n_clean=int(clean.sum()), n_light=int(lgrp.sum()), val=dict(n=int(len(vi)), median=val_med))  # fmt: skip
    short = int(clean.sum()) < MIN_CLEAN
    # 4 the ratio
    ci_idx = np.flatnonzero(clean)

    def ratio_block(sel_idx: np.ndarray, with_val: bool) -> dict:
        e = {n: er_all[n][sel_idx] for n in ARMS}
        G = groups_of(rs[sel_idx])
        o = dict(n=int(len(sel_idx)), n_episodes=len(G))
        for n in ARMS:
            o[n] = med(e[n])
            o[f"{n} ratio"] = o[n] / val_med[n]
        o["T-S"] = o[ARMS[1]] - o[ARMS[0]]
        B = {k: [] for k in o if k not in ("n", "n_episodes")}
        if with_val:
            for n in ARMS:
                B[f"{n} ratio (val resampled)"] = []
        for _ in range(NB):
            s = pick(G, rng)
            sv = pick(Gv, rng) if with_val else None
            for n in ARMS:
                m_ = med(e[n][s])
                B[n].append(m_)
                B[f"{n} ratio"].append(m_ / val_med[n])
                if with_val:
                    B[f"{n} ratio (val resampled)"].append(m_ / float(np.median(ev[n][sv])))
            B["T-S"].append(B[ARMS[1]][-1] - B[ARMS[0]][-1])
        o["ci"] = {k: (pct(v, 2.5), pct(v, 97.5)) for k, v in B.items()}
        return o

    def show(tag: str, o: dict, ratio: bool) -> None:
        f = lambda k, d=3: f"{o[k]:.{d}f} [{o['ci'][k][0]:.{d}f}, {o['ci'][k][1]:.{d}f}]"  # noqa: E731
        s = f"  {tag}: n {o['n']} ({o['n_episodes']} episodes); " + "; ".join(f"{n} median {f(n, 2)} cm" for n in ARMS)
        if ratio:
            s += "; ratio to HA val " + "; ".join(f"{n} {f(n + ' ratio')}" + (f" (val resampled too [{o['ci'][n + ' ratio (val resampled)'][0]:.3f}, "
                                                                            f"{o['ci'][n + ' ratio (val resampled)'][1]:.3f}])" if n + " ratio (val resampled)" in o["ci"] else "")
                                               for n in ARMS) + f"; T - S {f('T-S', 2)} cm"  # fmt: skip
        print(s)

    print(f"\n== clean reaches (kept, not touching): {len(ci_idx)}" + (f" < {MIN_CLEAN}: 样本不足 (no ratio; the arm by HA val)" if short else ""))
    print("  HA val median: " + "; ".join(f"{n} {val_med[n]:.3f} cm" for n in ARMS))
    res = {}
    if len(ci_idx):
        if short:
            for n in ARMS:
                print(f"  {n} errors (cm, each clean reach): {np.round(np.sort(er_all[n][ci_idx]), 2).tolist()}")
            res["all"] = dict(n=int(len(ci_idx)), **{n: med(er_all[n][ci_idx]) for n in ARMS})
        else:
            res["all"] = ratio_block(ci_idx, True)
            show("all tasks", res["all"], True)
    if short or not len(ci_idx):
        arm, rule = min(ARMS, key=lambda n: val_med[n]), "HA val (样本不足)"
    else:
        arm, rule = min(ARMS, key=lambda n: res["all"][n]), "the lower real median"
    print(f"  arm chosen: {arm} (by {rule})")
    out.update(short=short, arm=arm, arm_rule=rule, clean=res)
    groups = [("task", t, task == t) for t in sorted(set(task))] + [("reach_no", int(r), rno == r) for r in sorted(set(rno))]
    groups += [("task x reach_no", f"{t} {int(r)}", (task == t) & (rno == r)) for t in sorted(set(task)) for r in sorted(set(rno))]
    out["clean_by"] = {}
    for dim, lab, g in groups:
        sel = np.flatnonzero(clean & g)
        if not len(sel):
            print(f"  {dim} {lab}: no clean reaches")
            continue
        if short:
            o = dict(n=int(len(sel)), **{n: med(er_all[n][sel]) for n in ARMS})
            print(f"  {dim} {lab}: n {o['n']}; " + "; ".join(f"{n} median {o[n]:.2f} cm" for n in ARMS))
        else:
            o = ratio_block(sel, False)
            show(f"{dim} {lab}", o, True)
        out["clean_by"][f"{dim} {lab}"] = o
    # 5 light-touch group
    print(f"\n== light-touch group (revision 2a section 10; kept, touching, light_target): {int(lgrp.sum())}")
    out["light"] = {}
    for t in sorted(set(task)):
        sel, cs = np.flatnonzero(lgrp & (task == t)), np.flatnonzero(clean & (task == t))
        o = dict(n=int(len(sel)), n_clean=int(len(cs)), no_record=int((kp & touching & light_none & (task == t)).sum()))
        for n in ARMS:
            o[n], o[f"{n} clean"] = med(er_all[n][sel]), med(er_all[n][cs])
            if 0 < len(sel) <= 20:
                o[f"{n} each"] = np.round(np.sort(er_all[n][sel]), 2).tolist()
        out["light"][t] = o
        if not len(sel):
            print(f"  {t}: no light-touch reaches (touching kept without a record {o['no_record']}; clean n {o['n_clean']})")
            continue
        print(f"  {t}: n {o['n']} (touching kept without a record {o['no_record']}); " + "; ".join(
            f"{n} median {o[n]:.2f} cm (clean {o[n + ' clean']:.2f}, n {o['n_clean']})" + (f" each {o[n + ' each']}" if n + " each" in o else "")
            for n in ARMS))  # fmt: skip
        for r in sorted(set(rno)):
            s2 = np.flatnonzero(lgrp & (task == t) & (rno == r))
            if len(s2):
                print(f"    reach_no {r}: n {len(s2)}; " + "; ".join(f"{n} median {med(er_all[n][s2]):.2f} cm" for n in ARMS))
    # 6 range
    tpath = Path(D3.TABLE)
    table = json.loads(tpath.read_text())
    spec = ",".join(str(a.root / "data" / f) for f in ("d3.npz", "d3b.npz", "mid_data3.npz", "mid_data3b.npz"))
    DTr = DT.load(spec)
    tr = DT.splits(DTr)["train"]
    qt = SR.quantities(DTr, tr)
    lim = {g: (np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)) for g, v in qt.items()}
    X = np.asarray(DTr["x0"])[tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep_c = sd > 1e-6
    T = torch.tensor(((X - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
    trr = torch.tensor(np.asarray(DTr["reset"])[tr].astype(np.int64), device=dev)
    rng_nn = np.random.default_rng(0)
    pk = np.sort(rng_nn.choice(len(tr), size=min(SR.N_INTERNAL, len(tr)), replace=False))
    thr = float(np.percentile(SR.nn_dist(T[pk], trr[pk], T, trr), 99))
    print(f"\n== range (sd_range's checks): training set {len(tr)} segments ({len(np.unique(np.asarray(DTr['reset'])[tr]))} resets) from {spec}; "
          f"x0 {X.shape[1]} columns, {int(keep_c.sum())} not constant; nearest-neighbour p99 from {len(pk)} drawn; table {tpath} md5 "
          f"{hashlib.md5(tpath.read_bytes()).hexdigest()}")  # fmt: skip
    del X, DTr

    def assess(Dx, idx) -> dict:
        q = SR.quantities(Dx, idx)
        inr = {g: ((q[g] >= lim[g][0]) & (q[g] <= lim[g][1])).all(1) for g in q}
        x = np.asarray(Dx["x0"])[idx].astype(np.float64)
        Q = torch.tensor(((x - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
        rq = torch.tensor(np.asarray(Dx["reset"])[idx].astype(np.int64), device=dev)
        dd = SR.nn_dist(Q, rq, T, trr)
        sdv, pz, ct = np.asarray(Dx["side"])[idx], np.asarray(Dx["pelvis_z"])[idx], np.asarray(Dx["cmd_target"])[idx]
        tab = np.array([D3.in_table(table["rows"]["right" if s > 0 else "left"][D3.pz_key(None if p < 0 else float(p))], t)
                        for s, p, t in zip(sdv, pz, ct)], bool)  # fmt: skip
        c = dict(joints=inr["joints"], pelvis=inr["pelvis"], hands=inr["hands"], neighbour=dd <= thr, table=tab)
        c["range (39)"] = inr["joints"] & inr["pelvis"] & inr["hands"]
        c["all three"] = c["range (39)"] & c["neighbour"] & c["table"]
        return c

    av = assess(DV, vi)
    print("  HA val (reference) inside: " + ", ".join(f"{k} {100 * v.mean():.1f} %" for k, v in av.items()))
    out["range"] = dict(n_train=int(len(tr)), nn_p99=thr, table=str(tpath), val={k: float(v.mean()) for k, v in av.items()}, sets={})
    cir = np.asarray(D["cmd_in_range"]).astype(bool)
    for setname, mask in (("clean", clean), ("light-touch", lgrp)):
        idx = np.flatnonzero(mask)
        if not len(idx):
            print(f"  {setname}: no segments")
            continue
        chk = assess(D, idx)
        chk["cmd_in_range (hr_real)"] = cir[idx]
        G = groups_of(rs[idx])
        so = {}
        print(f"  {setname}: {len(idx)} reaches from {len(G)} episodes")
        for k, v in chk.items():
            bs = [v[pick(G, rng)].mean() for _ in range(NB)]
            o = dict(inside=float(v.mean()), inside_ci=(pct(bs, 2.5), pct(bs, 97.5)), n_in=int(v.sum()), n_out=int((~v).sum()))
            for n in ARMS:
                e = er_all[n][idx]
                o[f"{n} in"], o[f"{n} out"] = med(e[v]), med(e[~v])
                if not short:
                    o[f"{n} in ratio"], o[f"{n} out ratio"] = o[f"{n} in"] / val_med[n], o[f"{n} out"] / val_med[n]
            o["by"] = {}
            for t in sorted(set(task[idx])):
                for r in sorted(set(rno[idx])):
                    g = (task[idx] == t) & (rno[idx] == r)
                    if g.any():
                        o["by"][f"{t} {r}"] = dict(n=int(g.sum()), inside=float(v[g].mean()),
                                                   **{f"{n} in": med(er_all[n][idx][g & v]) for n in ARMS},
                                                   **{f"{n} out": med(er_all[n][idx][g & ~v]) for n in ARMS})  # fmt: skip
            so[k] = o
            rr = "" if short else " (" + ", ".join(f"{n} {o[n + ' in ratio']:.2f}x / {o[n + ' out ratio']:.2f}x" for n in ARMS) + ")"
            print(f"    {k:22s}: inside {100 * o['inside']:.1f} % [{100 * o['inside_ci'][0]:.1f}, {100 * o['inside_ci'][1]:.1f}] "
                  f"({o['n_in']} in / {o['n_out']} out); error in / out " + "; ".join(f"{n} {o[n + ' in']:.2f} / {o[n + ' out']:.2f} cm" for n in ARMS) + rr)  # fmt: skip
            if k in ("all three", "cmd_in_range (hr_real)"):
                print("      by task x reach_no: " + "; ".join(f"{b} n {x['n']} inside {100 * x['inside']:.0f} %" for b, x in o["by"].items()))
        out["range"]["sets"][setname] = so
    # 7 descriptive: contacts and the pelvis displacement, every merged segment
    n_ = np.asarray(D["n_steps"]).astype(int)
    so_ = np.asarray(D["so"]).astype(int)
    r0, r1 = so_ + np.asarray(D["k0"]).astype(int), so_ + n_ - 1
    W = np.asarray(D["W"]).astype(np.float64)
    dp = DT.hv(np.c_[W[r1, :2] - W[r0, :2], np.zeros(N)], W[r0, 3])[:, :2] * 100
    tn, tz = np.asarray(D["touch_n"]).astype(float), np.asarray(D["touch_z"]).astype(float)
    parts = np.array([x["part"] for x in info])
    print("\n== contacts and pelvis displacement (every merged segment; descriptive)")
    out["descriptive"] = {}
    for t in sorted(set(task)):
        for r in sorted(set(rno)):
            g = (task == t) & (rno == r)
            if not g.any():
                continue
            h = g & touching
            o = dict(n=int(g.sum()), kept=int((g & kp).sum()), touching=int(h.sum()), light=int((h & light).sum()), light_none=int((h & light_none).sum()),
                     parts={p: int((h & (parts == p)).sum()) for p in ("own hand", "other hand", "body")},
                     force_median=med(tn[h]), force_max=float(tn[h].max()) if h.any() else float("nan"), height_median=med(tz[h]),
                     fwd_median=med(dp[g, 0]), fwd_min=float(dp[g, 0].min()), fwd_max=float(dp[g, 0].max()),
                     lat_median=med(dp[g, 1]), lat_min=float(dp[g, 1].min()), lat_max=float(dp[g, 1].max()))  # fmt: skip
            out["descriptive"][f"{t} {r}"] = o
            print(f"  {t} reach_no {r}: n {o['n']} (kept {o['kept']}); touching {o['touching']} {o['parts']}, light-touch {o['light']} "
                  f"(no record {o['light_none']}); peak force median {o['force_median']:.1f} N, max {o['force_max']:.1f} N; touched body "
                  f"height median {o['height_median']:.2f} m; pelvis forward {o['fwd_median']:.1f} [{o['fwd_min']:.1f}, {o['fwd_max']:.1f}] cm, "
                  f"lateral {o['lat_median']:.1f} [{o['lat_min']:.1f}, {o['lat_max']:.1f}] cm")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=float))


def DT_kept(D: dict, sp: np.ndarray) -> np.ndarray:  # t3b_eval.kept
    se = np.asarray(D["settled_end"]) > 0 if "settled_end" in D else np.ones(len(D["n_steps"]), bool)
    return sp[(np.asarray(D["fell"])[sp] == 0) & se[sp]]


if __name__ == "__main__":
    main()
