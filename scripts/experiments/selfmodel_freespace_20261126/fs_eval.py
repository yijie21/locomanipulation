"""selfmodel_freespace_20261126: the entry's metrics, cell verdicts and criteria for one controller (read-only).

    cd $G; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 PYTHONPATH=$G:$W/src FS_OUT=$O \
        $PY fs_eval.py --backend scalebfm [--json F] [--no-start]

Pairs: in a cell, an episode id with both groups' records valid -- no worker error, no fall / lost, a landing that
settled (label_kind 0, settled_end). Differences X - A of the landing error (cm), per cell: median and lc3_eval's
median_ci (sign-test order statistics, 95 %).
Cell verdicts (on R - A), in this order: 局数不够 (< MIN_PAIRS valid pairs); 无余地 (A's median over the valid pairs
< 2 cm); 有效 (median <= -1 cm and the interval's upper end < 0); 有害 (median >= +1 cm and the lower end > 0);
分不开 (the rest).
Labels: IK (grid.json: reach / edge / out), target coverage (grid.json, this controller: 有 / 稀), start coverage
(per A episode, simpledata's evaluation 4 on model D's training split -- the 39 range quantities at the start row inside
the training set's p0.5..p99.5; the standardized-x0 nearest training segment of another reset not farther than the
training set's own p99 (10 000 drawn, seed 0); the command inside the reach table for its pelvis class -- all three; a
cell is "在" when >= 80 % of its A episodes with a segment pass). In the training range = 有 and 在.
Main set: IK reach, in the training range, >= MIN_PAIRS valid pairs, not 无余地.
Criteria (revision 1's mechanics for the entry's table): see verdict().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_run as FR  # noqa: E402
import fs_grid as FG  # noqa: E402
import fs_load as FL  # noqa: E402

LC, D3, L, LT = FR.LC, FR.D3, FR.L, FR.LT
MIN_PAIRS = 10
EFF, NOROOM = -1.0, 2.0
START_SHARE = 0.8
N_INTERNAL = 10_000


def median_ci(d: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """lc3_eval.median_ci (copied: lc3_eval imports scipy at the top level only through this)."""
    d = np.sort(d)
    n = len(d)
    if n < 6:
        return float("nan"), float("nan")
    from scipy.stats import binom

    k = int(binom.ppf((1 - conf) / 2, n, 0.5))
    while k > 0 and binom.cdf(k - 1, n, 0.5) > (1 - conf) / 2:
        k -= 1
    k = max(k, 1)
    return float(d[k - 1]), float(d[n - k])


def valid(r: dict | None) -> bool:
    if r is None or "error" in r or r.get("err_cm") is None or r.get("fell") is not None:
        return False
    w = r.get("reach") or {}
    return w.get("label_kind") == 0 and bool(w.get("settled_end"))


# -- start coverage (simpledata's evaluation 4 on LocoManiLab's D) -------------------------------------------------------
def start_cover(backend: str, z: dict, idx: np.ndarray) -> dict:
    import torch

    sys.path[:0] = [str(FG.EXP / "reachwm_20261030")]
    import data as DT

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    table = json.loads(D3.TABLE.read_text())
    DTr = DT.load(FG.d_spec(backend))
    tr = DT.splits(DTr, 1.0)["train"]

    def quantities(D: dict, ii: np.ndarray) -> dict:
        n = np.asarray(D["n_steps"]).astype(int)
        so = np.r_[0, np.cumsum(n)[:-1]]
        r = so[ii] + np.asarray(D["k0"])[ii].astype(int)
        S, W = np.asarray(D["S"]), np.asarray(D["W"])
        st = np.asarray(D["start"])[ii].astype(np.float64)
        yaw, org = st[:, 3], np.c_[st[:, :2], np.zeros(len(ii))]
        hands = [DT.hv(W[r, a:b].astype(np.float64) - org, yaw) for a, b in ((4, 7), (7, 10))]
        return dict(joints=S[r, 0:29].astype(np.float64), pelvis=np.c_[S[r, 64], S[r, 58:61]].astype(np.float64), hands=np.concatenate(hands, 1))

    qt = quantities(DTr, tr)
    lim = {g: (np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)) for g, v in qt.items()}
    X = np.asarray(DTr["x0"])[tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep_c = sd > 1e-6
    T = torch.tensor(((X - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32)
    rtr = torch.tensor(np.asarray(DTr["reset"])[tr].astype(np.int64))
    del X, DTr

    def nn(Q, qr, chunk_q=1024, chunk_t=200_000):
        out = []
        tn = (T * T).sum(1)
        for i in range(0, len(Q), chunk_q):
            q, rq = Q[i : i + chunk_q], qr[i : i + chunk_q]
            qn = (q * q).sum(1)
            best = torch.full((len(q),), float("inf"))
            for j in range(0, len(T), chunk_t):
                d2 = qn[:, None] + tn[None, j : j + chunk_t] - 2 * q @ T[j : j + chunk_t].T
                d2 = d2.masked_fill(rq[:, None] == rtr[None, j : j + chunk_t], float("inf"))
                best = torch.minimum(best, d2.min(1).values)
            out.append(best.clamp_min(0).sqrt().numpy())
        return np.concatenate(out)

    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(tr), size=min(N_INTERNAL, len(tr)), replace=False))
    thr = float(np.percentile(nn(T[pick], rtr[pick]), 99))
    q = quantities(z, idx)
    inr = {g: ((q[g] >= lim[g][0]) & (q[g] <= lim[g][1])).all(1) for g in q}
    x = np.asarray(z["x0"])[idx].astype(np.float64)
    Q = torch.tensor(((x - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32)
    d = nn(Q, torch.full((len(idx),), -1, dtype=torch.int64))
    pz, ct = np.asarray(z["pelvis_z"])[idx], np.asarray(z["cmd_target"])[idx]
    tab = np.array([D3.in_table(table["rows"]["right"][D3.pz_key(None if p < 0 else float(p))], t) for p, t in zip(pz, ct)])
    ok = inr["joints"] & inr["pelvis"] & inr["hands"] & (d <= thr) & tab
    return dict(ok=ok, joints=inr["joints"], pelvis=inr["pelvis"], hands=inr["hands"], neighbour=d <= thr, table=tab, nn=d, nn_p99=thr, n_train=int(len(tr)))


# -- per cell ------------------------------------------------------------------------------------------------------------
def cell_stats(E: dict, cid: int, eps: list[int], X: str, base: str = "alone") -> dict:
    a, b = [], []
    n_run = 0
    for e in eps:
        ra, rb = E.get((cid, e, base)), E.get((cid, e, X))
        if ra is None or rb is None:
            continue
        n_run += 1
        if valid(ra) and valid(rb):
            a.append(ra["err_cm"])
            b.append(rb["err_cm"])
    a, b = np.array(a), np.array(b)
    d = b - a
    out = dict(n_run=n_run, n=len(d))
    if len(d):
        lo, hi = median_ci(d)
        out.update(base_med=float(np.median(a)), x_med=float(np.median(b)), diff_med=float(np.median(d)), ci=(lo, hi),
                   base_lt2=float((a < 2).mean()), base_lt5=float((a < 5).mean()), x_lt2=float((b < 2).mean()), x_lt5=float((b < 5).mean()),
                   d=d.tolist(), a=a.tolist(), b=b.tolist())  # fmt: skip
    return out


def verdict_cell(s: dict) -> str:
    if s["n"] < MIN_PAIRS:
        return "局数不够"
    if s["base_med"] < NOROOM:
        return "无余地"
    lo, hi = s["ci"]
    if s["diff_med"] <= EFF and hi < 0:
        return "有效"
    if s["diff_med"] >= -EFF and lo > 0:
        return "有害"
    return "分不开"


def verdict(main: list[dict], cand: list[dict]) -> dict:
    """The entry's criteria table on the main set (main) and the reachable, in-range cells (cand: main + the cells left out
    for too few pairs or for no room). Mechanics fixed in revision 1:
      - a level (height, distance, direction, posture) with no main-set cell counts as not >= 60 %;
      - 证据不足: cells with too few pairs > 20 % of cand -> no other verdict;
      - otherwise 窄 (effective < 50 %, or every effective cell in two adjacent heights, or in one posture), else 部分
        (effective < 80 %, or a level < 60 %, or any harmful cell), else 广泛; 广泛 drops to 部分 when the main set has
        < 30 cells or a posture has < 5 main-set cells;
      - flags beside the verdict: 有害 >= 10 % of the main set; 无余地 >= 20 % of cand (excluding too-few-pairs cells)."""
    n = len(main)
    eff = [c for c in main if c["verdict"] == "有效"]
    harm = [c for c in main if c["verdict"] == "有害"]
    share = len(eff) / n if n else float("nan")
    levels = {}
    for key, vals in (("z", FG.Z), ("r", FG.R), ("th", FG.TH), ("posture", FG.POSTURES)):
        for v in vals:
            sel = [c for c in main if c[key] == v]
            levels[f"{key}={v}"] = (len(sel), (sum(c["verdict"] == "有效" for c in sel) / len(sel)) if sel else None)
    low = [k for k, (m, s) in levels.items() if s is None or s < 0.6]
    few = [c for c in cand if c["verdict"] == "局数不够"]
    noroom = [c for c in cand if c["verdict"] == "无余地"]
    enough = [c for c in cand if c["verdict"] != "局数不够"]
    iz = sorted({FG.Z.index(c["z"]) for c in eff})
    squeezed = bool(eff) and iz[-1] - iz[0] <= 1
    one_posture = bool(eff) and len({c["posture"] for c in eff}) == 1
    post_n = {p: sum(c["posture"] == p for c in main) for p in FG.POSTURES}
    flags = []
    if cand and len(few) / len(cand) > 0.2:
        v = "证据不足"
    elif not n:
        v = "主集合为空"
    elif share < 0.5 or squeezed or one_posture:
        v = "只在窄范围内成立"
    elif share < 0.8 or low or harm:
        v = "部分成立"
    else:
        v = "广泛成立"
    if v == "广泛成立" and (n < 30 or min(post_n.values()) < 5):
        v = "部分成立"
        flags.append("主集合 < 30 格或某姿态 < 5 格：不下广泛成立")
    if n and len(harm) / n >= 0.1:
        flags.append("在这些位置有害（有害格 >= 主集合的 10 %）")
    if enough and len(noroom) / len(enough) >= 0.2:
        flags.append("直接执行本来就很准（无余地 >= 20 %）")
    return dict(verdict=v, flags=flags, n_main=n, n_eff=len(eff), n_harm=len(harm), share_eff=share, levels=levels, low_levels=low,
                squeezed=squeezed, one_posture=one_posture, posture_main=post_n, n_cand=len(cand), n_few=len(few), n_noroom=len(noroom))  # fmt: skip


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f} %"


def main(backend: str, out_json: Path | None, do_start: bool) -> None:
    base = FR.OUT / backend
    E, dup = FL.runs(base)
    gridd = json.loads(FR.GRID.read_text())
    grid = {r["cid"]: r for r in gridd["cells"]}
    groups = FR.GROUPS[backend]
    eps = sorted({k[1] for k in E})
    print(f"{backend}: {len(E)} records ({sum('error' in r for r in E.values())} worker errors, {len(dup)} duplicates); episode ids "
          f"{eps[0] if eps else '-'}..{eps[-1] if eps else '-'} ({len(eps)}); groups {groups}; grid {FR.GRID} ({gridd['made_utc']})")  # fmt: skip
    # per group: falls, lost, unsettled
    for m in groups:
        rs = [r for k, r in E.items() if k[2] == m]
        nf = sum(r.get("fell") is not None and r.get("lost") is None for r in rs)
        nl = sum(r.get("lost") is not None for r in rs)
        nu = sum(r.get("err_cm") is not None and r.get("fell") is None and not valid(r) for r in rs)
        ne = sum("error" in r for r in rs)
        print(f"  {m}: {len(rs)} episodes; fell {nf} ({pct(nf / max(len(rs), 1))}), lost {nl} ({pct(nl / max(len(rs), 1))}), landing not settled {nu}, errors {ne}")
    # start coverage on A's segments
    start = {}
    if do_start:
        z, _ = FL.segs(base)
        a_id = LC.METHODS.index("alone")
        idx = np.where(np.asarray(z["method_id"]) == a_id)[0]
        sc = start_cover(backend, z, idx)
        for j, i in enumerate(idx):
            start[(int(z["cell"][i]), int(z["ep"][i]))] = {k: bool(sc[k][j]) for k in ("ok", "joints", "pelvis", "hands", "neighbour", "table")}
        print(f"start coverage: {len(idx)} A segments, training split {sc['n_train']} segments, nearest-neighbour p99 {sc['nn_p99']:.3f}; inside: "
              + ", ".join(f"{k} {pct(float(np.mean(sc[k])))}" for k in ("joints", "pelvis", "hands", "neighbour", "table", "ok")))  # fmt: skip
        for ip, p in enumerate(FG.POSTURES):
            v = [s["ok"] for (c, e), s in start.items() if FG.cell(c)["posture"] == p]
            print(f"  {p}: {len(v)} A episodes, all three inside {pct(float(np.mean(v)) if v else None)}")
    # cells
    cells = []
    for c in FG.cells():
        g = grid[c["cid"]]
        s = cell_stats(E, c["cid"], eps, "reachable")
        s_eps = [start[(c["cid"], e)]["ok"] for e in eps if (c["cid"], e) in start]
        st_share = float(np.mean(s_eps)) if s_eps else None
        row = dict(cid=c["cid"], posture=c["posture"], z=c["z"], r=c["r"], th=c["th"], support=g["support"], ik=g["ik_label"], ik_err_cm=g["ik_err_cm"],
                   pitch=g["pitch_deg"], cover_n=g["cover"][backend]["n"], cover=g["cover"][backend]["label"], start_share=st_share,
                   start=None if st_share is None else ("在" if st_share >= START_SHARE else "不在"), R=s)  # fmt: skip
        row["in_range"] = row["cover"] == "有" and row["start"] == "在"
        row["verdict"] = verdict_cell(s)
        if "selfmodel" in groups:
            row["S"] = cell_stats(E, c["cid"], eps, "selfmodel")
            row["S_verdict"] = verdict_cell(row["S"])
            row["RS"] = cell_stats(E, c["cid"], eps, "reachable", base="selfmodel")
        # R's corrections and shrinks, R == A episodes
        corr, nsh, same = [], 0, 0
        for e in eps:
            rr, ra = E.get((c["cid"], e, "reachable")), E.get((c["cid"], e, "alone"))
            if rr is None or "error" in rr or "reach" not in rr or "cmd_h" not in (rr.get("reach") or {}):
                continue
            w = rr["reach"]
            corr.append(float(np.linalg.norm(np.asarray(w["cmd_h"]) - np.asarray(rr["goal_h"]))) * 100)
            nsh += sum(bool(x.get("shrunk")) for x in w.get("shrink", []))
            if ra is not None and "reach" in ra and (ra.get("reach") or {}).get("cmd_h") == w["cmd_h"] and ra.get("err_cm") == rr.get("err_cm"):
                same += 1
        row.update(corr_med_cm=float(np.median(corr)) if corr else None, n_shrunk=nsh, n_r_eq_a=same, n_r=len(corr))
        cells.append(row)
    main_set = [c for c in cells if c["ik"] == "reach" and c["in_range"] and c["verdict"] in ("有效", "有害", "分不开")]
    cand = [c for c in cells if c["ik"] == "reach" and c["in_range"]]
    V = verdict(main_set, cand)
    # tables
    print("\n== cells (R - A): posture z r th | support IK cover(n) start | pairs A R diff [CI] verdict")
    for c in cells:
        s = c["R"]
        dm = f"{s['base_med']:.2f} {s['x_med']:.2f} {s['diff_med']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]" if s["n"] else "-"
        print(f"  {c['cid']:3d} {c['posture']:10s} {c['z']:.2f} {c['r']:.2f} {c['th']:+4.0f} | {c['support']:3s} {c['ik']:5s} {c['cover']}({c['cover_n']}) "
              f"{c['start'] or '-'}({pct(c['start_share'])}) | {s['n']}/{s['n_run']} {dm} {c['verdict']}"
              + (f" | S-A {c['S']['diff_med']:+.2f} {c['S_verdict']}, R-S {c['RS']['diff_med']:+.2f} [{c['RS']['ci'][0]:+.2f}, {c['RS']['ci'][1]:+.2f}]" if "S" in c and c["S"]["n"] and c["RS"]["n"] else "")
              + f" | corr {c['corr_med_cm'] if c['corr_med_cm'] is None else round(c['corr_med_cm'], 1)} shrunk {c['n_shrunk']} R=A {c['n_r_eq_a']}/{c['n_r']}")  # fmt: skip
    print(f"\n== main set: {V['n_main']} cells (reachable and in range: {V['n_cand']}; of those too few pairs {V['n_few']}, no room {V['n_noroom']})")
    print(f"  effective {V['n_eff']} ({pct(V['share_eff'])}), harmful {V['n_harm']}; per posture in the main set {V['posture_main']}")
    print("  levels: " + "; ".join(f"{k} {m} cells {pct(s)}" for k, (m, s) in V["levels"].items()))
    for sup in ("C", "W"):
        sel = [c for c in main_set if c["support"].startswith(sup)]
        print(f"  support {sup}: {len(sel)} cells, effective {pct(sum(c['verdict'] == '有效' for c in sel) / len(sel) if sel else None)}")
    pa = np.concatenate([c["R"]["a"] for c in main_set]) if main_set else np.zeros(0)
    pb = np.concatenate([c["R"]["b"] for c in main_set]) if main_set else np.zeros(0)
    if len(pa):
        d = pb - pa
        lo, hi = median_ci(d)
        print(f"  pooled ({len(d)} pairs): A median {np.median(pa):.2f} cm (<2 {pct((pa < 2).mean())}, <5 {pct((pa < 5).mean())}); R median "
              f"{np.median(pb):.2f} cm (<2 {pct((pb < 2).mean())}, <5 {pct((pb < 5).mean())}); R - A median {np.median(d):+.2f} [{lo:+.2f}, {hi:+.2f}]")  # fmt: skip
        for p in FG.POSTURES:
            sel = [c for c in main_set if c["posture"] == p]
            if sel:
                bb = np.concatenate([c["R"]["b"] for c in sel])
                print(f"    {p}: {len(sel)} cells, effective {pct(sum(c['verdict'] == '有效' for c in sel) / len(sel))}, R median {np.median(bb):.2f} cm")
    print(f"  VERDICT: {V['verdict']}" + (f"; flags: {V['flags']}" if V["flags"] else "") + (f"; levels below 60 % or empty: {V['low_levels']}" if V["low_levels"] else ""))
    # outside the main set
    print("\n== outside the main set")
    ikx = [c for c in cells if c["ik"] != "reach"]
    nr = sum(c["n_r"] for c in ikx)
    print(f"  IK edge / out: {len(ikx)} cells; R = A bit for bit {sum(c['n_r_eq_a'] for c in ikx)}/{nr} episodes; verdicts "
          f"{dict(sorted(defaultdict(int, {v: sum(c['verdict'] == v for c in ikx) for v in {c['verdict'] for c in ikx}}).items()))}")  # fmt: skip
    oor = [c for c in cells if c["ik"] == "reach" and not c["in_range"]]
    oor_j = [c for c in oor if c["verdict"] in ("有效", "有害", "分不开")]
    so_eff = sum(c["verdict"] == "有效" for c in oor_j) / len(oor_j) if oor_j else None
    print(f"  reachable, not in range: {len(oor)} cells (target 稀 {sum(c['cover'] == '稀' for c in oor)}, start 不在 "
          f"{sum(c['start'] == '不在' for c in oor)}); judged {len(oor_j)}, effective {pct(so_eff)}")  # fmt: skip
    if so_eff is not None and V["n_main"]:
        m_eff = V["share_eff"]
        oo = "放宽到这些格子" if so_eff >= m_eff - 0.1 else "限定在训练范围内" if so_eff < m_eff / 2 else "照实报，不放宽"
        print(f"    against the main set's {pct(m_eff)}: {oo}")
    print(f"  too few pairs (reachable, in range): {V['n_few']}; no room: {V['n_noroom']}")
    if "selfmodel" in groups:
        js = [c for c in main_set if c["S"]["n"] >= MIN_PAIRS]
        same = 0
        tot = 0
        for c in main_set:
            for e in eps:
                rs_, rr_ = E.get((c["cid"], e, "selfmodel")), E.get((c["cid"], e, "reachable"))
                if rs_ and rr_ and "cmd_h" in (rs_.get("reach") or {}) and "cmd_h" in (rr_.get("reach") or {}):
                    tot += 1
                    same += rs_["reach"]["cmd_h"] == rr_["reach"]["cmd_h"]
        allrs = [c for c in cells if c["RS"]["n"] >= MIN_PAIRS]
        rb = [c for c in allrs if c["RS"]["ci"][1] < 0]
        sb = [c for c in allrs if c["RS"]["ci"][0] > 0]
        s_eff = sum(c["S_verdict"] == "有效" for c in js) / len(js) if js else None
        print(f"\n== ScaleBFM's S: main set S = R commands {same}/{tot} ({pct(same / tot if tot else None)}); R better than S (interval < 0) "
              f"{len(rb)}/216 cells ({pct(len(rb) / 216)}), S better {len(sb)} ({pct(len(sb) / 216)}); S - A effective in the main set {pct(s_eff)} "
              f"(R - A {pct(V['share_eff'])}); R better than S at {sorted((c['posture'], c['z'], c['r'], c['th']) for c in rb)[:20]}")  # fmt: skip
    if out_json:
        out_json.write_text(json.dumps(dict(backend=backend, verdict=V, cells=cells), default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="scalebfm")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-start", action="store_true")
    a = ap.parse_args()
    main(a.backend, a.json, not a.no_start)
