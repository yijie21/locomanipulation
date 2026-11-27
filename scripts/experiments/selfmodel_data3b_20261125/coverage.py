"""Read-only coverage measurement for selfmodel_data3b_20261125's ranges (2026-11-25).

Real-task reaches (SIMPLE A / J-as-JR rows, ScaleBFM and SONIC; LML hardgrasp 'alone', ScaleBFM) against data3 smoke
(right-hand single-hand segments, optionally with mirrored left-hand ones), per item and jointly (nearest neighbour).

Conventions: hand point = data3's (wrist_yaw_link + 0.13 m along its x); SIMPLE's 0.0415 m point is moved there with the
wrist rotation. Positions in the pelvis heading frame at the command (x ahead, y left; z world). Joint order of S:
IsaacLab (checked against film qpos).

    python3 -I coverage.py <tmp dir>
"""

import glob
import json
import math
import sys

import numpy as np

T = sys.argv[1]
I2M = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
RARM = [I2M[j] for j in range(22, 29)]  # S indices of the right arm (sh pitch, roll, yaw, elbow, wr roll, pitch, yaw)
LARM = [I2M[j] for j in range(15, 22)]
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1.0])  # left arm -> right arm sign map
D_OFF = 0.13 - 0.0415


def rz(y):
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def r6R(v):
    a, b = np.asarray(v[:3], float), np.asarray(v[3:6], float)
    a = a / np.linalg.norm(a)
    b = b - a * (a @ b)
    b = b / np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], 1)


def qR(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ang(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def yaw_q(q):  # wxyz
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def hframe(pw, p, yaw):
    d = rz(yaw).T @ (np.asarray(pw, float) - np.asarray(p, float))
    return np.r_[d[:2], pw[2]]


# ---------------------------------------------------------------- data3 smoke
def data3(ctrl, mirror_left=False):
    z = np.load(f"{T}/d3/{ctrl}/all.npz")
    off = np.r_[0, np.cumsum(z["n_steps"])[:-1]]
    out, chk = [], []
    for s in range(len(off)):
        side = int(z["side"][s])
        if side != 1 and not mirror_left:
            continue
        h = 0 if side == 1 else 1
        c = off[s] + z["k0"][s]
        e = off[s] + z["n_steps"][s] - 1
        p, yaw = z["start"][s][:3], float(z["start"][s][3])
        Rh = rz(yaw)
        palm = z["KP"][c, h, 0]
        tw = np.r_[p[:2] + (Rh @ np.r_[z["target"][s][:2], 0])[:2], z["target"][s][2]]
        Rc = Rh @ r6R(z["rot6"][s])
        Rcur = r6R(z["J"][c])
        Rend = r6R(z["J"][e])
        chk.append((ang(Rend, Rc), float(z["att_err_deg"][s]), ang(Rcur, qR(z["HQ"][c, h]))))
        kp = z["KP"][c - 3: c + 1, h, 0]
        v = np.linalg.norm(np.diff(kp, axis=0), axis=1) / 0.1
        pv = np.linalg.norm(np.diff(z["W"][c - 3: c + 1, :3], axis=0), axis=1) / 0.1
        oh = 1 - h
        Ro = qR(z["HQ"][c, oh])
        other = z["KP"][c, oh, 0] - D_OFF * Ro[:, 0]  # the other hand's 0.0415 m point, as SIMPLE's W
        q = z["S"][c, RARM if side == 1 else LARM] * (1 if side == 1 else MIRROR)
        dvec = rz(yaw).T @ (tw - palm)
        r = dict(src="data3", ctrl=ctrl, side=side, chain=int(z["chain"][s]), walked=bool(z["walked"][s]),
                 palm_h=hframe(palm, p, yaw), pel_z=float(z["S"][c, 64]), v05=float(v.max()) * 100, v_win=0.3,
                 pel_v=float(pv.max()) * 100, dist=float(np.linalg.norm(tw - palm)) * 100, dvec=dvec,
                 att=ang(Rcur, Rc), ref_s=float(z["ref_s"][s]), q=q, other_h=hframe(other, p, yaw),
                 pz_cmd=float(z["pelvis_z"][s]))
        r["eff_v"] = r["dist"] / 100 / r["ref_s"]
        if side != 1:
            for k in ("palm_h", "other_h", "dvec"):
                r[k] = r[k] * np.array([1, -1, 1.0])
        out.append(r)
    return out, np.array(chk)


# ---------------------------------------------------------------- SIMPLE
def rows(path):
    return [json.loads(l) for l in open(path) if l.startswith("{")]


def film_ref_end(f, t_cmd, stage):
    """The reference's own duration: from the command to the last frame of this stage's phase where the reference
    joints still change (the hold after it and the next stage's PRE_S keep the reference still)."""
    t, ref, ph = f["t"], f["ref"], f["phase"]
    k0 = int(np.argmin(np.abs(t - t_cmd)))
    k1 = k0
    while k1 + 1 < len(t) and str(ph[k1 + 1]) == stage:
        k1 += 1
    j = ref[k0: k1 + 1, 7:36]
    moving = np.where(np.abs(np.diff(j, axis=0)).max(1) > 1e-5)[0]
    return float(t[k0 + moving[-1] + 1] - t_cmd) if len(moving) else 0.0, k0


def simple(ctrl, col):
    out = []
    src = "JR" if col == "J" else "A"
    tasks = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
    for task in tasks:
        A = {(r["level"], r["index"]): r for r in rows(f"{T}/simple/{ctrl}/rows/A_{task}.jsonl")}
        JR = {(r["level"], r["index"]): r for r in rows(f"{T}/simple/{ctrl}/rows/JR_{task}.jsonl")}
        for key, rj in JR.items():
            r = rj if col == "J" else A[key]
            routed = bool(rj.get("avoid") and rj["avoid"].get("n_via", 0) > 0)
            if len(r.get("stages", [])) < 2 or len(rj.get("stages", [])) < 2:
                continue
            fj = np.load(f"{T}/simple/{ctrl}/films_JR/{task}_L{key[0]}_e{key[1]}_perception_direct.npz")
            fa = np.load(f"{T}/simple/{ctrl}/films_A/{task}_L{key[0]}_e{key[1]}_perception_direct.npz")
            tr = r["trace"]
            tt = np.array([x[0] for x in tr])
            jaw = np.array([x[2] for x in tr]) / 1000.0
            ph = [x[1] for x in tr]
            pw_cmd = np.array(r["stages"][0]["goal_mm"]) / 1000  # the pregrasp command (direct: the goal itself)
            for si, stg in enumerate(("pregrasp", "entry")):
                s, sj = r["stages"][si], rj["stages"][si]
                exact = col == "J" or not routed or stg == "pregrasp"  # A == JR on unrouted episodes and at the pregrasp
                if exact:
                    t_cmd, W, Rw = sj["t_cmd"], np.array(sj["W"]), r6R(sj["J"])
                    S0 = np.array(sj["S3"][0])
                else:  # A's entry on a routed episode: trace / film / the pregrasp's settled wrist
                    t_first = tt[ph.index("entry")]
                    t_cmd = t_first + 0.28
                    k = int(np.argmin(np.abs(fa["t"] - t_cmd)))
                    q7 = fa["qpos"][k]
                    W = np.r_[q7[:3], yaw_q(q7[3:7]), np.full(9, np.nan)]
                    Rw = np.array(r["stages"][0]["rot_jaw"])
                    S0 = np.array(s["S0"] + [np.nan] * 36)
                p, yaw = W[:3], float(W[3])
                kt = int(np.argmin(np.abs(tt - t_cmd)))
                palm4 = jaw[kt] if not exact else W[4:7]
                palm = palm4 + D_OFF * Rw[:, 0]
                Rg = np.array(s["rot_want"])
                goal4 = np.array(s["goal_mm"]) / 1000
                tgt = goal4 + D_OFF * Rg[:, 0]
                win = (tt >= t_cmd - 0.5 - 1e-6) & (tt <= t_cmd + 1e-6)
                # 0.1 s displacements (5 ticks) inside the window (pregrasp: the trace starts 0.28 s before)
                idx = np.where(win)[0]
                vs = [np.linalg.norm(jaw[i] - jaw[i - 5]) / 0.1 for i in idx if i - 5 >= 0 and win[i - 5]]
                vwin = (tt[idx[-1]] - tt[idx[0]]) if len(idx) else 0
                # pelvis speed over 0.5 s from the JR / A film root (25 Hz)
                ff = fj if exact and col == "J" else fa
                kk = np.where((ff["t"] >= t_cmd - 0.5) & (ff["t"] <= t_cmd + 1e-6))[0]
                pv = [np.linalg.norm(ff["qpos"][i, :3] - ff["qpos"][i - 2, :3]) / (ff["t"][i] - ff["t"][i - 2])
                      for i in kk if i - 2 >= 0 and ff["t"][i - 2] >= t_cmd - 0.5]
                # reference duration: JR film's reference; A: straight reach at 0.15 m/s or the entry rule
                if col == "J":
                    ref_s, _ = film_ref_end(fj, t_cmd, stg)
                else:
                    if stg == "pregrasp":
                        ref_s = film_ref_end(fj, sj["t_cmd"], stg)[0] if not routed else np.nan
                    else:
                        ref_s = film_ref_end(fj, sj["t_cmd"], stg)[0] if not routed else np.nan
                if stg == "entry":
                    start_ref = palm4 + 0.5 * (pw_cmd - palm4)
                    ref_rule = float(np.linalg.norm(goal4 - start_ref)) / 0.08
                    if col == "A" and routed:
                        ref_s = ref_rule
                else:
                    ref_rule = np.nan
                other = W[7:10]
                d = dict(src="simple", ctrl=ctrl, col=col, task=task, ep=key, stage=stg, routed=routed, exact=exact,
                         palm_h=hframe(palm, p, yaw), pel_z=float(p[2]), v05=(max(vs) * 100 if vs else np.nan), v_win=vwin,
                         pel_v=(max(pv) * 100 if pv else np.nan), dist=float(np.linalg.norm(tgt - palm)) * 100,
                         dist4=float(np.linalg.norm(goal4 - palm4)) * 100, dvec=rz(yaw).T @ (tgt - palm),
                         att=ang(Rw, Rg), ref_s=ref_s, ref_rule=ref_rule, q=S0[RARM],
                         other_h=(hframe(other, p, yaw) if np.isfinite(other).all() else np.full(3, np.nan)),
                         pz_cmd=float(r.get("pz", -1)))
                d["eff_v"] = d["dist"] / 100 / ref_s if ref_s and np.isfinite(ref_s) and ref_s > 0 else np.nan
                out.append(d)
    # A's routed entries: the other hand from the pregrasp command of the same episode (the left arm is held)
    byep = {(d["task"], d["ep"], d["col"]): d for d in out if d["stage"] == "pregrasp"}
    for d in out:
        if not np.isfinite(d["other_h"]).all():
            d["other_h"] = byep[(d["task"], d["ep"], d["col"])]["other_h"]
            d["other_approx"] = True
    return out


# ---------------------------------------------------------------- LML hardgrasp
def lml(method="alone"):
    out = []
    for f in sorted(glob.glob(f"{T}/lml/episodes/*/{method}/*.json")):
        r = json.load(open(f))
        rs = r.get("reaches") or []
        byk = {w["k"]: w for w in rs if "f_end" in w}
        for k in (-1, 0, 1):
            w = byk.get(k)
            if w is None:
                continue
            prev = byk.get(k - 1)
            first = prev is None
            d = dict(src="lml", ctrl="scalebfm", task=r["task"], ep=r["ep"], k=k, first=first,
                     pel_z=float(w["p_cmd"][2]), pz_cmd=-1.0 if w["pz"] is None else float(w["pz"]),
                     ref_s=(w["f_end"] - w["f_cmd"]) / 50.0, q_out=w.get("q_out"), routed="via" in (r.get("plan") or {}) and
                     (r.get("plan") or {}).get("via") is not None)
            if prev is not None and "act_h" in prev:  # hand at this command = previous reach's settled hand
                p0, y0 = np.asarray(prev["p_cmd"]), float(prev["yaw_cmd"])
                hw = np.r_[p0[:2] + (rz(y0) @ np.r_[prev["act_h"][:2], 0])[:2], prev["act_h"][2]]
                p1, y1 = np.asarray(w["p_cmd"]), float(w["yaw_cmd"])
                d["palm_h"] = hframe(hw, p1, y1)
                gw = np.asarray(w["goal_w"])
                d["dist"] = float(np.linalg.norm(gw - hw)) * 100
                d["dvec"] = rz(y1).T @ (gw - hw)
                d["eff_v"] = d["dist"] / 100 / d["ref_s"]
            out.append(d)
    return out


# ---------------------------------------------------------------- summaries
def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return "  --"
    return f"{np.median(x):7.2f} [{np.percentile(x, 10):7.2f},{np.percentile(x, 90):7.2f}] n={len(x)}"


ITEMS = [("palm x", lambda d: d["palm_h"][0] * 100), ("palm y", lambda d: d["palm_h"][1] * 100),
         ("palm z", lambda d: d["palm_h"][2] * 100), ("pelvis z", lambda d: d["pel_z"] * 100),
         ("palm v max cm/s", lambda d: d["v05"]), ("pelvis v max cm/s", lambda d: d.get("pel_v", np.nan)),
         ("dist cm", lambda d: d["dist"]), ("wrist turn deg", lambda d: d["att"]), ("ref s", lambda d: d["ref_s"]),
         ("dist/ref m/s", lambda d: d["eff_v"])] + [
    (f"q{j} {n}", (lambda j: lambda d: math.degrees(d["q"][j]))(j))
    for j, n in enumerate(("shP", "shR", "shY", "elbow", "wrR", "wrP", "wrY"))] + [
    ("other x", lambda d: d["other_h"][0] * 100), ("other y", lambda d: d["other_h"][1] * 100),
    ("other z", lambda d: d["other_h"][2] * 100)]


def val(f, d):
    try:
        v = f(d)
        return float(v) if v is not None else np.nan
    except (KeyError, TypeError, IndexError):
        return np.nan


def inside(real, ref, f):
    rv = np.array([val(f, d) for d in real])
    xv = np.array([val(f, d) for d in ref])
    rv, xv = rv[np.isfinite(rv)], xv[np.isfinite(xv)]
    if not len(rv) or not len(xv):
        return np.nan
    lo, hi = np.percentile(xv, 1), np.percentile(xv, 99)
    return float(((rv >= lo) & (rv <= hi)).mean())


def feat(d, keys):
    v = []
    for k in keys:
        if k == "palm":
            v += list(d["palm_h"])
        elif k == "pel_z":
            v.append(d["pel_z"])
        elif k == "dvec":
            v += list(d["dvec"])
        elif k == "att":
            v.append(math.radians(d["att"]))
        elif k == "ref_s":
            v.append(d["ref_s"])
        elif k == "q":
            v += list(d["q"])
        elif k == "other":
            v += list(d["other_h"])
    return np.array(v, float)


def nn(real, ref, keys):
    X = np.array([feat(d, keys) for d in ref])
    Y = np.array([feat(d, keys) for d in real])
    X = X[np.isfinite(X).all(1)]
    Y = Y[np.isfinite(Y).all(1)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs, Ys = (X - mu) / sd, (Y - mu) / sd
    dxx = np.linalg.norm(Xs[:, None] - Xs[None], axis=2)
    np.fill_diagonal(dxx, np.inf)
    self_nn = dxx.min(1)
    dyx = np.linalg.norm(Ys[:, None] - Xs[None], axis=2).min(1)
    thr = np.percentile(self_nn, 90)
    return (f"real->data NN med {np.median(dyx):.2f} p90 {np.percentile(dyx, 90):.2f} | data self-NN med "
            f"{np.median(self_nn):.2f} p90 {thr:.2f} | ratio med {np.median(dyx) / np.median(self_nn):.2f} | "
            f"real beyond data p90 {np.mean(dyx > thr) * 100:.0f}% (n real {len(Y)}, n data {len(X)}, dims {X.shape[1]})")


def main():
    D3 = {}
    for c in ("scalebfm", "heft", "sonic"):
        D3[c], chk = data3(c)
        D3[c + "+L"], _ = data3(c, mirror_left=True)
        print(f"data3 {c}: right {len(D3[c])} segs, +mirrored left {len(D3[c + '+L'])}; att check |end-angle - att_err| "
              f"med {np.median(np.abs(chk[:, 0] - chk[:, 1])):.2f} deg; J vs HQ med {np.median(chk[:, 2]):.2f} deg")
    SM = {(c, col): simple(c, col) for c in ("scalebfm", "sonic") for col in ("A", "J")}
    LM = lml("alone")
    # checks
    for (c, col), L in SM.items():
        e = [d for d in L if d["stage"] == "entry" and np.isfinite(d["ref_rule"]) and np.isfinite(d["ref_s"]) and d["exact"]]
        if e:
            dd = np.array([d["ref_s"] - d["ref_rule"] for d in e])
            print(f"check {c} {col}: entry ref duration film - rule: med {np.median(dd):.3f} s, p90 |.| "
                  f"{np.percentile(np.abs(dd), 90):.3f} s (n {len(e)})")
    groups = []
    for c in ("scalebfm", "sonic"):
        for col in ("A", "J"):
            for stg in ("pregrasp", "entry"):
                groups.append((f"SIMPLE {c} {col} {stg}", [d for d in SM[(c, col)] if d["stage"] == stg], c))
        for task in ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"):
            for stg in ("pregrasp", "entry"):
                groups.append((f"SIMPLE {c} A+J {task} {stg}",
                               [d for col in ("A", "J") for d in SM[(c, col)] if d["stage"] == stg and d["task"] == task], c))
    groups.append(("LML scalebfm entry (k=1)", [d for d in LM if d["k"] == 1], "scalebfm"))
    groups.append(("LML scalebfm pregrasp after via (k=0)", [d for d in LM if d["k"] == 0 and not d["first"]], "scalebfm"))
    groups.append(("LML scalebfm first reach", [d for d in LM if d["first"]], "scalebfm"))
    for c in ("scalebfm", "heft", "sonic"):
        groups.append((f"DATA3 {c} right, all", D3[c], c))
        groups.append((f"DATA3 {c} right, chain0", [d for d in D3[c] if d["chain"] == 0], c))
        groups.append((f"DATA3 {c} right, chain1", [d for d in D3[c] if d["chain"] == 1], c))
        groups.append((f"DATA3 {c} right+mirrored left", D3[c + "+L"], c))
    for name, G, c in groups:
        print(f"\n== {name}  (n {len(G)})")
        for iname, f in ITEMS:
            vals = [val(f, d) for d in G]
            if not np.isfinite(vals).any():
                continue
            line = f"  {iname:18s} {pct(vals)}"
            if not name.startswith("DATA3"):
                line += f"   inside data3[p1,p99]: R {inside(G, D3[c], f) * 100:5.1f}%  R+L {inside(G, D3[c + '+L'], f) * 100:5.1f}%"
            print(line)
        if not name.startswith("DATA3"):
            settled = [d["v05"] < 2.0 for d in G if np.isfinite(d.get("v05", np.nan))]
            if settled:
                print(f"  settled (<2 cm/s) {np.mean(settled) * 100:.0f}%")
    print("\n== nearest neighbour (standardized by data3)")
    full = ["palm", "pel_z", "dvec", "att", "ref_s", "q"]
    red = ["palm", "pel_z", "dvec", "ref_s"]
    for c in ("scalebfm", "sonic"):
        for col in ("A", "J"):
            for stg in ("pregrasp", "entry"):
                G = [d for d in SM[(c, col)] if d["stage"] == stg]
                for ref in (c, c + "+L"):
                    print(f"  SIMPLE {c} {col} {stg} vs {ref} full: {nn(G, D3[ref], full)}")
                print(f"  SIMPLE {c} {col} {stg} vs {c} reduced: {nn(G, D3[c], red)}")
    G = [d for d in LM if d["k"] == 1]
    for ref in ("scalebfm", "scalebfm+L"):
        print(f"  LML entry vs {ref} reduced: {nn(G, D3[ref], red)}")
    # data3's own chain1 against chain0+1 is the self-NN above; also: data3 segments starting within 15 / 25 cm
    for c in ("scalebfm", "heft", "sonic"):
        dd = np.array([d["dist"] for d in D3[c + "+L"]])
        at = np.array([d["att"] for d in D3[c + "+L"]])
        print(f"  DATA3 {c} R+L: dist<15 {np.mean(dd < 15) * 100:.1f}%  <25 {np.mean(dd < 25) * 100:.1f}%  att<15 "
              f"{np.mean(at < 15) * 100:.1f}%  dist<25&att<15 {np.mean((dd < 25) & (at < 15)) * 100:.1f}%")
    np.save(f"{T}/coverage_samples.npy", np.array(dict(SM={f'{k[0]}_{k[1]}': v for k, v in SM.items()}, LML=LM, D3=D3),
                                                     dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
