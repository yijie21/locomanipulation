"""selfmodel_rootframe_20261125: the offline gate of the self model retrained in the motor's reference-root frame (mR,
jref_data.py build --frame root, via_point.py train), on SIMPLE episodes already run. Nothing is run or replanned: J's rows
(the measured jaw at the pregrasp settle) and the command states (s_check.py state of J's filmed run) are read.

With jw the jaw centre at the end of J's reference placed by the motor's own root ref0[0:7] (precision.ref_root_end) and the
reference root's heading frame at the command (x ahead; precision.to_head with that root's yaw):
  er = jw - the look's goal (precision3's), t = the measured jaw centre - jw (what is left after er), both cm.
Each predictor's offline miss is |t - its predicted t| (where the hand would stop had the goal been moved by it and the
execution added nothing). The predictors:
  c_rr     precision3's constant (ctrack_rr_<ctrl>.json; development episodes: without the episode's own position, by name)
  aff      t = a + B [er, goal], the goal the look's minus the root (heading frame), one ridge per axis on the development
           episodes (precision's 4 repeats of the second development set), lambda by leave-one-layout there (`aff`, frozen
           before mR is trained; development episodes: their leave-one-layout predictions)
  aff_far  the same on the development positions whose look goals lie over --far-cm from every test position's (the
           entry's 5 cm, revised to 1 cm on 11-25: no position is 5 cm away)
  old      the jrefwm model's implied t, p - u (p its predicted palm offset, u = jw - the reference end carried onto the
           true pelvis; precision3's review)
  mR       the new model's predicted settled palm offset (the mean of the palm's three spheres, as modelf takes it), its input
           from the robot's own readings (jref_data.Belief) and the reference rebuilt from J's plan as s_check.reference
           builds it, in the root's frame

    cd $REPO && SIMPLE_ROOT=... REACH_HAND_OFF=0.0415,0,0 PYTHONPATH=src $PY -I rootframe_eval.py aff <ctrl> <out dir>
        -> <out dir>/aff/aff_rr_<ctrl>.json, aff_rr_far_<ctrl>.json
    ... rootframe_eval.py aff <ctrl> <out dir> --far-cm 1      (aff_rr is kept as frozen; only the far file is new)
    ... rootframe_eval.py table <out dir> [scalebfm sonic] [--model <mR dir>] > <out dir>/table.txt
        (also <out dir>/table_rows_<ctrl>.jsonl, one line per episode-repeat, and range_<ctrl>.txt)
Sets: test (precision's jittered test episodes, 4 repeats: the main set), dev (its development episodes, 4 repeats) and jr
(the official JR episodes, bench_rgbd_20261124; command states s_check.py state into <out dir>/states_jr_<ctrl>). Layouts
as precision.layouts (look goals within 1 cm), the bootstrap and the layout-mean sign test precision.by_layout's, plus a
sign test on the layouts' medians. The checks (<out dir>/check_<ctrl>/<set>[_r<k>]: jref_data.py build --frame root of J's
films) give the model's own error (the filmed settled palm offset against mR's), range.txt the joints' range.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import precision as PR  # noqa: E402
import s_check as SC  # noqa: E402

JD, VP, VM = SC.JD, SC.VP, SC.VM
TEST = Path("outputs/simple_eval_20261112/precision_20261124")
DEV = Path("outputs/simple_dev_20261124/precision_20261124")
JRB = Path("outputs/simple_eval_20261112/bench_rgbd_20261124")
SFULL = Path("outputs/simple_eval_20261112/sfull_20261125")
CONST = Path("outputs/simple_dev_20261124/precision_const")
OLD = Path("outputs/simple_eval_20261112/jrefwm_20261124")
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)  # ridge penalties, standardized features
FAR_M = 0.05


def md5(p) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def episodes(fk, ctrl: str, kind: str, out: Path) -> list[dict]:
    """One dict per routed episode-repeat of the set: its command state, J's plan, the references and the frame terms."""
    if kind == "jr":
        reps = [(0, out / f"states_jr_{ctrl}", PR.rows_of(JRB / ctrl, "JR"), SFULL / ctrl / "P" / "shifts_sf.jsonl")]
    else:
        base = TEST if kind == "test" else DEV
        reps = [(k, base / ctrl / f"r{k}" / "P", PR.rows_of(base / ctrl / f"r{k}", "JR"), base / ctrl / f"r{k}" / "P" / "shifts3.jsonl")
                for k in range(4)]  # fmt: skip
    E = []
    for rep, P, R, shf in reps:
        sh = {json.loads(x)["name"]: json.loads(x) for x in open(shf)}
        names = sorted(Path(f).stem[len("state_") :] for f in glob.glob(str(P / "state_*.npz")))
        assert names == sorted(sh), (kind, rep, "the states are not the routed episodes")
        for name in names:
            st = np.load(P / f"state_{name}.npz")
            Q = np.load(Path(str(st["plan_src"])) / f"plan_{name}.npz")["Q"]
            ref, total, _, end = SC.reference(fk, st, Q)
            jw, yaw_rr = PR.ref_root_end(fk, st, Q, end)
            pj = PR.pregrasp(R[name])
            goal, jaw = np.asarray(pj["goal_mm"], float) / 1000, np.asarray(pj["jaw_mm"], float) / 1000
            root = np.asarray(st["ref0"][0:7], float)
            E.append(dict(set=kind, rep=rep, name=name, goal=goal, st=st, Q=Q, ref=ref, total=total, end=end, jw=jw,
                          yaw_rr=yaw_rr, yaw_W=float(st["W"][3]), root=root,
                          er=100 * PR.to_head(jw - goal, yaw_rr), t=100 * PR.to_head(jaw - jw, yaw_rr),
                          goal_h=100 * PR.to_head(goal - root[0:3], yaw_rr), u=100 * PR.to_head(jw - end[0], yaw_rr),
                          g=100 * PR.to_head(end[0] - goal, float(st["W"][3])), sh=sh[name]))  # fmt: skip
    return E


def layout_ids(E: list[dict]) -> np.ndarray:
    goals = {}
    for e in E:
        goals.setdefault(e["name"], []).append(e["goal"])
    lay = PR.layouts({k: np.mean(v, 0) for k, v in goals.items()})
    return np.array([lay[e["name"]] for e in E])


# --- the development affine fits -------------------------------------------------------------------------------------


def ridge(X, Y, lam):
    """per output column: Y ~ a + X b, X standardized here, the intercept free -> (mu, sd, b (d, k), a (k,))."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    ym = Y.mean(0)
    b = np.zeros((X.shape[1], Y.shape[1]))
    for k in range(Y.shape[1]):
        b[:, k] = np.linalg.solve(Z.T @ Z + lam[k] * np.eye(X.shape[1]), Z.T @ (Y[:, k] - ym[k]))
    return mu, sd, b, ym


def ridge_predict(fit, X):
    mu, sd, b, ym = fit
    return ym + ((X - mu) / sd) @ b


def lol(X, Y, lay, lam) -> np.ndarray:
    """leave-one-layout predictions with penalties lam (per output)."""
    P = np.zeros_like(Y)
    for L in np.unique(lay):
        m = lay == L
        P[m] = ridge_predict(ridge(X[~m], Y[~m], lam), X[m])
    return P


def fit_aff(E: list[dict], lay: np.ndarray) -> dict:
    X = np.array([np.r_[e["er"], e["goal_h"]] for e in E])
    Y = np.array([e["t"] for e in E])
    sse = np.array([[((lol(X, Y, lay, [lm] * 3) - Y)[:, k] ** 2).sum() for k in range(3)] for lm in LAMBDAS])
    lam = [LAMBDAS[int(np.argmin(sse[:, k]))] for k in range(3)]
    fit = ridge(X, Y, lam)
    mu, sd, b, ym = fit
    B = (b / sd[:, None]).T  # t = a + B f in cm
    a = ym - B @ mu
    P = lol(X, Y, lay, lam)
    return dict(lam=lam, lam_grid=list(LAMBDAS), cv_sse=sse.round(3).tolist(), mu=mu.tolist(), sd=sd.tolist(), b=b.tolist(),
                ym=ym.tolist(), a_cm=a.round(5).tolist(), B=B.round(6).tolist(), n=len(E), positions=len({e["name"] for e in E}),
                layouts=len(np.unique(lay)), lol_miss_median_cm=round(float(np.median(np.linalg.norm(P - Y, axis=1))), 4),
                in_sample_miss_median_cm=round(float(np.median(np.linalg.norm(ridge_predict(fit, X) - Y, axis=1))), 4),
                lol_pred={f"r{e['rep']}/{e['name']}": P[i].round(5).tolist() for i, e in enumerate(E)})  # fmt: skip


def aff(a) -> None:
    fk = SC.ArmFK(SC.SCENE)
    out = Path(a.aff_dir or Path(a.out) / "aff")
    out.mkdir(parents=True, exist_ok=True)
    D = episodes(fk, a.ctrl, "dev", Path(a.out))
    test_goals = np.array([e["goal"] for e in episodes(fk, a.ctrl, "test", Path(a.out)) + episodes(fk, a.ctrl, "jr", Path(a.out))])
    lay = layout_ids(D)
    near = {}
    for e in D:
        near[e["name"]] = min(near.get(e["name"], np.inf), float(np.linalg.norm(test_goals - e["goal"], axis=1).min()))
    far_m = a.far_cm / 100
    far = sorted(n for n, d in near.items() if d > far_m)
    code = dict(rootframe_eval=md5(__file__), precision=md5(PR.__file__), s_check=md5(SC.__file__))
    base = dict(ctrl=a.ctrl, frame="the reference root's yaw at the command; features [er, goal - root] cm, target t = jaw - jw cm",
                dev=str(DEV / a.ctrl), written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_md5=code,
                nearest_test_goal_cm={n: round(100 * d, 3) for n, d in sorted(near.items())},
                test_goals=f"{len(test_goals)} look goals: precision test r0-r3 and the official JR")  # fmt: skip
    full = dict(base, which="aff: every development episode-repeat", **fit_aff(D, lay))
    files = [(f"aff_rr_{a.ctrl}.json", full)]
    print(f"development positions over {a.far_cm:g} cm from every test look goal: {len(far)} of {len(near)}; with "
          + ", ".join(f"{c} cm: {sum(d > c / 100 for d in near.values())} positions "
                      f"({sum(near[e['name']] > c / 100 for e in D)} episode-repeats)" for c in (1, 1.5, 2, 2.5, 3, 4, 5)))  # fmt: skip
    if far:
        Df = [e for e in D if e["name"] in far]
        ff = fit_aff(Df, lay[[e["name"] in far for e in D]])
        rest = [e for e in D if e["name"] not in far]  # not in the far fit: its full fit predicts them
        fit = ridge(np.array([np.r_[e["er"], e["goal_h"]] for e in Df]), np.array([e["t"] for e in Df]), ff["lam"])
        for e in rest:
            ff["lol_pred"][f"r{e['rep']}/{e['name']}"] = ridge_predict(fit, np.r_[e["er"], e["goal_h"]][None])[0].round(5).tolist()
        files.append((f"aff_rr_far_{a.ctrl}.json", dict(base, which=f"aff_far: the development positions whose look goals lie "
                      f"over {a.far_cm:g} cm from every test position's ({len(far)} of {len(near)}); the others' lol_pred is "
                      "this fit's", far_positions=far, **ff)))  # fmt: skip
    else:
        print(f"aff_rr_far_{a.ctrl}.json NOT written: no development position qualifies")
    for nm, js in files:
        if (out / nm).exists():  # a frozen file is never rewritten
            print(f"{out / nm}: exists, kept (md5 {md5(out / nm)})")
            continue
        json.dump(js, open(out / nm, "w"), indent=1)
        print(f"{out / nm}: n {js['n']} episode-repeats, {js['positions']} positions, {js['layouts']} layouts; lambda per axis "
              f"{js['lam']}; a {js['a_cm']} cm; B {js['B']}; leave-one-layout miss median {js['lol_miss_median_cm']} cm "
              f"(in sample {js['in_sample_miss_median_cm']}); md5 {md5(out / nm)}")  # fmt: skip
    print(f"nearest test look goal per development position, cm: {full['nearest_test_goal_cm']}")


# --- the offline table -----------------------------------------------------------------------------------------------

METHODS = ("c_rr", "aff", "aff_far", "old", "mR")
COMPS = (("mR", "c_rr"), ("mR", "aff"), ("mR", "old"), ("mR", "aff_far"), ("aff", "c_rr"))  # the core two first
SETS = (("test", "precision test episodes (jittered), 4 repeats: THE MAIN SET"), ("dev", "precision development episodes, "
        "4 repeats"), ("jr", "official JR episodes (bench_rgbd_20261124)"))  # fmt: skip


def lid_of(name: str) -> int:
    task, rest = name.rsplit("_L", 1)
    lv, ix = rest.split("_e")
    return JD.layout(task, int(lv), int(ix))


def palm_settled(M, X, R, key, hand2, palm) -> np.ndarray:
    """the model's settled offset of the palm (the mean of its three spheres) for inputs X (n, 499), references R (n, Q, 24, 3)."""
    T = dict(x=np.asarray(X, np.float32), ref=np.asarray(R, np.float32), key=key, hand2=hand2)
    return VP.predict(M["point"], T, np.arange(len(X)))[:, -1][:, palm].mean(1)


def root_state(bel, st) -> dict:
    """the state at the command as the robot itself knows it, in the reference root's frame: S3 as it is, W and J the
    believed pelvis, jaw and wrist (jref_data.Belief), the true pelvis replaced by the reference root (so s_check's carry
    is the identity and its heading frame the root's)."""
    s = {k: st[k] for k in st.files}
    s["S3"], s["W"], s["J"] = bel.of(st["S3"], np.asarray(st["ref0"][0:7], float))
    q0 = np.asarray(st["qpos0"], float).copy()
    q0[0:7] = np.asarray(st["ref0"][0:7], float)
    s["qpos0"] = q0
    return s


def sign_medians(d, lay):
    from scipy.stats import binomtest

    m = np.array([np.median(d[lay == k]) for k in np.unique(lay)])
    neg, pos = int((m < 0).sum()), int((m > 0).sum())
    return neg, pos, (binomtest(neg, neg + pos).pvalue if neg + pos else float("nan"))


def q(v, p=50):
    return float(np.percentile(v, p))


def table(a) -> None:
    import contextlib
    import io

    import torch

    out = Path(a.out)
    fk = SC.ArmFK(SC.SCENE)
    bel = JD.Belief(fk)
    link = list(fk.links)
    key = np.array([link.index(x) for x in VM.KEY_LINKS])
    hand2 = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
    palm = np.array([x == "hand_palm_link" for x in link])
    pr = lambda *s: print(*s, flush=True)  # noqa: E731
    pr(f"selfmodel_rootframe_20261125 offline table, written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    code = dict(rootframe_eval=md5(__file__), jref_data=md5(JD.__file__), precision=md5(PR.__file__), s_check=md5(SC.__file__),
                via_point=md5(VP.__file__), via_model=md5(VM.__file__))  # fmt: skip
    pr("code md5: " + ", ".join(f"{k} {v}" for k, v in code.items()))
    pr("offline miss = |t - predicted t| per episode-repeat, cm, t = measured jaw centre - jw (the reference-root end), the "
       "root's heading frame; paired d = miss(first) - miss(second) < 0: the first is better. Interval: 95 % of the median of "
       "d, 4000 layout resamples (precision.by_layout, seed 0); sign tests over layouts (1 cm) on their means (registered) "
       "and on their medians. win = interval below 0 and the layout-mean sign p < 0.05 (the verdict reads only the main set)")  # fmt: skip
    for ctrl in a.ctrls:
        mdir = Path(a.model) if a.model else out / f"model_{ctrl}"
        Mn = torch.load(mdir / "via_point.pt", map_location="cpu", weights_only=False)
        Mo = torch.load(OLD / f"model_{ctrl}" / "via_point.pt", map_location="cpu", weights_only=False)
        crr = json.load(open(CONST / f"ctrack_rr_{ctrl}.json"))
        assert crr["frame"] == "the reference root's yaw", crr["frame"]
        mean_rr = np.asarray(crr["mean_crr_h_cm"], float)
        per_rr = [(np.asarray(x["crr_h_cm"], float), x["name"]) for x in crr["per"]]
        A = {k: json.load(open(out / "aff" / f"aff_rr{s}_{ctrl}.json")) for k, s in (("aff", ""), ("aff_far", "_far"))}
        fits = {k: (np.asarray(v["mu"]), np.asarray(v["sd"]), np.asarray(v["b"]), np.asarray(v["ym"])) for k, v in A.items()}
        pr(f"\n######## {ctrl}")
        pr(f"new model {mdir / 'via_point.pt'} md5 {md5(mdir / 'via_point.pt')}; old (jrefwm) {OLD / f'model_{ctrl}' / 'via_point.pt'} "
           f"md5 {md5(OLD / f'model_{ctrl}' / 'via_point.pt')}; c_rr {CONST / f'ctrack_rr_{ctrl}.json'} md5 "
           f"{md5(CONST / f'ctrack_rr_{ctrl}.json')} (n {crr['n']}); aff md5 {md5(out / 'aff' / f'aff_rr_{ctrl}.json')}, aff_far "
           f"md5 {md5(out / 'aff' / f'aff_rr_far_{ctrl}.json')} (n {A['aff_far']['n']}{', reported, not read' if A['aff_far']['n'] < 20 else ''})")  # fmt: skip
        dirs_new = VP.model_dirs(Mn, mdir)
        band = np.concatenate([(lambda t: t["x"][t["bucket"] < 8][:, :29])(np.load(Path(d) / "table.npz")) for d in dirs_new])
        lo_b, hi_b = np.percentile(band, [0.5, 99.5], axis=0)
        rows_out = open(out / f"table_rows_{ctrl}.jsonl", "w")
        rng_lines = []
        for kind, title in SETS:
            E = episodes(fk, ctrl, kind, out)
            lay = layout_ids(E)
            n = len(E)
            # the old model's input as precision built it (the true pelvis) and the new one's (the robot's own readings)
            Xo = np.array([SC.model_input(fk, e["st"], e["ref"], e["total"], e["end"]) for e in E])
            Xn, Rn, d_end, d_tot = [], [], 0.0, 0.0
            for e in E:
                s = root_state(bel, e["st"])
                ref2, tot2, _, end2 = SC.reference(fk, s, e["Q"])
                d_end = max(d_end, float(np.abs(end2[0] - e["jw"]).max()))
                d_tot = max(d_tot, abs(tot2 - e["total"]))
                Xn.append(SC.model_input(fk, s, ref2, tot2, end2))
                Rn.append(ref2)
            assert d_end < 1e-6 and d_tot < 1e-9, (d_end, d_tot)  # the stored root quaternion is float32: R R^T = I to ~1e-7
            p_old = 100 * palm_settled(Mo, Xo, np.array([e["ref"] for e in E]), key, hand2, palm)
            p_new = 100 * palm_settled(Mn, np.array(Xn), np.array(Rn), key, hand2, palm)
            t = np.array([e["t"] for e in E])
            P = {m: np.zeros((n, 3)) for m in METHODS}
            P["old_lit"] = np.zeros((n, 3))
            chk = dict(er=0.0, crr=0.0, p_old=0.0)
            for i, e in enumerate(E):
                sh = e["sh"]
                crr_i = mean_rr if kind != "dev" else np.mean([m for m, nm in per_rr if nm != e["name"]], 0)
                P["c_rr"][i] = crr_i
                f = np.r_[e["er"], e["goal_h"]][None]
                for k in ("aff", "aff_far"):
                    P[k][i] = A[k]["lol_pred"][f"r{e['rep']}/{e['name']}"] if kind == "dev" else ridge_predict(fits[k], f)[0]
                p_rr = 100 * PR.to_head(PR.from_head(p_old[i] / 100, e["yaw_W"]), e["yaw_rr"])
                P["old"][i] = p_rr - e["u"]
                P["old_lit"][i] = np.asarray(sh["pred_h_cm"]) - (np.asarray(sh["er_h_cm"]) - np.asarray(sh["g_h_cm"]))
                P["mR"][i] = p_new[i]
                chk["er"] = max(chk["er"], float(np.abs(e["er"] - np.asarray(sh["er_h_cm"])).max()))
                chk["crr"] = max(chk["crr"], float(np.abs(crr_i - np.asarray(sh["crr_h_cm"])).max()))
                chk["p_old"] = max(chk["p_old"], float(np.abs(p_old[i] - np.asarray(sh["pred_h_cm"])).max()))
            miss = {m: np.linalg.norm(t - P[m], axis=1) for m in P}
            miss["t itself (kin_rr)"] = np.linalg.norm(t, axis=1)
            pr(f"\n==== {ctrl} {kind}: {title}")
            pr(f"{n} episode-repeats, {len({e['name'] for e in E})} positions, {len(np.unique(lay))} layouts (1 cm)")
            pr(f"  checks: |er - shifts' er_h_cm| max {chk['er']:.4f} cm; |c_rr - shifts' crr_h_cm| max {chk['crr']:.4f} cm "
               f"({'leave-one-out by name' if kind == 'dev' else 'the full mean'}); |old p - shifts' pred_h_cm| max "
               f"{chk['p_old']:.4f} cm; root-frame reference end - jw max {d_end:.1e} m")  # fmt: skip
            # the same inputs from J's film (jref_data.py build --frame root: the check datasets)
            cdirs = sorted((out / f"check_{ctrl}").glob(f"{kind}_r*" if kind != "jr" else "jr"))
            cdirs = [d for d in cdirs if d.is_dir()]
            TC = VP.load(cdirs)
            crep = np.array([int(d.name.rsplit("_r", 1)[1]) if kind != "jr" else 0 for d in cdirs])[TC["src"]]
            idx = {(e["rep"], lid_of(e["name"])): i for i, e in enumerate(E)}
            match = np.array([idx.get((int(r), int(l)), -1) for r, l in zip(crep, TC["reset"])])
            assert (match >= 0).all() and len(set(match.tolist())) == len(match), "check rows without their episode"
            s0 = float(np.abs(TC["S0"] - np.array([E[i]["st"]["S3"][0, :29] for i in match], np.float32)).max())
            dx = np.abs(TC["x"] - np.array(Xn, np.float32)[match])
            pf = 100 * VP.predict(Mn["point"], TC, np.arange(len(match)))
            pf_palm = pf[:, -1][:, palm].mean(1)
            lab = 100 * TC["off"][:, -1][:, palm].mean(1)  # the filmed settled palm offset: what mR predicts
            err = np.linalg.norm(lab - pf_palm, axis=1)
            pr(f"  film route ({len(match)} of {n} have a check row; S0 max |diff| {s0:.1e}): x max |diff| from the plan-built "
               f"input {dx.max():.2e} (the reference keys {dx[:, 379:499].max():.2e}, the rest {np.delete(dx, np.s_[379:499], 1).max():.2e} "
               f"at column {int(np.argmax(np.delete(dx, np.s_[379:499], 1).max(0)))}); "
               f"mR from the film's input - from the plan's: max {np.abs(pf_palm - p_new[match]).max():.3f} cm")  # fmt: skip
            pr(f"  mR's own error (the filmed settled palm offset - mR's, {len(match)} reaches): median {q(err):.2f} p90 {q(err, 90):.2f} cm; "
               f"the reference itself (the offset): median {q(np.linalg.norm(lab, axis=1)):.2f} p90 {q(np.linalg.norm(lab, axis=1), 90):.2f} cm; "
               f"the filmed palm offset - the measured t: median {q(np.linalg.norm(lab - t[match], axis=1)):.2f} cm")  # fmt: skip
            buf = []
            VP.report(f"  via_point report, {kind} root-frame checks", cdirs, TC, Mn, np.arange(len(match)), buf.append)
            for s in buf:
                pr("  " + s)
            S0 = np.array([e["st"]["S3"][0, :29] for e in E])
            oor = ((S0 < lo_b) | (S0 > hi_b)).sum(1)
            pr(f"  joints at the command outside the training paths' 0.5-99.5 % band ({n} episode-repeats): median "
               f"{np.median(oor):.1f}, mean {oor.mean():.2f}, max {oor.max()}, counts {np.bincount(oor).tolist()}")  # fmt: skip
            for d in cdirs:
                with contextlib.redirect_stdout(io.StringIO()) as so:
                    JD.joint_range(argparse.Namespace(check=str(d), train=[str(x) for x in dirs_new]))
                rng_lines.append(f"{d}: {so.getvalue().strip()}")
            pr("  offline miss, cm            median   mean    p90")
            for m in ("t itself (kin_rr)", "c_rr", "aff", "aff_far", "old", "old_lit", "mR"):
                lab_m = {"old": "old (p - u)", "old_lit": "old, review's p-(er-g)"}.get(m, m)
                pr(f"    {lab_m:24s} {q(miss[m]):6.2f} {miss[m].mean():6.2f} {q(miss[m], 90):6.2f}")
            pr("  paired d                    median [95 % interval]      layout means better/worse p   layout medians "
               "better/worse p   win")
            for x, y in COMPS:
                d = miss[x] - miss[y]
                b = PR.by_layout(d, lay, 4000, 0)
                nb, nw, pm = sign_medians(d, lay)
                win = b["hi"] < 0 and b["sign_p"] < 0.05
                note = ""
                if y == "aff_far" and ctrl == "sonic":
                    note = " (SONIC's far fit: reported, not read)"
                if (x, y) == ("aff", "c_rr"):
                    note = " (extra, not registered)"
                pr(f"    {x} - {y:8s} {b['median']:+6.2f} [{b['lo']:+5.2f}, {b['hi']:+5.2f}]   {b['layouts_better']:2d}/"
                   f"{b['layouts_worse']:<2d} p {b['sign_p']:.3f}   {nb:2d}/{nw:<2d} p {pm:.3f}   {'WIN' if win else 'no'}{note}")  # fmt: skip
            for i, e in enumerate(E):
                rows_out.write(json.dumps(dict(set=kind, rep=e["rep"], name=e["name"], layout=int(lay[i]), t=t[i].round(4).tolist(),
                               **{f"pred_{m}": P[m][i].round(4).tolist() for m in P}, **{f"miss_{m}": round(float(miss[m][i]), 4) for m in P},
                               joints_out=int(oor[i]))) + "\n")  # fmt: skip
        rows_out.close()
        open(out / f"range_{ctrl}.txt", "w").write("\n".join(rng_lines) + "\n")
        # jref's own test bucket (9): the settled point of each model in its own data
        pr(f"\n==== {ctrl} jref test bucket 9 (each model on its own data and frame; via_point eval's lines below)")
        for nm, M, dirs in (("new (root frame)", Mn, dirs_new), ("old (jrefwm)", Mo, VP.model_dirs(Mo, OLD / f"model_{ctrl}"))):
            T = VP.load(dirs)
            r9 = np.where((~T["sector"]) & (T["bucket"] == 9))[0]
            pp = 100 * VP.predict(M["point"], T, r9)[:, -1]
            off = 100 * T["off"][r9, -1]
            e_palm = np.linalg.norm(off[:, palm].mean(1) - pp[:, palm].mean(1), axis=1)
            e_all = np.linalg.norm(off - pp, axis=2).mean(1)
            r_palm = np.linalg.norm(off[:, palm].mean(1), axis=1)
            pr(f"  {nm:16s} {len(r9)} paths: settled palm error median {q(e_palm):.2f} p90 {q(e_palm, 90):.2f} cm (the reference "
               f"itself {q(r_palm):.2f} / {q(r_palm, 90):.2f}); settled, mean over the 24 spheres: median {q(e_all):.2f} cm")  # fmt: skip
        for nm, f in (("new", mdir / "eval.log"), ("old", OLD / f"model_{ctrl}" / "eval.log")):
            pr(f"  {nm} eval.log ({f}, md5 {md5(f)}), its first block (bucket 9):")
            for s in open(f).read().splitlines()[:6]:
                pr("    " + s)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("aff")
    p.add_argument("ctrl")
    p.add_argument("out")
    p.add_argument("--aff-dir", default="", help="where the two files go (default <out>/aff)")
    p.add_argument("--far-cm", type=float, default=100 * FAR_M, help="aff_far's distance to every test look goal")
    t = sub.add_parser("table")
    t.add_argument("out")
    t.add_argument("ctrls", nargs="*", default=["scalebfm", "sonic"])
    t.add_argument("--model", default="", help="the mR model dir (one controller only; default <out>/model_<ctrl>)")
    a = ap.parse_args()
    dict(aff=aff, table=table)[a.cmd](a)


if __name__ == "__main__":
    main()
