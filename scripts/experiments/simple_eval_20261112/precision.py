"""selfmodel_precision_20261124: the direct test of one-shot precision on SIMPLE. On episodes whose reach varies (the
stance jittered, simple_bench.py --stance-jitter), does the controller's self model's per-episode prediction of where the
hand stops beat a constant offset that needs no model?

Every arm starts from J (the right-arm joint path cuRobo plans, bench_via.sh step JR: J filmed, which gives the state at
the command) and differs only in where the pregrasp goal is moved before one replan (plan2.py, unchanged but for its
IK seed). An arm whose replan fails sends J's own plan (no episode is given up). The arms:

  replan  the same goal, cuRobo's IK seed changed (another path to the same goal: what replanning alone does)
  const   the goal moved by minus J's mean pregrasp miss on the development episodes (heading frame; no model)
  model   the goal moved by minus the self model's predicted settled miss of J's plan (s_check.py's end check, one round,
          no threshold: it always moves)
  sham    the model's shift length in a random direction (seeded per episode)
  oracle  the goal moved by minus this episode's own measured J miss (a reference only: it uses the measured hand)

    cd $REPO && PYTHONPATH=src $PY -I precision.py const <dev bench dir> > const.json
    ... precision.py shifts <P dir> <model dir> <bench dir> --const const.json --rep k
        (<P dir> holds s_check.py state's files) -> <P>/<arm>/step1, step2 (the planner's inputs), <P>/shifts.jsonl
    plan2.py per arm and episode (precision_run.sh)
    ... precision.py pick <P dir> <bench dir> -> <P>/<arm>/via (J's via files, the plan source moved to the arm's plan,
        J's own when the replan found none), <P>/pick.jsonl
    ... precision.py table <ctrl dir> [<ctrl dir> ...]   (the repeats r*/ under each; metrics by layout)

selfmodel_precision2_20261125 adds three arms on the same repeats (A, plan, JR, the state and the five arms above are not
rerun). With g the reference's jaw centre at the end of J's plan (s_check.reference on the state at the command) minus the
look's goal, p the model's predicted palm offset (the model arm's pred) and c_t the tracking constant (the mean of the
measured jaw centre minus the reference end's on the development JR rows, precision.py ctrack):

  kin     the goal moved by -g (only the reference's kinematic gap to the goal; no model, nothing measured)
  kinc    the goal moved by -(g + c_t) (the model-free per-episode baseline)
  modelf  the goal moved by -(g + p) (the model's full prediction)

    ... precision.py ctrack <dev bench dir> <states dir (s_check.py state of it)> > ctrack.json
    ... precision.py shifts2 <P dir> <model dir> <bench dir> --ctrack ctrack.json --rep k [--loo]
        -> <P>/<arm>/step1, step2* for the three, <P>/shifts2.jsonl
    ... precision.py pick <P> <bench> --arms kin,kinc,modelf --shifts shifts2.jsonl --out pick2.jsonl
    ... precision.py table2 <ctrl dir> ...   (all eight arms; modelf against const and against kinc)

selfmodel_sfull_20261125 (sfull_run.sh): S-full, SF (modelf's shift, -(g + p)), with const and kinc, on SIMPLE's official
episodes without the stance jitter: shifts2 --arms SF,const,kinc --const <const json> --out shifts_sf.jsonl, pick
--row-fmt 'plan_row_{task}_{ep}.json' (episode numbers repeat across tasks), table_sfull.

selfmodel_precision3_20261125 (and S-full's revision of 11-25) adds two model-free arms in the motor's own frame. With er
the jaw centre at the end of J's reference placed by the motor's reference root ref0[0:7] (no carry onto the true pelvis)
minus the look's goal, in the heading frame of that root's yaw, and c_rr the mean of the measured jaw centre minus er's
jaw on the development JR rows (precision.py ctrack_rr, the same frame):

  kin_rr   the goal moved by -er
  kinc_rr  the goal moved by -(er + c_rr) (development runs: c_rr without the episode's own position, by name)

    ... precision.py ctrack_rr <dev bench dir> <states dir> > ctrack_rr.json
    ... precision.py shifts2 <P> <model> <bench> --arms kin_rr,kinc_rr --ctrack-rr ctrack_rr.json --out shifts3.jsonl
    ... precision.py pick <P> <bench> --arms kin_rr,kinc_rr --shifts shifts3.jsonl --out pick3.jsonl
    ... precision.py table3 <ctrl dir> ...   (all ten arms; modelf against kinc_rr, kinc_rr against const)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s_check as SC  # noqa: E402

ARMS = ("replan", "const", "model", "sham", "oracle")
ARMS2 = ("kin", "kinc", "modelf")  # selfmodel_precision2_20261125
ARMS3 = ("kin_rr", "kinc_rr")  # selfmodel_precision3_20261125: the reference-root kinematic arms
# selfmodel_sfull_20261125: SF shifts as modelf does, const and kinc as above; kin_rr and kinc_rr (its revision of 11-25)
SF_ARMS = ("SF", "const", "kinc") + ARMS3
SF_COMPS = (("SF", "const"), ("SF", "kinc_rr"), ("SF", "kinc"))  # the core two first (the revision of 11-25), then kinc
SHIFTED = ("const", "model", "sham", "oracle") + ARMS2 + ("SF",) + ARMS3
# a shifted goal cuRobo cannot plan (IK_FAIL in the table's margin, selfmodel_precision_20261124's revision of 11-25) is
# tried again along the same direction at 2/3, then 1/3 of the shift; only then the arm sends J's own plan
SCALES = (("1", 1.0, "step2"), ("2/3", 2 / 3, "step2_s2of3"), ("1/3", 1 / 3, "step2_s1of3"))
REPLAN_IK_SEED = 7  # plan2.py --ik-seed for the replan arm (cuRobo's default is 1531)


def pregrasp(row: dict) -> dict:
    st = [s for s in row["stages"] if s["stage"] == "pregrasp"]
    return st[0] if st else {}


def stage(row: dict, name: str) -> dict:
    st = [s for s in row["stages"] if s["stage"] == name]
    return st[0] if st else {}


def to_head(v_w, yaw: float) -> np.ndarray:  # a world displacement into the heading frame of yaw
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v_w, float)
    return np.r_[c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]


def from_head(v_h, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v_h, float)
    return np.r_[c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def ref_root_end(fk, st, P, end) -> tuple:
    """selfmodel_precision3_20261125's reference-root end: the jaw centre at the end of J's reference (s_check.reference's
    joints_at(1.0), which is P[-1]) placed by the motor's own reference root ref0[0:7], not carried onto the true pelvis
    (world, m), and that root's yaw. end is s_check.reference's (the same jaw carried onto the true pelvis): the carry of
    this jaw must give it back."""
    ref0 = st["ref0"]
    pos, quat, base, fing = ref0[0:3], ref0[3:7], ref0[7:36].copy(), ref0[36:43]
    idx = [int(i) for i in st["idx"]]
    j = base.copy()
    j[idx] = np.asarray(P, float)[-1]
    jw = np.array(fk.spheres("right", pos, quat, j, "mj", fing)[1], float)
    _, _, _, Ra, pa = SC.frames(st)
    dev = float(np.abs(Ra @ jw + pa - np.asarray(end[0], float)).max())
    assert dev < 1e-9, ("the reference-root end carried onto the pelvis differs from s_check.reference's", dev)
    R = SC.JD.quat_R(quat)
    return jw, float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_deg(x: float) -> float:
    return float((np.degrees(x) + 180.0) % 360.0 - 180.0)


def rows_of(B: Path, arm: str) -> dict:
    out = {}
    for f in sorted((B / "rows").glob(f"{arm}_*.jsonl")):
        if not f.name[len(arm) + 1].isupper():  # <arm>_<Task>.jsonl: kin's and kinc's must not take kin_rr's, kinc_rr's
            continue
        for line in open(f):
            r = json.loads(line)
            out[f"{r['task']}_L{r['level']}_e{r['index']}"] = r
    return out


# --- the constant ----------------------------------------------------------------------------------------------------


def const(a) -> None:
    """J's mean pregrasp miss (jaw centre against the look's pregrasp goal) over the episodes it ran a plan on, in the
    heading frame of the pelvis at the command (the row's W), cm."""
    R = rows_of(Path(a.bench), a.j_arm)
    v, per = [], []
    for name, r in sorted(R.items()):
        p = pregrasp(r)
        if not p.get("joint_src"):
            continue
        v.append(to_head(p["miss"], float(p["W"][3])))
        per.append(dict(name=name, miss_h_cm=v[-1].round(3).tolist(), goal_mm=p["goal_mm"]))
    v = np.array(v)
    print(json.dumps(dict(bench=str(a.bench), arm=a.j_arm, n=len(v), mean_miss_h_cm=v.mean(0).round(3).tolist(),
                          sd_h_cm=v.std(0, ddof=1).round(3).tolist(), per=per)))  # fmt: skip


def ctrack(a) -> None:
    """selfmodel_precision2_20261125's tracking constant c_t: the mean over the development JR rows const() uses (the
    episodes J ran a plan on) of the measured jaw centre at the pregrasp settle (the row's jaw_mm) minus the reference's
    jaw centre at the end of J's plan (s_check.reference on the state at the command, s_check.py state of the same
    bench), in the heading frame of the pelvis at the command, cm. Also per episode g (the reference end minus the goal)
    and the miss, so that mean g + c_t = const's mean miss is checked, and the jaw's definition: the planner side's jaw
    (ArmFK) on the film's last pregrasp frame against the row's jaw_mm."""
    B, S = Path(a.bench), Path(a.states)
    fk = SC.ArmFK(SC.SCENE)
    R = rows_of(B, a.j_arm)
    per, C, G, Mi, Dj = [], [], [], [], []
    for name, r in sorted(R.items()):
        p = pregrasp(r)
        if not p.get("joint_src"):
            continue
        st = np.load(S / f"state_{name}.npz")
        pl = np.load(Path(str(st["plan_src"])) / f"plan_{name}.npz")
        _, _, _, end = SC.reference(fk, st, pl["Q"])
        yaw = float(st["W"][3])
        assert abs(yaw - float(p["W"][3])) < 1e-9, (name, yaw, p["W"][3])
        goal, jaw = np.asarray(p["goal_mm"], float) / 1000, np.asarray(p["jaw_mm"], float) / 1000
        g, c, m = (
            to_head(end[0] - goal, yaw),
            to_head(jaw - end[0], yaw),
            to_head(np.asarray(p["miss"], float) / 100, yaw),
        )
        z = np.load(B / f"films_{a.j_arm}" / f"{name}_{r['column']}_{r['method']}.npz")
        q = z["qpos"][np.where(z["phase"] == "pregrasp")[0][-1]].astype(float)
        jaw_fk = fk.spheres("right", q[0:3], q[3:7], q[np.asarray(st["qa"])], "mj", q[np.asarray(st["fa"])])[1]
        C.append(c), G.append(g), Mi.append(m), Dj.append(1000 * np.linalg.norm(jaw_fk - jaw))
        per.append(dict(name=name, ctrack_h_cm=(100 * c).round(3).tolist(), g_h_cm=(100 * g).round(3).tolist(),
                        miss_h_cm=(100 * m).round(3).tolist(), goal_mm=p["goal_mm"], jaw_fk_vs_row_mm=round(float(Dj[-1]), 2)))  # fmt: skip
    C, G, Mi = 100 * np.array(C), 100 * np.array(G), 100 * np.array(Mi)
    print(json.dumps(dict(bench=str(a.bench), states=str(a.states), arm=a.j_arm, n=len(C),
                          mean_ctrack_h_cm=C.mean(0).round(3).tolist(), sd_ctrack_h_cm=C.std(0, ddof=1).round(3).tolist(),
                          mean_g_h_cm=G.mean(0).round(3).tolist(), mean_g_plus_ctrack_h_cm=(G.mean(0) + C.mean(0)).round(3).tolist(),
                          mean_miss_h_cm=Mi.mean(0).round(3).tolist(),
                          jaw_fk_vs_row_mm=dict(median=round(float(np.median(Dj)), 2), max=round(float(np.max(Dj)), 2)),
                          per=per)))  # fmt: skip


def ctrack_rr(a) -> None:
    """selfmodel_precision3_20261125's c_rr: over the development JR rows ctrack() and const() use (the episodes J ran a
    plan on), the mean of the measured jaw centre at the pregrasp settle (jaw_mm) minus er's jaw (ref_root_end: J's
    reference end placed by the motor's own reference root, no carry onto the pelvis), in the heading frame of the
    reference root's yaw, cm. Also per episode er (that jaw minus the look's goal) and the miss in the same frame (so mean
    er + c_rr = the mean miss is checked), the yaw of the reference root against W's, and how far the hand stops from the
    reference-root end and from the carried one."""
    B, S = Path(a.bench), Path(a.states)
    fk = SC.ArmFK(SC.SCENE)
    R = rows_of(B, a.j_arm)
    per, C, Er, Mi, Dy, Drr, Dcar = [], [], [], [], [], [], []
    for name, r in sorted(R.items()):
        p = pregrasp(r)
        if not p.get("joint_src"):
            continue
        st = np.load(S / f"state_{name}.npz")
        pl = np.load(Path(str(st["plan_src"])) / f"plan_{name}.npz")
        _, _, _, end = SC.reference(fk, st, pl["Q"])
        jw, yaw_rr = ref_root_end(fk, st, pl["Q"], end)
        assert abs(float(st["W"][3]) - float(p["W"][3])) < 1e-9, (name, st["W"][3], p["W"][3])
        goal, jaw = np.asarray(p["goal_mm"], float) / 1000, np.asarray(p["jaw_mm"], float) / 1000
        c, er, m = (
            to_head(jaw - jw, yaw_rr),
            to_head(jw - goal, yaw_rr),
            to_head(np.asarray(p["miss"], float) / 100, yaw_rr),
        )
        dy = wrap_deg(yaw_rr - float(st["W"][3]))
        C.append(c), Er.append(er), Mi.append(m), Dy.append(dy)
        Drr.append(100 * np.linalg.norm(jaw - jw)), Dcar.append(100 * np.linalg.norm(jaw - end[0]))
        per.append(dict(name=name, crr_h_cm=(100 * c).round(3).tolist(), er_h_cm=(100 * er).round(3).tolist(),
                        miss_rr_h_cm=(100 * m).round(3).tolist(), goal_mm=p["goal_mm"], yaw_rr_minus_W_deg=round(dy, 3),
                        jaw_to_er_cm=round(float(Drr[-1]), 3), jaw_to_carried_end_cm=round(float(Dcar[-1]), 3)))  # fmt: skip
    C, Er, Mi, Dy = 100 * np.array(C), 100 * np.array(Er), 100 * np.array(Mi), np.abs(Dy)
    print(json.dumps(dict(bench=str(a.bench), states=str(a.states), arm=a.j_arm, n=len(C), frame="the reference root's yaw",
                          mean_crr_h_cm=C.mean(0).round(3).tolist(), sd_crr_h_cm=C.std(0, ddof=1).round(3).tolist(),
                          mean_er_h_cm=Er.mean(0).round(3).tolist(), mean_er_plus_crr_h_cm=(Er.mean(0) + C.mean(0)).round(3).tolist(),
                          mean_miss_rr_h_cm=Mi.mean(0).round(3).tolist(),
                          abs_yaw_rr_minus_W_deg=dict(median=round(float(np.median(Dy)), 3), max=round(float(np.max(Dy)), 3)),
                          jaw_to_er_cm_median=round(float(np.median(Drr)), 3), jaw_to_carried_end_cm_median=round(float(np.median(Dcar)), 3),
                          per=per)))  # fmt: skip


# --- the shifts ------------------------------------------------------------------------------------------------------


def shifts(a) -> None:
    import torch

    P, B = Path(a.p), Path(a.bench)
    c = json.load(open(a.const))
    mean_miss = np.asarray(c["mean_miss_h_cm"], float) / 100
    per = [(np.asarray(x["miss_h_cm"], float) / 100, np.asarray(x["goal_mm"], float) / 1000) for x in c["per"]]
    M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
    fk = SC.ArmFK(SC.SCENE)
    link = list(fk.links)
    key = np.array([link.index(x) for x in SC.VM.KEY_LINKS])
    hand2 = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
    palm = np.array([x == "hand_palm_link" for x in link])
    J = rows_of(B, "JR")
    for arm in ARMS:
        (P / arm / "step1").mkdir(parents=True, exist_ok=True)
        for _, _, d in SCALES if arm in SHIFTED else SCALES[:1]:
            (P / arm / d).mkdir(parents=True, exist_ok=True)
    out = open(P / "shifts.jsonl", "w")
    for f in sorted(P.glob("state_*.npz")):
        name = f.stem[len("state_") :]
        st = np.load(f)
        pl = np.load(Path(str(st["plan_src"])) / f"plan_{name}.npz")
        ref, total, length, end = SC.reference(fk, st, pl["Q"])
        T = dict(
            x=SC.model_input(fk, st, ref, total, end)[None], ref=ref[None].astype(np.float32), key=key, hand2=hand2
        )
        off = SC.VP.predict(M["point"], T, np.arange(1))[0]  # (Q + 1, 24, 3), heading frame
        pred = off[-1, palm].mean(
            0
        )  # the palm's predicted settled miss of the plan's end (s_check.py's end_off, round 0)
        yaw = float(st["W"][3])
        miss_h = to_head(np.asarray(pregrasp(J[name])["miss"], float) / 100, yaw)
        cst, n_c = mean_miss, len(per)
        if a.loo:  # the constant without the episodes of this layout (development runs: the constant came from them)
            g = np.asarray(pregrasp(J[name])["goal_mm"], float) / 1000
            rest = [m for m, gg in per if np.linalg.norm(gg - g) >= a.loo_tol]
            cst, n_c = np.mean(rest, 0), len(rest)
        rng = np.random.default_rng([a.seed, zlib.crc32(name.encode()), a.rep, 5])
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        sh = dict(replan=np.zeros(3), const=-cst, model=-pred, sham=float(np.linalg.norm(pred)) * u, oracle=-miss_h)
        pr = dict(np.load(B / "step2" / f"prep_{name}.npz"))
        for arm, s in sh.items():
            for _, k, d in SCALES if arm in SHIFTED else SCALES[:1]:
                prz = dict(pr)
                prz["goal_wrist"] = np.asarray(pr["goal_wrist"], float) + from_head(k * s, yaw)
                np.savez(P / arm / d / f"prep_{name}.npz", **prz)
            shutil.copy(B / "step1" / f"obst_{name}.npz", P / arm / "step1" / f"obst_{name}.npz")
        out.write(json.dumps(dict(name=name, ep=int(st["ep"]), task=str(st["task"]), rep=a.rep,
                                  act=0.02 if "try_act02" in str(st["plan_src"]) else 0.03,
                                  pred_h_cm=(100 * pred).round(2).tolist(), const_n=n_c, miss_J_h_cm=(100 * miss_h).round(2).tolist(),
                                  shift_h_cm={k: (100 * v).round(2).tolist() for k, v in sh.items()})) + "\n")  # fmt: skip
        out.flush()
    out.close()
    print(open(P / "shifts.jsonl").read(), end="")


def shifts2(a) -> None:
    """selfmodel_precision2_20261125: the goals of kin, kinc and modelf from the same state at the command, J's plan and
    model as shifts(); writes <P>/<arm>/step1, step2* (every scale) and <P>/shifts2.jsonl. --arms picks others
    (selfmodel_sfull_20261125: SF, the same shift as modelf, and const, minus --const's mean miss as shifts() moves it),
    --out the file.

    selfmodel_precision3_20261125: kin_rr (-er) and kinc_rr (-(er + c_rr)), with er the reference-root end
    (ref_root_end) minus the look's goal and c_rr --ctrack-rr's constant (ctrack_rr), both in the heading frame of the
    reference root's yaw, the shift carried into the prep's (world) frame with that yaw; --loo leaves c_rr's episodes of
    this episode's own position out by name (the const and c_t ones stay as they were, by goal distance). Their rows add
    er_h_cm, crr_h_cm, crr_n, yaw_rr_minus_W_deg and shift_frame (W or rr: the yaw each arm's shift_h_cm is in).
    --ctrack is needed only by kin, kinc, modelf, SF (their rows carry ctrack_h_cm)."""
    import torch

    P, B = Path(a.p), Path(a.bench)
    arms = tuple(a.arms.split(","))
    rr = any(x in ARMS3 for x in arms)
    if "const" in arms:  # shifts()'s constant, its leave-one-layout version with --loo
        k_ = json.load(open(a.const))
        mean_miss = np.asarray(k_["mean_miss_h_cm"], float) / 100
        per_m = [(np.asarray(x["miss_h_cm"], float) / 100, np.asarray(x["goal_mm"], float) / 1000) for x in k_["per"]]
    if a.ctrack:
        c = json.load(open(a.ctrack))
        mean_ct = np.asarray(c["mean_ctrack_h_cm"], float) / 100
        per = [(np.asarray(x["ctrack_h_cm"], float) / 100, np.asarray(x["goal_mm"], float) / 1000) for x in c["per"]]
    elif any(x in ("kin", "kinc", "modelf", "SF") for x in arms):
        raise SystemExit("--ctrack is needed for kin, kinc, modelf and SF")
    if rr:
        c_rr = json.load(open(a.ctrack_rr))
        assert c_rr["frame"] == "the reference root's yaw", c_rr["frame"]
        mean_rr = np.asarray(c_rr["mean_crr_h_cm"], float) / 100
        per_rr = [(np.asarray(x["crr_h_cm"], float) / 100, x["name"]) for x in c_rr["per"]]
    M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
    fk = SC.ArmFK(SC.SCENE)
    link = list(fk.links)
    key = np.array([link.index(x) for x in SC.VM.KEY_LINKS])
    hand2 = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
    palm = np.array([x == "hand_palm_link" for x in link])
    J = rows_of(B, "JR")
    old = (
        {json.loads(x)["name"]: json.loads(x) for x in open(P / "shifts.jsonl")}
        if (P / "shifts.jsonl").exists()
        else None
    )
    for arm in arms:
        (P / arm / "step1").mkdir(parents=True, exist_ok=True)
        for _, _, d in SCALES:
            (P / arm / d).mkdir(parents=True, exist_ok=True)
    out = open(P / a.out, "w")
    for f in sorted(P.glob("state_*.npz")):
        name = f.stem[len("state_") :]
        st = np.load(f)
        pl = np.load(Path(str(st["plan_src"])) / f"plan_{name}.npz")
        ref, total, length, end = SC.reference(fk, st, pl["Q"])
        T = dict(
            x=SC.model_input(fk, st, ref, total, end)[None], ref=ref[None].astype(np.float32), key=key, hand2=hand2
        )
        pred = SC.VP.predict(M["point"], T, np.arange(1))[0][-1, palm].mean(0)  # the model arm's pred (shifts())
        yaw = float(st["W"][3])
        pj = pregrasp(J[name])
        goal = np.asarray(pj["goal_mm"], float) / 1000
        g = to_head(end[0] - goal, yaw)  # the reference's jaw centre at J's end minus the look's goal, heading frame
        miss_h = to_head(np.asarray(pj["miss"], float) / 100, yaw)
        every = {}
        if a.ctrack:
            ct, n_c = mean_ct, len(per)
            if a.loo:  # c_t without the development episodes of this layout, as shifts() leaves the constant's out
                rest = [m for m, gg in per if np.linalg.norm(gg - goal) >= a.loo_tol]
                ct, n_c = np.mean(rest, 0), len(rest)
            every.update(kin=-g, kinc=-(g + ct), modelf=-(g + pred), SF=-(g + pred))
        if "const" in arms:
            cst, n_k = mean_miss, len(per_m)
            if a.loo:
                rest = [m for m, gg in per_m if np.linalg.norm(gg - goal) >= a.loo_tol]
                cst, n_k = np.mean(rest, 0), len(rest)
            every["const"] = -cst
        yaw_of = {}  # the yaw each arm's shift is in: W's but for the reference-root arms
        if rr:
            jw, yaw_rr = ref_root_end(fk, st, pl["Q"], end)
            er = to_head(jw - goal, yaw_rr)  # the reference-root end minus the look's goal, the root's heading frame
            crr, n_rr = mean_rr, len(per_rr)
            if a.loo:  # c_rr without this position's own development episodes (by name: selfmodel_precision3_20261125)
                rest = [m for m, nm in per_rr if nm != name]
                crr, n_rr = np.mean(rest, 0), len(rest)
            every.update(kin_rr=-er, kinc_rr=-(er + crr))
            yaw_of.update(kin_rr=yaw_rr, kinc_rr=yaw_rr)
        sh = {k: every[k] for k in arms}
        pr = dict(np.load(B / "step2" / f"prep_{name}.npz"))
        for arm, s in sh.items():
            for _, k, d in SCALES:
                prz = dict(pr)
                prz["goal_wrist"] = np.asarray(pr["goal_wrist"], float) + from_head(k * s, yaw_of.get(arm, yaw))
                np.savez(P / arm / d / f"prep_{name}.npz", **prz)
            shutil.copy(B / "step1" / f"obst_{name}.npz", P / arm / "step1" / f"obst_{name}.npz")
        row = dict(name=name, ep=int(st["ep"]), task=str(st["task"]), rep=a.rep,
                   act=0.02 if "try_act02" in str(st["plan_src"]) else 0.03,
                   g_h_cm=(100 * g).round(3).tolist(), pred_h_cm=(100 * pred).round(3).tolist())  # fmt: skip
        if old is not None:
            row["pred_vs_shifts_cm"] = round(float(np.abs(100 * pred - np.asarray(old[name]["pred_h_cm"])).max()), 4)
        if a.ctrack:
            row.update(ctrack_h_cm=(100 * ct).round(3).tolist(), ctrack_n=n_c)
        if "const" in arms:
            row["const_n"] = n_k
        if rr:
            row.update(er_h_cm=(100 * er).round(3).tolist(), crr_h_cm=(100 * crr).round(3).tolist(), crr_n=n_rr,
                       yaw_rr_minus_W_deg=round(wrap_deg(yaw_rr - yaw), 3),
                       shift_frame={k: "rr" if k in yaw_of else "W" for k in sh})  # fmt: skip
        row.update(
            miss_J_h_cm=(100 * miss_h).round(3).tolist(),
            shift_h_cm={k: (100 * v).round(3).tolist() for k, v in sh.items()},
        )
        out.write(json.dumps(row) + "\n")
        out.flush()
    out.close()
    print(open(P / a.out).read(), end="")


# --- the plans each arm sends ----------------------------------------------------------------------------------------


def pick(a) -> None:
    P, B = Path(a.p), Path(a.bench)
    arms = tuple(a.arms.split(",")) if a.arms else ARMS
    out = open(P / a.out, "w")
    for arm in arms:
        (P / arm / "via").mkdir(exist_ok=True)
    for line in open(P / a.shifts):
        r = json.loads(line)
        name = r["name"]
        v0 = dict(np.load(B / "via" / f"via_{name}.npz", allow_pickle=True))
        Q0 = np.load(Path(str(v0["src"])) / f"plan_{name}.npz")["Q"]
        for arm in arms:
            tried, pl, scale = [], None, "fallback"
            for lab, _, d in SCALES if arm in SHIFTED else SCALES[:1]:
                pf, rf = P / arm / d / f"plan_{name}.npz", P / arm / d / a.row_fmt.format(ep=r["ep"], task=r["task"])
                if not (pf.exists() or rf.exists()):
                    continue  # not tried: an earlier scale planned
                st_ = (
                    json.loads(rf.read_text().splitlines()[0]).get("status")
                    if rf.exists() and rf.stat().st_size
                    else None
                )
                tried.append(dict(scale=lab, status=st_))
                z = np.load(pf) if pf.exists() else None
                if z is not None and "Q" in z.files:
                    pl, scale = z, lab
                    break
            ok = pl is not None
            v = dict(v0)
            row = dict(name=name, ep=r["ep"], arm=arm, planned=ok, scale=scale, tried=tried)
            if ok:
                v["src"] = np.array(str((P / arm / dict((x[0], x[2]) for x in SCALES)[scale]).resolve()))
                Q = pl["Q"]
                row.update(same_as_J=bool(Q.shape == Q0.shape and np.allclose(Q, Q0, atol=1e-6)),
                           T=len(Q), T_J=len(Q0), end_dq_deg=round(float(np.degrees(np.abs(Q[-1] - Q0[-1]).max())), 3))  # fmt: skip
            np.savez(P / arm / "via" / f"via_{name}.npz", **v)
            out.write(json.dumps(row) + "\n")
    out.close()
    print(open(P / a.out).read(), end="")


# --- the table -------------------------------------------------------------------------------------------------------


def seen_by_model(name: str) -> bool:
    """the position's layout was collected for the self model (levels 0-2, crc32 % 3 != 0; simple_bench.held_out)."""
    task, lv, ix = name.rsplit("_L", 1)[0], int(name.rsplit("_L", 1)[1].split("_e")[0]), int(name.rsplit("_e", 1)[1])
    return lv <= 2 and zlib.crc32(f"{task}:{lv}:{ix}".encode()) % 3 != 0


def layouts(goals: dict, tol: float = 0.01) -> dict:
    """positions whose look's pregrasp goals (world, m; the mean over repeats) lie within tol of each other are one layout
    -> {position: layout id}."""
    names = sorted(goals)
    par = {n: n for n in names}

    def root(n):
        while par[n] != n:
            n = par[n]
        return n

    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            if np.linalg.norm(goals[x] - goals[y]) < tol:
                par[root(y)] = root(x)
    ids = {r: k for k, r in enumerate(sorted({root(n) for n in names}))}
    return {n: ids[root(n)] for n in names}


def by_layout(d: np.ndarray, lay: np.ndarray, n_boot: int = 4000, seed: int = 0):
    """d (episodes,) paired differences, lay their layouts: -> the median of d, the layouts' mean differences, and a 95 %
    interval of the median of d resampling layouts."""
    from scipy.stats import binomtest

    if len(d) == 0:
        return dict(
            median=np.nan,
            mean=np.nan,
            lo=np.nan,
            hi=np.nan,
            layouts=0,
            layouts_better=0,
            layouts_worse=0,
            sign_p=np.nan,
        )
    L = np.unique(lay)
    m = np.array([d[lay == k].mean() for k in L])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick_ = rng.choice(L, size=len(L))
        bs.append(np.median(np.concatenate([d[lay == k] for k in pick_])))
    neg, pos = int((m < 0).sum()), int((m > 0).sum())
    p = binomtest(neg, neg + pos).pvalue if neg + pos else float("nan")
    return dict(median=float(np.median(d)), mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                layouts=len(L), layouts_better=neg, layouts_worse=pos, sign_p=float(p))  # fmt: skip


# the state at the command an arm's row must share with J's (selfmodel_precision2_20261125's pipeline rule)
SAME_ROW = ("jitter", "jitter_rep", "walk", "stand")
SAME_PRE = ("S0", "S3", "W", "J", "t_cmd", "goal_mm")


def table(a) -> None:
    _table(a, ARMS, (("model", "const"),), level=1)


def table2(a) -> None:
    """selfmodel_precision2_20261125: the eight arms; the core comparisons modelf - const and modelf - kinc."""
    _table(a, ARMS + ARMS2, (("model", "const"), ("modelf", "const"), ("modelf", "kinc")), level=2)


def table3(a) -> None:
    """selfmodel_precision3_20261125: the ten arms; the core comparison modelf - kinc_rr, then kinc_rr - const, kin_rr - J
    and precision2's modelf - const, modelf - kinc."""
    comps = (("modelf", "kinc_rr"), ("kinc_rr", "const"), ("kin_rr", "J"), ("modelf", "const"), ("modelf", "kinc"))
    _table(a, ARMS + ARMS2 + ARMS3, comps, level=3)


def rot_deg(v, deg: float) -> np.ndarray:  # a heading-frame vector into the frame deg less in yaw (rr's into W's)
    return to_head(from_head(v, np.radians(deg)), 0.0)


def _table(a, arms: tuple, comps: tuple, level: int) -> None:
    second = level >= 2
    picks = ("pick.jsonl", "pick2.jsonl", "pick3.jsonl")[:level]
    for C in map(Path, a.ctrl):
        reps = sorted(p for p in C.glob("r*") if all((p / "P" / f).exists() for f in picks))
        E = []  # one per routed episode and repeat
        main = {x: [0, 0] for x in ("A", "J") + arms}
        state_bad = {x: 0 for x in arms}
        n_all = 0
        fell = 0
        for B in reps:
            R = {"A": rows_of(B, "A"), "J": rows_of(B, "JR")} | {x: rows_of(B, x) for x in arms}
            routed = {json.loads(x)["name"]: json.loads(x) for x in open(B / "P" / "shifts.jsonl")}
            if second:  # the new arms' goals came from the same routed episodes' states
                routed2 = {json.loads(x)["name"]: json.loads(x) for x in open(B / "P" / "shifts2.jsonl")}
                if set(routed2) != set(routed):
                    raise SystemExit(f"{B.name}: shifts2.jsonl's episodes differ from shifts.jsonl's")
            if level >= 3:
                routed3 = {json.loads(x)["name"]: json.loads(x) for x in open(B / "P" / "shifts3.jsonl")}
                if set(routed3) != set(routed):
                    raise SystemExit(f"{B.name}: shifts3.jsonl's episodes differ from shifts.jsonl's")
            pk = {}
            for f in picks:
                for x in open(B / "P" / f):
                    y = json.loads(x)
                    pk[(y["name"], y["arm"])] = y
            for name, j in R["J"].items():
                n_all += 1
                for x in ("A", "J") + arms:
                    r = (
                        R[x].get(name) if (x in ("A", "J") or name in routed) else j
                    )  # the arms run only routed episodes
                    if r is None:
                        r = {}
                    main[x][0] += bool(r.get("success_45"))
                    main[x][1] += bool(r.get("success_official"))
            for name, sh in routed.items():
                e = dict(
                    rep=B.name, name=name, pos=name, goal=np.asarray(pregrasp(R["J"][name])["goal_mm"], float) / 1000
                )
                for x in ("J",) + arms:
                    r = R[x if x != "J" else "J"].get(name)
                    if r is None:
                        e[x] = None
                        continue
                    fell += bool(r.get("fell"))
                    if r.get("jitter") != R["J"][name].get("jitter"):  # every arm must stand where J stood
                        raise SystemExit(
                            f"{B.name} {name} {x}: stance jitter {r.get('jitter')} != J's {R['J'][name].get('jitter')}"
                        )
                    if second and x != "J":
                        j_ = R["J"][name]
                        state_bad[x] += any(r.get(k) != j_.get(k) for k in SAME_ROW) or any(
                            pregrasp(r).get(k) != pregrasp(j_).get(k) for k in SAME_PRE
                        )
                    e[x] = dict(pre=pregrasp(r).get("err_cm"), entry=stage(r, "entry").get("err_cm"),
                                s45=bool(r.get("success_45")), off=bool(r.get("success_official")),
                                fallback=x != "J" and not pk.get((name, x), {}).get("planned", False),
                                scale=None if x == "J" else pk.get((name, x), {}).get("scale", "1" if pk.get((name, x), {}).get("planned") else "fallback"))  # fmt: skip
                e["shift"] = sh
                if second:
                    e["shift2"] = routed2[name]
                if level >= 3:
                    e["shift3"] = routed3[name]
                E.append(e)
        goals = {}
        for e in E:
            goals.setdefault(e["pos"], []).append(e["goal"])
        lay_of = layouts({k: np.mean(v, 0) for k, v in goals.items()})
        print(
            f"== {C}  repeats {len(reps)}, episodes {n_all}, routed episode-repeats {len(E)}, layouts {len(set(lay_of.values()))}, falls in routed {fell}"
        )
        print(
            "main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit"
        )
        print("  " + "  ".join(f"{x} {main[x][0]} / {main[x][1]}" for x in main))
        ok = [e for e in E if all(e[x] is not None and e[x]["pre"] is not None for x in ("J",) + arms)]
        lay = np.array([lay_of[e["pos"]] for e in ok])
        print(f"\nroutes with every arm's row: {len(ok)}; fallback to J's plan: " +
              ", ".join(f"{x} {sum(e[x]['fallback'] for e in ok)} ({100 * sum(e[x]['fallback'] for e in ok) / max(len(ok), 1):.0f} %)" for x in arms))  # fmt: skip
        print("shift scale used (1, 2/3, 1/3, fallback): " + "; ".join(
            f"{x} " + " ".join(f"{lab} {sum(e[x]['scale'] == lab for e in ok)}" for lab in ("1", "2/3", "1/3", "fallback"))
            for x in arms))  # fmt: skip
        if second:
            print(f"state at the command equal to J's ({', '.join(SAME_ROW)}; pregrasp {', '.join(SAME_PRE)}), routed "
                  f"episode-repeats differing: " + ", ".join(f"{x} {state_bad[x]}" for x in arms))  # fmt: skip
        if level >= 3:
            s3 = [e["shift3"] for e in ok]
            dy = np.abs([s["yaw_rr_minus_W_deg"] for s in s3])
            jer = [
                np.linalg.norm(np.asarray(s["miss_J_h_cm"]) - rot_deg(s["er_h_cm"], s["yaw_rr_minus_W_deg"]))
                for s in s3
            ]
            jg = [np.linalg.norm(np.asarray(s["miss_J_h_cm"]) - np.asarray(s["g_h_cm"])) for s in s3]
            print(f"reference root (kin_rr, kinc_rr): |yaw of the root - W's| median {np.median(dy):.2f}, max {np.max(dy):.2f} deg; "
                  f"|er| median {np.median([np.linalg.norm(s['er_h_cm']) for s in s3]):.2f} cm (|g| {np.median([np.linalg.norm(s['g_h_cm']) for s in s3]):.2f}); "
                  f"J's jaw to the root's end median {np.median(jer):.2f} cm, to the carried end {np.median(jg):.2f} cm; "
                  f"c_rr episodes {min(s['crr_n'] for s in s3)}-{max(s['crr_n'] for s in s3)}")  # fmt: skip

        def paired(a_, b_):  # NaN-aware: pairs with either side missing are dropped
            m_ = np.isfinite(a_) & np.isfinite(b_)
            return by_layout(a_[m_] - b_[m_], lay[m_]), int(m_.sum()), int((a_[m_] < b_[m_]).sum()), m_

        for what in ("pre", "entry"):
            print(f"\n{'pregrasp' if what == 'pre' else 'entry'} settled miss, cm (jaw centre against the look's goal; "
                  "medians skip missing rows, pairs with a missing side are dropped)")  # fmt: skip
            v = {x: np.array([np.nan if e[x][what] is None else e[x][what] for e in ok], float) for x in ("J",) + arms}
            for x in ("J",) + arms:
                miss_ = int(np.isnan(v[x]).sum())
                line = f"  {x:7s} median {np.nanmedian(v[x]):6.2f}  mean {np.nanmean(v[x]):6.2f}  missing {miss_}"
                if x != "J":
                    s, n_, c_, _ = paired(v[x], v["J"])
                    line += (f" | minus J: median {s['median']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}], closer in "
                             f"{c_}/{n_}; layouts closer {s['layouts_better']}/{s['layouts']} (sign p {s['sign_p']:.3g})")  # fmt: skip
                print(line)
            for x1, x2 in comps:
                s, n_, c_, m_ = paired(v[x1], v[x2])
                print(f"  {x1} minus {x2}: median {s['median']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] cm, {x1} closer in "
                      f"{c_}/{n_} episodes, layouts {s['layouts_better']}/{s['layouts']} "
                      f"(worse {s['layouts_worse']}, sign p {s['sign_p']:.3g})")  # fmt: skip
                # the self model's data (selfmodel_jrefwm_20261124) stood at levels 0-2's layouts but for a third held out by
                # crc32; a layout counts as seen when any of its positions was collected
                seen_lay = {lay_of[e["pos"]] for e in ok if seen_by_model(e["name"])}
                for lab, sel in (("seen by the model's data", np.isin(lay, list(seen_lay))),
                                 ("never seen", ~np.isin(lay, list(seen_lay)))):  # fmt: skip
                    sel = sel & m_
                    if sel.sum() == 0:
                        print(f"    {lab}: none")
                        continue
                    d = v[x1][sel] - v[x2][sel]
                    s = by_layout(d, lay[sel])
                    print(f"    {lab}: {int(sel.sum())} episodes, {x1} minus {x2} median {s['median']:+.2f} "
                          f"[{s['lo']:+.2f}, {s['hi']:+.2f}], layouts closer {s['layouts_better']}/{s['layouts']} (sign p {s['sign_p']:.3g})")  # fmt: skip
        print("\nroutes: successes within 45 s / official limit")
        print(
            "  "
            + "  ".join(f"{x} {sum(e[x]['s45'] for e in ok)} / {sum(e[x]['off'] for e in ok)}" for x in ("J",) + arms)
        )
        if a.per_episode:
            for e in ok:
                print(f"  {e['rep']} {e['name']:26s} L{lay_of[e['pos']]:<3d}" + " ".join(f"{x if second else x[:3]} {e[x]['pre']:6.2f}" for x in ("J",) + arms)
                      + f" | pred {e['shift']['pred_h_cm']} J miss {e['shift']['miss_J_h_cm']}")  # fmt: skip


# --- selfmodel_sfull_20261125's table --------------------------------------------------------------------------------


def table_sfull(a) -> None:
    """S-full on SIMPLE's official episodes (no jitter): the four tasks' successes for A, J, the old S (its given-up
    episodes failures, s_check.py table), SF, const, kinc, kin_rr and kinc_rr (an episode J ran no plan on takes J's
    row); on the routed episodes the paired successes and the pregrasp and entry misses against J, SF against const and
    kinc_rr (the core, the revision of 11-25) and kinc by layout; the scales, the pipeline checks and the model's error on
    JR (metric 6)."""
    D, Bn = Path(a.d), Path(a.bench)
    U = rows_of(D, "JR")  # this run's episodes: its copy of JR's rows
    names = sorted(U)
    R = {"A": rows_of(Bn, "A"), "J": rows_of(Bn, a.j_arm)} | {x: rows_of(D, x) for x in SF_ARMS}
    S_rows, gave_up = {}, set()
    if a.s_choice:
        S_rows = rows_of(Bn, "S")
        gave_up = {json.loads(x)["name"] for x in open(a.s_choice) if json.loads(x)["given_up"]}
    sh = {json.loads(x)["name"]: json.loads(x) for x in open(D / "P" / "shifts_sf.jsonl")}
    pk = {(y["name"], y["arm"]): y for y in map(json.loads, open(D / "P" / "pick_sf.jsonl"))}
    j_routed = {n for n in names if pregrasp(R["J"].get(n, {"stages": []})).get("joint_src")}
    cols = ("A", "J") + (("S",) if S_rows else ()) + SF_ARMS

    def row_of(x, n):  # an arm's row for the main metric: the shifted arms ran only the routed episodes
        return (R[x].get(n) if n in sh else R["J"].get(n)) if x in SF_ARMS else R[x].get(n)

    def ok(x, n, k):
        if x == "S":
            return bool(S_rows.get(n, {}).get(k)) and n not in gave_up
        return bool((row_of(x, n) or {}).get(k))

    print(f"== {D}  episodes {len(names)}, routed {len(sh)} (J's rows with a plan {len(j_routed)}, the same episodes: "
          f"{set(sh) == j_routed}), the others take J's rows: {len(names) - len(sh)}")  # fmt: skip
    print(
        "\n1. main metric: successes within 45 s / official limit"
        + (f" (old S: {len(gave_up)} given up, failures)" if S_rows else "")
    )
    print(f"  {'task':26s}" + "".join(f"{x:>11s}" for x in cols))
    tasks = sorted({U[n]["task"] for n in names}, key=lambda t: SC.JD.TASKS.index(t) if t in SC.JD.TASKS else 9)
    for t in tasks + ["all"]:
        nn = [n for n in names if t == "all" or U[n]["task"] == t]
        print(f"  {t + f' ({len(nn)})':26s}" + "".join(
            f"{sum(ok(x, n, 'success_45') for n in nn):>5d} / {sum(ok(x, n, 'success_official') for n in nn):<3d}" for x in cols))  # fmt: skip
    rt = sorted(sh)
    print(f"\n2. routed episodes ({len(rt)}): paired successes against J")
    for x in cols[2:]:
        for key, lab in (("success_45", "45 s"), ("success_official", "official")):
            j = np.array([ok("J", n, key) for n in rt], bool)
            v = np.array([ok(x, n, key) for n in rt], bool)
            print(f"  {x:6s} {lab:8s}: J {int(j.sum())}, {x} {int(v.sum())}, {x} - J {int(v.sum()) - int(j.sum()):+d}; only {x} "
                  f"{int((v & ~j).sum())}, only J {int((j & ~v).sum())}" + (f"  [{', '.join(n for n in rt if j[rt.index(n)] != v[rt.index(n)])}]" if (j != v).any() else ""))  # fmt: skip
    print("\n4. shift scale used (1, 2/3, 1/3, fallback) and fallback to J's plan:")
    for x in SF_ARMS:
        sc = [pk.get((n, x), {}).get("scale", "fallback") for n in rt]
        print(f"  {x:6s} " + " ".join(f"{lab} {sc.count(lab)}" for lab in ("1", "2/3", "1/3", "fallback")) +
              f"; fallback {sc.count('fallback')} / {len(rt)} ({100 * sc.count('fallback') / max(len(rt), 1):.0f} %)")  # fmt: skip
    lens = {x: [float(np.linalg.norm(sh[n]["shift_h_cm"][x])) for n in rt] for x in SF_ARMS}
    print("  full shift length median, cm: " + ", ".join(f"{x} {np.median(v):.2f}" for x, v in lens.items()) +
          f"; |g| {np.median([np.linalg.norm(sh[n]['g_h_cm']) for n in rt]):.2f}, |p| {np.median([np.linalg.norm(sh[n]['pred_h_cm']) for n in rt]):.2f}"
          f", |er| {np.median([np.linalg.norm(sh[n]['er_h_cm']) for n in rt]):.2f}")  # fmt: skip
    dy = np.abs([sh[n]["yaw_rr_minus_W_deg"] for n in rt])
    jer = [
        np.linalg.norm(np.asarray(sh[n]["miss_J_h_cm"]) - rot_deg(sh[n]["er_h_cm"], sh[n]["yaw_rr_minus_W_deg"]))
        for n in rt
    ]
    jg = [np.linalg.norm(np.asarray(sh[n]["miss_J_h_cm"]) - np.asarray(sh[n]["g_h_cm"])) for n in rt]
    print(f"  reference root (kin_rr, kinc_rr): |yaw of the root - W's| median {np.median(dy):.2f}, max {np.max(dy):.2f} deg; "
          f"J's jaw to the root's end median {np.median(jer):.2f} cm, to the carried end {np.median(jg):.2f} cm; "
          f"c_rr episodes {min(sh[n]['crr_n'] for n in rt)}-{max(sh[n]['crr_n'] for n in rt)}")  # fmt: skip
    print("\n5. pipeline checks")
    bad = {x: [n for n in rt if R[x].get(n) is None or any(R[x][n].get(k) != U[n].get(k) for k in SAME_ROW)
               or any(pregrasp(R[x][n]).get(k) != pregrasp(U[n]).get(k) for k in SAME_PRE)] for x in SF_ARMS}  # fmt: skip
    print(f"  state at the command equal to JR's ({', '.join(SAME_ROW)}; pregrasp {', '.join(SAME_PRE)}): " +
          ", ".join(f"{x} {len(rt) - len(bad[x])}/{len(rt)}" for x in SF_ARMS) +
          ("" if not any(bad.values()) else f"  differing: {bad}"))  # fmt: skip
    jit = sum("jitter" in R[x].get(n, {}) for x in SF_ARMS for n in rt)
    print(f"  rows carrying a stance jitter: {jit}")
    fails, nrow = {x: 0 for x in SF_ARMS}, {x: 0 for x in SF_ARMS}
    if (D / "pipeline.log").exists():
        for line in open(D / "pipeline.log"):
            w = line.split()
            if len(w) > 3 and w[1] == "arm" and w[2] in fails and "failed" in w:
                fails[w[2]] += int(w[w.index("failed") - 1])
    for x in SF_ARMS:
        nrow[x] = sum(n in R[x] for n in rt)
    print("  failed processes: " + ", ".join(f"{x} {fails[x]}" for x in SF_ARMS) + "; rows on the routed episodes: " +
          ", ".join(f"{x} {nrow[x]}/{len(rt)}" for x in SF_ARMS))  # fmt: skip
    # 3. the misses
    E = [n for n in rt if all(R[x].get(n) is not None for x in ("J",) + SF_ARMS)]
    goals = {n: np.asarray(pregrasp(R["J"][n])["goal_mm"], float) / 1000 for n in E}
    lay_of = layouts(goals)
    lay = np.array([lay_of[n] for n in E])
    seen_lay = {lay_of[n] for n in E if seen_by_model(n)}
    for what in ("pre", "entry"):
        print(f"\n3. {'pregrasp' if what == 'pre' else 'entry'} settled miss, cm (jaw centre against the look's goal), "
              f"{len(E)} routed episodes, {len(set(lay.tolist()))} layouts; medians skip missing rows")  # fmt: skip
        val = lambda r: (pregrasp(r) if what == "pre" else stage(r, "entry")).get("err_cm")  # noqa: E731
        v = {
            x: np.array([np.nan if val(R[x][n]) is None else val(R[x][n]) for n in E], float) for x in ("J",) + SF_ARMS
        }
        for x in ("J",) + SF_ARMS:
            line = f"  {x:6s} median {np.nanmedian(v[x]):6.2f}  mean {np.nanmean(v[x]):6.2f}  missing {int(np.isnan(v[x]).sum())}"
            if x != "J":
                m_ = np.isfinite(v[x]) & np.isfinite(v["J"])
                s = by_layout(v[x][m_] - v["J"][m_], lay[m_])
                line += (f" | minus J: median {s['median']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}], closer in "
                         f"{int((v[x][m_] < v['J'][m_]).sum())}/{int(m_.sum())}; layouts closer {s['layouts_better']}/{s['layouts']} (sign p {s['sign_p']:.3g})")  # fmt: skip
            print(line)
        for x1, x2 in SF_COMPS:
            m_ = np.isfinite(v[x1]) & np.isfinite(v[x2])
            s = by_layout(v[x1][m_] - v[x2][m_], lay[m_])
            print(f"  {x1} minus {x2}: median {s['median']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] cm, {x1} closer in "
                  f"{int((v[x1][m_] < v[x2][m_]).sum())}/{int(m_.sum())} episodes, layouts {s['layouts_better']}/{s['layouts']} "
                  f"(worse {s['layouts_worse']}, sign p {s['sign_p']:.3g})")  # fmt: skip
            for lab, sel in (
                ("seen by the model's data", np.isin(lay, list(seen_lay))),
                ("never seen", ~np.isin(lay, list(seen_lay))),
            ):
                sel = sel & m_
                if sel.sum() == 0:
                    print(f"    {lab}: none")
                    continue
                s = by_layout(v[x1][sel] - v[x2][sel], lay[sel])
                print(f"    {lab}: {int(sel.sum())} episodes, {x1} minus {x2} median {s['median']:+.2f} [{s['lo']:+.2f}, "
                      f"{s['hi']:+.2f}], layouts closer {s['layouts_better']}/{s['layouts']} (sign p {s['sign_p']:.3g})")  # fmt: skip
    # 6. the model on JR: the filmed settled palm offset (actual - reference, jref_data.py build's last point) against p
    print(
        "\n6. the model's error on JR's films (actual palm minus reference at rest, against p = the SF arm's prediction)"
    )
    if a.check:
        T = np.load(Path(a.check) / "table.npz")
        pal = T["palm"]
        err, ref_only, miss_ = [], [], 0
        for n in rt:
            S0 = np.load(D / "P" / f"state_{n}.npz")["S3"][0, :29].astype(np.float32)
            i = np.where((T["S0"] == S0).all(1))[0]
            if len(i) != 1:
                miss_ += 1
                continue
            act = 100 * T["off"][i[0], -1, pal].mean(0)
            err.append(np.linalg.norm(act - np.asarray(sh[n]["pred_h_cm"])))
            ref_only.append(np.linalg.norm(act))
        err, ref_only = np.array(err), np.array(ref_only)
        print(f"  {len(err)} episodes ({miss_} without a check row: fell or too short): |actual - p| median {np.median(err):.2f}, "
              f"90 % {np.percentile(err, 90):.2f} cm; |actual| (the reference alone) median {np.median(ref_only):.2f}, 90 % "
              f"{np.percentile(ref_only, 90):.2f} cm")  # fmt: skip
    if a.range:
        print("  jref_data.py range on JR's command states: " + open(a.range).read().strip().replace("\n", "\n  "))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("const")
    p.add_argument("bench")
    p.add_argument("--j-arm", default="JR")
    p = sub.add_parser("shifts")
    p.add_argument("p")
    p.add_argument("model")
    p.add_argument("bench")
    p.add_argument("--const", required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loo", action="store_true", help="the constant without the const file's episodes of this layout")
    p.add_argument(
        "--loo-tol", type=float, default=0.02, help="m: const episodes whose goal lies within this are left out"
    )
    p = sub.add_parser("pick")
    p.add_argument("p")
    p.add_argument("bench")
    p.add_argument("--arms", default="", help="comma-separated (default the five of selfmodel_precision_20261124)")
    p.add_argument("--shifts", default="shifts.jsonl")
    p.add_argument("--out", default="pick.jsonl")
    p.add_argument(
        "--row-fmt", default="plan_row_{ep}.json", help="the planner's row file of an arm, scale and episode"
    )
    for nm in ("ctrack", "ctrack_rr"):
        p = sub.add_parser(nm)
        p.add_argument("bench")
        p.add_argument("states", help="s_check.py state's output for the bench")
        p.add_argument("--j-arm", default="JR")
    p = sub.add_parser("shifts2")
    p.add_argument("p")
    p.add_argument("model")
    p.add_argument("bench")
    p.add_argument("--ctrack", default="", help="c_t (precision.py ctrack); needed by kin, kinc, modelf, SF")
    p.add_argument("--ctrack-rr", default="", help="c_rr (precision.py ctrack_rr); needed by kin_rr, kinc_rr")
    p.add_argument("--arms", default="kin,kinc,modelf")
    p.add_argument("--const", default="", help="the const arm's file (const_<ctrl>.json)")
    p.add_argument("--out", default="shifts2.jsonl")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--loo", action="store_true",
                   help="c_t and const without their files' episodes of this layout (goal distance); c_rr without this position's own (by name)")  # fmt: skip
    p.add_argument("--loo-tol", type=float, default=0.02, help="m, as shifts --loo-tol")
    p = sub.add_parser("table_sfull")
    p.add_argument("d", help="the S-full controller dir (sfull_run.sh's)")
    p.add_argument("--bench", required=True, help="the source bench dir (A, J and the old S rows)")
    p.add_argument("--j-arm", default="J")
    p.add_argument("--s-choice", default="", help="the old S's choice.jsonl (its given-up episodes)")
    p.add_argument("--check", default="", help="jref_data.py build of JR's films and rows")
    p.add_argument("--range", default="", help="jref_data.py range's output")
    for nm in ("table", "table2", "table3"):
        p = sub.add_parser(nm)
        p.add_argument("ctrl", nargs="+", help="a controller's dir of repeats r0, r1, ...")
        p.add_argument("--per-episode", action="store_true")
    a = ap.parse_args()
    dict(
        const=const,
        shifts=shifts,
        pick=pick,
        table=table,
        ctrack=ctrack,
        ctrack_rr=ctrack_rr,
        shifts2=shifts2,
        table2=table2,
        table3=table3,
        table_sfull=table_sfull,
    )[a.cmd](a)


if __name__ == "__main__":
    main()
