"""selfmodel_s_20261124 (TODO 38.10.4): the S arm -- cuRobo's right-arm joint path (the J arm) checked by the
controller's own self model before it is sent, repaired by replanning when the check fails, given up when it cannot be.

What S knows at the command is what J knows (the look's points, the plan, the robot's state) plus the self model
(selfmodel_jrefwm_20261124: per controller, where the whole arm really goes for a reference built from a joint path).
J and S are the same run up to the command, so the state at the command is read from J's own run filmed with its
reference (bench_via.sh step JR): the rows carry the model's inputs at the command (S3, W, J, t_cmd), the film the motor's
reference pose then. Nothing after the command is read.

For a candidate path the motor's reference is rebuilt as Motor.reach_joints builds it (the path retimed min-jerk in the
jaw centre's arc length, the start's gap faded out, the pelvis and every other joint held at the reference at the
command), its spheres by the scene robot (arm_fk.ArmFK), the model's input as jref_data.build makes it, and the model
(via_point.py) gives every sphere's offset at 24 moments and the settled end: the predicted sweep.

Two checks on the predicted sweep:
  end    the palm's predicted settled miss of the planned end (the mean of its three spheres), cm
  clear  the smallest gap between a predicted sphere's path and the look's points (obstacles_box.py's voxels, where the
         robot believes they are, less those at the arm's start and at the target, as plan2.py drops them), minus the
         sphere's radius and the error radius r, cm; moments with u >= --clear-until are left out
Repairs (one replan with cuRobo, plan2.py unchanged, per round): the goal moved back by the predicted end miss when
`end` > --tau-end; the voxels moved by minus the predicted offset of the sphere that comes closest, when `clear` <
-delta. A candidate passes when neither fires. Rounds stop at --rounds; a path with neither check passing after them
is given up (the S arm does not reach in that episode).

    cd $REPO && SIMPLE_EVAL_OUT=... PYTHONPATH=src $PY -I s_check.py state <bench dir> <S dir>
    ... s_check.py check <S dir> <model dir> <bench dir> <round> [--tau-end 2] [--delta 0] [--r 3.0] [--clear-until 1]
    ... s_check.py choose <S dir> <bench dir> <rounds> -> <S dir>/via (J's via files, their plan source moved to the
        chosen round's plan), choice.jsonl
    ... s_check.py table <S dir> <bench dir> [--arm S]   (A, J and S side by side, S's given-up episodes failures)
    s_run.sh drives the rounds (plan2.py between them)
    ... s_check.py calib <model dir> <dataset dir> ...   (jref_data.py build of J's filmed runs) -> the error radius
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
sys.path.insert(0, str(Path.cwd() / "src"))
import episodes as EP  # noqa: E402
import jref_data as JD  # noqa: E402
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402
from arm_fk import FINGERS, ArmFK  # noqa: E402

REACH_SPEED, REACH_MIN_S = 0.15, 0.6  # scalebfm/motor.py MotorConfig: J sends no speed of its own
SCENE = Path("outputs/scene/simple_g1_sonic")
FREE, TGT = 0.01, 0.02  # plan2.py's --free and --tgt: voxels at the arm's start and at the target are not obstacles


def minjerk(u: float) -> float:  # scalebfm/motor.py minjerk
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def name_of(task: str, e: int) -> str:
    return f"{task}_L{e // 10}_e{e % 10}"


# --- the state at the command (from J's run filmed with its reference) ---------------------------------------------


def state(a) -> None:
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    B, S = Path(a.bench), Path(a.s)
    S.mkdir(parents=True, exist_ok=True)
    fk = ArmFK(SCENE)
    out = open(S / "state.jsonl", "w")
    for f in sorted((B / "rows").glob("JR_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            st = [s for s in r["stages"] if s["stage"] == "pregrasp" and s.get("joint_src") and "t_cmd" in s]
            if not st:
                continue
            s = st[0]
            name = f"{r['task']}_L{r['level']}_e{r['index']}"
            z = dict(np.load(B / "films_JR" / f"{name}_{r['column']}_{r['method']}.npz"))
            jn = [str(x) for x in z["jnt"]]
            adr = np.asarray(z["jnt_adr"])
            qa = [int(adr[jn.index(n)]) for n in MUJOCO_JOINT_NAMES]
            fa = [int(adr[jn.index(f"right_hand_{x}_joint")]) for x in FINGERS]
            t_cmd = float(s["t_cmd"])
            sel = np.where(z["phase"] == "pregrasp")[0]
            j0 = sel[int(np.argmin(np.abs(z["t"][sel] - t_cmd)))]
            pl = np.load(Path(s["joint_src"]) / f"plan_{name}.npz")
            idx = [MUJOCO_JOINT_NAMES.index(str(n)) for n in pl["names"]]
            np.savez(S / f"state_{name}.npz", ref0=z["ref"][j0].astype(float), qpos0=z["qpos"][j0].astype(float), qa=qa,
                     fa=fa, idx=idx, names=pl["names"], S3=np.asarray(s["S3"], float), W=np.asarray(s["W"], float),
                     J=np.asarray(s["J"], float), t_cmd=t_cmd, dt_film=float(z["t"][j0] - t_cmd),
                     dur_run=float(s["joint_dur_s"]), path_run=float(s["joint_path_cm"]), success_45=bool(r.get("success_45")),
                     ep=int(r["ep"]), task=r["task"], plan_src=s["joint_src"])  # fmt: skip
            # the check of the rebuilt reference: J's own plan against the reference the film holds
            ref, dur, length, _ = reference(fk, np.load(S / f"state_{name}.npz"), pl["Q"])
            Xf = film_reference(fk, z, j0, t_cmd, dur, np.load(S / f"state_{name}.npz"))
            dev = 100 * np.linalg.norm(ref - Xf, axis=-1)
            out.write(json.dumps(dict(name=name, ep=int(r["ep"]), dur=round(dur, 3), dur_run=s["joint_dur_s"],
                                      path_cm=round(100 * length, 2), path_run=s["joint_path_cm"],
                                      rebuilt_vs_film_cm_max=round(float(dev.max()), 3),
                                      rebuilt_vs_film_cm_median=round(float(np.median(dev)), 3),
                                      dt_film_ms=round(1000 * float(z["t"][j0] - t_cmd), 1))) + "\n")  # fmt: skip
            out.flush()
    out.close()
    print(open(S / "state.jsonl").read(), end="")


def frames(st):
    """The heading frame of the true pelvis at the command (jref_data.build's): -> head(x), Rh, p0, Ra, pa."""
    W = st["W"]
    p0, yaw = W[:3], float(W[3])
    Rh = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    Rt, Rr = JD.quat_R(st["qpos0"][3:7]), JD.quat_R(st["ref0"][3:7])
    Ra = Rt @ Rr.T
    pa = st["qpos0"][0:3] - Ra @ st["ref0"][0:3]

    def head(x):
        x = np.asarray(x, float)
        return np.c_[(x[..., :2].reshape(-1, 2) - p0[:2]) @ Rh[:2, :2], x[..., 2].reshape(-1)].reshape(x.shape)

    return head, Rh, p0, Ra, pa


def reference(fk, st, P):
    """The motor's reference for joint path P (T, 7) from the state at the command, as Motor.reach_joints builds it:
    -> (ref (Q, 24, 3) heading frame, total s, jaw arc length m, (jaw_e, Rw_e) the end's jaw centre and wrist rotation
    carried onto the true pelvis)."""
    ref0 = st["ref0"]
    pos, quat, base, fing = ref0[0:3], ref0[3:7], ref0[7:36].copy(), ref0[36:43]
    idx = [int(i) for i in st["idx"]]
    P = np.asarray(P, float)
    gap = base[idx] - P[0]

    def pose(q):
        j = base.copy()
        j[idx] = q
        return j

    jaw = np.array([fk.spheres("right", pos, quat, pose(q), "mj", fing)[1] for q in P])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(jaw, axis=0), axis=1))]
    length = float(arc[-1])
    total = max(REACH_MIN_S, length / REACH_SPEED)
    frac = arc / max(length, 1e-9)

    def joints_at(u):
        return pose(np.array([np.interp(u, frac, P[:, i]) for i in range(len(idx))]) + (1.0 - u) * gap)

    head, Rh, p0, Ra, pa = frames(st)
    u = np.linspace(0.0, 1.0, VM.Q)
    X = []
    for t in u * total:
        x, _ = fk.spheres("right", pos, quat, joints_at(minjerk(t / total)), "mj", fing)
        X.append(x @ Ra.T + pa)
    _, jw = fk.spheres("right", pos, quat, joints_at(1.0), "mj", fing)
    Rw = Ra @ fk.d.xmat[fk.wrist["right"]].reshape(3, 3)
    return head(np.array(X)), total, length, (Ra @ jw + pa, Rw)


def film_reference(fk, z, j0, t_cmd, dur, st):
    """jref_data.build's reference from the film (for the check of reference())."""
    head, Rh, p0, Ra, pa = frames(st)
    sel = (z["phase"] == "pregrasp") & (z["t"] >= t_cmd - 0.041)
    T, RF = z["t"][sel].astype(float), z["ref"][sel].astype(float)
    XR = np.array([fk.spheres("right", f[0:3], f[3:7], f[7:36], "mj", f[36:43])[0] @ Ra.T + pa for f in RF])
    return VM.interp(T - t_cmd, head(XR), np.linspace(0.0, 1.0, VM.Q) * dur)


def model_input(fk, st, ref, total, end) -> np.ndarray:
    """jref_data.build's x for this reference (no via points, speed 0.15, standing, the right hand)."""
    head, Rh, p0, Ra, pa = frames(st)
    W = st["W"]
    jaw = np.r_[(W[4:6] - p0[:2]) @ Rh[:2, :2], W[6]]
    R0 = Rh.T @ JD.r6m(st["J"])
    jaw_e, Rw_e = end
    Rg = Rh.T @ Rw_e
    key = [list(fk.links).index(x) for x in VM.KEY_LINKS]
    return np.r_[
        st["S3"].ravel(), jaw, R0[:, 0], R0[:, 1], head(jaw_e[None])[0], Rg[:, 0], Rg[:, 1], np.zeros(48 + 96 + 16), 0.0,
        JD.SPEED, 0.0, 0.0, 1.0, total, ref[::3, key].ravel(),
    ].astype(np.float32)  # fmt: skip


# --- the check -------------------------------------------------------------------------------------------------------


def box_sd(X, c, h):  # plan2.py's: signed distance of points X to axis-aligned cubes (centres c, half size h)
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def seen(X, pr):  # plan2.py's: the look's points where the robot believes they lie from its pelvis at the stance
    X = np.asarray(X, float)
    if "bel_pb" not in pr.files:
        return X
    dy = float(pr["bel_yt"]) - float(pr["bel_yb"])
    c_, s_ = np.cos(dy), np.sin(dy)
    out = X.copy()
    out[:, :2] = pr["bel_pt"][:2] + (X[:, :2] - pr["bel_pb"][:2]) @ np.array([[c_, s_], [-s_, c_]])
    return out


def obstacles(fk, st, ob, pr, cloud) -> tuple[np.ndarray, float]:
    """The voxels as plan2.py keeps them (world, true frame of the command), with this arm's spheres for the start."""
    V = seen(ob["vox"], pr)
    vs = float(ob["vox_size"])
    q = st["qpos0"]
    X0, _ = fk.spheres("right", q[0:3], q[3:7], q[st["qa"]], "mj", q[st["fa"]])
    drop = np.zeros(len(V), bool)
    for i, js in enumerate(cKDTree(V).query_ball_point(X0, fk.radius + FREE + vs)):
        for j in js:
            if box_sd(X0[i], V[j], 0.5 * vs) - fk.radius[i] < FREE:
                drop[j] = True
    tc = seen(np.asarray(cloud, float), pr)
    if len(V):
        drop |= cKDTree(tc).query(V)[0] < TGT + 0.5 * vs
    return V[~drop], vs


def sweep_clearance(Pw, radius, V, vs, until):
    """Pw (Q + 1, 24, 3) world: per sphere the smallest signed gap of its path (each leg cut in 4) to the cubes; legs
    from moments u < until only. -> (gap (24,) m, moment index of each sphere's smallest gap)."""
    u = np.r_[np.linspace(0.0, 1.0, VM.Q), 1.0]
    gap, at = np.full(Pw.shape[1], np.inf), np.zeros(Pw.shape[1], int)
    if len(V) == 0:
        return gap, at
    tree, kn = cKDTree(V), min(8, len(V))
    for k in range(len(u) - 1):
        if u[k] >= until:
            break
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            X = (1 - s) * Pw[k] + s * Pw[k + 1]
            nn = tree.query(X, k=kn)[1].reshape(len(X), kn)
            d = np.min(box_sd(X[:, None], V[nn], 0.5 * vs), axis=1) - radius
            better = d < gap
            gap[better], at[better] = d[better], k
    return gap, at


def check(a) -> None:
    import torch

    S, rnd = Path(a.s), int(a.round)
    M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
    fk = ArmFK(SCENE)
    link = list(fk.links)
    key = np.array([link.index(x) for x in VM.KEY_LINKS])
    hand2 = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
    palm = np.array([x == "hand_palm_link" for x in link])
    nxt = S / f"round{rnd + 1}"
    (nxt / "step1").mkdir(parents=True, exist_ok=True)
    (nxt / "step2").mkdir(parents=True, exist_ok=True)
    B = Path(a.bench)
    out = open(S / f"check_round{rnd}.jsonl", "w")
    for f in sorted(S.glob("state_*.npz")):
        name = f.stem[len("state_") :]
        if a.only and name not in a.only.split(","):
            continue
        st = np.load(f)
        pf = (Path(str(st["plan_src"])) if rnd == 0 else S / f"round{rnd}" / "step2") / f"plan_{name}.npz"
        row = dict(
            name=name,
            ep=int(st["ep"]),
            task=str(st["task"]),
            round=rnd,
            act=0.02 if "try_act02" in str(st["plan_src"]) else 0.03,
        )  # plan2.py's activation distance for J's plan
        if not pf.exists():
            out.write(json.dumps(dict(row, plan=False)) + "\n")
            continue
        pl = np.load(pf)
        assert [str(x) for x in pl["names"]] == [str(x) for x in st["names"]], "the plan's joints"
        ref, total, length, end = reference(fk, st, pl["Q"])
        T = dict(x=model_input(fk, st, ref, total, end)[None], ref=ref[None].astype(np.float32), key=key, hand2=hand2)
        off = VP.predict(M["point"], T, np.arange(1))[0]  # (Q + 1, 24, 3), heading frame
        P_h = np.concatenate([ref, ref[-1:]]) + off
        head, Rh, p0, Ra, pa = frames(st)
        to_w = lambda x: np.c_[p0[:2] + x[..., :2].reshape(-1, 2) @ Rh[:2, :2].T, x[..., 2].reshape(-1)].reshape(
            x.shape
        )  # noqa: E731
        pr = np.load(B / "step2" / f"prep_{name}.npz")  # the check is always against the look itself
        ob = np.load(B / "step1" / f"obst_{name}.npz")
        cur = S / f"round{rnd}" if rnd else B  # this round's planner inputs: the shifts so far add up
        pr_c, ob_c = np.load(cur / "step2" / f"prep_{name}.npz"), np.load(cur / "step1" / f"obst_{name}.npz")
        cloud = np.load(EP.OUT / a.cloud / f"{name}.npz")["pts"]
        V, vs = obstacles(fk, st, ob, pr, cloud)
        gap, at = sweep_clearance(to_w(P_h), fk.radius, V, vs, a.clear_until)
        gap0, _ = sweep_clearance(to_w(np.concatenate([ref, ref[-1:]])), fk.radius, V, vs, a.clear_until)
        end_off = off[-1, palm].mean(0)  # heading frame, m
        k = int(np.argmin(gap))
        clear = float(gap[k]) - a.r / 100
        fire_end, fire_clear = float(np.linalg.norm(end_off)) > a.tau_end / 100, clear < -a.delta / 100
        row.update(plan=True, dur=round(total, 3), path_cm=round(100 * length, 2), end_cm=round(100 * float(np.linalg.norm(end_off)), 2),
                   end_vec_cm=(100 * end_off).round(2).tolist(), clear_cm=round(100 * clear, 2), gap_pred_cm=round(100 * float(gap[k]), 2),
                   gap_ref_cm=round(100 * float(gap0.min()), 2), worst=link[k], worst_u=round(float(at[k]) / (VM.Q - 1), 3),
                   off_worst_cm=(100 * off[at[k], k]).round(2).tolist(), vox=len(V), fire_end=fire_end, fire_clear=fire_clear,
                   passed=not (fire_end or fire_clear), r_cm=a.r, tau_end_cm=a.tau_end, delta_cm=a.delta)  # fmt: skip
        if not row[
            "passed"
        ]:  # the next round's inputs: plan2.py reads prep_<name>.npz (goal) and obst_<name>.npz (voxels)
            prz = dict(pr_c)
            if fire_end:
                prz["goal_wrist"] = np.asarray(pr_c["goal_wrist"], float) - np.r_[Rh[:2, :2] @ end_off[:2], end_off[2]]
            np.savez(nxt / "step2" / f"prep_{name}.npz", **prz)
            obz = dict(ob_c)
            if fire_clear:
                o = off[at[k], k]
                obz["vox"] = np.asarray(ob_c["vox"], float) - np.r_[Rh[:2, :2] @ o[:2], o[2]]
            np.savez(nxt / "step1" / f"obst_{name}.npz", **obz)
            row["next"] = dict(goal_shift_cm=(100 * (prz["goal_wrist"] - pr["goal_wrist"])).round(2).tolist(),
                               vox_shift_cm=(100 * (obz["vox"][0] - ob["vox"][0])).round(2).tolist())  # fmt: skip
        out.write(json.dumps(row) + "\n")
        out.flush()
    out.close()
    print(open(S / f"check_round{rnd}.jsonl").read(), end="")


def choose(a) -> None:
    """The first candidate that passed, round by round; none: given up. -> <S>/via/via_<name>.npz (J's via file with its
    plan source moved to the chosen round's plan), <S>/choice.jsonl."""
    S = Path(a.s)
    (S / "via").mkdir(exist_ok=True)
    rows = {}
    for rnd in range(int(a.rounds) + 1):
        f = S / f"check_round{rnd}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            rows.setdefault(r["name"], []).append(r)
    out = open(S / "choice.jsonl", "w")
    for name, rs in sorted(rows.items()):
        ok = [r for r in rs if r.get("passed")]
        pick = ok[0] if ok else None
        c = dict(name=name, ep=rs[0]["ep"], rounds=len(rs), chosen=None if pick is None else pick["round"],
                 given_up=pick is None, first=rs[0])  # fmt: skip
        if pick is not None:
            src = Path(a.bench) / "via" / f"via_{name}.npz"
            v = dict(np.load(src, allow_pickle=True))
            if pick["round"]:
                v["src"] = np.array(str((S / f"round{pick['round']}" / "step2").resolve()))
            c["src"] = str(v["src"])
            np.savez(S / "via" / f"via_{name}.npz", **v)
        out.write(json.dumps(c) + "\n")
    out.close()
    print(open(S / "choice.jsonl").read(), end="")


def table(a) -> None:
    """A, J and S side by side: successes within 45 s / the official limit per task, S's given-up episodes failures;
    then the episodes J ran a plan on: S against J, paired (each J row and its S row are the same episode)."""
    B, S = Path(a.bench), Path(a.s)
    ch = {json.loads(x)["name"]: json.loads(x) for x in open(S / "choice.jsonl")}

    def rows(arm):
        out = {}
        for f in sorted((B / "rows").glob(f"{arm}_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                out[f"{r['task']}_L{r['level']}_e{r['index']}"] = r
        return out

    R = {"A": rows("A"), "J": rows(a.j_arm), a.arm: rows(a.arm)}
    gave_up = {n for n, c in ch.items() if c["given_up"]}
    ok = lambda arm, n, k: bool(R[arm].get(n, {}).get(k)) and not (arm == a.arm and n in gave_up)  # noqa: E731
    tasks = sorted({r["task"] for r in R["J"].values()}, key=lambda t: JD.TASKS.index(t) if t in JD.TASKS else 9)
    print(f"successes within 45 s / official limit ({a.arm}: {len(gave_up)} given up, counted failures)")
    print(f"  {'task':22s}" + "".join(f"{x:>12s}" for x in ("A", "J", "S")))
    tot = {x: [0, 0] for x in ("A", "J", a.arm)}
    for t in tasks + ["all"]:
        names = [n for n in R["J"] if t == "all" or R["J"][n]["task"] == t]
        cells = []
        for x in ("A", "J", a.arm):
            c45, co = sum(ok(x, n, "success_45") for n in names), sum(ok(x, n, "success_official") for n in names)
            cells.append(f"{c45} / {co}")
        print(f"  {t + f' ({len(names)})':22s}" + "".join(f"{c:>12s}" for c in cells))
    routed = sorted(ch)
    j = np.array([ok("J", n, "success_45") for n in routed])
    sx = np.array([ok(a.arm, n, "success_45") for n in routed])
    same_plan = [n for n in routed if ch[n]["chosen"] == 0]
    same = sum(R["J"][n].get("success_45") == R[a.arm].get(n, {}).get("success_45") and
               R["J"][n].get("t_success") == R[a.arm].get(n, {}).get("t_success") for n in same_plan)  # fmt: skip
    rng = np.random.default_rng(0)
    d = [float((sx[i] - j[i]).sum()) for i in (rng.integers(len(routed), size=len(routed)) for _ in range(2000))]
    print(
        f"\nepisodes J ran a plan on: {len(routed)}; S kept J's plan {len(same_plan)} (rows identical to J's: {same}), "
        f"replanned and passed {sum(1 for n in routed if (ch[n]['chosen'] or 0) > 0)}, gave up {len(gave_up)}"
    )
    print(
        f"  45 s successes: J {int(j.sum())}, S {int(sx.sum())}, S - J {int(sx.sum() - j.sum()):+d} (95 % interval "
        f"{np.percentile(d, 2.5):+.0f} to {np.percentile(d, 97.5):+.0f}), only S {int((sx & ~j).sum())}, only J {int((j & ~sx).sum())}"
    )
    for n in routed:
        c = ch[n]
        print(f"  {n:26s} chosen {str(c['chosen']):4s} J {int(ok('J', n, 'success_45'))} S {int(ok(a.arm, n, 'success_45'))} | round 0: end "
              f"{c['first'].get('end_cm')} cm, clear {c['first'].get('clear_cm')} cm (gap {c['first'].get('gap_pred_cm')}, reference "
              f"{c['first'].get('gap_ref_cm')}), worst {c['first'].get('worst')} at u {c['first'].get('worst_u')}")  # fmt: skip


def calib(a) -> None:
    """The error radius on paths the model never saw (a jref_data.py build of J's runs filmed with the reference): the
    swept-region error of the model's prediction (and of the bare reference), quantiles in cm."""
    import torch

    M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
    D = [Path(x) for x in a.data]
    T = VP.load(D)
    rows = np.arange(len(T["seg"]))
    off = VP.offsets(M, T, rows)
    out = {}
    for p in ("reference", "direction"):
        e, eh = VP.errors(D, T, off[p], rows)
        q = {f"q{int(k)}": round(100 * float(np.percentile(e, k)), 2) for k in (50, 80, 90, 95)}
        out[p] = dict(n=len(e), **q, max=round(100 * float(e.max()), 2), palm_median=round(100 * float(np.median(eh)), 2),
                      inside_fixed=round(float(np.mean(e <= M["fixed"][p])), 3), fixed_cm=round(100 * float(M["fixed"][p]), 2))  # fmt: skip
    print(json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("state")
    p.add_argument("bench")
    p.add_argument("s")
    p = sub.add_parser("check")
    p.add_argument("s")
    p.add_argument("model")
    p.add_argument("bench", help="the J arm's bench dir (step1 obstacles, step2 preps)")
    p.add_argument("round", help="0: J's own plan (its via file's source); k: <S>/round<k>/step2")
    p.add_argument("--cloud", default="cloud_blind")
    p.add_argument("--tau-end", type=float, default=2.0, help="cm")
    p.add_argument("--delta", type=float, default=0.0, help="cm of predicted penetration allowed")
    p.add_argument("--r", type=float, default=3.0, help="cm, the model's error radius")
    p.add_argument("--clear-until", type=float, default=1.01, help="moments u below this are checked for clearance")
    p.add_argument("--only", default="")
    p = sub.add_parser("choose")
    p.add_argument("s")
    p.add_argument("bench")
    p.add_argument("rounds")
    p = sub.add_parser("table")
    p.add_argument("s")
    p.add_argument("bench")
    p.add_argument("--arm", default="S")
    p.add_argument("--j-arm", default="J", help="the rows that stand for J (JR on the development episodes: J filmed)")
    p = sub.add_parser("calib")
    p.add_argument("model")
    p.add_argument("data", nargs="+")
    a = ap.parse_args()
    dict(state=state, check=check, choose=choose, table=table, calib=calib)[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ = shutil
    main()
