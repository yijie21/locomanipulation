"""bigym_selfmodel_20261124: BiGym films -> datasets that selfmodel_via_20261120/via_point.py trains and evaluates as it
is (table.npz + arm.npz in via_model.build's format), simple_eval_20261112/jref_data.py's way; the joint-range check; the
leave-one-out constant on the real episodes; the collection's statistics.

    source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh; cd <repo>    # numpy, mujoco, yaml
    python scripts/experiments/bigym_20261124/bigym_data.py build <out dir> <film dir> <rows.jsonl ...>
    python .../bigym_data.py range <check dataset> <training dataset ...>
    python .../bigym_data.py stats <rows.jsonl ...>                       # the collection's planning, falls, time
    $SIMPLE/.venv/bin/python .../bigym_data.py loo <replay dataset> [--model <via_point model dir>]   # torch for --model
    python .../bigym_data.py build --end early <out dir> <film dir> <rows.jsonl ...>   # bigym_endpoint_20261125's label
    python .../bigym_data.py freeze <replay dataset> <constants.npz>      # per-task mean offsets, frozen
    $SIMPLE/.venv/bin/python .../bigym_data.py frozen <replay dataset> <constants.npz> [--model <model dir>]

Rows of two kinds. bigym_collect.py's (one per stance, a stage per sample, phases jref<k> + post<k> of the stance's
film) and bigym_bench.py's J (--stance, --film: one per episode with a plan and no fall, phase "reach" and what follows
it before the arm moves again or the episode ends, at most SETTLE_MAX_S + POST_S after the reach: the reach task's hold,
the drawer's settle, 0.5 s).

The reference is what the right arm's PD is sent (the film's "sent": the planned joints retimed, the start gap faded);
its spheres by BiGym's own model (the robot's MJCF, standalone) with every other joint and the floating base at their
readings at the command (the controller is told to hold them) and the right fingers at their reading then (open since
the reset). The real spheres are the simulated robot's at every control step. The 24 spheres: spheres24.py. Everything in
the heading frame of the real pelvis at the command (origin its x, y; z world; x its heading). Times as jref_data: the
real sweep resampled at 10 Hz from the command to the window's end, the reference over its own duration (the reach's
dur_s); offsets at via_model.Q = 24 equal moments of it plus the settled point (the mean of the last TAIL = 5 ticks
against the reference's end). A window needs TAIL ticks after the reach (0.5 s; jref_data asked 0.6 s, but the drawer's
sequence moves on 0.5 s after the reach).

x is via_model.build's layout with no via points: the state at the command and 0.1 and 0.2 s before (65 each: the 29 body
joints in the MJCF's order, their speeds, gravity in the pelvis frame, the pelvis's angular velocity in its frame, its
height), right_end_effector's position and attitude, the goal (the reference's last pose's right_end_effector) and its
attitude, 160 zeros (the via points), 0 via points, speed 0.15, the height asked (crouch flag and value: the drawer's
squat command, standing -1), the right hand, the reach's duration, the reference sweep of the 5 key spheres. Table
columns as jref_data's, plus task (0 reach, 1 drawer) and seed. Splits: collection rows by stance, bucket = stance id %
10 (dealt in turn by seed; the two tasks' stances are separate id blocks, so each spreads over every bucket), route =
the stance id (the bootstrap resamples stances); bench rows bucket 9, route -1 (one path each).

loo: the swept-region error (via_model.errors, the same definition) of four predictions on a replay dataset: the
reference (offset 0), the constant from the model's training paths (via_point's, with --model), the leave-one-out
constant (each episode the mean offset of the OTHER episodes of the same task; and, as a second reading, of all other
episodes), and the model (--model); medians, drops with 95 % intervals resampling episodes, and the palm's settled-point
error (the 3 palm spheres' mean offset error at the settled point).

bigym_endpoint_20261125 (added; without --end the build is unchanged, bit for bit):
  build --end early  each window ends 0.5 s (TAIL_S) after the reach's last step (the film's reach / jref<k> phase), so
                     the settled point is the mean of the last TAIL = 5 ticks at 10 Hz in the 0.5 s after the reach: the
                     drawer protocol's (it pushes in then) and the same on the training paths; the real drawer windows
                     already end there; the 24 moments' offsets do not change, the real sweep (arm.npz) ends earlier
  freeze             a replay's per-task mean offsets (Q + 1, 24, 3) into a file, to score other episodes against later
  frozen             a replay scored against such a file: reference, the frozen constant (its task's), the
                     leave-one-out constant within the replay (same task), the model (--model); swept-region error and
                     the end point (the palm's 3 spheres' mean, the settled point), per task; drops with 95 % intervals
                     resampling episodes and how many episodes each first predictor is better on
  loo                also prints the end point's drops and the better-on counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
import via_model as VM  # noqa: E402

SPEED, TAIL_S, SETTLE_MAX_S, POST_S, DT = 0.15, 0.5, 1.5, 1.0, 0.02
EE_OFF = np.array([0.148, 0.0, 0.0])  # right_end_effector in right_wrist_yaw_link (BiGym's G1Dex1._EE_SITE_POS)
MJCF = "/wuji-vefps-D/wuji-il/yijie/tools/bigym2/BiGym2/bigym/envs/xmls/g1/g1_29dof_with_dex1_1.xml"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
TASK_ID = {"reach_target_multi_modal": 0, "drawer_top_open": 1}
SQUAT_H = 0.62  # bigym_bench.py's --stance squat (the bench rows' height asked)


def quat_R(q) -> np.ndarray:
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


class Body:
    """BiGym's robot alone (its MJCF): the 24 spheres and right_end_effector for a floating base and named joints."""

    def __init__(self):
        import mujoco
        from spheres24 import Spheres24

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        self.d = mujoco.MjData(self.m)
        self.sph = Spheres24()
        self.wrist = self.m.body("right_wrist_yaw_link").id
        self.free = self.m.jnt_qposadr[[j for j in range(self.m.njnt) if self.m.jnt_type[j] == 0][0]]
        self._names = None

    def bind(self, jnt: list[str]) -> None:
        if self._names != jnt:
            self._names = list(jnt)
            self.adr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in jnt])

    def pose(self, root, qj):
        """-> (24, 3) sphere centres, right_end_effector's position and rotation (world)."""
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.free : self.free + 7] = root
        d.qpos[self.adr] = qj
        self.mj.mj_kinematics(self.m, d)
        Rw = d.xmat[self.wrist].reshape(3, 3)
        return self.sph.world(self.m, d), d.xpos[self.wrist] + Rw @ EE_OFF, Rw.copy()


def windows(rows_files: list[str], films: Path):
    """-> (row, stage or None, film path, i_cmd, i_end, dur, h_cmd, kind) for every usable reach."""
    for f in rows_files:
        for line in open(f):
            r = json.loads(line)
            if "stages" in r:  # bigym_collect.py
                fp = films / f"{r['task']}_{r['seed']}.npz"
                for s in r["stages"]:
                    if "skip" in s or s.get("fell") or "i_end" not in s:
                        continue
                    yield r, s, fp, int(s["i_cmd"]), int(s["i_end"]), float(s["dur_s"]), float(r["h_cmd"]), "collect"
            elif r.get("method") == "J" and r.get("plan_ok") and not r.get("fell"):  # bigym_bench.py
                fp = films / f"{r['task']}_{r['seed']}_J{'_stance' if r.get('stance') else ''}.npz"
                h = SQUAT_H if (r["task"] == "drawer_top_open" and r.get("stance")) else -1.0
                yield r, None, fp, -1, -1, float(r["reach"]["dur_s"]), h, "bench"


def bench_window(z, dur: float) -> tuple[int, int]:
    ph = z["phase"]
    i_cmd = int(np.where(ph == "reach")[0][0]) - 1
    i_end = len(ph) - 1
    after = np.where((np.arange(len(ph)) > i_cmd) & (ph != "reach"))[0]
    if len(after):
        stop = [i for i in after if ph[i] in ("in", "in_hold", "close", "pull", "release")]
        i_end = (stop[0] - 1) if stop else len(ph) - 1
    cap = i_cmd + int(round((dur + SETTLE_MAX_S + POST_S) / DT))
    return i_cmd, min(i_end, cap)


def state(z, i: int, body: list[int]) -> np.ndarray:
    """The 65-D state at film entry i: body joints, their speeds, gravity and angular velocity (pelvis frame), height."""
    Rp = quat_R(z["root"][i][3:7])
    return np.r_[z["qj"][i][body], z["dqj"][i][body], Rp.T @ np.array([0.0, 0.0, -1.0]), z["root_v"][i][3:6],
                 z["root"][i][2]].astype(np.float32)  # fmt: skip


def build(a) -> None:
    B = Body()
    link = list(B.sph.links)
    key = [link.index(x) for x in VM.KEY_LINKS]
    u = np.linspace(0.0, 1.0, VM.Q)
    Xall = []
    rows = {k: [] for k in ("x", "ref", "off", "a", "b", "reset", "route", "sid", "k", "S0", "pelvis_z", "walked",
                            "bucket", "task", "seed", "ee_dev")}  # fmt: skip
    n_skip = dict(film=0, short=0, no_reach=0)
    cache, body_names = {}, None
    for r, s, fp, i_cmd, i_end, dur, h_cmd, kind in windows(a.rows, Path(a.films)):
        if not fp.exists():
            n_skip["film"] += 1
            continue
        if fp not in cache:
            cache.clear()
            cache[fp] = dict(np.load(fp))
        z = cache[fp]
        if kind == "bench":
            if not (z["phase"] == "reach").any():  # the episode ended (success) during the walk, before any reach
                n_skip["no_reach"] += 1
                continue
            i_cmd, i_end = bench_window(z, dur)
        if a.end == "early":  # the window ends TAIL_S after the reach's last step
            name = "reach" if kind == "bench" else f"jref{s['k']}"
            n_reach = int((z["phase"][i_cmd + 1 : i_end + 1] == name).sum())
            i_end = min(i_end, i_cmd + n_reach + int(round(TAIL_S / DT)))
        jnt = [str(x) for x in z["jnt"]]
        B.bind(jnt)
        body = [i for i, n in enumerate(jnt) if "dex1" not in n]
        assert len(body) == 29, len(body)
        assert body_names in (None, [jnt[i] for i in body]), "the films' joint orders differ"
        body_names = [jnt[i] for i in body]
        arm = [jnt.index(n) for n in ARM]
        t = z["t"].astype(float)
        T = t[i_cmd : i_end + 1] - t[i_cmd]
        if i_cmd < 10 or T[-1] < dur + TAIL_S - 1e-6:
            n_skip["short"] += 1
            continue
        root0, q0 = z["root"][i_cmd].astype(float), z["qj"][i_cmd].astype(float)
        p0 = root0[:3]
        R0 = quat_R(root0[3:7])
        yaw = math.atan2(R0[1, 0], R0[0, 0])
        Rh = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        org = np.array([p0[0], p0[1], 0.0])

        def head(X):
            return (np.asarray(X, float) - org) @ Rh

        Xa, XR, dev = [], [], 0.0
        for i in range(i_cmd, i_end + 1):
            X, ee, _ = B.pose(z["root"][i], z["qj"][i])
            Xa.append(X)
            dev = max(dev, float(np.linalg.norm(ee - z["ee_p"][i])))  # our FK against the simulator's site
            qr = q0.copy()
            qr[arm] = z["sent"][i]
            XR.append(B.pose(root0, qr)[0])
        _, ee_e, Rw_e = B.pose(root0, qr)  # the reference's last pose
        Xa, XR = head(np.array(Xa)), head(np.array(XR))
        t10 = 0.1 * np.arange(int(np.floor(T[-1] / 0.1 + 1e-9)) + 1)
        X10 = VM.interp(T, Xa, t10)
        ref = VM.interp(T, XR, u * dur)
        act = VM.interp(t10, X10, u * dur)
        off = np.concatenate([act - ref, (X10[-VM.TAIL :].mean(0) - ref[-1])[None]])
        S3 = np.stack([state(z, i, body) for i in (i_cmd, i_cmd - 5, i_cmd - 10)])
        jaw = head(z["ee_p"][i_cmd])
        Rj = Rh.T @ quat_R(z["ee_q"][i_cmd])
        Rg = Rh.T @ Rw_e
        crouch = h_cmd > 0
        rows["x"].append(np.r_[
            S3.ravel(), jaw, Rj[:, 0], Rj[:, 1], head(ee_e), Rg[:, 0], Rg[:, 1], np.zeros(48 + 96 + 16), 0.0, SPEED,
            float(crouch), max(h_cmd, 0.0), 1.0, dur, ref[::3, key].ravel(),
        ].astype(np.float32))  # fmt: skip
        b0 = sum(len(x) for x in Xall)
        Xall.append(X10.astype(np.float32))
        rows["ref"].append(ref.astype(np.float32))
        rows["off"].append(off.astype(np.float32))
        rows["a"].append(b0)
        rows["b"].append(b0 + len(X10))
        seed = int(r["seed"])
        rows["reset"].append(seed)
        rows["seed"].append(seed)
        rows["task"].append(TASK_ID[r["task"]])
        rows["pelvis_z"].append(h_cmd if crouch else -1.0)
        rows["S0"].append(S3[0, :29])
        rows["ee_dev"].append(dev)
        if kind == "collect":
            rows["route"].append(int(r["sid"]))
            rows["sid"].append(int(r["sid"]))
            rows["k"].append(int(s["k"]))
            rows["bucket"].append(int(r["sid"]) % 10)
            w = r.get("walk", {})
            rows["walked"].append(int(abs(w.get("moved_cm", 0)) + abs(w.get("side_moved_cm", 0)) > 0))
        else:
            rows["route"].append(-1)
            rows["sid"].append(-1)
            rows["k"].append(0)
            rows["bucket"].append(9)
            w = r.get("walk") or {}
            rows["walked"].append(int(abs(w.get("moved_cm", 0)) + abs(w.get("side_moved_cm", 0)) > 0))
    n = len(rows["x"])
    assert n, f"no rows ({n_skip})"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T_ = dict(x=np.array(rows["x"]), ref=np.array(rows["ref"]), off=np.array(rows["off"]), seg=np.arange(n),
              a=np.array(rows["a"]), b=np.array(rows["b"]), reset=np.array(rows["reset"]), side=np.ones(n),
              pelvis_z=np.array(rows["pelvis_z"]), bends=np.zeros(n), route=np.array(rows["route"]),
              walked=np.array(rows["walked"]), phi=np.full(n, -1.0), sector=np.zeros(n, bool),
              bucket=np.array(rows["bucket"]), palm=np.array([x == "hand_palm_link" for x in link]),
              held=np.zeros(n, bool), sid=np.array(rows["sid"]), k=np.array(rows["k"]), S0=np.array(rows["S0"]),
              task=np.array(rows["task"]), seed=np.array(rows["seed"]))  # fmt: skip
    np.savez_compressed(out / "table.npz", **T_)
    np.savez_compressed(out / "arm.npz", X=np.concatenate(Xall), radius=B.sph.radius, link=np.array(link),
                        link_bigym=np.array(B.sph.bodies), local=B.sph.local, jnt=np.array(body_names))  # fmt: skip
    e, eh = VM.errors(out, T_, np.zeros_like(T_["off"], dtype=float))
    palm_set = np.linalg.norm(T_["off"][:, VM.Q][:, T_["palm"]].mean(1), axis=1)
    print(f"{out}: {n} paths ({n_skip} skipped), tasks {np.bincount(T_['task'], minlength=2).tolist()}, buckets "
          f"{np.bincount(T_['bucket'], minlength=10).tolist()}; reference-vs-actual: swept-region error median "
          f"{100 * np.median(e):.2f} cm (p90 {100 * np.percentile(e, 90):.2f}), palm {100 * np.median(eh):.2f} cm, palm "
          f"settled point {100 * np.median(palm_set):.2f} cm (p90 {100 * np.percentile(palm_set, 90):.2f}); our FK "
          f"against the simulator's right_end_effector: max {1000 * max(rows['ee_dev']):.3f} mm")  # fmt: skip
    for tid, name in ((0, "reach"), (1, "drawer")):
        for k in (None, 0, 1):
            m_ = (T_["task"] == tid) & ((T_["k"] == k) if k is not None else True)
            if m_.any():
                print(f"  {name}{'' if k is None else f', reach {k + 1}'}: {int(m_.sum())} paths, swept median "
                      f"{100 * np.median(e[m_]):.2f} cm (p90 {100 * np.percentile(e[m_], 90):.2f}), palm settled median "
                      f"{100 * np.median(palm_set[m_]):.2f} cm, duration median {np.median(T_['x'][m_, 378]):.2f} s")  # fmt: skip


def joint_range(a) -> None:
    """jref_data.range for BiGym: the check's joints at the command against the training paths' 0.5-99.5 % band, the 7
    right-arm joints (the registration's count) and all 29."""
    jn = [str(x) for x in np.load(Path(a.check) / "arm.npz")["jnt"]]
    for d in a.train:
        assert [str(x) for x in np.load(Path(d) / "arm.npz")["jnt"]] == jn, "joint orders differ"
    X = np.concatenate([(lambda t: t["S0"][t["bucket"] < 8])(np.load(Path(d) / "table.npz")) for d in a.train])
    lo, hi = np.percentile(X, [0.5, 99.5], axis=0)
    ck = np.load(Path(a.check) / "table.npz")
    out29 = ((ck["S0"] < lo) | (ck["S0"] > hi)).sum(1)
    arm = np.array([jn.index(n) for n in ARM])
    out7 = ((ck["S0"][:, arm] < lo[arm]) | (ck["S0"][:, arm] > hi[arm])).sum(1)
    print(f"joints at the command outside the training paths' 0.5-99.5 % band ({len(out7)} reaches, {len(X)} training "
          f"paths): right arm (7) median {np.median(out7):.1f}, mean {out7.mean():.2f}, max {out7.max()}, counts "
          f"{np.bincount(out7).tolist()}; all 29 median {np.median(out29):.1f}, max {out29.max()}")  # fmt: skip


def stats(a) -> None:
    rows = [json.loads(line) for f in a.rows for line in open(f)]
    samples = [s for r in rows for s in r["stages"]]
    tries = [t for s in samples for t in s["tries"]]
    ok = [s for s in samples if "skip" not in s]
    print(f"{len(rows)} stances, {len(samples)} samples, planned {len(ok)} ({len(ok) / max(len(samples), 1):.2f} a "
          f"sample), attempts {len(tries)} ({sum(t['ok'] for t in tries) / max(len(tries), 1):.2f}"
          f" planned an attempt); falls {sum(r['fell'] for r in rows)}, episodes ended early "
          f"{sum(bool(r.get('done')) for r in rows)}")  # fmt: skip
    st = {}
    for t in tries:
        st[t["status"]] = st.get(t["status"], 0) + 1
    print(f"  attempt statuses {st}; near goals planned {np.mean([t['ok'] for t in tries if t['near']]):.2f}"
          f", far {np.mean([t['ok'] for t in tries if not t['near']]):.2f}; plan wall median "
          f"{np.median([t['t'] for t in tries]):.2f} s (ok {np.median([t['t'] for t in tries if t['ok']]):.2f}"
          f", failed {np.median([t['t'] for t in tries if not t['ok']] or [np.nan]):.2f})")  # fmt: skip
    wall = sum(r["wall_s"] for r in rows)
    print(f"  wall {wall:.0f} s over {len(ok)} reaches: {wall / max(len(ok), 1):.2f} s a reach in one process; stance "
          f"wall median {np.median([r['wall_s'] for r in rows]):.1f} s; plan attempts a planned sample "
          f"{len(tries) / max(len(ok), 1):.2f}; reach duration median {np.median([s['dur_s'] for s in ok]):.2f} s; "
          f"planner's start FK vs BiGym max {max([s.get('fk_mm', 0) for s in ok] or [0]):.3f} mm")  # fmt: skip
    for task in sorted({r["task"] for r in rows}):
        rr = [r for r in rows if r["task"] == task]
        ss = [s for r in rr for s in r["stages"]]
        print(f"  {task}: {len(rr)} stances, planned {sum('skip' not in s for s in ss)}/{len(ss)}, falls "
              f"{sum(r['fell'] for r in rr)}, walk {np.median([r['walk']['moved_cm'] for r in rr]):.1f} cm ahead "
              f"{np.median([r['walk']['side_moved_cm'] for r in rr]):.1f} cm aside (medians)")  # fmt: skip
        for k in sorted({s["k"] for s in ss}):
            sk = [s for s in ss if s["k"] == k and "skip" not in s]
            pen = np.array([s.get("body_contact_mm", 0.0) for s in sk])
            print(f"    reach {k + 1}: {len(sk)} planned, attempts a sample {np.mean([len(s['tries']) for s in sk]):.2f}, "
                  f"duration median {np.median([s['dur_s'] for s in sk]):.2f} s, settle median "
                  f"{np.median([s['settle_s'] for s in sk]):.2f} s (unsettled {sum(not s['settled'] for s in sk)}); the "
                  f"arm against its own body: {int((pen > 0).sum())} reaches touch, {int((pen > 5).sum())} over 5 mm, "
                  f"worst {pen.max():.1f} mm")  # fmt: skip
            pairs = {}
            for s in sk:
                for pr in s.get("body_contact_pairs", []):
                    pairs[" / ".join(pr)] = pairs.get(" / ".join(pr), 0) + 1
            if pairs:
                print(f"      touching pairs (reaches): {dict(sorted(pairs.items(), key=lambda x: -x[1]))}")


def loo(a) -> None:
    import via_point as VP

    R = Path(a.replay)
    T = VP.load([R])
    n = len(T["seg"])
    rows = np.arange(n)
    preds = dict(reference=np.zeros((n, VM.Q + 1, 24, 3)))
    same, allo = np.zeros_like(preds["reference"]), np.zeros_like(preds["reference"])
    for j in range(n):
        o = rows != j
        same[j] = T["off"][o & (T["task"] == T["task"][j])].mean(0)
        allo[j] = T["off"][o].mean(0)
    preds["loo_same_task"], preds["loo_all"] = same, allo
    if a.model:
        import torch

        M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
        off = VP.offsets(M, T, rows)
        preds["constant_train"], preds["model"] = off["constant"], off["direction"]
    res = {}
    for name, p in preds.items():
        e, eh = VP.errors([R], T, p, rows)
        ps = np.linalg.norm((T["off"][:, VM.Q] - p[:, VM.Q])[:, T["palm"]].mean(1), axis=1)
        res[name] = dict(e=e, eh=eh, ps=ps)
    rng = np.random.default_rng(0)
    for tname, m_ in (("all", np.ones(n, bool)), ("reach", T["task"] == 0), ("drawer", T["task"] == 1)):
        idx = np.where(m_)[0]
        if len(idx) < 3:
            continue
        print(f"{R.name}, {tname}: {len(idx)} episodes")
        for name, v in res.items():
            print(f"  {name:15s} swept-region error median {100 * np.median(v['e'][idx]):5.2f} p90 "
                  f"{100 * np.percentile(v['e'][idx], 90):5.2f} cm | hand {100 * np.median(v['eh'][idx]):5.2f} cm | palm "
                  f"settled point median {100 * np.median(v['ps'][idx]):5.2f} cm")  # fmt: skip

        def drop(x, y):
            b = []
            for _ in range(2000):
                j = idx[rng.integers(len(idx), size=len(idx))]
                b.append(1 - np.median(res[x]["e"][j]) / np.median(res[y]["e"][j]))
            lo, hi = np.percentile(b, [2.5, 97.5])
            return f"{1 - np.median(res[x]['e'][idx]) / np.median(res[y]['e'][idx]):.2f} ({lo:.2f} to {hi:.2f})"

        pairs = [("loo_same_task", "reference")]
        if "model" in res:
            pairs += [
                ("model", "reference"),
                ("model", "loo_same_task"),
                ("model", "loo_all"),
                ("model", "constant_train"),
            ]
        print("  drops of the median (95 % interval, resampling episodes): " + "; ".join(
            f"{x} vs {y} {drop(x, y)}" for x, y in pairs))  # fmt: skip
        better_lines(res, idx, pairs, f"loo{tname}")


def scores(R: Path, T: dict, preds: dict) -> dict:
    """Each prediction's swept-region error (all spheres, palm) and end-point error (the palm's 3 spheres' mean at the
    settled point) for every row of the replay T."""
    import via_point as VP

    rows, res = np.arange(len(T["seg"])), {}
    for name, p in preds.items():
        e, eh = VP.errors([R], T, p, rows)
        ps = np.linalg.norm((T["off"][:, VM.Q] - p[:, VM.Q])[:, T["palm"]].mean(1), axis=1)
        res[name] = dict(e=e, eh=eh, ps=ps)
    return res


def drops(res: dict, idx: np.ndarray, pairs: list, field: str, tag: str) -> str:
    """1 - median(x) / median(y) with a 95 % interval (2000 draws of the episodes, a seed per tag and field)."""
    out = []
    for x, y in pairs:
        rng = np.random.default_rng(int(hashlib.md5(f"{tag}/{field}/{x}/{y}".encode()).hexdigest(), 16) % 2**32)
        b = []
        for _ in range(2000):
            j = idx[rng.integers(len(idx), size=len(idx))]
            b.append(1 - np.median(res[x][field][j]) / np.median(res[y][field][j]))
        lo, hi = np.percentile(b, [2.5, 97.5])
        out.append(
            f"{x} vs {y} {1 - np.median(res[x][field][idx]) / np.median(res[y][field][idx]):.2f} ({lo:.2f} to {hi:.2f})"
        )
    return "; ".join(out)


def better_lines(res: dict, idx: np.ndarray, pairs: list, tag: str) -> None:
    print(f"  drops of the end point median (95 % interval, resampling episodes): {drops(res, idx, pairs, 'ps', tag)}")
    for field, label in (("e", "swept-region error"), ("ps", "end point")):
        print(f"  better on ({label}): " + "; ".join(
            f"{x} than {y} {int((res[x][field][idx] < res[y][field][idx]).sum())}/{len(idx)}" for x, y in pairs))  # fmt: skip


def same_task_loo(T: dict) -> np.ndarray:
    n = len(T["seg"])
    out = np.zeros((n, VM.Q + 1, 24, 3))
    for j in range(n):
        out[j] = T["off"][(np.arange(n) != j) & (T["task"] == T["task"][j])].mean(0)
    return out


def freeze(a) -> None:
    """A replay's per-task mean offsets, saved to score other episodes against (bigym_endpoint_20261125)."""
    R = Path(a.replay)
    T = dict(np.load(R / "table.npz"))
    const = np.zeros((2, VM.Q + 1, 24, 3))
    n = []
    for tid in (0, 1):
        m = T["task"] == tid
        const[tid] = T["off"][m].astype(float).mean(0)
        n.append(int(m.sum()))
    md5 = hashlib.md5(open(R / "table.npz", "rb").read()).hexdigest()
    np.savez(a.out, const=const, n=np.array(n), source=str(R.resolve()), table_md5=md5, seeds=T["seed"], task=T["task"])
    palm = T["palm"]
    print(f"{a.out}: per-task mean offsets of {R} (table.npz md5 {md5}): reach {n[0]} episodes, drawer {n[1]}; the end "
          f"point's offset (palm mean, cm): reach {np.round(100 * const[0, VM.Q][palm].mean(0), 2).tolist()}, drawer "
          f"{np.round(100 * const[1, VM.Q][palm].mean(0), 2).tolist()}")  # fmt: skip


def frozen(a) -> None:
    """A replay scored against frozen per-task constants (freeze), beside the reference, the leave-one-out constant
    within the replay and the model."""
    import via_point as VP

    R = Path(a.replay)
    T = VP.load([R])
    n = len(T["seg"])
    C = np.load(a.const)
    preds = dict(reference=np.zeros((n, VM.Q + 1, 24, 3)), frozen_constant=C["const"][T["task"]].astype(float),
                 loo_same_task=same_task_loo(T))  # fmt: skip
    if a.model:
        import torch

        M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
        preds["model"] = VP.offsets(M, T, np.arange(n))["direction"]
    res = scores(R, T, preds)
    print(f"{R} against {a.const} (from {C['source']}, {C['n'].tolist()} episodes; table md5 {C['table_md5']})"
          f"{', model ' + str(a.model) if a.model else ''}")  # fmt: skip
    pairs = [("frozen_constant", "reference"), ("loo_same_task", "reference")]
    if "model" in res:
        pairs += [("model", "reference"), ("model", "frozen_constant"), ("model", "loo_same_task")]
    for tname, m_ in (("all", np.ones(n, bool)), ("reach", T["task"] == 0), ("drawer", T["task"] == 1)):
        idx = np.where(m_)[0]
        if len(idx) < 3:
            continue
        print(f"{R.name}, {tname}: {len(idx)} episodes")
        for name, v in res.items():
            print(f"  {name:15s} swept-region error median {100 * np.median(v['e'][idx]):5.2f} p90 "
                  f"{100 * np.percentile(v['e'][idx], 90):5.2f} cm | hand {100 * np.median(v['eh'][idx]):5.2f} cm | end "
                  f"point median {100 * np.median(v['ps'][idx]):5.2f} p90 {100 * np.percentile(v['ps'][idx], 90):5.2f} cm")  # fmt: skip
        print(f"  drops of the swept-region error median (95 % interval, resampling episodes): "
              f"{drops(res, idx, pairs, 'e', f'frozen{tname}')}")  # fmt: skip
        better_lines(res, idx, pairs, f"frozen{tname}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("films")
    b.add_argument("rows", nargs="+")
    b.add_argument("--end", choices=("early",), default=None, help="early: the window ends 0.5 s after the reach")
    r = sub.add_parser("range")
    r.add_argument("check")
    r.add_argument("train", nargs="+")
    s = sub.add_parser("stats")
    s.add_argument("rows", nargs="+")
    lo = sub.add_parser("loo")
    lo.add_argument("replay")
    lo.add_argument("--model", default=None)
    fz = sub.add_parser("freeze")
    fz.add_argument("replay")
    fz.add_argument("out")
    fr = sub.add_parser("frozen")
    fr.add_argument("replay")
    fr.add_argument("const")
    fr.add_argument("--model", default=None)
    a = ap.parse_args()
    dict(build=build, range=joint_range, stats=stats, loo=loo, freeze=freeze, frozen=frozen)[a.cmd](a)


if __name__ == "__main__":
    main()
