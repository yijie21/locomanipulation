"""Hindsight relabelling, step 1 (2026-11-04, user: learn the self model's correction into the controller itself).

Every recorded reach is a correct demonstration of reaching the jaw pose the hand really ended in. This replays a subset of
gen_reach.py's resets (same seeds and flags, so the same reaches the self model was trained on) and records, for every
reach, each 50 Hz tick from the command to the end of the 1 s hold after it:
    what ScaleBFM saw    the policy's packed inputs: the 3-tick history (root quat, angular velocity, joints, joint
                         velocities, last actions), the root position, the 6 future keypoint frames of the 5 bodies mode 4
                         reads (pelvis, both ankles, both wrists), world frame
    what it did          the 29 raw actions it returned
    the relabelled frames  the same keypoint frames of the reference the reach WOULD have had if it had been commanded to
                         the final jaw centre and wrist rotation: reach()'s own whole-body IK plan from the reach's start
                         state (the simulator state is copied at the command and restored after), natural timing for that
                         target, pelvis height and the other hand as commanded; the ankles stay pinned where they were
relabel_train.py fine-tunes a copy of the controller on (relabelled frames, same history) -> the same action.
RELABEL_TARGET=wrist (2026-11-05, after the hybrid step: a controller that learned the position correction too became
harder for the self model to predict) relabels only the wrist: the relabelled command is the COMMANDED jaw centre with
the wrist rotation the hand really ended in, so the student learns to turn the wrist where it is asked and keeps the
frozen controller's position behaviour (where the hand lands for a command), which the self model inverts.

Checks: every job re-plans its first VERIFY reaches with the original command from the stored start state and asserts the
recomputed frames equal what the policy was fed, bit for bit (the relabelled frames come from the same code path); phase
`check` compares jaw_err_cm and reach_s with the stored datasets (the replay is deterministic).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/relabel_gen.py run --procs 6
    $PY scripts/experiments/reachwm_20261030/relabel_gen.py smoke     # 2 resets of data_wrist, then check
    $PY scripts/experiments/reachwm_20261030/relabel_gen.py check
Writes outputs/reachwm_20261030/relabel/data/<dataset>_<job>_w0_<chunk>.npz (+ <dataset>_<job>.done, logs/).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402

OUT = E.REPO / "outputs" / "reachwm_20261030" / "relabel"
DATA = Path(os.environ.get("RELABEL_DATA", OUT / "data"))
STORED = E.REPO / "outputs" / "reachwm_20261030"
# gen_reach.py's four reach datasets (the self model's training data, its docstring): flags, seed, resets; `every` picks
# the replayed subset (resets 0, every, 2 every, ...), `per` resets go to one job (about 600 reaches)
DATASETS = {
    "ref": dict(env={}, seed=3000000, resets=20000, every=5, per=300),
    "side": dict(env=dict(REACH_SIDESTEP="1", REACH_CHAIN="1", REACH_ID0="100000"), seed=5000000, resets=15000, every=3, per=600),
    "wrist": dict(env=dict(REACH_WRIST="1", REACH_SIDESTEP="1", REACH_CHAIN="1", REACH_ID0="200000"), seed=7000000,
                  resets=18000, every=3, per=600),
    "wrist2": dict(env=dict(REACH_WRIST="1", REACH_WRIST_MIN="45", REACH_SIDESTEP="1", REACH_CHAIN="1", REACH_ID0="300000"),
                   seed=9000000, resets=18000, every=3, per=600),
}  # fmt: skip
FLAGS = ("REACH_SIDESTEP", "REACH_CHAIN", "REACH_ID0", "REACH_WRIST", "REACH_WRIST_MIN", "REACH_HOME", "REACH_APPROACH",
         "REACH_PERTURB_M")  # fmt: skip
BODIES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
PINNED = ("left_ankle_roll_link", "right_ankle_roll_link")
VERIFY = 2  # reaches per job whose original frames are recomputed and compared
# RELABEL_ACT (2026-11-04, after r1/r2task: trained on the teacher's own ticks the student learned to continue its history
# and stopped reading the reference, relabel_train.py diag): after each reach, restore the reach's start state and run an
# ACTOR under the relabelled reference for as many ticks as the reach took, recording its inputs and the original command's
# frames at its own states; relabel_train.py labels those states with the frozen controller's answer to the original
# command (DAgger with the frozen controller + the hindsight command as the expert). "frozen" = the frozen controller acts
# (round 0), else a student checkpoint (relabel_train.py's student_<tag>.pt). Then everything is put back as it was.
ACT = os.environ.get("RELABEL_ACT", "")
TARGET = os.environ.get("RELABEL_TARGET", "landing")  # "landing": the jaw pose reached; "wrist": the module docstring
assert TARGET in ("landing", "wrist"), TARGET
ACT_KEYS = ("k", "rq", "av", "dp", "dv", "ac", "rp", "pw", "qw", "act", "pwc", "qwc")
TICK_KEYS = ("k", "rq", "av", "dp", "dv", "ac", "rp", "pw", "qw", "act", "pw1", "qw1")
SEG_KEYS = ("reset", "chain", "side", "walked", "jaw_err_cm", "att_err_deg", "reach_s", "pelvis_z", "ik_e_cm", "ik_e_deg",
            "total_c", "total_e", "n_ticks", "verified")  # fmt: skip
SEG_VEC = ("cmd_pos", "cmd_rot", "e_pos", "e_rot", "target", "rot6", "start", "wrist_dev")


def set_params(pol, params: dict) -> None:
    """Copy `params` into the policy's parameters in place (they are inference tensors: load_policy moved them inside
    inference_mode), so the CUDA graph, which captured their memory, runs them."""
    import torch

    with torch.inference_mode():
        for n, p in pol.w.named_parameters():
            p.copy_(params[n].to(p.device))


def plan_of(m, side: str, pos_w, rot_w, pelvis_z):
    """reach()'s whole-body IK plan (poses, leg durations) from the motor's current state, without moving
    (gen_reach.plan_only, keeping the plan itself)."""
    import gen_reach as G

    from locomani.backends.scalebfm.motor import Motor

    blend, got = m._blend, {}

    def stop(poses, durations):
        got["plan"] = (poses, durations)
        raise G._PlanOnly

    m._blend = stop
    try:
        Motor.reach(m, side, pos_w, rot_w, pelvis_z=pelvis_z, servo=False)
    except G._PlanOnly:
        pass
    finally:
        m._blend = blend
    return got["plan"]


class Capture:
    """Hooks on one Motor: the policy's packed inputs and outputs every tick while `on`, and a copy of the simulator at
    each reach command (Motor.reach wrapped on the instance, Policy.net likewise; the Recorder's own wrappers stay)."""

    def __init__(self, m):
        import mujoco

        self.mj, self.m = mujoco, m
        self.on, self.rows, self.snap, self.verified = False, [], None, 0
        pol = m.pol
        self.bi = [pol.isel[n] for n in BODIES]
        net, reach = pol.net, m.reach

        def part(host, k):
            o, sh = pol.parts[k]
            a = host[o : o + int(np.prod(sh))].reshape(sh)
            return (a[0] if len(sh) > 1 else a).copy()  # every part but rp has the batch axis

        def net_hook(host):
            out = net(host)
            if self.on:
                self.rows.append(dict(k=m.k, mode=m.mode, rq=part(host, "rq"), av=part(host, "av"), dp=part(host, "dp"),
                                      dv=part(host, "dv"), ac=part(host, "ac"), rp=part(host, "rp"),
                                      pw=part(host, "pw")[:, self.bi], qw=part(host, "qw")[:, self.bi],
                                      act=out[pol.n_tgt :].astype(np.float32).copy()))  # fmt: skip
            return out

        def reach_hook(side, pos, rot=None, **kw):
            assert self.snap is None, "one reach per segment"
            d0 = mujoco.MjData(m.sim.m)
            mujoco.mj_copyData(d0, m.sim.m, m.sim.d)
            self.snap = dict(d=d0, k=m.k, ref=m.ref, side=side, pos=np.array(pos, float), rot=np.array(rot, float),
                             pelvis_z=kw.get("pelvis_z"), buf={k: v.copy() for k, v in pol.buf.items()})  # fmt: skip
            self.on = True
            return reach(side, pos, rot, **kw)

        pol.net, m.reach = net_hook, reach_hook
        self.actor = None
        if ACT and ACT != "frozen":
            import torch

            self.actor = torch.load(ACT, map_location="cpu", weights_only=False)["params"]
            self.frozen = {n: p.detach().clone() for n, p in pol.w.named_parameters()}

    def clear(self) -> None:
        self.on, self.rows, self.snap = False, [], None

    def frames(self, at, rows=None) -> tuple[np.ndarray, np.ndarray]:
        """The keypoint frames reference `at` gives at every captured tick (Motor._futures, servo off), ankles as fed."""
        from locomani.backends.scalebfm import runtime as R

        m = self.m
        fidx = m.pol.fidx
        rows = self.rows if rows is None else rows
        pw = np.empty((len(rows), len(fidx), len(BODIES), 3), np.float32)
        qw = np.empty((len(rows), len(fidx), len(BODIES), 4), np.float32)
        for i, r in enumerate(rows):
            for jj, j in enumerate(fidx):
                pose = at(r["k"] * R.DT + j * R.DT)  # Motor._futures: ref(t + j DT), t = k DT
                kp = m.kp(pose.pos, pose.quat, pose.joints + pose.velocity * j * R.DT)
                for b, n in enumerate(BODIES):
                    if n in PINNED:
                        pw[i, jj, b], qw[i, jj, b] = r["pw"][jj, b], r["qw"][jj, b]
                    else:
                        pw[i, jj, b], qw[i, jj, b] = kp[n]
        return pw, qw

    def relabel(self, verify: bool) -> dict:
        """The relabelled frames for the reach just finished (see the module docstring); the simulator and the motor are
        left exactly as they were (MjData copied whole, so even the stale-by-a-substep body poses come back)."""
        from locomani.backends.scalebfm import runtime as R
        from locomani.backends.scalebfm.motor import Motor

        mj, m, sn = self.mj, self.m, self.snap
        sim, side = m.sim, sn["side"]
        assert all(r["mode"] == 4 for r in self.rows) and len(self.rows) > 0
        e_pos, e_rot = m.jaw_centre(side)
        rel_pos = e_pos if TARGET == "landing" else sn["pos"]
        d1 = mj.MjData(sim.m)
        mj.mj_copyData(d1, sim.m, sim.d)
        k1, ref1 = m.k, m.ref
        mj.mj_copyData(sim.d, sim.m, sn["d"])
        m.k, m.ref = sn["k"], sn["ref"]
        try:
            pe, de = plan_of(m, side, rel_pos, e_rot, sn["pelvis_z"])
            at_e, total_e = Motor._blend(m, pe, de)  # starts at the command's tick, like the reach's own
            if verify or ACT:
                pc, dc = plan_of(m, side, sn["pos"], sn["rot"], sn["pelvis_z"])
                at_c, total_c = Motor._blend(m, pc, dc)
            else:
                total_c = float("nan")
        finally:
            mj.mj_copyData(sim.d, sim.m, d1)
            m.k, m.ref = k1, ref1
        pw1, qw1 = self.frames(at_e)
        act = self.act(at_e, at_c, d1, verify) if ACT else {}
        if verify:
            pw0, qw0 = self.frames(at_c)
            fed_pw = np.stack([r["pw"] for r in self.rows])
            fed_qw = np.stack([r["qw"] for r in self.rows])
            assert np.array_equal(pw0, fed_pw) and np.array_equal(qw0, fed_qw), (
                f"recomputed frames differ from the fed ones: {np.abs(pw0 - fed_pw).max():.2e} m, {np.abs(qw0 - fed_qw).max():.2e}"
            )
            self.verified += 1
        g = pe[-1]
        kp = m.kp(g.pos, g.quat, g.joints)
        wp, wq = kp[f"{side}_wrist_yaw_link"]
        Rw = R.mat(wq)
        cos = (np.trace(e_rot.T @ Rw) - 1.0) / 2.0
        out = {f"RL_{k}": np.stack([r[k] for r in self.rows]) for k in TICK_KEYS if k not in ("pw1", "qw1")}
        out.update(RL_pw1=pw1, RL_qw1=qw1, cmd_pos=sn["pos"].astype(np.float32), cmd_rot=sn["rot"].reshape(9).astype(np.float32),
                   e_pos=e_pos.astype(np.float32), e_rot=e_rot.reshape(9).astype(np.float32),
                   ik_e_cm=float(100 * np.linalg.norm(wp + Rw @ R.OFF - rel_pos)),
                   ik_e_deg=float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), total_c=float(total_c),
                   total_e=float(total_e), n_ticks=len(self.rows), verified=bool(verify))  # fmt: skip
        out.update(act)
        return out

    def act(self, at_e, at_c, d_end, verify: bool) -> dict:
        """RELABEL_ACT: the actor from the reach's start state (simulator, motor tick, policy history) under the relabelled
        reference for as many ticks as the reach took; then the simulator, the motor and the policy history are put back to
        the reach's end (d_end and what the motor holds now)."""
        mj, m, sn = self.mj, self.m, self.snap
        sim, pol, side = m.sim, m.pol, sn["side"]
        keep = dict(
            k=m.k, ref=m.ref, buf=pol.buf, trace=len(m.trace), fell_at=m.fell_at, last_fut=m.last_fut, rows=self.rows
        )
        mj.mj_copyData(sim.d, sim.m, sn["d"])
        m.k, m.ref, pol.buf = sn["k"], at_e, {k: v.copy() for k, v in sn["buf"].items()}
        if self.actor is not None:
            set_params(pol, self.actor)
        self.rows, self.on = [], True
        try:
            for _ in range(len(keep["rows"])):
                if not m.tick():
                    break
            rows, fell = self.rows, m.fell_at is not None
            a_pos = m.jaw_centre(side)[0]
        finally:
            self.on, self.rows = False, keep["rows"]
            if self.actor is not None:
                set_params(pol, self.frozen)
            mj.mj_copyData(sim.d, sim.m, d_end)
            m.k, m.ref, pol.buf, m.fell_at, m.last_fut = (
                keep["k"],
                keep["ref"],
                keep["buf"],
                keep["fell_at"],
                keep["last_fut"],
            )
            del m.trace[keep["trace"] :]
        assert rows and all(r["mode"] == 4 for r in rows)
        t0 = self.rows[0]
        assert all(np.array_equal(rows[0][k], t0[k]) for k in ("rq", "av", "dp", "dv", "ac", "rp")), (
            "actor start differs"
        )
        pwc, qwc = self.frames(at_c, rows)
        if verify:  # the actor was fed the relabelled frames
            pwe, qwe = self.frames(at_e, rows)
            assert np.array_equal(pwe, np.stack([r["pw"] for r in rows])) and np.array_equal(
                qwe, np.stack([r["qw"] for r in rows])
            )
        out = {f"A_{k}": np.stack([r[k] for r in rows]) for k in ACT_KEYS if k not in ("pwc", "qwc")}
        out.update(A_pwc=pwc, A_qwc=qwc, a_n_ticks=len(rows), a_fell=bool(fell), a_pos=a_pos.astype(np.float32))
        return out


def write(segs: list[dict], path: Path) -> None:
    segs = [s for s in segs if "RL_k" in s]
    if not segs:
        return
    pack = {f"t_{k}": np.concatenate([s[f"RL_{k}"] for s in segs]) for k in TICK_KEYS}
    for k in SEG_KEYS:
        pack[k] = np.array([s[k] for s in segs])
    for k in SEG_VEC:
        pack[k] = np.stack([np.asarray(s[k], np.float32) for s in segs])
    if ACT:
        pack.update({f"a_{k}": np.concatenate([s[f"A_{k}"] for s in segs]) for k in ACT_KEYS})
        pack.update(a_n_ticks=np.array([s["a_n_ticks"] for s in segs]), a_fell=np.array([s["a_fell"] for s in segs]),
                    a_pos=np.stack([s["a_pos"] for s in segs]))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **pack)
    os.replace(tmp, path)


def worker(data: str, job: str, ids: list[int]) -> None:
    spec = DATASETS[data]
    for k in FLAGS:  # gen_reach reads its flags at import: this process must carry exactly the dataset's
        assert os.environ.get(k) == spec["env"].get(k), (k, os.environ.get(k), spec["env"].get(k))
    import gen_reach as G

    orig, cap = G.segment, {}

    def segment(env, rec, rng, walked, chain):
        if "c" not in cap:
            cap["c"] = Capture(env.m)
        c = cap["c"]
        s = orig(env, rec, rng, walked, chain)
        try:
            if s is not None and c.snap is not None and not s["fell"]:
                s.update(c.relabel(verify=c.verified < VERIFY))
        finally:
            c.clear()
        return s

    def save(segs, path):
        write(segs, DATA / f"{data}_{job}_{path.stem}.npz")

    G.segment, G.save = segment, save
    t0 = time.time()
    G.worker(0, ids, spec["seed"])
    (DATA / f"{data}_{job}.done").write_text(
        json.dumps(dict(ids=len(ids), s=round(time.time() - t0), verified=cap["c"].verified if cap else 0))
    )


def jobs() -> list[tuple[str, str, list[int]]]:
    out = []
    for name, sp in DATASETS.items():
        ids = list(range(0, sp["resets"], sp["every"]))
        for j in range(0, len(ids), sp["per"]):
            out.append((name, f"j{j // sp['per']:03d}", ids[j : j + sp["per"]]))
    return out


def launch(todo: list[tuple[str, str, list[int]]], procs: int, gpus: str) -> None:
    """Run the jobs `procs` at a time, each in its own process with its dataset's flags (gen_reach reads them at import)."""
    DATA.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    todo = [j for j in todo if not (DATA / f"{j[0]}_{j[1]}.done").exists()]
    cards, running, n = gpus.split(","), [], 0
    print(f"{len(todo)} jobs, {sum(len(j[2]) for j in todo)} resets", flush=True)
    while todo or running:
        for p in [p for p in running if p[0].poll() is not None]:
            running.remove(p)
            print(f"{time.strftime('%H:%M:%S')} {p[1]} exit {p[0].returncode}", flush=True)
        while todo and len(running) < procs:
            data, job, ids = todo.pop(0)
            envv = {k: v for k, v in os.environ.items() if k not in FLAGS}
            envv.update(DATASETS[data]["env"], OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[n % len(cards)],
                        REACH_OUT=str(OUT / "unused"), RELABEL_DATA=str(DATA))  # fmt: skip
            n += 1
            (OUT / "logs" / DATA.name).mkdir(parents=True, exist_ok=True)
            log = open(OUT / "logs" / DATA.name / f"{data}_{job}.log", "w")
            cmd = [str(E.SIM_PY), __file__, "worker", "--data", data, "--job", job, "--ids", ",".join(map(str, ids))]
            running.append(
                (
                    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)),
                    f"{data}_{job}",
                )
            )
        time.sleep(5)
    print("all jobs done", flush=True)


def check() -> None:
    """Replayed reaches vs the stored datasets, by (reset, chain): jaw_err_cm and reach_s must match."""
    rows = []
    for name in DATASETS:
        files = sorted(DATA.glob(f"{name}_*.npz"))
        if not files:
            continue
        z = np.load(STORED / f"data_{name}" / "all.npz")
        ref = {
            (int(r), int(c)): (float(e), float(s))
            for r, c, e, s in zip(z["reset"], z["chain"], z["jaw_err_cm"], z["reach_s"])
        }
        n = same = 0
        worst, ticks, ik = 0.0, 0, []
        for f in files:
            d = np.load(f)
            for r, c, e, s in zip(d["reset"], d["chain"], d["jaw_err_cm"], d["reach_s"]):
                e0, s0 = ref[(int(r), int(c))]
                n += 1
                same += int(e == e0 and s == s0)
                worst = max(worst, abs(e - e0))
            ticks += len(d["t_k"])
            ik += d["ik_e_cm"].tolist()
        ik = np.array(ik)
        rows.append(f"{name}: {n} reaches, {ticks} ticks, identical to the stored run {same}/{n} (worst jaw_err gap "
                    f"{worst:.3f} cm); relabelled plan's jaw to its target ({TARGET}) median {np.median(ik):.2f} cm, "
                    f"p90 {np.percentile(ik, 90):.2f}")  # fmt: skip
    print("\n".join(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "smoke", "check"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--data", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--jobs-every", type=int, default=1, help="run every n-th job of each dataset (a subset)")
    ap.add_argument("--max-jobs", type=int, default=0, help="at most this many jobs per dataset (after --jobs-every)")
    ap.add_argument("--jobs-offset", type=int, default=0, help="with --jobs-every n: jobs whose index is this mod n")
    a = ap.parse_args(argv)
    if a.what == "run":
        todo, per = [j for j in jobs() if int(j[1][1:]) % a.jobs_every == a.jobs_offset], {}
        todo = [j for j in todo if (per.setdefault(j[0], []).append(j) or len(per[j[0]]) <= (a.max_jobs or 10**6))]
        launch(todo, a.procs, a.gpus)
    elif a.what == "worker":
        worker(a.data, a.job, [int(x) for x in a.ids.split(",") if x])
    elif a.what == "smoke":
        global DATA
        DATA = OUT / (("smoke_act" if ACT else "smoke") + ("_wrist" if TARGET == "wrist" else ""))
        launch([("wrist", "smoke", [0, 3]), ("ref", "smoke", [0])], 2, a.gpus)
        check()
    else:
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
