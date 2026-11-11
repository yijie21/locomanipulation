"""Report media (2026-11-02, the report rewrite): chosen core goals replayed under several commands with every 50 Hz
motor tick recorded, then drawn as comparison videos.

A case is one core goal (servo_gate.plans' row: its final state after no step, one or two steps) and a list of runs;
every run of a case replays the same reset (and walk, and steps), so all start from the bit-identical state. A run is
(name, prep, command, servo):
    prep     plan    the goal's stored stance (the steps recheck's wm2 policy took)
             nostep  the stance IE.setup leaves (reset, walk), no step: the goal as gen_reach drew it
             stance  plan, then the move stance_wrist.py chose for the goal (STANCE_RUN: Motor.walk_to the step and turn,
                     restance.post_step); the clip starts before the move (2026-11-04)
    command  naive   the wanted position as the command, the wanted wrist rotation
             stored  the stored command (servo_gate: p["t"], wanted rotation)
             fp      the position fixed point (wrist_invert.fixed_point) seeded at the wanted position, best by loss
             pos, pose3g, w20, cap15   wrist_invert.refine's pick
    servo    open, late_pos (wrist_invert), orig (the old servo, wrist-attitude integral on)
A run records per tick: qpos, the reference pose the controller is asked to track (Motor.ref: pelvis + 29 joints),
both jaw centres and rotations, the jaw centres the tick commanded (reference + servo lead) and the servo's position
lead; plus the world model's prediction for its command (and the fixed point's iterates).

Videos (phase v): runs[0] solid, runs[1] a translucent ghost in the same scene (or, for one-run views, the reference
pose as a shadow), the goal, the predicted landings (sphere radius = ensemble std), jaw trails, a close-up of the goal
and a log strip (distance to the goal, or wrist angle, over time per run).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/report_media.py r hero_a step servo      # record (REACHWM_GOAL_REF set here)
    MUJOCO_GL=egl $PY scripts/experiments/reachwm_20261030/report_media.py v hero_a gap_a:hero_a   # render
    $PY scripts/experiments/reachwm_20261030/report_media.py n --wid 0 --procs 6   # naive baselines, all core goals
Writes outputs/reachwm_20261030/media/ (rec/ recordings, src/ videos, preview/ <= 500 KB copies).
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_invert as WI  # noqa: E402

PV = WI.PV
RS, RC, E, IE, PI, SG = PV.RS, PV.RC, PV.E, PV.IE, PV.PI, PV.SG
MED = IE.OUT.parent / "media"
MODEL = IE.OUT / "wrist_invert2" / "wm_ref.pt"  # the far-turn model, the latest

# case -> (goal id, runs); runs[0] is drawn solid, runs[1] as the ghost, the rest only in the log strip
CASES = {
    "hero_a": (213, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "hero_b": (18, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "hero_c": (342, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "hero_d": (154, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "hero_e": (300, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "hero_f": (10, [("inv", "plan", "fp", "open"), ("naive", "plan", "naive", "open")]),
    "step": (290, [("step", "plan", "fp", "open"), ("nostep", "nostep", "fp", "open")]),
    "step_b": (110, [("step", "plan", "fp", "open"), ("nostep", "nostep", "fp", "open")]),
    "servo": (
        389,
        [
            ("late_pos", "plan", "stored", "late_pos"),
            ("orig", "plan", "stored", "orig"),
            ("open", "plan", "stored", "open"),
        ],
    ),
    "servo_b": (
        276,
        [
            ("late_pos", "plan", "stored", "late_pos"),
            ("orig", "plan", "stored", "orig"),
            ("open", "plan", "stored", "open"),
        ],
    ),
    "wrist": (
        256,
        [("pose3g", "plan", "pose3g", "open"), ("pos", "plan", "pos", "open")],
    ),  # eyes closed (11-03; was late_pos)
    "wrist_b": (
        204,
        [("pose3g", "plan", "pose3g", "open"), ("pos", "plan", "pos", "open")],
    ),  # eyes closed (11-03; was late_pos)
    "curse": (364, [("pose3g", "plan", "pose3g", "open"), ("pos", "plan", "pos", "open")]),
    "curse_b": (139, [("pose3g", "plan", "pose3g", "open"), ("pos", "plan", "pos", "open")]),
    # the wrist-aware stance (stance_wrist.py, the post-walk model): a move that keeps the wrist-first pick, and one where
    # the look-again rule switches to the position pick (stance_report.look_again)
    "stance_a": (0, [("move", "stance", "pose3g", "open"), ("stay", "plan", "pose3g", "open")]),
    "stance_b": (
        213,
        [
            ("move_pos", "stance", "pos", "open"),
            ("move_p3", "stance", "pose3g", "open"),
            ("stay", "plan", "pose3g", "open"),
        ],
    ),
    # the relabelling report (2026-11-05): the hero goal under the controllers relabel_train.py fine-tuned; a run's optional
    # 5th field names the controller (relabel/student_<tag>.pt, from the command to the end of the hold, as relabel_eval.py
    # deploys it) and the self model its command is planned with
    "copy": (213, [("r1", "plan", "naive", "open", dict(ctl="r1")), ("frozen", "plan", "naive", "open")]),
    "drift": (213, [("r3dag", "plan", "naive", "open", dict(ctl="r3dag")), ("frozen", "plan", "naive", "open")]),
    "drift_b": (0, [("r3dag", "plan", "naive", "open", dict(ctl="r3dag")), ("frozen", "plan", "naive", "open")]),
    "dag": (213, [("r5dag", "plan", "naive", "open", dict(ctl="r5dag")), ("frozen", "plan", "naive", "open")]),
    "wronly": (213, [("wr2", "plan", "naive", "open", dict(ctl="wr2")), ("frozen", "plan", "naive", "open")]),
    "hyb": (
        213,
        [
            ("hyb", "plan", "pos", "open", dict(ctl="r5dag", model=IE.OUT / "hybrid_r5dag" / "wm_ref.pt")),
            ("inv", "plan", "pos", "open"),
        ],
    ),
}
STUDENTS = IE.OUT.parent / "relabel"
CASE_MODEL = {c: IE.OUT / "wrist_turn" / "wm_ref.pt" for c in ("stance_a", "stance_b")}
STANCE_RUN = IE.OUT / "stance_wrist_turn"  # the moves


def stance_move(gid: int) -> tuple[tuple[float, float], float]:
    """The step (heading frame, m) and turn (rad) stance_wrist.py's run chose for gid."""
    import json
    import math

    r = next(
        r for f in sorted(STANCE_RUN.glob("h_w*.json")) for r in json.load(open(f)) if r["gid"] == gid and "method" in r
    )
    assert r["moved"], gid
    return (float(r["d"][0]), float(r["d"][1])), math.radians(r["turn_deg"])


class Tape:
    """Every motor tick while `on`: wraps Motor.tick (after gen_reach.Recorder wrapped it)."""

    def __init__(self, env):
        from locomani.backends.scalebfm import motor as MM

        self.env, self.on, self.MM = env, False, MM
        self.clear()
        m = env.m
        orig = m.tick

        def tick():
            ok = orig()
            if self.on:
                self.sample()
            return ok

        m.tick = tick

    def clear(self) -> None:
        self.rows, self.marks = [], {}

    def mark(self, name: str) -> None:
        self.marks[name] = len(self.rows)

    def sample(self) -> None:
        m = self.env.m
        ref = m.ref(m.t) if m.ref is not None else None
        jaw, Rj, cj, lead = [], [], [], []
        for s in ("right", "left"):
            p, R = m.jaw_centre(s)
            jaw.append(p)
            Rj.append(R)
            cj.append(m.commanded_jaw(s) if m.last_fut else np.full(3, np.nan))
            ld = m.servo.get(("pos", s))
            lead.append(np.zeros(3) if ld is None else np.asarray(ld, float))
        self.rows.append(dict(t=float(m.t), q=m.sim.d.qpos.copy(),
                              ref=np.full(36, np.nan) if ref is None else np.r_[ref.pos, ref.quat, ref.joints],
                              jaw=np.array(jaw), Rj=np.array(Rj), cj=np.array(cj), lead=np.array(lead)))  # fmt: skip

    def arrays(self) -> dict:
        return {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}


def to_w(st: dict, v_h) -> np.ndarray:
    """A heading-frame point of the planning state (xy relative to the pelvis, z world) in the world."""
    v = np.asarray(v_h, float)
    return np.r_[st["p"][:2] + (st["Rh"] @ np.r_[v[:2], 0.0])[:2], v[2]]


def record(case: str) -> None:
    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    gid, runs = CASES[case]
    env = E.Env("open")
    rec = G.Recorder(env)
    tape = Tape(env)
    gate = SG.Gate(env.m)
    model = torch.load(CASE_MODEL.get(case, MODEL), map_location="cpu", weights_only=False).eval()
    p = {r["gid"]: r for r in SG.plans()["rows"]}[gid]
    out = dict(
        case=case, gid=gid, kind=p["kind"], runs=[], qadr=np.asarray(env.m.kp.qadr) if hasattr(env.m, "kp") else None
    )
    states = {}
    pol = env.m.pol
    frozen = {n: q.detach().clone() for n, q in pol.w.named_parameters()}
    models, students = {}, {}
    for name, prep, cmd_name, v, *opt in runs:
        o = opt[0] if opt else {}
        assert prep != "stance" or "model" not in o, "a planning model of its own: plan / nostep only"
        skey = prep if prep == "stance" else (prep, str(o.get("model", "")))
        run_model = (
            model
            if "model" not in o
            else models.setdefault(o["model"], torch.load(o["model"], map_location="cpu", weights_only=False).eval())
        )
        tic = time.time()
        base = "plan" if prep == "stance" else prep
        if base == "plan" and p["kind"] != "none":

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

            pp = p
        else:

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)

            if base == "plan":
                pp = p
            else:
                c0, _ = IE.setup(env, gid)
                pp = dict(p, kind="none", goal_t=np.asarray(c0["target"], np.float32))
        st0 = None
        if prep == "stance":
            st0 = states[("plan", "")][0] if ("plan", "") in states else PV.state(env, rec, pp, replay)
            d, tr = stance_move(gid)

            def replay(plan_replay=replay, st0=st0, d=d, tr=tr):
                r = plan_replay()
                if r is None:
                    return None
                tape.mark("pre_move")
                m = env.m
                rec.start(st0["c2"]["side"])
                m.hold(G.PRE_S)
                rec.on = False
                q, Rq = env.pelvis()
                yw = E.yaw_of(Rq)
                m.walk_to(q[:2] + (E.rot_z(yw) @ np.r_[d[0], d[1], 0.0])[:2], yw + tr, face="fixed")
                return None if m.fell_at is not None else r

        if prep not in states and prep == "stance":
            replay()
            st = RS.post_step(env, rec, st0["c2"], to_w(st0, st0["goal"]), st0["yaw"], st0["p"])
            plans_ = WI.refine(model, env.m, st, st["goal"], pp["pz"])
            fp = WI.fixed_point(model, env.m, st, st["goal"], pp["pz"], None)
            naive = WI.evaluate(model, env.m, st, st["goal"].astype(np.float32).astype(np.float64), pp["pz"], None)
            states[prep] = (st, plans_, fp, naive, naive, pp, replay)
        elif skey not in states:
            st = PV.state(env, rec, pp, replay)
            plans_ = WI.refine(run_model, env.m, st, pp["t"] if prep == "plan" else st["goal"], pp["pz"])
            fp = WI.fixed_point(run_model, env.m, st, st["goal"], pp["pz"], None)
            naive = WI.evaluate(run_model, env.m, st, st["goal"].astype(np.float32).astype(np.float64), pp["pz"], None)
            stored = WI.evaluate(run_model, env.m, st, pp["t"].astype(np.float32).astype(np.float64), pp["pz"], None)
            states[skey] = (st, plans_, fp, naive, stored, pp, replay)
        st, plans_, fp, naive, stored, pp, replay = states[skey]
        if cmd_name == "naive":
            best = naive
        elif cmd_name == "stored":
            best = stored
        elif cmd_name == "fp":
            best = min(fp, key=lambda ev: ev["l_pos"])
        else:
            best = plans_[cmd_name][0]
        cmd = best["cmd"]
        goal = st["goal"].astype(np.float32)
        tape.clear()
        tape.on = True
        tape.mark("start")
        if replay() is None:
            print(f"{case} {name}: fell in the replay", flush=True)
            tape.on = False
            continue
        tape.mark("ready")
        gate.mode, gate.radius = ("orig" if v in ("open", "orig") else v), None
        gate.reset()
        cfg = env.m.cfg
        if v != "orig":
            env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
        servo = v != "open"
        c_cmd, rot_h = st["c2"], None
        if isinstance(cmd, tuple):
            c_cmd = dict(st["c2"], dyaw=cmd[0], pitch=cmd[1])
        elif cmd is not None:
            rot_h = cmd
        walked = st.get("walked", True)
        reach = env.m.reach
        if o.get("ctl"):
            import relabel_gen as RG

            sw = students.setdefault(o["ctl"], torch.load(STUDENTS / f"student_{o['ctl']}.pt", map_location="cpu",
                                                          weights_only=False)["params"])  # fmt: skip

            def reach_with(*a, _w=sw, **kw):
                RG.set_params(pol, _w)
                return reach(*a, **kw)

            env.m.reach = reach_with
        try:
            seg = IE.execute(env, rec, c_cmd, best["t"].astype(np.float32), pp["pz"], servo, walked,
                             servo_goal_h=goal if servo else None, rot_h=rot_h)  # fmt: skip
        finally:
            if o.get("ctl"):
                env.m.reach = reach
                RG.set_params(pol, frozen)
        tape.on = False
        env.m.cfg = cfg
        s = PI.summarize(seg, PI.plan_record(rec.plan), goal[None])
        side = st["c2"]["side"]
        sp = st0 if st0 is not None else st  # the planning state: the world goal, the camera
        if st0 is not None:  # judged in the world, as stance_wrist.py
            s["err"][0] = 100 * float(np.linalg.norm(env.m.jaw_centre(side)[0] - to_w(st0, st0["goal"])))
            if "pre_move" in tape.marks:
                tape.marks["ready"] = tape.marks["pre_move"]
        Rw_cmd = st["Rh"] @ WI.cmd_h(st, cmd)
        att = float(np.degrees(PV.angle(st["rot_w"], env.m.jaw_centre(side)[1])))
        tr = tape.arrays()
        k_cmd = int(np.searchsorted(tr["t"], rec.t_cmd))
        out["runs"].append(dict(
            name=name, prep=prep, cmd=cmd_name, servo=v, ctl=o.get("ctl", ""), side=side, err=float(s["err"][0]), att_deg=att,
            fell=bool(s["fell"][0]), goal_w=to_w(sp, sp["goal"]), rot_w=st["rot_w"], pelvis0=sp["p"].copy(), yaw0=sp["yaw"],
            target_w=to_w(st, best["t"]), rot_cmd_w=Rw_cmd, pred_w=to_w(st, best["pos"]), pred_std=float(best["pred_std"]),
            pred_err=float(best["pred_err"]), pred_att=float(best["pred_att"]), att_spread=float(best["att_spread"]),
            dev=float(best["dev"]), pred_R_w=st["Rh"] @ best["Rp"],
            iters=[(to_w(st, ev["t"]), to_w(st, ev["pos"]), float(ev["pred_std"])) for ev in fp] if cmd_name == "fp" else [],
            naive_pred_w=to_w(st, naive["pos"]), naive_pred_std=float(naive["pred_std"]), naive_pred_err=float(naive["pred_err"]),
            k_ready=tape.marks["ready"], k_cmd=k_cmd, t_cmd=float(rec.t_cmd), tape=tr,
            gate=dict(t_open=-1.0 if gate.t_open is None else float(gate.t_open - rec.t_cmd), lead_max=gate.lead_max)))  # fmt: skip
        print(f"{case} gid {gid} {name}: err {s['err'][0]:.2f} cm att {att:.1f} deg | pred {best['pred_err']:.2f} "
              f"+- {best['pred_std']:.2f} | naive pred {naive['pred_err']:.2f} | {len(tr['t'])} ticks | {time.time() - tic:.0f} s",
              flush=True)  # fmt: skip
    gate.mode = "orig"
    (MED / "rec").mkdir(parents=True, exist_ok=True)
    with open(MED / "rec" / f"{case}.pkl", "wb") as f:
        pickle.dump(out, f)


def naive_batch(wid: int, procs: int) -> None:
    """Phase n: every core goal (the stored stance), the wanted position sent as the command, open loop and with the
    late position-only servo: the baselines beside wrist_invert2's pos rows (same goals, states, model)."""
    import json

    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = SG.Gate(env.m)
    model = torch.load(MODEL, map_location="cpu", weights_only=False).eval()
    rows = SG.plans()["rows"][wid::procs]
    res = []
    for p in rows:
        gid = p["gid"]
        if p["kind"] != "none":

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])
        else:

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)

        st = PV.state(env, rec, p, replay)
        if st is None:
            continue
        goal = st["goal"].astype(np.float32)
        ev = WI.evaluate(model, env.m, st, st["goal"].astype(np.float32).astype(np.float64), p["pz"], None)
        for v in ("open", "late_pos"):
            if replay() is None:
                break
            gate.mode, gate.radius = ("orig" if v == "open" else v), None
            gate.reset()
            cfg = env.m.cfg
            env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
            servo = v != "open"
            seg = IE.execute(env, rec, st["c2"], goal, p["pz"], servo, st.get("walked", True),
                             servo_goal_h=goal if servo else None)  # fmt: skip
            env.m.cfg = cfg
            s = PI.summarize(seg, PI.plan_record(rec.plan), goal[None])
            att = float(np.degrees(PV.angle(st["rot_w"], env.m.jaw_centre(st["c2"]["side"])[1])))
            res.append(dict(gid=gid, kind=p["kind"], servo=v, err=float(s["err"][0]), att_deg=att, fell=bool(s["fell"][0]),
                            pred_err=float(ev["pred_err"]), pred_std=float(ev["pred_std"])))  # fmt: skip
        print(
            f"n{wid} goal {gid} " + " ".join(f"{r['servo']} {r['err']:.2f}" for r in res if r["gid"] == gid), flush=True
        )
    gate.mode = "orig"
    (MED / "naive").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(MED / "naive" / f"n_w{wid}.json", "w"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["r", "v", "n"])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args(argv)
    if a.phase == "n":
        naive_batch(a.wid, a.procs)
        return 0
    for c in a.cases:
        if a.phase == "r":
            record(c)
        else:
            import report_video as RV

            case, _, src = c.partition(":")  # gap_a:hero_a draws hero_a's recording as the gap view
            RV.render(case, src or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
