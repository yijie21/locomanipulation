"""Wrist attitude in the inversion (2026-11-02, after servo_gate.py).

With the position-only servo (servo_gate late_pos) the hand lands within 1 cm on 99 % of the core goals, but the wrist
ends 20 deg (median, p90 ~48) off the wanted rotation: the command asks for the wanted rotation and the controller does
not get there. The plan-aware ensemble already predicts the final wrist rotation (rot6 head; offline on the test split
3.1 deg median error against 25.8 deg for "the hand ends at the commanded rotation", member spread / error Spearman 0.79),
so the inversion can pre-compensate the attitude the way it pre-compensates the position.

From each core goal's final state (recheck wm2: no step, one step or two steps) and its stored command (the position-only
inversion's best target and pelvis height), two inversions with the same model, state and start:
    pos    position only, as before: plan_invert's fixed point t <- t + (goal - predicted) for ITERS_POSE updates, the
           wrist command the wanted rotation
    pose   also the wrist command, searched inside the command space the training data spans: gen_reach commands are
           rot_z(heading + dyaw) rot_y(pitch), dyaw +-60 deg, pitch 0-90 deg, no roll. The wanted command and a 15 deg
           grid over that whole range (CMD_DYAWS x CMD_PITCHES) are scored at the stored target, then the NB best get
           the position fixed point; the best
           iterate by |miss|^2 + LAMBDA |std|^2 + MU (theta^2 + LAMBDA spread^2), theta the predicted attitude miss,
           spread the members' mean angle to their mean rotation, MU making DEG_PER_CM degrees weigh like 1 cm
The first version turned the command by the predicted residual in full 3D (R_cmd <- R_want R_pred^T R_cmd): every turned
command left the trained command family (it adds roll), the members split (attitude spread 2 -> 40 deg, position std
0.6 -> 5-11 cm) and the loss kept the uncompensated command on all 6 smoke goals. A +-30 deg grid around the wanted
command gave 1-5 deg on the same goals: the controller follows about half of a commanded wrist turn (on the data,
final yaw = 0.55 x commanded, pitch 0.48 x; the IK reference reaches the wanted rotation within 0-8 deg, the measured
wrist sits 19-39 deg off the reference), so the command has to turn about twice as far, hence the whole range.
Each command is executed open loop and with the position-only late servo (servo_gate.Gate late_pos). Position and final
wrist attitude errors are reported.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/pose_invert.py h --procs 6
    $PY scripts/experiments/reachwm_20261030/pose_invert.py d
Writes outputs/reachwm_20261030/invert/pose_invert/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import servo_gate as SG  # noqa: E402  (imports recheck, which sets RESTANCE_DATA=side)

RS, RC, E, IE, PI = SG.RS, SG.RC, SG.E, SG.IE, SG.PI
OUT = IE.OUT / "pose_invert"
ITERS_POSE = 6
DEG_PER_CM = 10.0
MU = (0.01 / np.radians(DEG_PER_CM)) ** 2  # m^2 per rad^2
CMD_DYAWS = np.radians(np.arange(-60.0, 60.1, 15.0))  # wrist command grid over gen_reach.sample_command's range
CMD_PITCHES = np.radians(np.arange(0.0, 90.1, 15.0))
NB = 3  # grid commands that get the position fixed point
METHODS = ("pos", "pose")
SERVOS = ("open", "late_pos")


def angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def predict_pose(model, b: dict):
    """EndEnsemble.predict plus the members' rotations: pos, pos_std, R_pred (mean rot6 by Gram-Schmidt), spread (rad).
    Refuses a model trained with the other form of the joint feature (data.check_qfix)."""
    import data as DT
    import models as M
    import torch

    DT.check_qfix(model)

    with torch.no_grad():
        os_ = [mm.predict(b) for mm in model.members]
        pos = torch.stack([o["pos"] for o in os_])
        r6 = torch.stack([o["rot6"] for o in os_])
        Rk = M.gs6(r6)[:, 0].numpy().astype(np.float64)
        Rp = M.gs6(r6.mean(0))[0].numpy().astype(np.float64)
    spread = float(np.mean([angle(Rk[k], Rp) for k in range(len(Rk))]))
    return (
        pos.mean(0)[0].numpy().astype(np.float64),
        pos.std(0, unbiased=False)[0].numpy().astype(np.float64),
        Rp,
        spread,
    )


def state(env, rec, p: dict, replay):
    """Replay to the start of the reach and judge it the way the stored inversion did; the dict restance.post_step
    returns (for no step: plan_invert's planning state, the goal in the heading frame after the hold)."""
    import gen_reach as G

    r = replay()
    if r is None:
        return None
    if p["kind"] != "none":
        c, _, goal_w, yaw0, p0, _ = r
        return RS.post_step(env, rec, c, goal_w, yaw0, p0)
    c, walked = r
    m = env.m
    sd = 1.0 if c["side"] == "right" else -1.0
    rec.start(c["side"])
    m.hold(G.PRE_S)
    rec.on = False
    pv, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    rot_w = E.rot_z(yaw + c["dyaw"]) @ G.rot_y(c["pitch"])
    rot_h = Rh.T @ rot_w
    start = np.r_[pv, yaw].astype(np.float32)
    return dict(sd=sd, p=pv, yaw=yaw, Rh=Rh, goal=p["goal_t"].astype(np.float64), c2=c, rot_w=rot_w,
                rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32), start=start, walked=walked,
                x0b=RS.x0_now(rec, sd, walked, start))  # fmt: skip


def evaluate(model, m, st: dict, t: np.ndarray, pz: float, cmd) -> dict:
    """One iterate: the IK plan for target t (heading frame) and wrist command `cmd` = (dyaw, pitch) relative to the
    heading (None: the wanted one, sent bit for bit as before), and the plan-aware ensemble's prediction."""
    import gen_reach as G
    import torch

    sd, p, Rh, goal, start, x0b = (st[k] for k in ("sd", "p", "Rh", "goal", "start", "x0b"))
    if cmd is None:
        rot_w, rot6 = st["rot_w"], st["rot6"]
    else:
        rot_w = E.rot_z(st["yaw"] + cmd[0]) @ G.rot_y(
            cmd[1]
        )  # invert_eval.execute's formula with the command's dyaw, pitch
        rot_h = Rh.T @ rot_w
        rot6 = np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32)
    tw = np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]
    pr = PI.plan_record(PI.plan_of(m, st["c2"]["side"], tw, rot_w, pz))
    x0 = np.r_[x0b, PI.plan_features(pr, x0b[:29], start)]
    u = np.r_[sd, t, rot6, float(pz < 0), 0.0 if pz < 0 else pz]
    F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
    pos, std, Rp, spread = predict_pose(
        model, dict(x0=F(x0), u=F(u), target=F(t), rot6=F(rot6), side=torch.tensor([sd]))
    )
    R_want = Rh.T @ st["rot_w"]
    return dict(t=t.copy(), cmd=cmd, pos=pos, pred_err=100 * np.linalg.norm(pos - goal), pred_std=100 * np.linalg.norm(std),
                pred_att=float(np.degrees(angle(Rp, R_want))), att_spread=float(np.degrees(spread)),
                dev=float(np.degrees(angle(Rh.T @ rot_w, R_want))), rec=pr, x0=x0,
                l_pos=float(((pos - goal) ** 2).sum() + IE.LAMBDA * (std**2).sum()),
                l_att=float(MU * (angle(Rp, R_want) ** 2 + IE.LAMBDA * spread**2)))  # fmt: skip


def fixed_point(model, m, st: dict, t0: np.ndarray, pz: float, cmd) -> list[dict]:
    """plan_invert's position fixed point (ITERS_POSE updates) with the wrist command held at `cmd`."""
    lo, hi = PI.WIDE
    sd, goal = st["sd"], st["goal"]
    t = t0.astype(np.float32).astype(np.float64)
    evs = []
    for k in range(ITERS_POSE + 1):
        ev = evaluate(model, m, st, t, pz, cmd)
        ev["k"] = k
        evs.append(ev)
        if k < ITERS_POSE:
            t = t + (goal - ev["pos"])
            y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
            t = np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])].astype(np.float32).astype(np.float64)
    return evs


def refine(model, m, st: dict, t0: np.ndarray, pz: float, att: bool) -> tuple[dict, list]:
    """pos: the position fixed point from the stored command. pose: first the wanted wrist command and the grid over
    the range the training commands span (CMD_DYAWS x CMD_PITCHES), scored at the stored target; then the position
    fixed point for the NB best commands. The best iterate by its loss (pose adds the attitude term)."""
    L = (lambda ev: ev["l_pos"] + ev["l_att"]) if att else (lambda ev: ev["l_pos"])  # noqa: E731
    if not att:
        evs = fixed_point(model, m, st, t0, pz, None)
        for ev in evs:
            ev["stage"] = "pos"
    else:
        dy0, p0 = st["c2"]["dyaw"], st["c2"]["pitch"]
        cmds = {(0.0, 0.0): None}
        for a in CMD_DYAWS:
            for b in CMD_PITCHES:
                key = (round(float(a) - dy0, 9), round(float(b) - p0, 9))
                if key != (0.0, 0.0):
                    cmds.setdefault(key, (float(a), float(b)))
        grid = []
        for key, c in cmds.items():
            ev = evaluate(model, m, st, t0.astype(np.float32).astype(np.float64), pz, c)
            ev.update(stage="grid", k=0)
            grid.append(ev)
        top = sorted(grid, key=L)[:NB]
        evs = list(grid)
        for g in top:
            fp = fixed_point(model, m, st, t0, pz, g["cmd"])
            for ev in fp:
                ev["stage"] = "fp"
            evs += fp
    best = min(evs, key=L)
    best["loss"] = L(best)
    trace = [
        [ev["k"], L(ev), ev["pred_err"], ev["pred_std"], ev["pred_att"], ev["att_spread"], ev["dev"]] for ev in evs
    ]
    return best, trace


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + PI.REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = SG.Gate(env.m)
    model = torch.load(RS.WM_REF, map_location="cpu", weights_only=False).eval()
    rows = {r["gid"]: r for r in SG.plans()["rows"]}
    I = np.load(PI.PLAN / "init.npz")
    res = []
    for gid in gids:
        p = rows[gid]
        tic = time.time()
        if p["kind"] == "none":

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)
        else:

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

        st = state(env, rec, p, replay)
        if st is None:
            continue
        walked = st.get("walked", True)
        x0_check = (
            float(np.abs(st["x0b"] - I["x0"][int(np.where(I["gid"] == gid)[0][0])]).max())
            if p["kind"] == "none"
            else 0.0
        )
        msg = [f"p{wid} goal {gid} {p['kind']}"]
        # both inversions from the planning state (plan_of reads the robot's posture), then the executions
        plans_ = {meth: refine(model, env.m, st, p["t"], p["pz"], meth == "pose") for meth in METHODS}
        for meth in METHODS:
            best, trace = plans_[meth]
            for v in SERVOS:
                if replay() is None:
                    break
                gate.mode, gate.radius = ("orig" if v == "open" else v), None
                gate.reset()
                cfg = env.m.cfg
                env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
                servo = v != "open"
                goal = st["goal"].astype(np.float32)
                c_cmd = st["c2"] if best["cmd"] is None else dict(st["c2"], dyaw=best["cmd"][0], pitch=best["cmd"][1])
                seg = IE.execute(env, rec, c_cmd, best["t"].astype(np.float32), p["pz"], servo, walked,
                                 servo_goal_h=goal if servo else None)  # fmt: skip
                env.m.cfg = cfg
                tracked = PI.plan_record(rec.plan)
                same_plan = all(np.array_equal(tracked[k], best["rec"][k]) for k in tracked)
                s = PI.summarize(seg, tracked, goal[None])
                R_w = E.rot_z(float(seg["start"][3]) + st["c2"]["dyaw"]) @ G.rot_y(st["c2"]["pitch"])
                att = float(np.degrees(angle(R_w, env.m.jaw_centre(st["c2"]["side"])[1])))
                res.append(dict(gid=gid, kind=p["kind"], method=meth, servo=v, err=float(s["err"][0]), att_deg=att,
                                fell=bool(s["fell"][0]), same_plan=same_plan, x0_diff=float(np.abs(s["x0"][0] - best["x0"]).max()),
                                k=best["k"], stage=best["stage"], cmd_deg=[0.0, 0.0] if best["cmd"] is None else
                                [float(np.degrees(best["cmd"][0] - st["c2"]["dyaw"])), float(np.degrees(best["cmd"][1] - st["c2"]["pitch"]))],
                                pred_err=float(best["pred_err"]), pred_std=float(best["pred_std"]),
                                pred_att=float(best["pred_att"]), att_spread=float(best["att_spread"]), dev=float(best["dev"]),
                                stored_aim=p["aim"], x0_check=x0_check, trace=np.array(trace).tolist()))  # fmt: skip
            a = [x for x in res if x["gid"] == gid and x["method"] == meth]
            msg.append(f"{meth} {best['stage']} k {best['k']} pred att {best['pred_att']:.1f} spread {best['att_spread']:.1f} dev {best['dev']:.0f} | " +
                       " ".join(f"{x['servo']} {x['err']:.2f} cm {x['att_deg']:.1f} deg" for x in a))  # fmt: skip
        print(" | ".join(msg) + f" | x0 check {x0_check:.1e} | {time.time() - tic:.0f} s", flush=True)
    json.dump(res, open(OUT / f"h_w{wid}.json", "w"))


def launch(procs: int, gpus: str, limit: int) -> None:
    gids = [r["gid"] for r in SG.plans()["rows"]][: limit or None]
    print(f"{len(gids)} core goals x {METHODS} x {SERVOS}", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report() -> None:
    from scipy.stats import spearmanr

    res = [x for p in sorted(OUT.glob("h_w*.json")) for x in json.load(open(p))]
    by = {}
    for x in res:
        by[(x["method"], x["servo"])] = by.get((x["method"], x["servo"]), {})
        by[(x["method"], x["servo"])][x["gid"]] = x
    gids = sorted(set.intersection(*(set(v) for v in by.values())))
    sg = {(x["variant"], x["gid"]): x for p in sorted(SG.OUT.glob("h_w*.json")) for x in json.load(open(p))}
    E_ = lambda m, v: np.array([99.0 if by[(m, v)][g]["fell"] else by[(m, v)][g]["err"] for g in gids])  # noqa: E731
    A_ = lambda m, v: np.array([by[(m, v)][g]["att_deg"] for g in gids])  # noqa: E731
    X = [x for d in by.values() for x in d.values()]
    kinds = {g: by[("pos", "open")][g]["kind"] for g in gids}
    L = [f"{len(gids)} core goals ({', '.join(f'{k} {sum(v == k for v in kinds.values())}' for k in ('none', 'one', 'two'))}); "
         f"falls {sum(x['fell'] for x in X)}; plan matches the executed one {np.mean([x['same_plan'] for x in X]):.3f}, "
         f"max x0 diff {max(x['x0_diff'] for x in X):.1e}; no-step planning state vs plan_invert's x0 max diff "
         f"{max(x['x0_check'] for x in X):.1e}"]  # fmt: skip
    # pos from the stored command reproduces servo_gate (where the inversion keeps the stored command, k = 0)
    for v, sv in (("open", "open"), ("late_pos", "late_pos")):
        k0 = [g for g in gids if by[("pos", v)][g]["k"] == 0]
        same = sum(by[("pos", v)][g]["err"] == sg[(sv, g)]["err"] for g in k0)
        L.append(
            f"pos {v}: keeps the stored command on {len(k0)}/{len(gids)}, and there reproduces servo_gate {sv} exactly on {same}/{len(k0)}"
        )
    L += ["", f"wrist weight: {DEG_PER_CM:.0f} deg weigh like 1 cm; wrist command grid {len(CMD_DYAWS)} dyaw x {len(CMD_PITCHES)} pitch "
          f"over the trained range, {NB} best refined", "",
          "| inversion | servo | position median cm | <1 cm | >2 cm (n) | p90 | attitude median deg | <10 deg | <20 deg | >30 deg (n) | p90 deg |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    out = dict(n=len(gids), rows=[])
    for meth in METHODS:
        for v in SERVOS:
            e, a = E_(meth, v), A_(meth, v)
            r = dict(method=meth, servo=v, median=float(np.median(e)), lt1=float(np.mean(e < 1)), gt2=int(np.sum(e > 2)),
                     p90=float(np.percentile(e, 90)), att_median=float(np.median(a)), att_lt10=float(np.mean(a < 10)),
                     att_lt20=float(np.mean(a < 20)), att_gt30=int(np.sum(a > 30)), att_p90=float(np.percentile(a, 90)))  # fmt: skip
            out["rows"].append(r)
            L.append(f"| {meth} | {v} | {r['median']:.2f} | {r['lt1']:.2f} | {r['gt2']} | {r['p90']:.2f} | {r['att_median']:.1f} | "
                     f"{r['att_lt10']:.2f} | {r['att_lt20']:.2f} | {r['att_gt30']} | {r['att_p90']:.1f} |")  # fmt: skip
    # paired, and by how far the wrist ended off with the position-only inversion
    a0, a1 = A_("pos", "late_pos"), A_("pose", "late_pos")
    e0, e1 = E_("pos", "late_pos"), E_("pose", "late_pos")
    L += ["", "| attitude miss with pos, late_pos | n | attitude pos / pose median deg | pose better on | position pos / pose median cm | "
          "pose position worse by > 0.5 cm (n) | command turned off the wanted rotation median deg | pose kept the wanted command |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    dev = np.array([by[("pose", "late_pos")][g]["dev"] for g in gids])
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 180), (0, 180)):
        k = (a0 >= lo) & (a0 < hi)
        if k.any():
            L.append(f"| {lo}-{hi} deg | {int(k.sum())} | {np.median(a0[k]):.1f} / {np.median(a1[k]):.1f} | {np.mean(a1[k] < a0[k]):.2f} | "
                     f"{np.median(e0[k]):.2f} / {np.median(e1[k]):.2f} | {int(np.sum(e1[k] > e0[k] + 0.5))} | {np.median(dev[k]):.1f} | "
                     f"{np.mean(dev[k] == 0):.2f} |")  # fmt: skip
    # does the model's attitude prediction hold in closed loop
    for meth in METHODS:
        pa = np.array([by[(meth, "open")][g]["pred_att"] for g in gids])
        aa = A_(meth, "open")
        spr = np.array([by[(meth, "open")][g]["att_spread"] for g in gids])
        L.append(f"{meth} open loop: predicted attitude miss median {np.median(pa):.1f} deg vs measured {np.median(aa):.1f}; "
                 f"|predicted - measured| median {np.median(np.abs(pa - aa)):.1f} deg; spread / measured-miss Spearman "
                 f"{spearmanr(spr, np.abs(pa - aa)).correlation:.2f}")  # fmt: skip
    json.dump(out, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["h", "d", "worker"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N goals only (smoke test)")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.what == "h":
        OUT.mkdir(parents=True, exist_ok=True)
        launch(a.procs, a.gpus, a.limit)
    elif a.what == "d":
        report()
    else:
        worker(a.wid, [int(x) for x in a.gids.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
