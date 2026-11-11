"""A constrained servo on top of the world-model command (2026-11-02, after recheck.py).

After step, look again, step again, 23 of the 345 core goals still miss by more than 2 cm with feedforward + servo, and
15 of those were within 2 cm open loop (10 within 1 cm): the servo pulls an accurate hand away. A per-tick trace of the
servo (Motor._servo, an integral on the jaw-centre error to the wanted position, 8 cm per axis cap) shows why: it
integrates from the first tick of the reach, while the hand is still 50 cm out, so the lead hits the cap within 0.2 s
(12-14 cm), on top of the feedforward command that already aims ~10 cm past the wanted point; when the hand arrives the
lead is still saturated and the hand overshoots, and the integral swings it back and forth to the end of the settle.

A first smoke test (late and a lead cap on 6 of the misses) found a second cause: once the late servo starts the hand is
0.7-0.9 cm from the wanted point, and the wrist-attitude integral (to the reference rotation, 0.5 rad cap) saturates
within 0.25 s, i.e. the controller sits ~50 degrees off the commanded wrist rotation it cannot reach; twisting the
commanded wrist that far moves the jaw 6-8 cm, which a capped position lead then cannot pull back. The world model's
command was learned and inverted without that twist.

Constraints from the world model, alone and together (an instance-level wrapper of Motor._servo, the motor unchanged):
    late          integrate only once the reference has stopped (its future frames move < GATE_EPS): during the transit
                  the gap to the wanted point is the planned motion, not an error the model left
    pos           position only: no wrist-attitude integral (servo_ki_rot 0), the wrist rotation stays the commanded one
                  the model was trained and inverted with
    late_pos      both
    late_pos_cap  and the lead kept within a trust radius of the command, clip(K_SIG * std, R_MIN, R_MAX), std the
                  plan-aware ensemble's spread for that command: the model says the hand lands within ~std of the wanted
                  point, so a larger correction is the servo going wrong, not the model
    orig          the servo as before (reproduces the stored ff_servo error exactly: the check that the replay is the run)
The final wrist attitude error to the wanted rotation is reported next to the position error (what pos gives up).
Each core goal is replayed in its recheck.py wm2 final state (no step, one step, or two steps) with its stored command.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/servo_gate.py h --procs 6
    $PY scripts/experiments/reachwm_20261030/servo_gate.py d
Writes outputs/reachwm_20261030/invert/servo_gate/.
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
import recheck as RC  # noqa: E402  (sets RESTANCE_DATA=side before restance is imported)

RS, E, IE, PI = RC.RS, RC.E, RC.IE, RC.PI
OUT = IE.OUT / "servo_gate"
VARIANTS = ("open", "orig", "late", "pos", "late_pos", "late_pos_cap")  # open: no servo (the stored open-loop run)
GATE_EPS = 1e-3  # m the reference's future frames may still move once it counts as stopped
K_SIG, R_MIN, R_MAX = 3.0, 0.015, 0.08  # trust radius = clip(K_SIG * std, R_MIN, R_MAX), m


def plans() -> dict:
    """Per core goal, the final state recheck's wm2 policy leaves it in, its command and the plan-aware std."""
    P1 = dict(np.load(RS.OUT / "policy.npz"))
    P2 = dict(np.load(RC.OUT / "policy.npz"))
    C = np.load(RS.OUT / "choice.npz")
    cat = lambda d, pre: {k[len(pre):]: np.concatenate([np.load(p)[k] for p in sorted(d.glob("h_w*.npz"))])  # noqa: E731
                          for k in np.load(sorted(d.glob("h_w*.npz"))[0]) if k.startswith(pre)}  # fmt: skip
    H0, H1, H2 = cat(PI.PLAN, "ref_ff_servo_"), cat(RS.OUT, "wm_ff_servo_"), cat(RC.OUT, "wm2_ff_servo_")
    S0, S1, S2 = cat(PI.PLAN, "ref_aim_"), cat(RS.OUT, "wm_aim_"), cat(RC.OUT, "wm2_aim_")
    out = []
    for i in np.where(P1["core"])[0]:
        g = int(P1["gid"][i])
        j = int(np.where(C["gid"] == g)[0][0])
        if P2["wm2_again"][i]:
            kind, H, S = "two", H2, S2
        elif P1["wm_stepped"][i]:
            kind, H, S = "one", H1, S1
        else:
            kind, H, S = "none", H0, S0
        r = int(np.where(H["gid"] == g)[0][0])
        assert S["gid"][r] == g
        out.append(dict(gid=g, kind=kind, t=H["t"][r], pz=float(H["pz"][r]), std=float(S["pred_std"][r]), goal_t=H["goal_t"][r],
                        d1=C["wm_d"][j], d2=H["d2"][r] if kind == "two" else np.zeros(2), ff=float(H["err"][r]),
                        aim=float(S["err"][r])))  # fmt: skip
    return {"rows": out}


class Gate:
    """Motor._servo wrapped on one Motor instance; `mode` and `radius` are set before each execution."""

    def __init__(self, m):
        from locomani.backends.scalebfm import motor as MM

        self.m, self.orig, self.W, self.SIDES = m, m._servo, MM.WRIST, MM.SIDES
        self.mode, self.radius = "orig", None
        self.hold = (
            False  # True: while the reference moves the stored lead stays applied (late stops the integral only)
        )
        self.reset()
        m._servo = self

    def reset(self) -> None:
        self.t_open, self.capped, self.ticks, self.lead_max = None, 0, 0, 0.0

    def __call__(self, fut: dict) -> dict:
        m = self.m
        late = self.mode.startswith("late")
        held = []
        if late:
            for s in self.SIDES:
                p = fut[self.W[s]][0]
                if m.servo_on[s] and np.linalg.norm(p[-1] - p[0]) > GATE_EPS:  # the reference is still on its way
                    held.append(s)
                    m.servo_on[s] = False
        out = self.orig(fut)
        for s in held:
            m.servo_on[s] = True
            if self.hold and ("pos", s) in m.servo:
                out[self.W[s]] = (out[self.W[s]][0] + m.servo[("pos", s)], out[self.W[s]][1])
        for s in self.SIDES:
            if not m.servo_on[s] or s in held or ("pos", s) not in m.servo:
                continue
            if self.t_open is None:
                self.t_open = m.t
            self.ticks += 1
            lead = m.servo[("pos", s)]
            n = float(np.linalg.norm(lead))
            if self.mode.endswith("cap") and n > self.radius:
                lc = lead * (self.radius / n)
                out[self.W[s]] = (out[self.W[s]][0] - lead + lc, out[self.W[s]][1])
                m.servo[("pos", s)] = lc
                self.capped += 1
                n = self.radius
            self.lead_max = max(self.lead_max, n)
        return out


def worker(wid: int, gids: list[int], variants: tuple[str, ...]) -> None:
    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = Gate(env.m)
    rows = {r["gid"]: r for r in plans()["rows"]}
    res = []
    for gid in gids:
        p = rows[gid]
        tic = time.time()
        if p["kind"] == "none":

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)

            r0 = replay()
            if r0 is None:
                continue
            c2, walked, goal = r0[0], r0[1], p["goal_t"].astype(np.float32)
        else:

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

            r0 = replay()
            if r0 is None:
                continue
            c, _, goal_w, yaw0, p0, _ = r0
            st = RS.post_step(env, rec, c, goal_w, yaw0, p0)
            c2, walked, goal = st["c2"], True, st["goal"].astype(np.float32)
        msg = [f"s{wid} goal {gid} {p['kind']} std {p['std']:.2f} open {p['aim']:.2f} | stored ff {p['ff']:.2f}"]
        for v in variants:
            if replay() is None:
                break
            gate.mode, gate.radius = v, float(np.clip(K_SIG * p["std"] / 100, R_MIN, R_MAX))
            gate.reset()
            cfg = env.m.cfg
            if "pos" in v.split("_"):
                env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)  # MotorConfig is frozen
            servo = v != "open"
            seg = IE.execute(
                env, rec, c2, p["t"].astype(np.float32), p["pz"], servo, walked, servo_goal_h=goal if servo else None
            )
            env.m.cfg = cfg
            s = PI.summarize(seg, PI.plan_record(rec.plan), goal[None])
            R_w = E.rot_z(float(seg["start"][3]) + c2["dyaw"]) @ G.rot_y(c2["pitch"])
            Rm = env.m.jaw_centre(c2["side"])[1]
            att = float(np.degrees(np.arccos(np.clip((np.trace(R_w.T @ Rm) - 1) / 2, -1, 1))))
            res.append(dict(gid=gid, variant=v, err=float(s["err"][0]), att_deg=att, fell=bool(s["fell"][0]), settled=bool(s["settled"][0]),
                            radius=gate.radius, capped=gate.capped, ticks=gate.ticks, lead_max=gate.lead_max,
                            t_open=-1.0 if gate.t_open is None else float(gate.t_open - rec.t_cmd)))  # fmt: skip
            msg.append(f"{v} {s['err'][0]:.2f}")
        print(" | ".join(msg) + f" | {time.time() - tic:.0f} s", flush=True)
    gate.mode = "orig"
    json.dump(res, open(OUT / f"h_w{wid}.json", "w"))


def launch(procs: int, gpus: str, limit: int, variants: str) -> None:
    gids = [r["gid"] for r in plans()["rows"]][: limit or None]
    print(f"{len(gids)} core goals x {variants}", flush=True)
    cards = gpus.split(",")  # as in the runs being replayed
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / f"h_w{w}.log", "w")
        cmd = [
            str(E.SIM_PY),
            __file__,
            "worker",
            "--wid",
            str(w),
            "--gids",
            ",".join(map(str, gids[w::procs])),
            "--variants",
            variants,
        ]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report() -> None:
    rows = {r["gid"]: r for r in plans()["rows"]}
    res = [x for p in sorted(OUT.glob("h_w*.json")) for x in json.load(open(p))]
    by = {}
    for x in res:
        by.setdefault(x["variant"], {})[x["gid"]] = x
    gids = sorted(set.intersection(*(set(v) for v in by.values())))
    aim = np.array([rows[g]["aim"] for g in gids])
    stored = np.array([rows[g]["ff"] for g in gids])
    std = np.array([rows[g]["std"] for g in gids])
    err = {v: np.array([99.0 if by[v][g]["fell"] else by[v][g]["err"] for g in gids]) for v in by}
    stat = lambda e: dict(median=float(np.median(e)), lt1=float(np.mean(e < 1)), gt2=int(np.sum(e > 2)), p90=float(np.percentile(e, 90)),  # noqa: E731
                          worst=float(e.max()))  # fmt: skip
    L = [f"{len(gids)} core goals (of {len(rows)}) in their final recheck wm2 state, kinds " +
         ", ".join(f"{k} {sum(rows[g]['kind'] == k for g in gids)}" for k in ("none", "one", "two")) +
         (f"; orig reproduces the stored ff+servo error exactly on {int(np.sum(err['orig'] == stored))}/{len(gids)} "
          f"(max diff {np.abs(err['orig'] - stored).max():.1e} cm)" if "orig" in err else "") +
         (f", open the stored open-loop error on {int(np.sum(err['open'] == aim))}/{len(gids)}" if "open" in err else "") +
         f"; falls {sum(by[v][g]['fell'] for v in by for g in gids)}", "",
         f"trust radius clip({K_SIG} x std, {100 * R_MIN:.1f}, {100 * R_MAX:.0f}) cm: median {100 * np.median([by[next(iter(by))][g]['radius'] for g in gids]):.2f} cm", "",
         "| servo | median cm | <1 cm | >2 cm (n) | p90 | worst | worse than open loop by > 1 cm (n) | open loop > 2 cm rescued to < 2 (n) | "
         "wrist attitude error median / p90 deg |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    out = dict(n=len(gids), rows={})
    for nm, e in [(v, err[v]) for v in VARIANTS if v in err]:
        s = stat(e)
        out["rows"][nm] = s
        at = np.array([by[nm][g]["att_deg"] for g in gids]) if nm in by else None
        L.append(f"| {nm} | {s['median']:.2f} | {s['lt1']:.2f} | {s['gt2']} | {s['p90']:.2f} | {s['worst']:.2f} | "
                 f"{int(np.sum(e > aim + 1))} | {int(np.sum((aim > 2) & (e < 2)))} | "
                 f"{'-' if at is None else f'{np.median(at):.1f} / {np.percentile(at, 90):.1f}'} |")  # fmt: skip
    L += ["", "| servo | lead hit the cap (runs) | lead max median cm | servo starts after the reach command, median s | "
          "per-goal better than orig (share) | worse than orig by > 0.5 cm (n) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for v in VARIANTS[1:]:
        if v not in by:
            continue
        X = [by[v][g] for g in gids]
        base = err["orig"] if "orig" in err else stored
        L.append(f"| {v} | {sum(x['capped'] > 0 for x in X)} | {100 * np.median([x['lead_max'] for x in X]):.2f} | "
                 f"{np.median([x['t_open'] for x in X]):.2f} | {np.mean(err[v] < base):.2f} | {int(np.sum(err[v] > base + 0.5))} |")  # fmt: skip
    # the misses left by the best variant, split by what the open loop did
    for v in ("late", "late_pos", "late_pos_cap"):
        if v not in err:
            continue
        e = err[v]
        miss = e > 2
        L.append(f"{v}: misses > 2 cm {int(miss.sum())}: open loop also > 2 on {int(np.sum(miss & (aim > 2)))}, open loop <= 2 on "
                 f"{int(np.sum(miss & (aim <= 2)))}; std of the misses median {np.median(std[miss]) if miss.any() else float('nan'):.2f} "
                 f"(all {np.median(std):.2f}); min(open loop, {v}) > 2 cm on {int(np.sum(np.minimum(aim, e) > 2))}")  # fmt: skip
    json.dump(out, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["h", "d", "worker"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N goals only (smoke test)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.what == "h":
        OUT.mkdir(parents=True, exist_ok=True)
        launch(a.procs, a.gpus, a.limit, a.variants)
    elif a.what == "d":
        report()
    else:
        worker(a.wid, [int(x) for x in a.gids.split(",") if x], tuple(a.variants.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
