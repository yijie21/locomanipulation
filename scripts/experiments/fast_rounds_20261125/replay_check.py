"""TODO 52 mechanism 2, correctness: a run that replays from saved states against the same run from scratch, step by step.

    python replay_check.py <Task> --out DIR [--id <Task>/Q/<n>] [--slots 1]

In one bed (one server), under the loop's lazy render with warm-up renders:
  B0  base setting, memo on: runs for real, stores a state after every primitive, records every perception output.
  D   variant (the move speed changed: a post-approach parameter), memo OFF, perception frozen to B0's outputs: from scratch.
  D2  the same again: the baseline (is the bed itself deterministic?).
  C   variant, memo ON, perception frozen to B0's outputs: the common primitives are restored from B0's states, the rest steps.
  L   base setting again, memo on, LIVE perception: do the chain keys of two live runs of one start agree (hits)?
Compared: per counted step the server's joint and base readings (qpos, qvel) of C against D (and D2 against D), the final
record's fields, and the object poses at the end. Prints and writes check.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import common as X
import numpy as np

from locomani.bench.simple import client as C
from locomani.bench.simple import run as R

TRACE: list = []
EYES = dict(mode="live", rec=[], idx=0)


def install_patches():
    step0, eyes0 = C.OffCfgSim.step, C.OffCfgSim.eyes

    def step(self, tgt):
        step0(self, tgt)
        st = self.last
        if self.counted:
            TRACE.append((int(st["steps"]), np.concatenate([st["qpos"], st["qvel"]]).astype(np.float64)))

    def eyes(self):
        if EYES["mode"] == "replay":
            rgb, lb, le = EYES["rec"][EYES["idx"]]
            EYES["idx"] += 1
            self.last_eyes = dict(le)
            return rgb, lb
        out = eyes0(self)
        if EYES["mode"] == "record":
            EYES["rec"].append((out[0], out[1], dict(self.last_eyes)))
        return out

    C.OffCfgSim.step, C.OffCfgSim.eyes = step, eyes


def run_one(bed, eid, setting, memo, mode):
    TRACE.clear()
    EYES.update(mode=mode, idx=0)
    if mode == "record":
        EYES["rec"] = []
    bed.cfg = dataclasses.replace(bed.cfg, memo=memo)
    row = R.run(bed, eid, setting, None)
    return row, {s: v for s, v in TRACE}


FIELDS = ("success", "t_success", "t_check", "loss", "end", "counted_steps", "fell", "rise_cm", "knocked", "truth_end", "phases", "marks",
          "official_steps", "official_reward", "official_reward_max", "comply", "art", "art_touch", "reach_err_cm", "walk_stop_cm")  # fmt: skip


def same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def compare(tr_a, tr_b, steps_from=0):
    """Per counted step present in both: max abs difference of the joint / base readings."""
    common = sorted(set(tr_a) & set(tr_b))
    if not common:
        return dict(n=0)
    d = [float(np.abs(tr_a[s] - tr_b[s]).max()) for s in common]
    bad = [s for s, x in zip(common, d) if x > 0]
    return dict(n=len(common), max_abs=max(d), first_diff_step=bad[0] if bad else None, identical=not bad)


def one_id(bed, a, eid):
    res = dict(id=eid)
    K0 = bed.S.defaults(a.task, bed.S.K_CUT)
    base = (
        dict(X.SETTING, art_speed=float(K0["art_speed"]) * a.base_factor) if a.base_factor != 1.0 else dict(X.SETTING)
    )
    var = dict(X.SETTING, art_speed=float(K0["art_speed"]) * a.factor)
    res["variant"] = dict(art_speed=var["art_speed"], base=K0["art_speed"])
    rb, tb = run_one(bed, eid, base, True, "record")  # stores the states; its perception outputs are the frozen ones
    rd, td = run_one(bed, eid, var, False, "replay")
    rd2, td2 = run_one(bed, eid, var, False, "replay")
    rc, tc = run_one(bed, eid, var, True, "replay")
    rl, tl = run_one(
        bed, eid, base, True, "live"
    )  # the same start again with live perception: stored perception, so the keys hit
    res["base"] = {k: rb.get(k) for k in ("success", "t_success", "counted_steps", "wall_s", "memo")}
    for n, r in (("D", rd), ("D2", rd2), ("C", rc), ("live_again", rl)):
        res[n] = {k: r.get(k) for k in ("success", "t_success", "counted_steps", "wall_s", "memo", "loss")}
    res["steps"] = dict(D=len(td), D2=len(td2), C=len(tc))
    res["D2_vs_D"], res["C_vs_D"] = compare(td2, td), compare(tc, td)
    res["live_again_vs_base"] = compare(tl, tb)
    first_real = min(tc) if tc else 0
    res["base_vs_D_prefix"] = compare(
        {s: v for s, v in tb.items() if s < first_real}, td
    )  # the shared beginning, run twice from scratch
    res["rows_equal"] = {
        n: {k: same(rd.get(k), r.get(k)) for k in FIELDS} for n, r in (("D2", rd2), ("C", rc), ("live_again", rl))
    }
    res["rows_equal"]["live_again"] = {
        k: same(rb.get(k), rl.get(k)) for k in FIELDS
    }  # (against the base run it replays)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--slots", default="1")
    ap.add_argument("--factor", type=float, default=1.3)
    ap.add_argument("--base-factor", dest="base_factor", type=float, default=1.0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--warm", type=int, default=32)
    ap.add_argument("--sub", type=int, default=1)
    a = ap.parse_args()
    install_patches()
    ov = ("lazy_render=true", f"lazy_warm={a.warm}", f"lazy_subframes={a.sub}")
    out = []
    with X.bed_on_slot(a.task, Path(a.out) / a.task, a.slots, ov) as (bed, sess):
        for eid in [a.only] if a.only else X.pool_ids(a.task)[: a.n]:
            out.append(one_id(bed, a, eid))
            Path(a.out, a.task, "check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
