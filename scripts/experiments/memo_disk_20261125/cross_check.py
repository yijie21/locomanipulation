"""TODO 53, correctness across processes: states stored by process A, restored by a fresh process B (fresh server), step by step.

    python cross_check.py A <Task> --out DIR --store STORE       # base setting, memo on, perception recorded; writes A_<Task>.pkl
    python cross_check.py B <Task> --out DIR --store STORE       # a new process, a new server, an empty memory:
        D, D2  variant (move speed x 1.3), memo OFF, perception frozen to A's outputs: from scratch (twice: is the bed deterministic?)
        C      the same variant, memo ON, the disk store: the common beginning is restored from A's files, the rest steps
        L      the base setting again, memo ON, LIVE perception: everything comes from the disk (states and looks)
Compared per counted step: the server's joint and base readings (qpos, qvel), and the record's fields. Method of TODO 52's
replay_check.py; the memo's memory is empty in B, so every hit is a read from disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
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


def run_one(bed, eid, setting, memo, mode, rec=None):
    TRACE.clear()
    EYES.update(mode=mode, idx=0)
    if mode == "record":
        EYES["rec"] = []
    if rec is not None:
        EYES["rec"] = rec
    bed.cfg = dataclasses.replace(bed.cfg, memo=memo)
    bed.memo_store.clear()  # (a fresh process has an empty memory; so has every run here: every hit is a read from disk)
    bed.memo_eyes.clear()
    row = R.run(bed, eid, setting, None)
    return row, {s: v for s, v in TRACE}


FIELDS = ("success", "t_success", "t_check", "loss", "end", "counted_steps", "fell", "rise_cm", "knocked", "truth_end", "phases", "marks",
          "official_steps", "official_reward", "official_reward_max", "comply", "art", "art_touch", "reach_err_cm", "walk_stop_cm")  # fmt: skip


def same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def compare(tr_a, tr_b):
    common = sorted(set(tr_a) & set(tr_b))
    if not common:
        return dict(n=0)
    d = [float(np.abs(tr_a[s] - tr_b[s]).max()) for s in common]
    bad = [s for s, x in zip(common, d) if x > 0]
    return dict(n=len(common), max_abs=max(d), first_diff_step=bad[0] if bad else None, identical=not bad)


def settings(bed, task, base_factor, factor):
    K0 = bed.S.defaults(task, bed.S.K_CUT)
    base = dict(X.SETTING, art_speed=float(K0["art_speed"]) * base_factor) if base_factor != 1.0 else dict(X.SETTING)
    var = dict(X.SETTING, art_speed=float(K0["art_speed"]) * factor)
    return base, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["A", "B"])
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--slots", default="1,2,3")
    ap.add_argument("--factor", type=float, default=1.3)
    ap.add_argument("--base-factor", dest="base_factor", type=float, default=1.0)
    a = ap.parse_args()
    install_patches()
    out_dir = Path(a.out)
    ov = ("lazy_render=true", "lazy_warm=48", f"memo_dir={a.store}", "memo_sig=cross_check", "memo=true")
    ids = X.pool_ids(a.task)[: a.n]
    res = []
    with X.bed_on_slot(a.task, out_dir / f"run_{a.phase}_{a.task}", a.slots, ov) as (bed, sess):
        base, var = settings(bed, a.task, a.base_factor, a.factor)
        if a.phase == "A":
            saved = {}
            for eid in ids:
                rb, tb = run_one(bed, eid, base, True, "record")
                saved[eid] = dict(eyes=list(EYES["rec"]), trace=tb, row={k: rb.get(k) for k in FIELDS})
                res.append(dict(id=eid, wall_s=rb["wall_s"], success=rb["success"], memo=rb["memo"]))
            pickle.dump(saved, open(out_dir / f"A_{a.task}.pkl", "wb"))
        else:
            saved = pickle.load(open(out_dir / f"A_{a.task}.pkl", "rb"))
            for eid in ids:
                s = saved[eid]
                rd, td = run_one(bed, eid, var, False, "replay", s["eyes"])
                rd2, td2 = run_one(bed, eid, var, False, "replay", s["eyes"])
                rc, tc = run_one(bed, eid, var, True, "replay", s["eyes"])
                rl, tl = run_one(bed, eid, base, True, "live")
                first_real = min(tc) if tc else 0
                res.append(
                    dict(
                        id=eid,
                        wall_s=dict(D=rd["wall_s"], D2=rd2["wall_s"], C=rc["wall_s"], L=rl["wall_s"]),
                        steps=dict(D=len(td), C=len(tc), L=len(tl), A=len(s["trace"])),
                        steps_stepped_after_restore=len([x for x in tc if x >= first_real]) if tc else 0,
                        D2_vs_D=compare(td2, td),
                        C_vs_D=compare(tc, td),
                        L_vs_A=compare(tl, s["trace"]),
                        rows_equal=dict(
                            D2=all(same(rd.get(k), rd2.get(k)) for k in FIELDS),
                            C=all(same(rd.get(k), rc.get(k)) for k in FIELDS),
                            L=all(same(s["row"].get(k), rl.get(k)) for k in FIELDS),
                            C_fields={k: same(rd.get(k), rc.get(k)) for k in FIELDS},
                        ),
                        memo_C=rc["memo"],
                        memo_L=rl["memo"],
                        success=dict(D=rd["success"], C=rc["success"], L=rl["success"]),
                    )
                )
    (out_dir / f"result_{a.phase}_{a.task}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
