"""Re-run chosen episodes of room_grasp with a per-tick trace of jaw, object and robot-object contacts. Behaviour unchanged."""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_reach as GR
import room_grasp as RG
import torch
import walk_model as WK

torch.set_num_threads(1)
wid, spec, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
jobs = [tuple(map(int, s.split(","))) for s in spec.split(";")]
by = {}
for k, j in jobs:
    by.setdefault(k, []).append(j)
models = {
    "post": torch.load(RG.AE.WM_POST, map_location="cpu", weights_only=False).eval(),
    "post_wkp": torch.load(WK.WALK_MODEL_POST, map_location="cpu", weights_only=False).eval(),
}
out = []
for k, slots in by.items():
    env = RG.ROOM.make_env(k)
    layout = json.load(open(RG.ROOM.OUT / f"layout_{k}" / "layout.json"))
    cache = RG.load_cache(layout)
    rec = GR.Recorder(env)
    con = RG.Contacts(env)
    m, sim = env.m, env.sim
    sm = sim.m
    root = sm.body_rootid[sm.body("pelvis").id]
    state = dict(b=None, side=None, T=[])
    base = m.tick

    def tick():
        ok = base()
        b = state["b"]
        if b is not None and con.phase in ("pregrasp", "entry", "grip", "lift"):
            d = sim.d
            touch = set()
            for i in range(d.ncon):
                b1, b2 = int(sm.geom_bodyid[d.contact[i].geom1]), int(sm.geom_bodyid[d.contact[i].geom2])
                o = b2 if b1 == b else b1 if b2 == b else None
                if o is not None and sm.body_rootid[o] == root:
                    touch.add(sm.body(o).name)
            jp, Rj = m.jaw_centre(state["side"])
            state["T"].append(
                (
                    con.phase,
                    float(m.t),
                    jp.copy(),
                    Rj.copy(),
                    d.xpos[b].copy(),
                    d.xquat[b].copy(),
                    tuple(sorted(touch)),
                    m.jaw_opening(state["side"]),
                )
            )
        return ok

    m.tick = tick
    m.reset((0.0, 0.0), 0.0)
    m.hold(1.0)
    for j in slots:
        slot = layout["slots"][j]
        state.update(b=sm.body(slot["object"]).id, T=[])
        st, c, a, ax, perp = RG.station_frame(slot["station"])
        state["side"] = "right" if float(perp @ (np.array(slot["pos"])[:2] - c)) < 0 else "left"
        tic = time.time()
        r = RG.episode(env, rec, con, models, layout, k, j, "model", cache)
        r["trace2"] = state["T"]
        out.append(r)
        print(
            f"w{wid} L{k} s{j} {r['station']} {r['object']} success {r.get('success')} err {r.get('at_grip_err', float('nan')):.2f} pushed {r.get('pushed_cm', float('nan')):.1f} | {time.time() - tic:.0f} s n {len(state['T'])}",
            flush=True,
        )
        pickle.dump(out, open(f"{outdir}/t_w{wid}.pkl", "wb"))
