"""TODO 52 mechanism 1, step 1: the head images of the first look, continuous rendering against the lazy one-shot render.

Run under the sim venv with the repo's src on PYTHONPATH:
    python look_compare.py <Task> --out DIR [--n 10] [--reps 2] [--slot-set 1] [--seq 8]

For each of the task's pool_diag ids (the loop's 15 diagnostic starts) and each mode (cont = every-step rendering, lazy = the
lazy render of the loop's diagnostic rounds) it sets the episode up exactly as a round does (reset, stabilising hold, the
clock), then reads: the head stereo pair as `eyes` gives it (cont: the last step's frame, lazy: one render of the state), then
`seq` further renders of the SAME frozen state (eyes_seq), the stereo depth of each pair, and the camera view's first look
(env.view.art(): SAM 3 object and parts). One npz per (id, mode, repeat) under DIR/<Task>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from locomani.bench import options
from locomani.bench.simple import __main__ as M
from locomani.bench.simple import client as C
from locomani.bench.simple import episodes as EP
from locomani.bench.simple import run as R
from locomani.bench.simple.config import SimpleConfig
from locomani.bench.simple.session import Session

POOL = "/workspace/code/rsi-scene/simple_art_r2/outputs/scene_rsi_20261016/simple_art/pool_diag"
PKG = "/workspace/code/rsi-scene/simple_art_r2/outputs/scene_rsi_20261016/simple_art/incumbents/fda7af7386a1/skills"


def pool_ids(task: str) -> list[str]:
    ids = [json.load(open(f))["episode_id"] for f in sorted(Path(POOL).glob(f"{task}_s*.json"))]
    return sorted(ids, key=lambda i: int(i.split("/")[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--slots", default="1,2,3")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--seqsub", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out) / a.task
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimpleConfig().with_overrides([f"run_dir={out}/run", f"slots={a.slots}"])
    ids = pool_ids(a.task)[: a.n]
    idf = out / "ids.txt"
    idf.write_text("".join(i + "\n" for i in ids))
    sess = Session(cfg, a.task, idf, out / "rows.jsonl")
    sess.slot, lock = __import__("locomani.bench.simple.session", fromlist=["x"]).take_slot(cfg)
    try:
        assert sess.start_depth() and sess.start_server(), "services did not start"
        args = M.build_parser().parse_args(["batch", "--slot", str(sess.slot)])
        resolved = options.from_args(args, allow_planned=True)
        bed = R.Bed(cfg, R.load_package(PKG), sess.slot, dict(), resolved.options)
        S = bed.S
        K = S.defaults(a.task, S.K_CUT)
        K["look_near"] = True
        K["feed_torque"] = True
        link, depth = bed.links()
        for eid in ids:
            ep = EP.load(eid, cfg)
            for rep in range(a.reps):
                for mode in ("cont", "lazy"):
                    f = out / f"{eid.replace('/', '_')}_{mode}_{rep}.npz"
                    if f.exists():
                        continue
                    t0 = time.time()
                    link.send(("reset_cfg", a.task, C.state_cfg(ep["cfg"])))
                    link.recv()  # makes the server's Episode of this task before the first message below
                    lazy = mode == "lazy"
                    link.send(("lazy_render", lazy, a.warm if lazy else 0, a.sub))
                    link.recv()
                    bed.cfg = dataclasses.replace(cfg, lazy_render=lazy, lazy_warm=a.warm, lazy_subframes=a.sub)
                    env = R.setup(bed, eid, dict(K), None, bed.cache_dir(a.task, None))
                    t_set = time.time() - t0
                    L = env.sim.link
                    L.send(("eyes",))
                    e0 = L.recv()
                    pairs = [(np.asarray(e0["left"]), np.asarray(e0["right"]))]
                    tq = time.time()
                    L.send(("eyes_seq", a.seq, a.seqsub))
                    pairs += list(L.recv())
                    t_seq = time.time() - tq
                    dep = []
                    for lft, rgt in pairs:
                        env.sim.depth_link.send((lft, rgt))
                        _, sg, lb = env.sim.depth_link.recv()
                        dep.append(np.asarray(lb, np.float32))
                    t1 = time.time()
                    seen = env.view.art()
                    rec = dict(
                        id=eid, mode=mode, rep=rep, t_setup=t_set, t_seq=t_seq, seq=a.seq, seqsub=a.seqsub, t_art=time.time() - t1,
                        n_obj=0 if seen["obj"] is None else int(len(seen["obj"])),
                        obj_med=None if seen["obj"] is None else np.median(seen["obj"], axis=0).tolist(),
                        parts=[dict(p=np.asarray(r["p"]).tolist(), n=int(r["n"]), score=float(r["score"])) for r in seen["parts"]],
                    )  # fmt: skip
                    np.savez_compressed(
                        f,
                        left=np.stack([p[0] for p in pairs]), right=np.stack([p[1] for p in pairs]),
                        depth=np.stack(dep), rgb_art=np.asarray(seen["rgb"]), rec=json.dumps(rec),
                    )  # fmt: skip
                    print(
                        eid,
                        mode,
                        rep,
                        f"setup {t_set:.0f}s art {rec['t_art']:.0f}s obj {rec['n_obj']} parts {len(rec['parts'])}",
                        flush=True,
                    )
                    R.free(env)
    finally:
        sess.stop_all()
        lock.close()


if __name__ == "__main__":
    main()
