# notebook: skip render for paper Figure 1, not an experiment
"""fig1_replay.py <task> <ep> <methods>: replay one lmlroute ScaleBFM episode with the frozen lrt_run.run (the formal
run's loop body run_one, unchanged), records to $LC_OUT; at each group's close (the step on which lrt_run.episode records
close_dist_cm) save the full MuJoCo state (mj_getState, mjSTATE_INTEGRATION), qpos / qvel / mocap, body poses and the
model (MJB) to $FIG_STATE for rendering.

The only hook is a read-only wrapper around HGDriver.hand_point: it returns the original value; when its caller is
lrt_run.episode in phase 'close' before close_dist_cm is set, it copies state out of the simulator (nothing written back).
Launch exactly as dev/lrt_queue.sh starts a worker (cwd g1_backend, PYTHONPATH=$G:$CODE/src, CUDA_VISIBLE_DEVICES empty,
one thread, nice 19), plus LC_CODE_ROOT = the private copy (lrt_run's code check then verifies every module came from it).
"""

import json
import os
import sys

import numpy as np

task, ep, methods = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
CODE = os.environ["LC_CODE_ROOT"]
STATE = os.environ["FIG_STATE"]
sys.path.insert(0, f"{CODE}/scripts/experiments/selfmodel_lmlroute_20261126")
import lrt_run as T  # noqa: E402

HG, L = T.HG, T.L
_orig = HG.HGDriver.hand_point
_done: set = set()


def _snap(dr, loc: dict, hp: np.ndarray) -> None:
    mj, m, d = dr.mj, dr.m, dr.d
    meth, f = loc["method"], int(loc["f"])
    spec = mj.mjtState.mjSTATE_INTEGRATION
    st = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, st, spec)
    mj.mj_saveModel(m, f"{STATE}/{meth}.mjb", None)
    goal = np.asarray(loc["plan"]["goal"], float)
    obj = dr.obj_pos()
    np.savez(f"{STATE}/{meth}.npz", state=st, qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(),
             mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(), time=np.float64(d.time),
             xpos=d.xpos.copy(), xmat=d.xmat.copy(), geom_xpos=d.geom_xpos.copy(), geom_xmat=d.geom_xmat.copy(),
             hp=np.asarray(hp, float), obj_pos=obj, goal=goal, frame=np.int64(f))
    wb = int(dr.wrist["right"])
    meta = dict(task=dr.task, ep=int(loc["ep"]), method=meth, frame=f, sim_time=float(d.time),
                wrist_body=wb, wrist_name=mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, wb), hand_off=L.HAND_OFF.tolist(),
                obj_bid=int(dr.obj_bid), obj_name=dr.obj_name, obj_tree=sorted(int(b) for b in dr.obj_tree),
                hp=np.asarray(hp, float).tolist(), obj_pos=obj.tolist(), goal=goal.tolist(),
                close_dist_cm=100 * float(np.linalg.norm(hp - goal)), close_obj_cm=100 * float(np.linalg.norm(hp - obj)),
                nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody), ngeom=int(m.ngeom), state_size=int(st.size),
                mujoco=mj.__version__, host=os.uname().nodename)
    with open(f"{STATE}/{meth}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"fig1 snapshot {meth} frame {f}: hp {np.round(hp, 6).tolist()} obj {np.round(obj, 6).tolist()} "
          f"close_dist_cm {meta['close_dist_cm']:.6f}", flush=True)


def _hook(self, x):
    v = _orig(self, x)
    fr = sys._getframe(1)
    if fr.f_code is T.episode.__code__:
        loc = fr.f_locals
        if loc.get("phase") == "close" and "close_dist_cm" not in loc["R"] and loc["method"] not in _done:
            _done.add(loc["method"])
            _snap(self, loc, v)
    return v


HG.HGDriver.hand_point = _hook
print(f"fig1 replay: {task} ep {ep} methods {methods}; LC_OUT {T.OUT}; looks {os.environ.get('LRT_LOOKS')}", flush=True)
T.run("scalebfm", task, [ep], methods, 0, 1)
print(f"fig1 replay done: snapshots {sorted(_done)}", flush=True)
