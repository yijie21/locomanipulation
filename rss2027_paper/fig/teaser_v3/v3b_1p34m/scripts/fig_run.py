"""fig_run.py <code copy> <arm: A|R|servo> <episodes a-b> <out dir>: Figure 1 v3's episodes. notebook: skip figure
render, not an experiment.

Runs the frozen simple_bench.py (selfmodel_simplecmp_20261126's code, private copy; md5 checked by the launcher) on the
illustrative shelf scene (shelf_scene.py) instead of a SIMPLE episode. Nothing of simple_bench's episode logic is
changed; only its inputs are swapped:
  - the task "ShelfReach": EP.episodes -> the 20 placements, EP.build -> shelf_scene.build (scene dir under <out>/scene),
    EP.LIFT / SB.LIMIT_S get an entry (the success test is reported, not used);
  - the truth column's grasp: P.stored_grasp -> the placement's ground-truth palm pose in the vase's frame and the
    fingers (shelf_scene); simple_bench reads the vase's pose from the simulator after the settle (grasp_now);
  - the stance: the robot starts at the origin; per placement a.stand / a.side_off = the grasp's offset from the start,
    so simple_bench's stance is the start and its walk is ~0 (< 2 cm -> no walk: the robot does not step);
  - arm A: --correct D --correct-rule dryR (R's inversion run and recorded, the goal itself sent: simplecmp's AL, rows
    equal to A's); arm R: --correct D --correct-rule R (simplecmp's R_D); arm servo: --method servo (development only,
    the palm servo reads the true palm: checks the grasp pose and the fingers).
Records added (read only, no effect on the run): per stage the exact goal (Corrector.command's argument), the state at
each stage's end (Corrector.landed: after POST_S -- the entry's end is the close's start) and at the close's end
(pick.cont, called by simple_bench right after the close and its 0.5 s hold, before the lift): qpos, qvel, MuJoCo
integration state, the palm (Motor.jaw_centre) and the vase pose -> <out>/snap/<arm>_p<index>.npz.
v3b (11-27): also the same state at fixed times after the close command (the entry's end): when the entry's landing is
recorded, the episode's own per-tick callback (m.tick, simple_bench's closure) is wrapped by one that calls it and then,
at the first tick with m.t - t_close >= dt, records "after<ms>" (dt 0.25 .. 5.00 s, step 0.25; those past the episode's
end are absent). Read only: the callback's result is returned unchanged.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

CODE, ARM, EPS, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
HERE = Path(__file__).resolve().parent
SD = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126")
MODEL = SD / "train/models_D_s0/E3_ensemble_f1"
TEMPLATE = SD / "real/task/data/all.npz"
ARM_ID = {"A": 61, "R": 62, "servo": 69}[ARM]
AFTER_S = tuple(round(0.25 * k, 2) for k in range(1, 21))  # s after the close command

os.chdir(CODE)
sys.path[:0] = [str(CODE / "src"), str(CODE / "scripts/experiments/simple_eval_20261112"), str(HERE)]
import mujoco  # noqa: E402

import simple_bench as SB  # noqa: E402
import shelf_scene as SS  # noqa: E402

EP, P, E = SB.EP, SB.P, SB.E
PL = SS.placements()
for d in ("scene", "snap", "rows", "film", "corr"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# -- the task -----------------------------------------------------------------------------------------------------
SB.PICK_TASKS = tuple(SB.PICK_TASKS) + (SS.TASK,)
SB.LIMIT_S[SS.TASK] = 45.0
EP.LIFT[SS.TASK] = 0.05
EP.episodes = lambda task: [dict(task=task, level="0", index=p["index"], cfg=p) for p in PL]  # noqa: E731
CUR: dict = {}


def build(ep, out=None):
    p = ep["cfg"]
    info = SS.build(p, OUT / "scene" / f"{ARM}_p{p['index']:02d}", EP.robot_spec)
    CUR.update(p=p, info=info, snaps={}, goals={})
    return info


EP.build = build


def stored_grasp(side, obj="0"):
    p = CUR["p"]
    assert side == "right" and obj == f"vase_p{p['index']}", (side, obj)
    return SS.grasp_in_vase(p), None, SS.F_CLOSE.copy(), SS.F_PRE.copy()


P.stored_grasp = stored_grasp

# -- read-only records ------------------------------------------------------------------------------------------
env_init0 = E.Env.__init__


def env_init(self, *a, **kw):
    env_init0(self, *a, **kw)
    CUR["env"] = self


E.Env.__init__ = env_init


def snap(tag: str) -> None:
    env = CUR["env"]
    m, sim = env.m, env.sim
    M, D = sim.m, sim.d
    st = np.zeros(mujoco.mj_stateSize(M, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(M, D, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    p, R = m.jaw_centre("right")
    vb = M.body("vase").id
    CUR["snaps"][tag] = dict(t=float(m.t), qpos=D.qpos.copy(), qvel=D.qvel.copy(), state=st, palm=np.asarray(p, float).copy(),
                             palm_R=np.asarray(R, float).copy(), vase_pos=D.xpos[vb].copy(), vase_quat=D.xquat[vb].copy(),
                             xpos=D.xpos.copy(), xmat=D.xmat.copy(), geom_xpos=D.geom_xpos.copy(), geom_xmat=D.geom_xmat.copy())


cont0 = P.cont


def cont(mo, side):
    snap("close_end")
    return cont0(mo, side)


P.cont = cont

episode0 = SB.episode


def episode(ep, method, a, models, rep=None):
    p = ep["cfg"]
    g = SS.grasp_world(p)[:3, 3]
    a.stand, a.side_off = float(g[0]), float(g[1])  # the start is the stance: no walk
    row = episode0(ep, method, a, models, rep)
    sn = CUR["snaps"]
    np.savez_compressed(OUT / "snap" / f"{ARM}_p{p['index']:02d}.npz",
                        **{f"{k}__{f}": v for k, d in sn.items() for f, v in d.items()},
                        **{f"goal__{k}": np.asarray(v, float) for k, v in CUR["goals"].items()})
    row["fig"] = dict(arm=ARM, placement=p, snaps=sorted(sn), goals={k: np.asarray(v, float).tolist() for k, v in CUR["goals"].items()},
                      palm_close_end=sn["close_end"]["palm"].tolist() if "close_end" in sn else None,
                      walk_none=bool(np.linalg.norm(row.get("walk", [9, 9])) <= 0.02))
    return row


SB.episode = episode

argv = ["simple_bench.py", SS.TASK, "--method", "servo" if ARM == "servo" else "direct", "--bench-set", "controller=scalebfm",
        "--bench-set", "target=truth", "--trace", "--episodes", EPS, "--film", str(OUT / "film" / ARM),
        "--out", str(OUT / "rows" / f"{ARM}_{EPS}.jsonl")]
if ARM in ("A", "R"):
    sys.path.insert(0, str(CODE / "scripts/experiments/selfmodel_simplecmp_20261126"))
    import scmp_correct as CORR  # noqa: E402

    CORR.TASKS = tuple(CORR.TASKS) + (SS.TASK,)
    cmd0, landed0 = CORR.Corrector.command, CORR.Corrector.landed

    def command(self, stage, goal_w, rot_w, pz, kw):
        CUR["goals"][stage] = np.asarray(goal_w, float).copy()
        CUR["goals"][stage + "_rot"] = np.asarray(rot_w, float).copy()
        return cmd0(self, stage, goal_w, rot_w, pz, kw)

    def after_close() -> None:
        m = CUR["env"].m
        t_c, todo, tick_ep = float(m.t), list(AFTER_S), m.tick

        def tick():
            ok = tick_ep()
            while todo and float(m.t) - t_c >= todo[0] - 1e-9:
                snap(f"after{int(round(1000 * todo.pop(0))):04d}ms")
            return ok

        m.tick = tick

    def landed(self, srec, palm_w, R_w):
        snap(f"{srec['stage']}_end")
        if srec["stage"] == "entry":
            after_close()
        return landed0(self, srec, palm_w, R_w)

    CORR.Corrector.command, CORR.Corrector.landed = command, landed
    argv += ["--correct", str(MODEL), "--correct-rule", "dryR" if ARM == "A" else "R", "--correct-template", str(TEMPLATE),
             "--correct-out", str(OUT / "corr" / ARM), "--correct-arm-id", str(ARM_ID)]
sys.argv = argv
print("fig_run argv", json.dumps(argv), flush=True)
SB.main()
