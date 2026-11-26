"""heft_walkstop_20261126 revision 2: the same finish-line trials (hw_run.trial, unchanged) in HumanoidArena's (HA's) body,
with HA's HEFT wiring (hadata / hacollect: sx_ha.HAG1Host through ha_controllers' ha_g1ctrl provider, LocoManiLab's
G1_BRIDGE control path on HA's articulation, the reference pushed through the motion buffer's episode transform) and
SONIC's (HA's own SonicActionProvider, native stance). Runs in HA's runtime (hw_ha.sh: env.sh, one card).

    hw_ha.sh <gpu> --controller heft --arms p0,p2 --cells 0,2,4,5 --reps 0:5 --out DIR [--traces]
    hw_ha.sh <gpu> --controller heft --probe --out DIR          (dev: the LML-vs-HA candidates, then exit)

What is HA's here and what is hw_run's:
  host    hacollect's hc_ha.HCG1 / HCSonic (hadata's host classes plus the spawn site and LML's start noise): the robot is
          moved to hc_ha.SITE (ppbox's clearest floor, 2.20 m to the nearest collision box) after HA's episode reset
          (episode seed EP_SEED, the same scene every trial) and gets LocoManiLab's start noise (N(0, 0.02 rad) on the 43
          joints) from default_rng([noise_seed, 1]), noise_seed drawn from hw_run.trial's default_rng([TAG, rep, 1]):
          every arm and cell sees the same start for the same repeat. Cameras off (--no-camera: no lerobot refresh; the
          g1 provider steps physics with render=False)
  feet    load from HA's ContactSensor (the vertical net force on each ankle_roll_link, hr_real's HAOdo); the odometry
          is hr_real's HAOdo (hg_walk.LegOdo with that load)
  truth   the frontmost point of the feet: the MuJoCo mirror sx_ha keeps (SIMPLE's G1, posed every sample at HA's
          measured pelvis and joints), its ankle_roll_link geoms' world AABB -- HA's own foot collision shapes are probed
          (--probe), not used, so the toe is the same G1 foot in both simulators; reach_own is the same geoms by FK of
          the encoders with the IMU attitude, as in LML
  settle  hw_run's (lc3_run.PropSettle, 3 s at most); SONIC's hand limit 5 cm/s (hc_ha's, the user's 11-26 decision:
          SONIC's hands in HA keep shaking at 1-14 cm/s), HEFT's 2 cm/s
Everything else -- cells, the line, the target, the walks, re-anchoring, the metrics, success -- is hw_run.trial's.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfmodel_hacollect_20261126"))
import hc_ha as HC  # noqa: E402  (sx_ha, ha_sim: isaaclab.app)

SXH, ha_sim = HC.SXH, HC.ha_sim
EP_SEED = 200  # HA's episode reset seed (hadata's first dev seed); the robot is then moved to the site
SONIC_HAND_V = 0.05


def parser():
    p = ha_sim.build_parser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controller", choices=["heft", "sonic"], default="heft")
    p.add_argument("--arms", default="p0,p2")
    p.add_argument("--cells", default="0,2,4,5")
    p.add_argument("--reps", default="0:5")
    p.add_argument("--site", default=None, help="x,y world (default hc_ha.SITE)")
    p.add_argument("--sigma", type=float, default=HC.SIGMA)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--traces", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    return p


def make_host(a, sim, L, D3):
    site = tuple(float(v) for v in (a.site or f"{HC.SITE[0]},{HC.SITE[1]}").split(","))
    if a.controller == "sonic":
        base, args = HC.HCSonic, (sim, L, D3, SXH.SONIC_DELAY_F, "native", a.sigma, site)
    else:
        base, args = HC.HCG1, (sim, L, D3, "heft", "lml", None, a.sigma, site)

    class HWHost(base):
        def reset(self) -> None:  # hw_run.trial sets env.rng, then resets: the noise follows the repeat
            rng = getattr(getattr(self, "env", None), "rng", None)
            self.ep_seed = EP_SEED
            self.noise_seed = int(rng.integers(2**31)) if rng is not None else 0
            super().reset()

    h = HWHost(*args)
    h.env = types.SimpleNamespace(rng=None)
    if not hasattr(h, "buf"):  # HA's SONIC host has no motion buffer (hw_run.trial asks for one)
        h.buf = None
    return h


def ha_body(H, h):
    class HABody(H.Body):
        """hw_run.Body on sx_ha's mirror (unprefixed names), the feet's load from HA's ContactSensor."""

        def __init__(self, h):
            mj, m = h.mj, h.m
            self.mj, self.m, self.h = mj, m, h
            bn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
            root = int(m.body_rootid[h.pelvis])
            self.robot_body = np.array([int(m.body_rootid[i]) == root for i in range(m.nbody)])
            self.robot_geoms = np.array([g for g in range(m.ngeom) if self.robot_body[m.geom_bodyid[g]]])
            feet = [bn.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
            self.foot_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in feet])
            self.bname = [bn[int(m.geom_bodyid[g])] for g in range(m.ngeom)]
            self.scratch = mj.MjData(m)
            self.rq, self.qadr, self.c, self.to_mj = h.rq, h.qadr_mj, np.asarray(h.c, float), np.asarray(h.to_mj)
            names = list(h.cs.body_names)
            self.i_foot = [names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]

        def load(self, d) -> np.ndarray:
            F = self.h.cs.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
            return np.array([abs(float(F[i, 2])) for i in self.i_foot])

        def reach_own(self, x: dict, hd) -> float:
            s = self.scratch
            s.qpos[:] = 0.0
            s.qpos[self.rq + 3 : self.rq + 7] = x["root"][3:]
            s.qpos[self.qadr] = np.asarray(x["q"], float)[self.to_mj]
            self.mj.mj_kinematics(self.m, s)
            pel = H.L._qmat(x["root"][3:]) @ self.c
            return self._front(s, self.foot_geoms, pel[:2], hd, None)[0]

    return HABody(h)


def ha_odo_cls(HW):
    class HAOdo(HW.LegOdo):  # hr_real.ha_odo
        def _foot_geoms(self, dr):
            names = list(dr.cs.body_names)
            self.i_foot = {s: names.index(f"{s}_ankle_roll_link") for s in ("left", "right")}

        def load(self, dr):
            F = dr.cs.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
            return {s: abs(float(F[i, 2])) for s, i in self.i_foot.items()}

    return HAOdo


def probe(a, sim, h, body) -> None:
    """dev: the candidates for "HEFT steps in LML but not in HA" that can be read off the two bodies (rates, PD, friction,
    the feet's collision shapes, masses). Read-only; printed."""
    rob, env = sim.robot, sim.env
    out = {}

    def tr(k, f):
        try:
            out[k] = f()
        except Exception as e:  # noqa: BLE001 -- a probe that does not apply
            out[k] = f"n/a ({type(e).__name__}: {e})"

    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    jn = list(rob.data.joint_names)
    legs = [jn.index(n) for n in MUJOCO_JOINT_NAMES[:12]]
    t = lambda v: np.asarray(v[0].detach().cpu().numpy(), float)  # noqa: E731
    tr("control_dt", lambda: float(sim.dt))
    tr("physics_dt", lambda: float(env.cfg.sim.dt))
    tr("decimation", lambda: int(env.cfg.decimation))
    tr("solver_iters", lambda: str(rob.cfg.spawn.articulation_props))
    tr("actuators", lambda: {k: type(v).__name__ for k, v in rob.actuators.items()})
    tr("legs", lambda: [MUJOCO_JOINT_NAMES[i] for i in range(12)])
    tr("stiffness_legs", lambda: t(rob.data.joint_stiffness)[legs].round(2).tolist())
    tr("damping_legs", lambda: t(rob.data.joint_damping)[legs].round(3).tolist())
    tr("armature_legs", lambda: t(rob.data.joint_armature)[legs].round(4).tolist())
    tr("friction_legs", lambda: t(rob.data.joint_friction_coeff)[legs].round(4).tolist())
    tr("effort_legs", lambda: t(rob.data.joint_effort_limits)[legs].round(1).tolist())
    tr("mass_total", lambda: float(t(rob.data.default_mass).sum()))
    tr("mass_mirror", lambda: float(h.m.body_mass[np.asarray(body.robot_body)].sum()))

    def mats():
        mp = rob.root_physx_view.get_material_properties()[0].cpu().numpy()
        u, c = np.unique(mp.round(3), axis=0, return_counts=True)
        return {"robot shapes [static, dynamic, restitution]": [(v.tolist(), int(n)) for v, n in zip(u, c)]}

    tr("robot_material", mats)
    tr("sim_material", lambda: str(env.cfg.sim.physics_material))
    tr("terrain_material", lambda: str(getattr(getattr(env.cfg.scene, "terrain", None), "physics_material", None)))

    def usd():
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        st = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "proxy", "render"])
        feet, mats = [], []
        for p in st.Traverse():
            path = str(p.GetPath())
            if "ankle_roll_link" in path and p.HasAPI(UsdPhysics.CollisionAPI):
                b = cache.ComputeWorldBound(p).ComputeAlignedRange()
                feet.append(dict(path=path[-70:], type=p.GetTypeName(), lo=np.round(np.array(b.GetMin()), 4).tolist(),
                                 hi=np.round(np.array(b.GetMax()), 4).tolist()))  # fmt: skip
            if p.HasAPI(UsdPhysics.MaterialAPI) and len(mats) < 20:
                m = UsdPhysics.MaterialAPI(p)
                mats.append((path[-70:], m.GetStaticFrictionAttr().Get(), m.GetDynamicFrictionAttr().Get()))
        return dict(foot_collisions=feet[:16], materials=mats)

    tr("usd", usd)
    x = h.sample()
    hd = np.array([1.0, 0.0])
    p0 = x["root"][:2]
    tr("toe_ahead_of_pelvis_mirror_m", lambda: float(body.frontmost(h.d, p0, hd)[0]))
    tr("ankle_xy_w", lambda: {s: np.round(t(rob.data.body_link_pos_w)[list(rob.data.body_names).index(f"{s}_ankle_roll_link")], 4).tolist()
                              for s in ("left", "right")})  # fmt: skip
    tr("pelvis_w", lambda: np.round(x["root"], 4).tolist())
    tr("feet_load_N", lambda: body.load(None).round(1).tolist())
    tr("reach_own_m", lambda: float(body.reach_own(x, hd)))
    tr("dq_legs_abs_mean_at_stand", lambda: float(np.abs(np.asarray(x["dq"], float)[np.asarray(h.to_mj)[:12]]).mean()))
    tr("delay_f", lambda: int(h.ref_delay_f))
    for k, v in out.items():
        print(f"hw_ha probe {a.controller} {k}: {json.dumps(v, default=str)}", flush=True)


def run(a) -> int:
    _orig = ha_sim.sev._configure_renderer_gpu

    def _cfg(args):
        _orig(args)
        SXH.kit_single_gpu(args)

    ha_sim.sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    a.no_walker = True
    ha_sim.check_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    t_up = time.time()
    sim = ha_sim.Sim(a)
    sim.provider._should_refresh_lerobot_visuals_next_step = lambda: False  # cameras off
    SXH.setup_paths()
    sys.path.insert(0, str(HC.W_REPO / "src"))  # SX_REPO: the round's copy (hw_ha.sh)
    sys.path.insert(0, str(HERE))
    import hw_run as H

    H.code_check("start")
    L, D3, HW, LC = H.L, H.D3, H.HW, H.LC
    import sx_collect as SX

    if a.controller == "sonic":
        LC.HAND_V = SONIC_HAND_V
    h = make_host(a, sim, L, D3)
    body = ha_body(H, h)
    H.HW = types.SimpleNamespace(**{k: getattr(HW, k) for k in dir(HW) if not k.startswith("__")})
    H.HW.LegOdo = ha_odo_cls(HW)
    rig, ps = SX.SRig(h), LC.PropSettle(h)
    print(f"hw_ha {a.controller}: up in {time.time() - t_up:.0f} s; dt {sim.dt:.3f}; c {np.round(h.c, 6).tolist()}; delay_f {h.ref_delay_f}; "
          f"site {a.site or HC.SITE}; hand limit {LC.HAND_V} m/s", flush=True)  # fmt: skip
    if a.probe:
        h.env.rng = np.random.default_rng([H.TAG, 0, 1])
        h.reset()
        fr = h.start_frames()
        for k in range(100):
            h.push(fr[min(k, len(fr) - 1)])
            h.step()
        probe(a, sim, h, body)
        return 0
    G = H.gaits()
    arms = a.arms.split(",")
    cells = H.rng_of(a.cells)
    jobs = [(arm, cid, rep) for arm in arms for cid in H.cells_of(arm) if cid in cells for rep in H.rng_of(a.reps)]
    jobs = jobs[a.shard :: a.shards]
    done = H.done_jobs(a.out)
    todo = [j for j in jobs if j not in done]
    sub = f"ha_{a.controller}"
    d = a.out / sub / "runs"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"n{a.shards}_{a.shard}_{os.getpid()}.jsonl"
    print(f"hw_ha {a.controller} shard {a.shard}/{a.shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    for arm, cid, rep in todo:
        if (a.out / "STOP").exists():
            break
        assert H.ARMS[arm]["backend"] == a.controller, (arm, a.controller)
        try:
            r = H.trial(h, body, rig, ps, arm, cid, rep, G, None, a.out / sub / "traces" if a.traces else None)
            r.update(sim="ha", placed=dict(h.placed))
        except Exception as e:  # noqa: BLE001 -- recorded, the shard goes on
            r = dict(arm=arm, cid=cid, rep=rep, sim="ha", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])
        with open(fn, "a") as fh:
            fh.write(json.dumps(r, default=H._js) + "\n")
        H.report(r)
    H.code_check("end")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = run(parser().parse_args())
    finally:
        os._exit(code)
