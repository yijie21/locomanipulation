"""selfmodel_hacollect_20261126 (developed 2026-11-26; the collector of selfmodel_hadata_20261126 for ScaleBFM / HEFT and
selfmodel_hadata_sonic_20261126 for SONIC): data3 / data3b's self-model collection in HumanoidArena's (HA's) body. Runs
in HA's runtime:

    hc_ha.sh <gpu> --phase site|prof|drive --controller scalebfm|heft|sonic --out DIR [--b] [--resets N --shards P --shard K]

The host is selfmodel_xfer_20261125's (sx_ha.HAHost: HA's own SonicActionProvider, the 40-D frame through the documented
VLA hook; sx_ha.HAG1Host: ha_controllers_20261125's ha_g1ctrl provider for ScaleBFM / HEFT, LML's G1_BRIDGE path), used as
it is: ha_g1ctrl and HA's provider are not touched, and the joint velocity the controllers read is HA's own. What is new:

  interface     LocoManiLab's HostDriver calls (push / step / _find_buffer / buf, with sample / start_frames / reset / pl /
                c from sx_ha), so data3's collector runs unchanged: d3_collect.reset_plan + run_reset (data3: a third of
                the resets walk first, 0.3-1.5 m, then two reaches) or, with --b, reset_plan_b + run_reset_b (data3b: four
                slots: pre + entry pairs, rel reaches, a detour first in a third of the resets), d3_collect.segments for
                the record, sx_collect's d_this / d_prev and sx_ha's scene contact added. push() stores the frame, step()
                runs sx_ha's step on it (HAG1Host turns it through the motion buffer's episode transform from the second
                frame on, as HostDriver.push does); _find_buffer() returns None, HAG1Host.step sets buf after the first
                controller tick (so frame 0 goes in untransformed and frame 1 through the transform, as in LML).
  start noise   LocoManiLab's reset (robosuite EmptyEnv, initialization_noise "default"): N(0, 0.02 rad) on each of the
                robot's 43 joints (29 body + 14 Dex3) at the spawn pose, zero velocity, the base as spawned (probed
                2026-11-26 on lml_host.HostDriver: noise cfg {'magnitude': 0.02, 'type': 'gaussian'}, all 43
                robot_joints, base std 0). Here: the same draw on HA's 43 joints, after HA's episode reset and before the
                controller is told of it (its reset reads the perturbed pose, as LML's first tick does), from
                default_rng([seed + i, 1]) (the reset's plan uses default_rng(seed + i)); clipped to the soft limits.
  spawn site    ppbox spawns the robot in a corner of its room (a wall 0.4 m behind, the table 0.96 m ahead, the
                baseboard 1.45 m to the right; xfer dev/ha_scene.log): walks need open floor, so the robot is moved
                (root xy; height and heading as spawned) to --site, a point the `site` phase found clear.
  no render     --no-camera (or SX_NORENDER=1): SONIC's per-step camera / observation refresh off, as xfer's revision 2
                (nothing here reads a camera; the g1 provider steps physics with render=False anyway).
  settle        fixed for the formal runs (set on d3_collect's module constants, which its waits read every time):
                the longest wait SETTLE_MAX_S = 6 s (d3_collect: 3 s), the reference hold after each splice HOLD_S =
                7.5 s (d3_collect: 4.5 s for its 3 s wait: it must outlast the buffer delay + the wait), the threshold
                SETTLE_V per controller: 2 cm/s (d3_collect's) for ScaleBFM and HEFT, 5 cm/s for SONIC (the user's
                decision 11-26: SONIC's hands in HA keep shaking at 1-14 cm/s). The rule is otherwise d3_collect's:
                pelvis and both hand points below SETTLE_V for 0.5 s (25 steps), a wait that runs out skips the next reach.
  extra fields  per segment (all controllers): settled_2cm (the end label also settled under 2 cm/s: settled_end and
                v_end_cms < 2, the same 25-step window), jit_pp_cm / jit_rms_cm (the segment's own hand point over the
                25 steps ending at its settled end label: the largest distance between two of its positions, and the RMS
                distance to their mean; NaN without a settled label).

Phases: site (clear floor: the scene's USD collision-prim boxes around candidate points), prof (per-step wall time by
part), drive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W_REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
XFER = Path(os.environ.get("HC_XFER", W_REPO / "scripts/experiments/selfmodel_xfer_20261125"))
sys.path.insert(0, str(XFER))
import sx_ha as SXH  # noqa: E402  (imports ha_sim: isaaclab.app only)

ha_sim = SXH.ha_sim
SIGMA = 0.02  # rad, robosuite's "default" initialization noise
SITE = (-1.71, -0.09)  # world xy, ppbox: the site phase's clearest point (2.20 m to the nearest prim box; dev/site.log)
SEED, ID0 = 89_000_000, 90_000_000  # dev only (xfer 79M / 40M; data3 73M / 20M; data3b 77M / 30M)
SETTLE_MAX_S, HOLD_S = 6.0, 7.5  # s (d3_collect: 3.0, 4.5)
SETTLE_V = {"scalebfm": 0.02, "heft": 0.02, "sonic": 0.05}  # m/s
EXTRA = ("settled_2cm", "jit_pp_cm", "jit_rms_cm")


def parser():
    p = ha_sim.build_parser()
    p.add_argument("--phase", choices=["site", "prof", "drive"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controller", choices=["sonic", "scalebfm", "heft"], default="sonic")
    p.add_argument("--b", action="store_true", help="data3b's slots (reset_plan_b / run_reset_b)")
    p.add_argument("--b-config", default=None, help="data3b's b_config.json (default: d3_collect.DEFAULT_B)")
    p.add_argument("--no-walk", action="store_true")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--site", default=None, help="x,y world (default SITE); 'none' keeps HA's spawn")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--delay-f", type=int, default=None)
    p.add_argument("--stance", choices=["native", "lml"], default=None)
    p.add_argument("--resets", type=int, default=10)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default="")
    p.add_argument("--hc-seed", type=int, default=SEED)
    p.add_argument("--hc-id0", type=int, default=ID0)
    p.add_argument("--vtrace", action="store_true", help="dev: per-frame settle speed (all, pelvis, right, left hand point) "
                   "and whether the wait tested quietness on that frame, into the reset log")
    return p


# -- the host: HostDriver's interface on sx_ha's ------------------------------------------------------------------------
class HCMixin:
    def _hc_init(self, sigma: float, site, D3):
        self.sigma, self.site, self.noise_seed, self._f = float(sigma), site, 0, 0
        self.placed = {}

    def push(self, fw: np.ndarray) -> None:
        self.frames = np.asarray(fw, np.float64)[None]

    def step(self, f: int | None = None) -> bool:  # HostDriver.step(); sx_ha's step(f) on the pushed frame
        r = super().step(self._f)
        self._f += 1
        return r

    def _find_buffer(self):
        return None

    def place(self) -> None:
        """After HA's episode reset, before the controller hears of it: the site, then LML's spawn noise."""
        import torch

        rob, env = self.sim.robot, self.sim.env
        rng = np.random.default_rng([int(self.noise_seed), 1])
        rec = {}
        if self.site is not None:
            rp = rob.data.root_link_pose_w.clone()
            rec["spawn_xy"] = rp[0, :2].detach().cpu().numpy().round(4).tolist()
            rp[:, 0] = float(self.site[0])
            rp[:, 1] = float(self.site[1])
            rob.write_root_link_pose_to_sim(rp)
            rob.write_root_velocity_to_sim(torch.zeros((1, 6), device=rp.device))
        jp = rob.data.joint_pos.clone()
        n = jp.shape[1]
        eps = rng.standard_normal(n) * self.sigma
        jp = jp + torch.as_tensor(eps, dtype=jp.dtype, device=jp.device)[None]
        lim = rob.data.soft_joint_pos_limits
        jp = torch.minimum(torch.maximum(jp, lim[..., 0]), lim[..., 1])
        rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        psv = getattr(env.sim, "physics_sim_view", None)
        if psv is not None:
            psv.update_articulations_kinematic()
        rec.update(noise_std=float(eps.std()), noise_max=float(np.abs(eps).max()), n=int(n))
        self.placed = rec

    def reset(self) -> None:
        sev = ha_sim.sev
        orig = sev._notify_action_provider_env_reset

        def notify(p):
            self.place()
            orig(p)

        sev._notify_action_provider_env_reset = notify
        try:
            super().reset()
        finally:
            sev._notify_action_provider_env_reset = orig
        self._f = 0


class HCSonic(HCMixin, SXH.HAHost):
    def __init__(self, sim, L, D3, delay_f, stance, sigma, site):
        self._hc_init(sigma, site, D3)
        super().__init__(sim, L, D3, delay_f, stance)


class HCG1(HCMixin, SXH.HAG1Host):
    def __init__(self, sim, L, D3, backend, stance, delay_f, sigma, site):
        self._hc_init(sigma, site, D3)
        super().__init__(sim, L, D3, backend, stance, delay_f)


# -- the record --------------------------------------------------------------------------------------------------------
def save_chunk(SX, D3, segs: list[dict], d: Path, name: str, rig, scalars) -> None:
    """sx_collect.save_chunk with data3b's scalars when --b."""
    (d / "arm").mkdir(parents=True, exist_ok=True)
    if not segs:
        return
    z, a = D3.pack(segs, scalars)
    z["d_this"] = np.stack([s["d_this"] for s in segs])
    z["d_prev"] = np.stack([s["d_prev"] for s in segs])
    for k in ("touch_n", "touch_z"):
        if all(k in s for s in segs):
            z[k] = np.array([s[k] for s in segs], np.float32)
    a.update(radius=rig.radius, link=np.array(rig.links))
    for path, arr in ((d / f"{name}.npz", z), (d / "arm" / f"{name}.npz", a)):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, path)


def settle_fields(segs: list[dict], rows: dict, reaches: list[dict], L, D3) -> None:
    """EXTRA per segment: settled_2cm, jit_pp_cm, jit_rms_cm (module docstring)."""
    for sg in segs:
        rc = reaches[sg["chain"]]
        side = "right" if int(sg["side"]) == 1 else "left"
        f = rc.get("f_label")
        ok = bool(sg["settled_end"]) and not bool(sg["fell"]) and f is not None
        sg["settled_2cm"] = bool(ok and float(sg["v_end_cms"]) < 2.0)
        if ok:
            P = np.array([rows[g]["wr"][side][0] + rows[g]["wr"][side][1] @ L.HAND_OFF
                          for g in range(f - D3.SETTLE_F + 1, f + 1) if g in rows], float)
            sg["jit_pp_cm"] = float(100 * np.linalg.norm(P[:, None] - P[None], axis=2).max())
            sg["jit_rms_cm"] = float(100 * np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
        else:
            sg["jit_pp_cm"] = sg["jit_rms_cm"] = float("nan")


def md5(p: Path) -> str:
    import hashlib

    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# -- the clear-floor search ----------------------------------------------------------------------------------------------
def find_site(sim, tag: str) -> None:
    """Clear floor from the USD stage (PhysX scene-query rays return no hit in this runtime): every collision prim not on
    the robot, its world AABB; the room's outline from the ceiling's AABB and the baseboards (thin pads along the walls);
    furniture = the other prims with an AABB under 6 m across reaching 0.02-2.0 m in height. A candidate on a 0.25 m grid
    inside the outline scores its xy distance to the nearest furniture box and to the outline."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    rs = sim.robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(float)
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    boxes, ceil, base, floors = [], None, [], []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot" in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
        low = path.lower()
        if "ceiling" in low:
            ceil = (mn, mx) if ceil is None else (np.minimum(ceil[0], mn), np.maximum(ceil[1], mx))
            continue
        if "baseboard" in low:
            base.append((path, mn, mx))
            continue
        if mx[2] - mn[2] < 0.06 and mn[2] < 0.03 and (mx[:2] - mn[:2]).max() > 3:
            floors.append((path, mn, mx))
            continue
        if (mx[:2] - mn[:2]).max() < 6 and mx[2] > 0.02 and mn[2] < 2.0:
            boxes.append((path, mn, mx))
    print(f"{tag}: spawn {np.round(rs, 3).tolist()}; ceiling xy {np.round(ceil[0][:2], 2).tolist()}..{np.round(ceil[1][:2], 2).tolist()}; "
          f"{len(base)} baseboards, {len(floors)} floor-like, {len(boxes)} furniture prims", flush=True)
    for pth, mn, mx in floors[:10]:
        print(f"  floor-like {pth}: {np.round(mn, 2).tolist()}..{np.round(mx, 2).tolist()}", flush=True)
    bb = np.array([np.r_[mn[:2], mx[:2]] for _, mn, mx in base]) if base else np.zeros((0, 4))
    if len(bb):
        print(f"  baseboards span x {bb[:, 0].min():.2f}..{bb[:, 2].max():.2f}, y {bb[:, 1].min():.2f}..{bb[:, 3].max():.2f}", flush=True)
    lo = np.maximum(ceil[0][:2], bb[:, :2].min(0)) if len(bb) else ceil[0][:2]
    hi = np.minimum(ceil[1][:2], bb[:, 2:].max(0)) if len(bb) else ceil[1][:2]
    F = np.array([np.r_[mn[:2], mx[:2]] for _, mn, mx in boxes])
    out = []
    for x in np.arange(lo[0] + 0.5, hi[0] - 0.49, 0.25):
        for y in np.arange(lo[1] + 0.5, hi[1] - 0.49, 0.25):
            dx = np.maximum(np.maximum(F[:, 0] - x, x - F[:, 2]), 0.0)
            dy = np.maximum(np.maximum(F[:, 1] - y, y - F[:, 3]), 0.0)
            d = np.hypot(dx, dy)
            # the outline: walls along the baseboards (each a thin box: distance to it), else the outline box's edges
            dw = min(x - lo[0], hi[0] - x, y - lo[1], hi[1] - y)
            if len(bb):
                bx = np.maximum(np.maximum(bb[:, 0] - x, x - bb[:, 2]), 0.0)
                by = np.maximum(np.maximum(bb[:, 1] - y, y - bb[:, 3]), 0.0)
                dw = min(dw, float(np.hypot(bx, by).min()))
            k = int(np.argmin(d))
            out.append((min(float(d[k]), dw), float(x), float(y), float(d[k]), dw, boxes[k][0]))
    out.sort(reverse=True)
    for o in out[:12]:
        print(f"  clear {o[0]:.2f} m at ({o[1]:.2f}, {o[2]:.2f}): furniture {o[3]:.2f} m ({o[5]}), outline {o[4]:.2f} m", flush=True)
    near = sorted(boxes, key=lambda b: np.hypot(*(np.clip(out[0][1:3], b[1][:2], b[2][:2]) - np.array(out[0][1:3]))))[:8]
    for pth, mn, mx in near:
        print(f"  near the best: {pth}: {np.round(mn, 2).tolist()}..{np.round(mx, 2).tolist()}", flush=True)


# -- run -----------------------------------------------------------------------------------------------------------------
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
    if os.environ.get("SX_NORENDER") == "1" or a.no_camera:
        sim.provider._should_refresh_lerobot_visuals_next_step = lambda: False
    SXH.setup_paths()
    sys.path.insert(0, str(W_REPO / "src"))
    import sx_collect as SX

    D3, L = SX.D3, SX.L
    site = None if (a.site or "").lower() == "none" else tuple(float(v) for v in (a.site or f"{SITE[0]},{SITE[1]}").split(","))
    tag = f"hc_ha {a.controller} {a.phase}"
    if a.phase == "site":
        find_site(sim, tag)
        return 0
    if a.controller == "sonic":
        h = HCSonic(sim, L, D3, SXH.SONIC_DELAY_F if a.delay_f is None else a.delay_f, a.stance or "native", a.sigma, site)
    else:
        h = HCG1(sim, L, D3, a.controller, a.stance or "lml", a.delay_f, a.sigma, site)
    rig = SX.SRig(h)
    # the settle rule (module constants d3_collect reads on every wait / splice; Rig.quiet reads SETTLE_V)
    D3.SETTLE_MAX_F = int(round(SETTLE_MAX_S * D3.FPS))
    D3.HOLD_S = HOLD_S
    D3.SETTLE_V = SETTLE_V[a.controller]
    trace = []
    if a.vtrace:
        rec0, quiet0 = rig.record, rig.quiet

        def rec(x):
            v = rec0(x)
            sp = (np.linalg.norm(rig.pts[-1] - rig.pts[0], axis=1) / (D3.TICKS * L.DT)).tolist() if len(rig.pts) > D3.TICKS else [None] * 3
            trace.append([None if not np.isfinite(v) else round(v, 4)] + [None if u is None else round(u, 4) for u in sp] + [0])
            return v

        def quiet():
            if trace:
                trace[-1][-1] = 1
            return quiet0()

        rig.record, rig.quiet = rec, quiet
    h.sample()
    import torch

    gpu_mb = torch.cuda.max_memory_allocated() / 2**20
    print(f"{tag}: up in {time.time() - t_up:.0f} s (sim {sim.startup_s:.0f} s); dt {sim.dt:.3f}; c {np.round(h.c, 6).tolist()}; "
          f"root check {rig.check_root():.1e} m; placed {h.placed}; torch max alloc {gpu_mb:.0f} MB", flush=True)  # fmt: skip
    table = json.loads(Path(D3.TABLE).read_text())
    cfg = D3.b_config(a.b_config) if a.b else None
    walk_ok = not a.no_walk

    def one(i):
        h.ep_seed = a.hc_id0 + i
        h.noise_seed = a.hc_seed + i
        h.dev = dict(xy=0.0, z=0.0, rot_deg=0.0)
        rng = np.random.default_rng(a.hc_seed + i)
        if a.b:
            plan = D3.reset_plan_b(rng, table, walk_ok, cfg)
            rows, info, frames, reaches = D3.run_reset_b(h, rig, plan, table, cfg)
        else:
            plan = D3.reset_plan(rng, table, walk_ok)
            rows, info, frames, reaches = D3.run_reset(h, rig, plan)
        s = D3.segments(rows, info, frames, reaches, a.hc_id0 + i, h.pl, rig)
        if a.b:
            D3.augment_b(s, reaches)
        SX.add_fields(s, reaches)
        settle_fields(s, rows, reaches, L, D3)
        info["touch"] = SXH.add_touch(h, s, reaches)
        info["placed"] = dict(h.placed)
        if a.vtrace:
            info["vtrace"] = [list(t) for t in trace]
            trace.clear()
        return plan, s, info

    if a.phase == "prof":  # wall time per control step by part, over --limit resets
        acc = {}

        def timed(obj, name, key):
            f0 = getattr(obj, name)

            def w(*args, **kw):
                t = time.perf_counter()
                try:
                    return f0(*args, **kw)
                finally:
                    acc[key] = acc.get(key, 0.0) + time.perf_counter() - t
                    acc[key + "_n"] = acc.get(key + "_n", 0) + 1

            setattr(obj, name, w)

        env = sim.env
        timed(env.sim, "step", "physx_step")
        timed(env.scene, "update", "scene_update")
        timed(env.scene, "write_data_to_sim", "write_data")
        timed(env.sim, "render", "render")
        timed(h, "sample", "sample")
        timed(rig, "record", "rig_record")
        timed(h, "touch_now", "touch_now")
        ctl = getattr(h, "ctl", None) or sim.controller
        timed(ctl, "step", "controller_step")
        if hasattr(h, "prov"):
            p0, f0 = h.prov.t_policy, h.prov.t_phys
        t0 = time.time()
        steps, plan_s = 0, 0.0
        for i in range(a.limit or 3):
            plan, s, info = one(i)
            steps += info["steps"]
            plan_s += info["plan_s"]
            print(f"{tag}: reset {i}: {info['steps']} steps, {info['wall_s']:.1f} s wall, why {info['why']!r}, walked "
                  f"{info['walked']}, slots {[(x['status'], round(x.get('wait_s', 0), 2)) for x in info['slots']]}, segs {len(s)}", flush=True)
        wall = time.time() - t0
        ms = lambda k: 1000 * acc.get(k, 0.0) / max(steps, 1)  # noqa: E731
        parts = {k: round(ms(k), 2) for k in acc if not k.endswith("_n")}
        if hasattr(h, "prov"):
            parts["g1_policy_tick"] = round(1000 * (h.prov.t_policy - p0) / steps, 2)
            parts["g1_phys_loop"] = round(1000 * (h.prov.t_phys - f0) / steps, 2)
        parts["planning"] = round(1000 * plan_s / steps, 2)
        calls = {k[:-2]: round(acc[k] / steps, 2) for k in acc if k.endswith("_n")}
        print(f"{tag}: {steps} control steps in {wall:.1f} s = {1000 * wall / steps:.1f} ms/step ({steps * sim.dt / wall:.2f}x real time); "
              f"ms per control step by part: {json.dumps(parts)}; calls per step {json.dumps(calls)}; torch max alloc "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)  # fmt: skip
        return 0
    # drive
    ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(a.shard, a.resets, a.shards))
    n = min(a.limit or len(ids), len(ids))
    subs = ("data", "data_both") + (("data_detour",) if a.b else ())
    dirs = {k: a.out / k for k in subs}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    rp = a.out / "run.json"
    if a.shard == 0 and not rp.exists():
        code = SX.code_md5() | {"sx_ha.py": md5(XFER / "sx_ha.py"), "hc_ha.py": md5(Path(__file__))}
        rp.write_text(json.dumps(dict(controller=a.controller, planner="lml", stance=h.stance, delay_f=h.ref_delay_f, seed=a.hc_seed,
                                      id0=a.hc_id0, resets=a.resets, calib=h.calib, task=a.ha_task, host="humanoidarena",
                                      body_fk="robot_fk", mode="d3b" if a.b else "d3", walk=walk_ok, sigma=a.sigma,
                                      site=site, no_camera=bool(a.no_camera), b_config=cfg, settle_max_f=D3.SETTLE_MAX_F, hold_s=D3.HOLD_S, settle_v=D3.SETTLE_V,
                                      norender=os.environ.get("SX_NORENDER") == "1" or bool(a.no_camera),
                                      body=dict(REACH_SCENE_DIR=str(SXH.SCENE.parent)), md5=code), indent=1))  # fmt: skip
    scalars = (D3.SCALARS_B if a.b else D3.SCALARS) + EXTRA
    t0, nres, nseg = time.time(), 0, 0
    for c in range(0, n, D3.CHUNK):
        name = f"w{a.shard:03d}_{c // D3.CHUNK:03d}"
        if (dirs["data"] / f"{name}_log.json").exists():
            continue
        out, log = {k: [] for k in subs}, []
        for j in range(c, min(c + D3.CHUNK, n)):
            i = ids[j]
            plan, s, info = one(i)
            for x in s:
                k = "data_detour" if a.b and x["kind"] == D3.KIND["detour"] else "data_both" if x["hands"] == 2 else "data"
                out[k].append(x)
            ex = dict(kinds=[sl["kind"] for sl in plan["slots"]]) if a.b else {}
            log.append(dict(reset=a.hc_id0 + i, n_segs=len(s), walk_asked=plan["walk"] is not None, ha_fell_at=h.ha_fell_at,
                            roundtrip=dict(h.dev), **ex, **info))  # fmt: skip
            nres, nseg = nres + 1, nseg + len(s)
            print(f"{tag} shard {a.shard}: {j + 1}/{n} resets, {nseg} segments, {time.time() - t0:.0f} s "
                  f"({(time.time() - t0) / nres:.1f} s/reset); last: walked {info['walked']} why {info['why']!r} slots "
                  f"{[x['status'] for x in info['slots']]}", flush=True)  # fmt: skip
        for k, d in dirs.items():
            save_chunk(SX, D3, out[k], d, name, rig, scalars)
        (dirs["data"] / f"{name}_log.json").write_text(json.dumps(log, default=D3._json))
    print(f"{tag} shard {a.shard} done: {nres} resets, {nseg} segments, {time.time() - t0:.0f} s; torch max alloc "
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    args = parser().parse_args()
    ha_sim.main_guard(lambda: run(args))
