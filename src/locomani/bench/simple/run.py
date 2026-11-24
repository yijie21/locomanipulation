"""One SIMPLE episode (or one look) in SIMPLE's own environment, and a batch of them (TODO 49.2).

Moved from escalation_20261116/bed_simple_official.py (setup, look, run, each, layout_check, cache key, the see_self filter) and
the parts of bed_simple.py they use (comply_fields, policy, scene_dir). Runs under the SIMPLE sim venv (mujoco, torch, the
ScaleBFM runtime) as `python -m locomani.bench.simple batch|look|cachedir|check`, with the three services up (the environment
server and the stereo depth service of this call's slot, session.py; SAM 3). The step count, the success and the time limit are
the environment's own: the episode ends at the step the task's check_success holds or when TimeLimit truncates it; after a
truncation it runs on (raw steps, no judge) up to CAP_S only to tell a slow episode (loss "timeout") from one that would never
have succeeded.

The skill package (--skills: a folder with __init__.py, skill.py, pick.py, camera_view.py) is loaded by path under a unique
module name (`load_package`); nothing is put on sys.path. A package gets the episode facade `OEnv` (client.py): its task, the
layout, the Motor and the Belief, the parameters, the record, the clock's phases, and its view; `sam3`, `cloud_dir` and
`mask_post` are what its camera view needs of the run.

Only the switches the loops ran with are implemented (target=perception, sensor=stereo, depth_noise=none, self_pose=odometry,
controller_root=odometry, controller=scalebfm): `check_scope` refuses the others with a pointer to TODO 49.2 step 6 / 49.3.
Not ported (hand switches that were off in every loop, record only): PHASE_BOX, FOOT_LOG, ART_FRAMES, NEAR_DIAG; the named speed
settings L2/L3/L4 (sweep_simple); the legacy non-loop cache folders (cloud, cloud_clean, cloud_self_n*).
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import itertools
import json
import resource
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from types import ModuleType

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import MotorConfig
from locomani.bench.simple import body as BODY
from locomani.bench.simple import client as C
from locomani.bench.simple import episodes as EP
from locomani.bench.simple import feed as FL
from locomani.bench.simple import judge
from locomani.bench.simple.config import (  # noqa: F401
    DEFAULT_VIEW,
    SCOPE,
    SEED_DIR,
    ScopeError,
    SimpleConfig,
    check_scope,
)
from locomani.bench.simple.judge import loss_of
from locomani.bench.simple.privileged import SimplePrivileged
from locomani.perception import head
from locomani.perception.truth import TruthView

DT, CAP_S, LIMIT = C.DT, C.CAP_S, C.LIMIT
PACKAGE_FILES = ("__init__.py", "skill.py", "pick.py", "camera_view.py")

# The parameters the look path reads (the look cache is keyed by them and by the package's camera_view.py): CameraView.look
# (see_clean: the cleaned cloud; see_foot: the foot level), the bed's setup / look (see_self + self_band: the robot's own pixels
# out of the mask; start_hold: the stabilising hold before the look; feed_torque / seat_feed / odo_stop: the estimator the pose
# of the look comes from) and the Motor keys that shape the start hold (skill.MOTOR_KEYS).
LOOK_KEYS = ("see_clean", "see_foot", "see_self", "self_band", "start_hold", "feed_torque", "seat_feed", "odo_stop")
BED_KEYS = (
    "feed_torque",
    "seat_feed",
    "amo_arm_pos",
    "amo_hand_gain",
    "amo_hand_limit",
    "see_self",
    "self_band",
)  # settings the bed reads, not the skill
COMPLY_FACTS = (
    "t_hand",
    "t_carry",
    "t_right",
    "t_left_ok",
    "best_left_s",
    "t_container",
    "last",
    "t_check",
    "t_held",
    "best_held_s",
)
ART_FACTS = ("t_body", "hand_deg", "body_links")  # the articulated tasks' verdict also reads these (t_hand is above)

_COUNTER = itertools.count()


# ---- the skill package, loaded by path


def load_package(path) -> ModuleType:
    """Load the skill package in folder `path` under a unique module name: spec_from_file_location with the folder as the
    package's search location, so its relative imports (`from . import pick as P`) find its own files and two copies of the
    package (the loop's drafts) load side by side without clashing. Nothing is put on sys.path."""
    path = Path(path).resolve()
    missing = [f for f in PACKAGE_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: skill package files missing: {missing}")
    name = f"_simple_skill_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        for sub in ("skill", "pick", "camera_view"):
            importlib.import_module(f"{name}.{sub}")
    except BaseException:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        raise
    mod.path = path
    return mod


# ---- what a run holds


@dataclasses.dataclass
class Bed:
    """One process's run context: the configs, the skill package, the slot's sockets, the policy."""

    cfg: SimpleConfig
    pkg: ModuleType
    slot: int = 1
    record: dict = dataclasses.field(default_factory=dict)  # the `bench` and `simple_config` fields of every row
    options: object = None  # the run's resolved BenchOptions (None: the loops' defaults)
    _links: dict = dataclasses.field(default_factory=dict)
    _pol: object = None

    @property
    def S(self):
        return self.pkg.skill

    def switch(self, axis: str) -> str:
        """A run switch's value: the resolved options', else the loops' default."""
        o = self.options
        return getattr(o, axis) if o is not None else dict(zip(("target", "sensor", "depth_noise"), DEFAULT_VIEW))[axis]

    @property
    def depth_source(self) -> str:
        """What the camera view's depth comes from, for the row: "stereo" (the stereo model), "rgbd" / "rgbd:realsense", or "none" (truth)."""
        if self.switch("target") == "truth":
            return "none"
        if self.switch("sensor") == "rgbd":
            return "rgbd" if self.switch("depth_noise") == "none" else f"rgbd:{self.switch('depth_noise')}"
        return "stereo"

    def links(self):
        """The two connections of this slot (environment server, depth service), opened on first use, kept for the process."""
        for k in ("isaac", "depth"):
            if k == "depth" and self.depth_source != "stereo":
                self._links[k] = None  # rgbd and truth runs have no stereo depth service
            if k not in self._links:
                self._links[k] = Client(str(Path(self.cfg.slots_dir) / f"s{self.slot}_{k}.sock"), family="AF_UNIX")
        return self._links["isaac"], self._links["depth"]

    def policy(self):
        if self._pol is None:
            # the runtime's jaw offset is a module array (no argument for it): set once to the Dex3 palm, as the old bed did
            R.OFF[:] = BODY.PALM
            self._pol = R.Policy("cuda", ScaleBfmBackendConfig())
        return self._pol

    def cache_dir(self, task: str, setting: dict | None) -> Path:
        """The look + GraspGen-X cache of this package and these parameters: <cache_dir>/<key>."""
        return Path(self.cfg.cache_dir) / cache_key(self.pkg, task, setting, self.switches())

    def switches(self) -> tuple:
        return tuple(self.switch(a) for a in ("target", "sensor", "depth_noise"))


def resolved(S, task: str, setting: dict | None) -> dict:
    """The parameters of a job: the task's defaults with the setting applied (bed keys and skill keys alike)."""
    return {**S.defaults(task, S.K_CUT), **(setting or {})}


def cache_key(pkg: ModuleType, task: str, setting: dict | None, switches: tuple = DEFAULT_VIEW) -> str:
    """First 10 hex of sha1 over the package's camera_view.py and the sorted JSON of the look's parameters. (Not in the key:
    the fixed perception code, the depth service, SAM 3's weights: as before.) `switches` = (target, sensor, depth_noise): a
    look made from the stereo depth must never be read by an rgbd run or the other way round, so they are hashed too, but
    only when they differ from the loops' defaults: with the default switches the key is what it always was (the esc bed's
    folders and the parity runs' caches stay valid)."""
    S = pkg.skill
    K = resolved(S, task, setting)
    h = hashlib.sha1((Path(pkg.path) / "camera_view.py").read_bytes())
    keys = tuple(LOOK_KEYS) + tuple(S.MOTOR_KEYS)
    h.update(json.dumps({k: K.get(k) for k in sorted(keys)}, sort_keys=True, default=str).encode())
    if tuple(switches) != DEFAULT_VIEW:
        h.update(json.dumps(dict(target=switches[0], sensor=switches[1], depth_noise=switches[2])).encode())
    return h.hexdigest()[:10]


def comply_fields(
    task: str, success: bool, state: dict | None, marks: dict | None = None, limit_s: float | None = None
) -> dict:
    """The record's `comply` = dict(ok, why) and `comply_facts` (the watcher's state without its timeline); an errored or
    lost episode (no state): ok False, why not_success."""
    v = judge.verdict(task, bool(success), state or {}, marks, limit_s) if state else dict(ok=False, why="not_success")
    facts = COMPLY_FACTS + (ART_FACTS if task in judge.ART else ())
    return dict(comply=dict(ok=v["ok"], why=v["why"]), comply_facts={k: (state or {}).get(k) for k in facts})


def scene_dir(cfg: SimpleConfig, episode_id: str) -> Path:
    """The built scene of an episode, under this call's run directory (one per process group)."""
    return Path(cfg.run_dir) / "scene" / episode_id.replace("/", "_")


# ---- the robot's own pixels out of the object's mask (see_self)


class SelfFilter:
    """see_self: keep the robot's own body out of the object's cloud (two rules, applied to the object's mask: `post`).
    `install` wraps the Belief's scene so that each look also renders the robot ALONE (every other geom hidden, so no object
    truth enters) from the head camera at the official joint readings and keeps its silhouette and depth for the rules."""

    def __init__(self, env, band: int):
        self.env, self.band, self.on, self.dropped, self.state = env, int(band), True, None, {}
        sim, bel = env.sim, env.bel
        m = sim.m
        pel = m.body("pelvis").id
        robot = np.array([int(m.body_rootid[b]) == int(m.body_rootid[pel]) for b in range(m.nbody)])
        self.hide = ~robot[m.geom_bodyid]  # geoms of everything but the robot
        scene = bel.scene

        def filtered():
            rgb, pts, ok = scene()
            self.render()
            self.state.update(meas=sim.last_eyes["lb"], ok=ok.copy(), rgb=rgb)
            return rgb, pts, ok

        bel.scene = filtered

    def render(self) -> None:
        sim = self.env.sim
        m, d = sim.m, sim.d
        H, W = head.HEAD_CAM_WH[1], head.HEAD_CAM_WH[0]
        mujoco.mj_forward(m, d)
        g = m.vis.global_
        g.offwidth, g.offheight = max(W, g.offwidth), max(H, g.offheight)
        grp0 = m.geom_group.copy()
        m.geom_group[self.hide] = 5
        opt = mujoco.MjvOption()
        opt.geomgroup[5] = 0
        r = mujoco.Renderer(m, H, W)
        r.update_scene(d, camera=head.HEAD_CAM, scene_option=opt)
        r.enable_depth_rendering()
        dep = r.render().copy()
        r.close()
        m.geom_group[:] = grp0
        self.state.update(sil=(dep > 0.05) & (dep < 5), dep=dep)

    def post(self, mask):
        """The two rules on the object's mask. (1) Within `band` pixels of the rendered robot silhouette, drop the mask pixels where
        the robot (the silhouette's nearest pixel's rendered depth) is nearer the camera than the measured depth. (2) Drop the mask
        pixels more than 2 cm nearer the camera than the median measured depth of the other mask pixels within a 15-pixel
        neighbourhood (a foreground veil; no robot model). The counts of pixels dropped (of pixels with a usable depth) are kept
        in `dropped`."""
        m1, r1, r2 = self._rules(mask)
        self.dropped = (r1, r2)
        return m1

    def _rules(self, mask):
        import cv2
        from scipy import ndimage

        if mask is None or "sil" not in self.state:
            return mask, 0, 0
        st = self.state
        sil, dep, meas, ok = st["sil"], st["dep"], st["meas"], st["ok"]
        near = cv2.dilate(sil.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=self.band).astype(bool)
        _, idx = ndimage.distance_transform_edt(~sil, return_indices=True)
        rdep = dep[idx[0], idx[1]]  # the rendered depth of the nearest silhouette pixel
        drop1 = mask & near & (rdep < meas - 0.005)
        m1 = mask & ~drop1
        ys, xs = np.nonzero(m1)
        drop2 = np.zeros_like(mask)
        if len(ys) > 50:
            y0, y1, x0, x1 = max(ys.min() - 16, 0), ys.max() + 17, max(xs.min() - 16, 0), xs.max() + 17
            sub = np.where(m1[y0:y1, x0:x1], meas[y0:y1, x0:x1], np.nan).astype(float)
            med = ndimage.generic_filter(
                sub, np.nanmedian, footprint=np.ones((31, 31), bool), mode="constant", cval=np.nan
            )
            veil = m1[y0:y1, x0:x1] & (med - meas[y0:y1, x0:x1] > 0.02)
            drop2[y0:y1, x0:x1] = veil
        return m1 & ~drop2, int((drop1 & ok).sum()), int((drop2 & ok).sum())


# ---- one episode


class RecBelief(BODY.Belief):
    """The skill's Belief that also records where each walk really ended against its goal (record only: row["walk_stop_cm"])."""

    row: dict | None = None

    def walk(self, xy, yaw, **kw):
        r = super().walk(xy, yaw, **kw)
        if self.row is not None:
            pt, _ = self._truth()
            d = np.asarray(self.from_sim(pt)[:2]) - np.asarray(xy)
            c, s_ = np.cos(yaw), np.sin(yaw)
            self.row.setdefault("walk_stop_cm", []).append(
                [round(float(100 * (d[0] * c + d[1] * s_)), 1), round(float(100 * (-d[0] * s_ + d[1] * c)), 1)]
            )
        return r


def setup(bed: Bed, episode_id: str, K: dict, stance: dict | None, cloud_dir: Path) -> C.OEnv:
    """The episode in SIMPLE's environment, the robot standing where SIMPLE put it, the clock running."""
    cfg, S = bed.cfg, bed.S
    task = episode_id.split("/")[0]
    ep = EP.load(episode_id, cfg)  # refuses released ids and pools this run was not given
    info = EP.build(ep, scene_dir(cfg, episode_id), cfg.simple_root, cfg.visual_dir)
    pol = bed.policy()
    link, depth = bed.links()
    with EP.no_model_cache():
        sim = C.OffCfgSim(pol.meta, ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""), link, depth)
        t0 = time.time()
        st = sim.begin_cfg(task, C.state_cfg(ep["cfg"]))
        echo: dict = {}
        for key, msg in (
            ("amo_arm_mode", "amo_arm_pos"),
            ("amo_hand_gain", "amo_hand_gain"),
            ("amo_hand_limit", "amo_hand_limit"),
        ):
            link.send(
                (key, bool(K.get(msg)))
            )  # AMO tasks only matter; default off. The server answers with the value it holds
            echo[key] = bool(link.recv())
        if cfg.lazy_render:  # default off: nothing is sent, the cameras render in every step
            link.send(("lazy_render", True))
            link.recv()
        if (
            bed.switch("sensor") == "rgbd" and bed.switch("target") == "perception"
        ):  # the annotator is attached on this message only
            seed = int(hashlib.sha1(episode_id.encode()).hexdigest()[:8], 16)  # a rerun of an id sees the same noise
            sim.use_rgbd(bed.switch("depth_noise"), seed)
        if (
            ep["cfg"]["robot_cfg"]["uid"] != "g1_sonic"
        ):  # AMO tasks: the root starts low, the pelvis 0.785 m over the ground
            sim.zoff = float(EP.AMO_PELVIS - st["qpos"][2])
        mo = C.BedMotor(
            sim,
            pol,
            dataclasses.replace(MotorConfig(), **{k: K[k] for k in S.MOTOR_KEYS}),
            model_xml=info["dir"] + "/scene_locomani.xml",
        )
    bel = RecBelief(
        sim, mo, True
    )  # odo=True: self_pose=odometry; it also hooks OdoRoot into the policy (controller_root=odometry)
    bel.walk_hold = K["walk_hold"]
    if K.get("feed_torque") or K.get(
        "seat_feed"
    ):  # the robot's own torque command where the odometry reads the foot loads
        FL.patch_torques(sim)
        if K.get("seat_feed"):
            bel.feed = FL.SlipFeed(bel.body)
    if K.get("odo_stop") and bel.odo:  # Belief switched the stop servo off for odometry: back on, steering on odometry
        bel.odo_stop = True
        mo.cfg = dataclasses.replace(mo.cfg, stop_ki=MotorConfig().stop_ki, stop_timeout=K["stop_timeout"])
    env = C.OEnv(task, info, sim, mo, bel, K, stance, sam3=cfg.sam3(), cloud_dir=cloud_dir)
    bel.row = mo.row = env.row
    env.limit, env.limit_eff = st["limit"], st["limit"] or LIMIT[task]
    env.row["flags"] = dict(
        server_echo=echo,
        feed_torque=bool(K.get("feed_torque")),
        see_clean=bool(K.get("see_clean")),
        odo_stop=bool(K.get("odo_stop")),
    )  # what the server and the bed hold: a null result is not a flag left off
    env.row["reset_s"] = round(st.get("reset_s", time.time() - t0), 2)
    env.row["limit_steps"] = st["limit"]
    env.clock = C.OClock(env)
    if K.get("see_self"):
        env.self_filter = SelfFilter(env, int(K.get("self_band", 6)))
        env.mask_post = env.self_filter.post
    if bed.switch("target") == "truth":  # the view from the simulator; the skill package's camera view is not built
        priv = SimplePrivileged(sim, info, task, ep["instruction"])
        env.view = TruthView(priv, bel, S.choose_grasp, cfg.grasp_lib_dir, K, task)
    else:
        env.view = bed.pkg.camera_view.CameraView(env, episode_id, ep["instruction"])
    mo.reset(info["start"]["xy"], info["start"]["yaw"])
    EP.pull_objects(sim.m, sim.d, info)
    sim.pull()  # the robot stands where SIMPLE put it; the hold is SIMPLE's stabilising phase, outside the step count
    mo._stand()
    mo.stand_pose = mo.ref(0.0)
    mo.hold(K["start_hold"])
    env.clock.start()
    bel.start()
    env.mark("start")
    return env


def free(env) -> None:
    env.mo.clock = None  # the clock holds the Motor and the Motor's tick is the clock's: break the ring
    env.sim = env.mo = env.bel = env.view = env.clock = None
    gc.collect()


def layout_check(bed: Bed, episode_id: str) -> dict:
    """Read the environment's state back after the reset and compare it with the layout asked for (cm, deg)."""
    task = episode_id.split("/")[0]
    ep = EP.load(episode_id, bed.cfg)
    A = ep["cfg"]["layout"]["actors"]
    link, _ = bed.links()
    link.send(("reset_cfg", task, C.state_cfg(ep["cfg"])))
    st = link.recv()
    link.send(("truth",))
    t = link.recv()["objects"]
    out = {}
    for k, a in A.items():
        if k in t:
            want = np.asarray(a["pose"]["position"], float)
            out[k] = round(float(100 * np.linalg.norm(np.asarray(t[k][:3]) - want)), 2)
    r = ep["cfg"]["dr_state_dict"]["spatial"][ep["cfg"]["robot_cfg"]["uid"]]
    out["robot_xy"] = round(float(100 * np.linalg.norm(st["qpos"][:2] - np.asarray(r["position"][:2]))), 2)
    out["instruction"] = st["instruction"]
    return out


def look(bed: Bed, episode_id: str, setting: dict | None = None) -> dict:
    """The look pass: the target's points seen by the official camera to <cache>/<name>.npz for GraspGen-X."""
    S, task = bed.S, episode_id.split("/")[0]
    if bed.switch("target") == "truth":
        raise ValueError("target=truth has no look pass: the simulator is read when asked")
    K = resolved(S, task, setting)
    out = bed.cache_dir(task, setting)
    env = setup(bed, episode_id, K, None, out)
    if getattr(S, "ART_TASK", {}).get(env.task):  # the articulated tasks: the object and the part, no cloud file
        seen = env.view.art()
        rec = dict(episode_id=episode_id, success=bool(seen["parts"]), obj_px=int(0 if seen["obj"] is None else len(seen["obj"])),
                   parts=[dict(p=r["p"].round(3).tolist(), n=r["n"], score=round(r["score"], 3)) for r in seen["parts"]])  # fmt: skip
        free(env)
        return rec
    s = env.view.look()
    rec = dict(episode_id=episode_id, success=s["mask"] is not None, score=float(s["score"]))
    sf = getattr(env, "self_filter", None)
    if (
        sf is not None and sf.dropped is not None
    ):  # see_self: pixels dropped by rule 1 (silhouette band) and rule 2 (veil)
        rec["self_rule1_px"], rec["self_rule2_px"] = sf.dropped
    if s["mask"] is not None:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{bed.pkg.camera_view.cloud_name(episode_id)}.npz", pts=s["cloud"], p_o=np.median(s["cloud"], axis=0),
                            R_o=np.eye(3), iou=0.0, score=s["score"])  # fmt: skip
        rec["px"] = int(len(s["cloud"]))
        t = env.truth()["objects"]["target"][:3] + [
            0.0,
            0.0,
            env.sim.zoff,
        ]  # record only: the target in the mirror's frame
        c = np.median(s["cloud"], axis=0)
        rec["cloud_vs_truth_cm"] = np.round(100 * (c - t), 1).tolist()  # the seen face is nearer than the body's centre
    free(env)
    return rec


def run(bed: Bed, episode_id: str, setting: dict | None = None, stance: dict | None = None) -> dict:
    """One episode -> its record (the row)."""
    S = bed.S
    t_wall = time.time()
    task = episode_id.split("/")[0]
    K = S.defaults(task, S.K_CUT)
    K["look_near"] = True
    extra = {k: v for k, v in (setting or {}).items() if k in BED_KEYS}  # bed keys, not skill keys
    skill = {k: v for k, v in (setting or {}).items() if k not in extra}
    unknown = set(skill) - set(K)
    assert not unknown, f"unknown parameters {sorted(unknown)}"
    K.update(skill)
    K.update(extra)
    env = setup(bed, episode_id, K, stance, bed.cache_dir(task, setting))
    mo, t_run = env.mo, time.time()
    try:
        S.episode(env)
        end = "script"
    except C.Done as e:
        end = str(e)
    run_s = time.time() - t_run
    c = env.clock
    limit = env.limit_eff
    limit_s = limit * DT
    env.mark("end")
    truth = env.truth()
    env.sim.link.send(("comply",))
    comply_state = env.sim.link.recv()  # judge.Watcher.state() of the server, fed there each counted step
    s0, se = env.marks["start"], env.marks["end"]
    off = c.official or {}
    t_succ = c.t_success if (c.t_success is not None or not c.pick) else c.t_check
    official_success = bool(
        (c.t_check if c.pick else c.t_success) is not None and not c.late
    )  # the environment's check_success held within its limit
    rec = dict(
        episode_id=episode_id, track="camera", env="official", base="cut", setting=setting or {}, stance=stance,
        controller="frozen", success=official_success,
        t_success=t_succ, t_check=c.t_check, success_45=bool(t_succ is not None and t_succ <= 45.0), limit_s=limit_s, loss=None,
        end=end, t_end=c.steps * DT, fell=mo.fell_at, phases={k: round(v * DT, 2) for k, v in c.phases.items()},
        official_success=official_success, official_steps=off.get("steps"), official_truncated=off.get("truncated"),
        official_reward=c.reward_last, official_reward_max=c.reward_max, limit_steps=limit,
        steps_per_s=round(c.steps / max(1e-6, run_s), 2), counted_steps=c.steps,
        rise_cm=float(100 * (se["p"][2] - s0["p"][2])),
        knocked=int(sum(np.linalg.norm((se["others"][k] - s0["others"][k])[:2]) > 0.03 for k in s0["others"])),
        wall_s=round(time.time() - t_wall, 1),
        truth_end={k: np.round(v[:3], 3).tolist() for k, v in truth["objects"].items() if k in ("target", "container")},
        marks={} if task in judge.ART else {
            k: dict(
                t=round(v["t"], 2),
                rise_cm=round(float(100 * (v["p"][2] - s0["p"][2])), 1),
                moved_cm=round(float(100 * np.linalg.norm((v["p"] - s0["p"])[:2])), 1),
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(v["up"] @ s0["up"], -1, 1)))), 1),
                held=v["held"],
                basket_cm=round(float(100 * np.linalg.norm((v["basket"] - s0["basket"])[:2])), 1) if "basket" in v else None,
            )
            for k, v in env.marks.items()
            if k != "start"
        },
    )  # fmt: skip
    rec.update({k: v for k, v in env.row.items() if k not in rec and k != "carried"})
    rec.update(
        comply_fields(task, official_success, comply_state, None, limit_s)
    )  # the server's watcher, asked once at the end
    if (
        task in judge.ART
    ):  # truth for the record: the joint, its advance, and the contacts with the moving parts (the server's ArtWatcher)
        rec["art"], rec["art_touch"] = comply_state["art"], comply_state["art_touch"]
    rec["loss"] = loss_of(rec)
    del mo
    free(env)
    rec["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return json.loads(json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def each(bed: Bed, ids: list[str], out, fn) -> None:
    """`fn(id)` for every id without a row in `out`, one JSON line per id appended and flushed (resumable); every row, errors
    included, carries the run's `bench` and `simple_config`. A lost server exits the process with code 3 (no row): the
    session restarts it and runs the ids without a row again."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["episode_id"] for line in open(out)} if out.exists() else set()
    with open(out, "a") as f:
        for eid in ids:
            if eid in done:
                continue
            try:
                rec = fn(eid)
            except (
                EOFError,
                ConnectionError,
                BrokenPipeError,
            ):  # the Isaac server is gone: stop, the session restarts it
                print("server lost at", eid, flush=True)
                sys.exit(3)
            except Exception as e:  # one broken episode must not stop the batch; it is counted, not hidden
                rec = dict(
                    episode_id=eid,
                    success=False,
                    loss="error",
                    error=f"{type(e).__name__}: {e}",
                    **comply_fields("", False, None),
                )
                gc.collect()
            rec.update(copy.deepcopy(bed.record))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(eid, {k: rec.get(k) for k in ("success", "t_success", "loss", "end", "wall_s", "error", "px", "flags") if k in rec}, flush=True)  # fmt: skip
