"""One LocoManiLab episode (or one look) in the benchmark's own environment, and a batch of them (TODO 49.4).

Moved from escalation_20261116/bed_lml_cam.py (setup, look, run, flow, worker) and bed_lml.py (loop_fields, error_row,
spec_of): the camera ("official") track. Runs under the sim venv (mujoco, torch, the ScaleBFM runtime) as
`python -m locomani.bench.locomanilab batch|look|cachedir`; the environment server (server.py, LocoManiLab's venv) is started by
this process (client.ServerLink) and dies with it; SAM 3 is expected to be up. The step count, the success and the step limit
are the environment's own.

The robot's decisions use the instruction (the reset reply's `prompt`), the head camera's colour + depth image (320x240),
joint readings, IMU tilt and leg odometry; nothing is read from the simulator's scene by the skill package (camera_view.py).
The simulator's state is read only where a comment says "record" (the success check, the comparisons with the truth track) and
to put the robot where the episode starts.

The skill package (--skills: a folder with __init__.py, skill.py, pick.py, camera_view.py, flows.py) is loaded by path under a
unique module name (`load_package`); nothing is put on sys.path. It gets `client.PackageEnv`, not the sim. Its parameters are the
whole setting space (`pkg.apply_set`); the bed's own switches (feed_torque, seat_feed: the estimator, the loop's `bed_fixed`)
are not the package's. Every camera-only setting the old bed popped (relook, palm_bias, stop_lead, stop_fix, truth_stop,
over_wait, ref_tol, centre_pct, centre_bias) was held at its neutral value by the loop's package mode and cannot be given
(`apply_set` rejects unknown names): they are gone, with the monkey patches that applied them.
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
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import MotorConfig
from locomani.bench.locomanilab import client as C
from locomani.bench.locomanilab import comply as CL
from locomani.bench.locomanilab import episodes as EP
from locomani.bench.locomanilab import families, loss
from locomani.bench.locomanilab import record as RL
from locomani.bench.locomanilab import truth as SL
from locomani.bench.locomanilab.body import RecBelief
from locomani.bench.locomanilab.config import (  # noqa: F401
    DEFAULT_CONTROL,
    DEFAULT_VIEW,
    SCOPE,
    SEED_DIR,
    LocoManiLabConfig,
    ScopeError,
    check_scope,
)
from locomani.bench.simple import body as BODY
from locomani.bench.simple import feed as FL

DT = C.DT
PACKAGE_FILES = ("__init__.py", "skill.py", "pick.py", "camera_view.py", "flows.py")
BED_SWITCHES = ("feed_torque", "seat_feed")  # settings of the bed (the estimator), not parameters of the package
FLOW_NAMES = dict(navigation="approach", pick="pick_only", place="pick_place", articulate="articulate")
_COUNTER = itertools.count()
_SPECS: dict[str, dict] = {}


# ---- the skill package, loaded by path


def load_package(path) -> ModuleType:
    """Load the skill package in folder `path` under a unique module name: spec_from_file_location with the folder as the
    package's search location, so its relative imports (`from . import skill as S`) find its own files and two copies of the
    package (the loop's drafts) load side by side without clashing. Nothing is put on sys.path."""
    path = Path(path).resolve()
    missing = [f for f in PACKAGE_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: skill package files missing: {missing}")
    name = f"_lml_skill_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        for sub in ("skill", "pick", "camera_view", "flows"):
            importlib.import_module(f"{name}.{sub}")
    except BaseException:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        raise
    mod.path = path
    return mod


def spec_of(cfg: LocoManiLabConfig, task: str) -> dict:
    """What the task's success check reads (specs.json, written from the server's reset replies): the body to move, where to, the
    thresholds, the judge family."""
    key = cfg.specs_file
    if key not in _SPECS:
        _SPECS[key] = json.loads(Path(key).read_text())
    return _SPECS[key][task]


# ---- what a run holds


@dataclasses.dataclass
class Bed:
    """One process's run context: the configs, the skill package, the environment server link, the policy."""

    cfg: LocoManiLabConfig
    pkg: ModuleType
    record: dict = dataclasses.field(default_factory=dict)  # the `bench` and `lml_config` fields of every row
    options: object = None  # the run's resolved BenchOptions (None: the loops' defaults)
    _server: object = None
    _pol: object = None

    @property
    def S(self):
        return self.pkg.skill

    def switch(self, axis: str) -> str:
        """A run switch's value: the resolved options', else the loops' default."""
        o = self.options
        return (
            getattr(o, axis)
            if o is not None
            else dict(
                zip(("target", "sensor", "depth_noise"), DEFAULT_VIEW), self_pose="odometry", controller_root="odometry"
            )[axis]
        )

    @property
    def depth_source(self) -> str:
        return "rgbd" if self.switch("depth_noise") == "none" else f"rgbd:{self.switch('depth_noise')}"

    def server(self) -> C.ServerLink:
        if self._server is None:
            self._server = C.ServerLink(self.cfg, Path(self.cfg.run_dir) / "srv")
        return self._server

    def policy(self):
        if self._pol is None:
            # the runtime's jaw offset is a module array (no argument for it): set once to the Dex3 palm, as the old bed did
            R.OFF[:] = BODY.PALM
            self._pol = R.Policy("cuda", ScaleBfmBackendConfig())
        return self._pol

    def cache_dir(self) -> Path:
        """The look + GraspGen-X cache of this package and these switches: <cache_dir>/<key>."""
        return Path(self.cfg.cache_dir) / cache_key(self.pkg, self.switches(), self.control())

    def control(self) -> tuple:
        return tuple(self.switch(a) for a in ("self_pose", "controller_root"))

    def switches(self) -> tuple:
        return tuple(self.switch(a) for a in ("target", "sensor", "depth_noise"))


def cache_key(pkg: ModuleType, switches: tuple = DEFAULT_VIEW, control: tuple = DEFAULT_CONTROL) -> str:
    """First 10 hex of sha1 over the package's files (the look uses its view and the flows' `dest_of`; the loop keyed its cache
    by the whole package too). The look runs with the package's own default parameters and neutral bed switches, whatever the
    round's setting says (as the old bed's look pass did: `setup(id, NEUTRAL)`), so nothing of the setting is in the key. Not in
    the key: the fixed perception code, SAM 3's weights, the benchmark's scene: as before. `switches` = (target, sensor,
    depth_noise) and `control` = (self_pose, controller_root) join the hash only when they differ from the loops' defaults, so a
    look made from the exact depth is never read by a noisy-depth run, and with the default switches the key does not move when
    a switch is added."""
    h = hashlib.sha1()
    for f in PACKAGE_FILES:
        h.update((Path(pkg.path) / f).read_bytes())
    if tuple(control) != DEFAULT_CONTROL:
        h.update(json.dumps(dict(self_pose=control[0], controller_root=control[1])).encode())
    if tuple(switches) != DEFAULT_VIEW:
        h.update(json.dumps(dict(target=switches[0], sensor=switches[1], depth_noise=switches[2])).encode())
    return h.hexdigest()[:10]


def scene_dir(cfg: LocoManiLabConfig, task: str) -> Path:
    """The built scene of a task, under this call's run directory (one built scene per task: the benchmark's rooms are fixed)."""
    return Path(cfg.run_dir) / "scene" / task


# ---- one episode


def setup(bed: Bed, episode_id: str, setting: dict, stance: dict | None, cloud_dir: Path):
    """The episode's start in the benchmark's environment, the robot standing where it put it, odometry on, the camera's belief
    frame set, the clock running; -> (env, first reply of the environment, K)."""
    cfg, pkg = bed.cfg, bed.pkg
    task = EP.parse_id(episode_id)[0]
    truth = bed.switch("target") == "truth"
    msg = EP.reset_msg(
        cfg, episode_id, not truth
    )  # refuses the benchmark's own episodes, foreign pools and bad rows; no cameras under target=truth (as the old truth track)
    given = dict(setting or {})
    sw = {k: given.pop(k) for k in BED_SWITCHES if k in given}
    K = pkg.apply_set(
        given, task
    )  # the package's parameters with the setting on them; an unknown name raises KeyError (2026-11-23)
    link = bed.server()
    first = link.ask(*msg)
    scene = C.scene_of(first, scene_dir(cfg, task))
    pol = bed.policy()
    sim = C.LmlSim(pol.meta, link.link(), first, scene)
    seed = int(hashlib.sha1(episode_id.encode()).hexdigest()[:8], 16)  # a rerun of an id sees the same depth noise
    if not truth:
        sim.use_rgbd(bed.switch("depth_noise"), seed)
    mo = C.BedMotor(sim, pol, dataclasses.replace(MotorConfig(), **{k: K[k] for k in bed.S.MOTOR_KEYS}), model_xml=str(scene / "scene_locomani.xml"))  # fmt: skip
    xy, yaw = sim.start()  # the robot is put where the environment put it
    mo.reset(xy, yaw)
    sim.pull()
    mo._stand()
    mo.stand_pose = mo.ref(0.0)
    bel = RecBelief(sim, mo, bed.switch("self_pose") == "odometry", bed.switch("controller_root") == "odometry")
    bel.walk_hold = K["walk_hold"]
    if sw.get("seat_feed") or sw.get(
        "feed_torque"
    ):  # the motors' own torques (the PD command) fed to leg_odo's load weights
        FL.patch_torques(sim)
    if sw.get("seat_feed") and bel.odo:  # the hand estimate's feed re-seats a foot's anchor while it is off the floor
        bel.feed = FL.SlipFeed(bel.body)
    if K.get("odo_stop") and bel.odo:  # Belief set the stop servo off for odometry: back on, steering on odometry
        bel.odo_stop = True
        mo.cfg = dataclasses.replace(mo.cfg, stop_ki=MotorConfig().stop_ki, stop_timeout=K["stop_timeout"])
    spec = spec_of(cfg, task)  # record: the truth track's facts, for comparison; the flow is chosen from the prompt
    fam = spec["family"]
    if truth and fam in CL.ART:
        raise NotImplementedError(f"target=truth has no flow for the articulated tasks ({task}): see lml_port_notes.md")
    if fam == "articulation_pnp":  # the drawer pick-and-place: the target is the apple
        info = dict(target=spec["object_body"], objects={})
    elif fam in CL.ART:  # a door or drawer task: the target is the moving part's body
        info = dict(target=spec["door_body"], objects={})
    else:
        info = dict(target=C.root_of(sim.m, spec["target"]["body"] if fam == "navigation" else spec["object_body"]), objects={})  # fmt: skip
    env = C.Env(task, info, sim, mo, bel, K, stance, spec=spec, sam3=cfg.sam3(), cloud_dir=cloud_dir, skill=bed.S)
    env.bodies = SL.subtree(sim.m, sim.m.body(info["target"]).id)
    env.clock = C.Clock(env)
    bel.env = env
    env.prompt = first["prompt"]
    env.want = pkg.camera_view.parse(first["prompt"])
    if truth:  # the view from the simulator (the old bed's truth track); the package's camera view is not built
        env.view = SL.LmlTruthView(env, Path(cfg.truth_grasp_dir) / f"grasp_{task}_LX_e0.npz", bed.S)
    else:
        env.view = pkg.camera_view.LmlCameraView(env.facade(), pkg.camera_view.cloud_name(episode_id), first["prompt"])
        env.rec = bel.recorder = RL.Recorder(env)
    if fam in CL.ART:  # the per-step record of the door or drawer: record only, the package never sees it
        env.art = RL.ArtRecorder(env)
    env.marked = lambda name: on_mark(bed, env, name)
    mo.hold(K["start_hold"])
    bel.start()
    env.mark("start")
    return env, first, K


def on_mark(bed: Bed, env, name: str) -> None:
    """The records made at each mark (record only): the recorder's, the palm against the object's middle, the object against the
    receptacle."""
    if env.rec is not None:
        env.rec.on_mark(name)
    if name == "before_close":  # the true palm against the object's true middle, in grasp_desc's form
        try:
            ch = getattr(env.view, "chosen", None)
            yaw = float(ch["yaw"]) if ch else float(env.view.heading())
            O = SL.body_points(env.sim, env.bodies)
            p, Rm = env.mo.jaw_centre(BODY.SIDE)
            Tw = np.eye(4)
            Tw[:3, :3], Tw[:3, 3] = Rm, p
            env.row["close_desc"] = bed.S.grasp_desc(Tw, 0.5 * (O.min(0) + O.max(0)), yaw).round(3).tolist()
        except Exception as e:  # noqa: BLE001
            env.row["close_desc_error"] = str(e)[:60]
    if name in ("before_open", "end") and env.spec["family"] in (
        "pick_place",
        "receptacle",
        "composite",
    ):  # the object against the receptacle
        try:
            O = SL.body_points(env.sim, env.bodies)
            rc = SL.receptacle(env)
            c = 0.5 * (O.min(0) + O.max(0))
            env.row[f"obj_dest_{name}"] = dict(dxy_cm=(100 * (c[:2] - rc[0])).round(1).tolist(), dz_cm=round(float(100 * (O[:, 2].min() - rc[1])), 1))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            env.row[f"obj_dest_{name}_error"] = str(e)[:60]


def truth_facts(bed: Bed, env) -> dict:
    """Record only: the truth track's object and grasp at the start, for the comparisons."""
    out = {}
    if env.spec["family"] in CL.ART:  # no object to grasp: the door's facts are in rec["art"]
        return out
    try:
        tv = SL.LmlTruthView(env, Path(bed.cfg.truth_grasp_dir) / f"grasp_{env.task}_LX_e0.npz", bed.S)
        tgt = tv.target()
        out["true_centre"] = (0.5 * (tgt["lo"] + tgt["hi"])).round(3).tolist()
        out["true_top"] = round(float(tgt["hi"][2]), 3)
        out["true_foot"] = round(float(tgt["top"]), 3)
        g = tv.grasp(tgt, tv.heading())
        out["true_grasp"] = None if g is None else (tgt["T_wo"] @ g[0])[:3, 3].round(3).tolist()
    except Exception as e:  # counted, not hidden
        out["truth_error"] = f"{type(e).__name__}: {e}"
    return out


def flow(bed: Bed, env):
    """The skill for what the instruction asks (looked up by name: a package without `articulate` fails only on a door task)."""
    return getattr(bed.pkg.flows, FLOW_NAMES[env.want["kind"]])


def truth_flow(bed: Bed, env):
    """target=truth: the old bed's truth track (`bed_lml.FLOWS`): the navigation flow reading the room's geometry, the package's pick on
    the truth view, the pick-and-place with the truth receptacle. Returns a function of the package facade."""
    fam, S = env.spec["family"], bed.S
    if fam == "navigation":
        return lambda facade: SL.approach(env)
    if fam in ("lift", "pick"):
        return S.episode
    if fam in ("pick_place", "receptacle", "composite"):
        return lambda facade: SL.pick_place(env, S, facade)
    raise NotImplementedError(f"no truth flow for task family {fam}")


def art_found(env):
    """Whether the package's look for the handle found it: the flow's own record `art_look`, None when it made none."""
    a = env.row.get("art_look")
    return None if a is None else bool(a.get("found"))


def free(env) -> None:
    C.free(env)


def look(bed: Bed, episode_id: str) -> dict:
    """The look pass: the target's seen points to <cache>/<name>.npz for GraspGen-X."""
    cloud = bed.cache_dir()
    env, first, K = setup(
        bed, episode_id, {}, None, cloud
    )  # the package's defaults and neutral bed switches, as the old look pass
    view, rec = env.view, dict(episode_id=episode_id, prompt=first["prompt"], parse=env.want)
    if env.want["kind"] == "navigation" or (env.want["kind"] == "articulate" and not env.want.get("obj")):
        rec["found"] = (
            None  # a door or a drawer alone: nothing to look at before the round; the pnp's apple is looked at
        )
        free(env)
        return rec
    s = view.look()
    rec.update(found=bool(s["mask"] is not None), turns=env.row.get("turns"), cam_err=env.row.get("cam_err"))
    tf = truth_facts(bed, env)
    if s["mask"] is not None:
        cloud.mkdir(parents=True, exist_ok=True)
        full = bed.pkg.camera_view.complete(s["cloud"], s["cam_b"])
        np.savez_compressed(cloud / f"{view.name}.npz", pts=full, pts_seen=s["cloud"], p_o=np.median(s["cloud"], axis=0), R_o=np.eye(3),
                            rgb=s["rgb"], mask=s["mask"], score=s["score"])  # fmt: skip
        rec.update(px=int(len(s["cloud"])), score=round(s["score"], 2), phrase=s["phrase"])
        c = np.median(s["cloud"], axis=0)
        # record: the cloud against the object's true points (the world and the robot's frame agree at the start)
        if "true_centre" in tf and view.yaw0 is not None:
            rec["centre_err_cm"] = round(
                float(100 * np.linalg.norm(env.bel.to_sim(c)[:2] - np.array(tf["true_centre"])[:2])), 1
            )
            rec["top_err_cm"] = round(float(100 * (np.percentile(s["cloud"][:, 2], 99) - tf["true_top"])), 1)
    if env.want["kind"] == "place":
        rec.update({k: env.row.get(k) for k in ("dest_in_view", "dest_score", "dest_phrase", "dest_span_cm")})
        if view.dest is not None:
            try:  # record: the receptacle seen against the truth track's
                rc = SL.receptacle(env)
                tc = env.bel.from_sim(np.r_[rc[0], 0.0])[:2]
                rec["dest_err_cm"] = round(float(100 * np.linalg.norm(view.dest[0] - tc)), 1)
                rec["dest_rim_err_cm"] = round(float(100 * (view.dest[1] - rc[1])), 1)
                rec["dest_seen"] = np.round(view.dest[0], 3).tolist()
                rec["dest_true"] = np.round(tc, 3).tolist()
            except Exception as e:  # noqa: BLE001
                rec["dest_record_error"] = str(e)[:80]
    rec.update(tf)
    free(env)
    return rec


def loop_fields(rec: dict, env, first: dict, steps: int) -> dict:
    """The fields the automatic loop reads on every record: `comply` = {ok, why} and `comply_facts` (the Watcher's state without
    its timeline), `genuine` (the benchmark's own flag: a success of at least the reset reply's `genuine_min_steps` steps),
    `t_end` (steps x 0.02 s), `t_check` (seconds of the first step the check held), `instruction`, and `loss`
    (loss.loss_of). The family comes from the task's spec."""
    clock, ok = env.clock, bool(rec["success"])
    state = clock.watcher.state()
    art = env.art.result() if getattr(env, "art", None) is not None else None  # the drawer's own rule reads moved_by
    v = CL.verdict(env.spec["family"], ok, state, genuine=steps >= int(first["genuine_min_steps"]),
                   moved_by=None if art is None else art["moved_by"], hand_drives=bool(env.spec.get("hand_drives")))  # fmt: skip
    facts = {k: state[k] for k in ("t_check", "hand_at_check", "fell_at_check", "t_carry", "best_run_s", "at_check")}
    rec.update(comply=dict(ok=v["ok"], why=v["why"]), comply_facts=facts, t_check=None if state["t_check"] is None else round(state["t_check"], 2),
               genuine=bool(ok and steps >= int(first["genuine_min_steps"])), genuine_min_steps=int(first["genuine_min_steps"]),
               t_end=round(steps * DT, 2), instruction=first.get("prompt"), limit_s=round(first["max_steps"] * DT, 2))  # fmt: skip
    if clock.watch_error:
        rec["comply_error"] = clock.watch_error
    rec["loss"] = loss.loss_of(rec)
    return rec


def error_row(episode_id: str, e: Exception, **more) -> dict:
    """The row of an episode that raised: counted as a loss "error", never dropped."""
    rec = dict(episode_id=episode_id, success=False, end="error", error=f"{type(e).__name__}: {e}"[:300],
               comply=dict(ok=False, why="not_success"), comply_facts={}, genuine=False, loss="error")  # fmt: skip
    rec.update(more)
    return rec


def run(bed: Bed, episode_id: str, setting: dict | None = None, stance: dict | None = None) -> dict:
    """One episode -> its record (the row)."""
    t_wall = time.time()
    given = setting or {}
    env, first, K = setup(bed, episode_id, given, stance, bed.cache_dir())
    sim, mo, bel, spec = env.sim, env.mo, env.bel, env.spec
    truth = bed.switch("target") == "truth"
    tf = {} if truth else truth_facts(bed, env)  # record
    view = env.view

    def on_grasp(tgt, g):  # record: where the chosen grasp lies against the truth track's
        if g is not None and tf.get("true_grasp"):
            p = bel.to_sim((tgt["T_wo"] @ g[0])[:3, 3])
            env.row["grasp_vs_truth_cm"] = round(float(100 * np.linalg.norm(p - np.array(tf["true_grasp"]))), 1)

    if not truth:
        env.view = C.GraspRecorded(view, on_grasp)
    end = "script"
    try:
        (truth_flow(bed, env) if truth else flow(bed, env))(env.facade())
        while True:  # the script is through: the environment's check runs on to its verdict or the step limit
            mo.hold(0.1)
    except C.Done as e:
        end = str(e)
    except C.EpisodeOver:
        end = "success" if sim.success else "limit"
    env.mark("end")
    s0, se = env.marks.get("start", env.marks["end"]), env.marks["end"]
    rec = dict(
        episode_id=episode_id, track="official", setting=given, stance=stance, controller="frozen", target=env.info["target"],
        family=spec["family"], skill=spec["skill"], prompt=first["prompt"], parse=env.want,
        success=bool(sim.success), t_success=sim.steps * DT if sim.success else None, limit_s=first["max_steps"] * DT,
        end=end, steps=sim.steps, fell=mo.fell_at, phases={k: round(v * DT, 2) for k, v in env.clock.phases.items()},
        rise_cm=float(100 * (se["p"][2] - s0["p"][2])), sha=first["sha"], jitter=first.get("jitter"),
        wall_s=round(time.time() - t_wall, 1),
        found=True if truth else art_found(env) if env.want["kind"] == "articulate" else bool(view.s is not None and view.s.get("mask") is not None) if env.want["kind"] != "navigation" else env.row.get("nav", {}).get("seen") is not None,
        marks={k: dict(t=round(v["t"], 2), rise_cm=round(float(100 * (v["p"][2] - s0["p"][2])), 1),
                       moved_cm=round(float(100 * np.linalg.norm((v["p"] - s0["p"])[:2])), 1),
                       tilt_deg=round(float(np.degrees(np.arccos(np.clip(v["up"] @ s0["up"], -1, 1)))), 1), held=v["held"])
               for k, v in env.marks.items() if k != "start"},
    )  # fmt: skip
    rec.update({k: v for k, v in env.row.items() if k not in rec})
    rec.update({k: v for k, v in tf.items() if k not in rec})
    if getattr(env, "art", None) is not None:
        rec["art"] = env.art.result()
    loop_fields(rec, env, first, sim.steps)
    free(env)
    return json.loads(json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def each(bed: Bed, ids: list[str], out, fn) -> None:
    """`fn(id)` for every id without a row in `out`, one JSON line per id appended and flushed (resumable); every row, errors
    included, carries the run's `bench` and `lml_config`. A dead server is dropped: the next id gets a new one. The
    environment server is told to quit at the end."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["episode_id"] for line in open(out)} if out.exists() else set()
    with open(out, "a") as f:
        for eid in ids:
            if eid in done:
                continue
            try:
                rec = fn(eid)
            except Exception as e:  # counted, not hidden
                rec = error_row(eid, e, tb=traceback.format_exc()[-600:])
                if bed.server().dead():
                    bed.server().reset_link()
                gc.collect()
            rec.update(copy.deepcopy(bed.record))
            f.write(json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
            f.flush()
            print(eid, {k: rec.get(k) for k in ("success", "t_success", "loss", "end", "wall_s", "error", "px") if k in rec}, flush=True)  # fmt: skip
    if bed.server().conn is not None:
        bed.server().ask("quit")
