"""The one Isaac renderer both drivers call: a MuJoCo state in, the dataset's
camera images out.

Collection re-renders offline and evaluation renders live (DECISIONS
2026-09-23), and the policy is trained on one and run on the other -- so the
two must be the SAME function of the state, not two loops that happen to
agree. Both call :class:`IsaacView`:

    offline : a state trace's rows -> the dataset's video columns   (TODO 3.5.1)
    online  : the newest sim state, when the policy asks for it     (TODO 3.4.7)

Everything Isaac needs comes out of one ``mj_forward`` on the given qpos
(:mod:`replay`): the joints, the articulation root's pose, the object poses and
each camera's world pose. The cameras are the ones MuJoCo publishes
(``dataset_cameras.CAMERA_CONFIGS``: same MJCF camera, same resolution), mounted as
top-level prims and re-posed every frame, which is why a frame costs
``stage.SETTLE_STEPS`` renders. With ``cfg.supersample`` > 1 they render that
many times larger and ``grab`` box-averages back to the dataset resolution.

The caller owns Kit: start it with ``stage.start_app(cfg)`` before building a
view, and on failure ``os._exit(1)`` BEFORE ``app.close()`` -- close() ends the
process itself with status 0. ``run_in_kit`` does both for a run's Isaac look.

``film_setup`` turns a view that has done its dataset work into the renderer of
the run's third-person video (``view_render.film``): one more world camera at
the video's size, the ghost grippers (``ghost.add_isaac``), the head mesh shown
again (it is hidden only for the ego camera that sits inside it), and the
dataset cameras' render products stopped, since the video's insets are the
dataset's own filled frames. After it the view films and no longer renders
dataset images.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from locomani.sim.dataset_cameras import CAMERA_CONFIGS
from locomani.sim.isaac import frames as F
from locomani.sim.isaac import stage as S


def camera_specs(scene_xml: str, streams) -> dict:
    """`{mjcf camera: (pos, quat, fovy, width, height)}` for the dataset streams,
    in the form ``stage.mount_inset_cameras`` takes."""
    specs = {}
    for key in streams:
        if key not in CAMERA_CONFIGS:
            raise SystemExit(f"unknown camera stream {key!r}; the dataset streams are {sorted(CAMERA_CONFIGS)}")
        c = CAMERA_CONFIGS[key]
        pos, quat, fovy = F.camera_spec(scene_xml, c["mjcf_name"])
        specs[c["mjcf_name"]] = (pos, quat, fovy, c["width"], c["height"])
    return specs


class IsaacView:
    """An Isaac stage mirroring one MuJoCo scene, read through the dataset's cameras."""

    def __init__(self, cfg: S.StageCfg, scene_xml: str, streams=("ego_view",), rtx_settings=(), log=print):
        from locomani.sim.isaac.isaac_render import object_keys
        from locomani.sim.isaac.replay import MujocoReplay

        self.streams = tuple(streams)
        self.log = log
        self.cfg = cfg
        self.supersample = cfg.supersample
        self.stage = S.assemble_stage(cfg, log)
        S.apply_rtx_settings(rtx_settings, log)
        self.world = S.make_world()
        joint_map = F.build_joint_map()
        self.robot, _jidx, _jsrc = S.bind_robot(joint_map, log)
        mounted = S.mount_inset_cameras(self.stage, cfg, camera_specs(scene_xml, self.streams), log)
        # stream key -> (mjcf camera, sensor)
        self.cameras = {
            k: (CAMERA_CONFIGS[k]["mjcf_name"], mounted[CAMERA_CONFIGS[k]["mjcf_name"]][0]) for k in self.streams
        }
        self.replay = MujocoReplay(scene_xml, object_keys(scene_xml))
        dof_names = list(self.robot.dof_names)
        self._dofs = [n for n, _ in joint_map if n in dof_names]
        self._dof_idx = [dof_names.index(n) for n in self._dofs]
        self._root = S.root_body(self.robot)
        log(f"view: streams {list(self.streams)}; {len(self._dof_idx)} joints bound; root link {self._root}")

    def push(self, qpos, qvel=None) -> None:
        """One MuJoCo state into the stage: bodies, objects and every camera."""
        r = self.replay
        r.set(qpos, qvel)
        S.sync_replay(
            self.stage,
            self.robot,
            self._dof_idx,
            r.dof_positions(self._dofs),
            r.body_pose(self._root),
            r.object_poses(),
        )
        for name, sensor in self.cameras.values():
            pos, quat = r.camera_pose(name)
            sensor.set_world_pose(position=pos, orientation=quat, camera_axes="usd")

    def grab(self) -> dict | None:
        """`{stream: HxWx3 uint8}` from one settled render, or None if any stream
        came back missing or stale -- a previous step's image never fills in."""
        got = S.step_and_grab_all(self.world, {k: sensor for k, (_n, sensor) in self.cameras.items()})
        missing = [k for k, v in got.items() if v is None]
        if missing:
            self.log(f"view: missing/stale streams {missing}")
            return None
        return {k: np.ascontiguousarray(S.downsample(v[..., :3], self.supersample)) for k, v in got.items()}

    def render(self, qpos, qvel=None) -> dict:
        """The images of this state. Raises rather than return a partial set."""
        self.push(qpos, qvel)
        images = self.grab()
        if images is None:
            raise RuntimeError("Isaac returned missing or stale camera streams for this state")
        return images

    def warm_up(self, qpos, qvel=None) -> int:
        """Hold one state until every stream returns pixels and the picture stops
        changing; returns the steps spent.

        The first get_rgba() calls after a stage is built come back empty, and a
        non-empty sensor is not a finished picture yet: RTX streams the robot's
        meshes in over the first ~10 rendered frames (isaac_render pilot 21 r00:
        0.8 s of video with only two finger tips visible).
        """
        return self._settle(lambda: self.push(qpos, qvel), self.grab, "picture")

    def _settle(self, push, grab, what: str) -> int:
        """`push` then `grab` until `grab` returns pixels and they stop changing; returns the steps spent."""
        steps = 0
        for _ in range(8):
            steps += 1
            push()
            if grab() is not None:
                break
        else:
            raise RuntimeError(f"Isaac {what} still missing after 8 warm-up steps")
        prev, calm = None, 0
        for _ in range(80):
            steps += 1
            push()
            cur = grab()
            if cur is None:
                prev, calm = None, 0
                continue
            cur = {k: v.astype(np.float32) for k, v in cur.items()}
            quiet = prev is not None and all(float(np.abs(cur[k] - prev[k]).mean()) < 0.5 for k in cur)
            calm = calm + 1 if quiet else 0
            if calm >= 3:
                self.log(f"view: {what} settled after {steps} warm-up steps")
                return steps
            prev = cur
        self.log(f"view: WARNING {what} still changing after 80 extra warm-up steps")
        return steps

    def film_setup(self, width: int, height: int, fovy: float) -> None:
        """Mount the third-person camera and the ghost grippers; stop rendering the dataset cameras."""
        from pxr import UsdGeom

        from locomani.sim.isaac import ghost as G

        spec = {"third_person": (np.zeros(3), np.array([1.0, 0, 0, 0]), fovy, width, height)}
        self._film = S.mount_inset_cameras(self.stage, self.cfg, spec, self.log)["third_person"][0]
        self._ghosts = G.add_isaac(self.stage, self.replay.model, self.log)
        # Undo exactly assemble_stage's MakeInvisible. Not MakeVisible(): the robot layer's /visuals scope is itself
        # invisible (its prototypes sit at the origin), so MakeVisible would show the scope and author `invisible` on
        # every other link's visuals -- the instances reference them, and the robot vanished but for its head.
        for rel in self.cfg.hidden_visuals:
            UsdGeom.Imageable(self.stage.GetPrimAtPath(f"/visuals/{rel}")).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited
            )
        stopped = 0
        for _name, sensor in self.cameras.values():
            try:  # a speed-up only: Replicator's own switch, on a private handle of the sensor
                sensor._render_product.hydra_texture.set_updates_enabled(False)
                stopped += 1
            except AttributeError as exc:
                self.log(f"view: could not stop a dataset camera ({exc}); it renders along")
        self.log(f"view: filming {width}x{height} fovy {fovy:g}; {stopped} dataset cameras stopped")

    def _film_push(self, qpos, camera, ghosts) -> None:
        from locomani.sim.isaac import ghost as G

        self.push(qpos)
        self._film.set_world_pose(position=camera[0], orientation=camera[1], camera_axes="usd")
        G.set_isaac(self._ghosts, *ghosts)

    def _film_grab(self) -> dict | None:
        got = S.step_and_grab_all(self.world, {"third_person": self._film})["third_person"]
        return None if got is None else {"third_person": S.downsample(got[..., :3], self.supersample)}

    def film_warm_up(self, qpos, camera, ghosts) -> int:
        """Hold the first filmed state until the third-person picture settles (a new shot streams in too)."""
        return self._settle(lambda: self._film_push(qpos, camera, ghosts), self._film_grab, "third-person picture")

    def film(self, qpos, camera, ghosts) -> np.ndarray:
        """The third-person picture of this state: `camera` (pos, wxyz quat, -Z forward), `ghosts` (pos [2,3],
        quat [2,4]) as `ghost_trace.ghost_poses` gives them."""
        self._film_push(qpos, camera, ghosts)
        got = self._film_grab()
        if got is None:
            raise RuntimeError("Isaac returned a missing or stale third-person picture for this state")
        return np.ascontiguousarray(got["third_person"])


def run_in_kit(scene_xml: str, isaac: dict, gpu, work, log=print) -> None:
    """Start Kit on card `gpu`, build the view of the run's Isaac look (`run_context.isaac_scene`) over the dataset
    streams and call `work(view)`. A failure prints its traceback and exits 1 before Kit's close, which would exit
    0; after a success Kit's close ends the process, so nothing after this call runs."""
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA="YES")  # Kit takes the card at start
    cfg = S.StageCfg(
        scene_dir=isaac["dir"], robot_mjcf=scene_xml, absolute_camera=True, supersample=isaac["supersample"]
    )
    app = S.start_app(cfg)
    try:
        work(IsaacView(cfg, scene_xml, tuple(CAMERA_CONFIGS), isaac["rtx_settings"], log=log))
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # NOT app.close() first: it ends the process with status 0
    app.close()
