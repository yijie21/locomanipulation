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
``stage.SETTLE_STEPS`` renders.

The caller owns Kit: start it with ``stage.start_app(cfg)`` before building a
view, and on failure ``os._exit(1)`` BEFORE ``app.close()`` -- close() ends the
process itself with status 0.
"""

from __future__ import annotations

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
        return {k: np.ascontiguousarray(v[..., :3]) for k, v in got.items()}

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
        steps = 0
        for _ in range(8):
            steps += 1
            self.push(qpos, qvel)
            if self.grab() is not None:
                break
        else:
            raise RuntimeError("Isaac camera streams still missing after 8 warm-up steps")
        prev, calm = None, 0
        for _ in range(80):
            steps += 1
            self.push(qpos, qvel)
            cur = self.grab()
            if cur is None:
                prev, calm = None, 0
                continue
            cur = {k: v.astype(np.float32) for k, v in cur.items()}
            quiet = prev is not None and all(float(np.abs(cur[k] - prev[k]).mean()) < 0.5 for k in cur)
            calm = calm + 1 if quiet else 0
            if calm >= 3:
                self.log(f"view: picture settled after {steps} warm-up steps")
                return steps
            prev = cur
        self.log("view: WARNING picture still changing after 80 extra warm-up steps")
        return steps
