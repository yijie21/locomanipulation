"""Admit simulation bottle geometry through the two actual student camera streams.

Frustum/ray visibility is approximate, not RGB pose estimation. An admitted world
point may bridge a short occlusion; hidden truth never updates that cached point.
"""

import time
from copy import copy, deepcopy
from threading import Event, Lock, Thread

import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.sim import ports


class AssistedBottleTarget:
    streams = {"ego_view": "head_camera", "right_wrist": "right_wrist"}

    def __init__(self, ik, task, timeout, *, client=None):
        if client is None:
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

            client = ComposedCameraClientSensor(port=ports.camera())
        self.client, self.ik, self.task, self.timeout = client, ik, task, timeout
        self.point = self.seen = self.image_key = None
        self.diagnostics = {}
        m = ik.model
        self.body = m.body(task.target.body).id
        self.qadr = int(m.jnt_qposadr[m.body_jntadr[self.body]])
        half = np.asarray(task.target.box["half_extent"])
        centre = np.asarray(task.target.box.get("center", [0, 0, 0]))
        angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        self.samples = (
            np.array(
                [
                    [half[0] * np.cos(a), half[1] * np.sin(a), z]
                    for z in (-half[2] * 0.7, 0, half[2] * 0.7)
                    for a in angle
                ]
            )
            + centre
        )

    def observe(self, state, *, now=None):
        now = time.time() if now is None else now
        message = self.client.read()
        visible = {}
        if message is not None:
            timestamps = message["timestamps"]
            key = tuple(timestamps.get(name, 0) for name in self.streams)
            if key != self.image_key:
                self.image_key = key
                ik, obj = self.ik, state["objects"][self.task.target.key]
                m, d, mj = ik.model, ik.data, ik._mj
                d.qpos[:] = m.qpos0
                d.qpos[ik.qadr] = state["joints"]
                d.qpos[ik.base_qadr : ik.base_qadr + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
                d.qpos[self.qadr : self.qadr + 7] = [*obj["pos"], *obj["quat"]]
                for side, palm in state["palms"].items():
                    for i, value in enumerate(palm["hand_q"][:2], 1):
                        d.qpos[m.joint(f"{side}_hand_gripper_{i}_joint").qposadr[0]] = value
                mj.mj_fwdPosition(m, d)
                points = self.samples @ Primitives._rot(obj["quat"]).T + obj["pos"]
                ego_px = 0.0  # apparent height of the bottle in the ego view, the visible trigger for raising the arm
                wrist_roi = None
                for name, camera in self.streams.items():
                    stamp, rgb = timestamps.get(name, 0), message["images"].get(name)
                    if rgb is None or not 0 <= now - stamp <= self.timeout or np.std(rgb) < 1:
                        continue
                    cid = m.camera(camera).id
                    h, w = rgb.shape[:2]
                    local = (points - d.cam_xpos[cid]) @ d.cam_xmat[cid].reshape(3, 3)
                    depth = -local[:, 2]
                    focal = h / (2 * np.tan(np.radians(m.cam_fovy[cid]) / 2))
                    uv = local[:, :2] / np.maximum(depth[:, None], 1e-6) * [focal, -focal] + [w / 2, h / 2]
                    inside = (depth > 0.03) & np.all(uv > 2, axis=1) & np.all(uv < [w - 2, h - 2], axis=1)
                    hits = []
                    for index in np.flatnonzero(inside):
                        ray = points[index] - d.cam_xpos[cid]
                        ray /= np.linalg.norm(ray)
                        hit = np.array([-1], dtype=np.int32)
                        mj.mj_ray(
                            m,
                            d,
                            d.cam_xpos[cid],
                            ray,
                            np.array([1, 1, 1, 0, 0, 0], np.uint8),
                            True,
                            int(m.cam_bodyid[cid]),
                            hit,
                        )
                        if hit[0] >= 0 and m.geom_bodyid[hit[0]] == self.body:
                            hits.append(uv[index])
                    if len(hits) >= 3 and np.max(np.ptp(hits, axis=0)) >= 8:
                        visible[name] = float(stamp)
                        hit_uv = np.asarray(hits, dtype=float)
                        extent = np.ptp(hit_uv, axis=0) / np.array([w, h], dtype=float)
                        centroid = np.mean(hit_uv, axis=0) / np.array([w, h], dtype=float)
                        if name == "right_wrist":
                            # Only samples whose camera ray terminated on the
                            # target contribute. The image timestamp prevents
                            # one frame from satisfying stable acquisition.
                            camera_rotation = d.cam_xmat[cid].reshape(3, 3)
                            pelvis_rotation = Primitives._rot(state["pelvis"]["quat"])
                            # Moving the camera along +camera-x reduces +u;
                            # moving it along -camera-y reduces +v under the
                            # projection convention above. Express those
                            # directions in the pelvis frame used by ArmIK.
                            correction_basis = pelvis_rotation.T @ np.column_stack(
                                [camera_rotation[:, 0], -camera_rotation[:, 1]]
                            )
                            wrist_roi = dict(
                                timestamp=float(stamp),
                                centroid=np.clip(centroid, 0.0, 1.0),
                                extent=np.clip(extent, 0.0, 1.0),
                                occupancy=float(np.prod(np.clip(extent, 0.0, 1.0))),
                                quality=float(np.clip(len(hits) / len(self.samples), 0.0, 1.0)),
                                admitted_samples=len(hits),
                                image_size=[int(w), int(h)],
                                correction_basis_pelvis=correction_basis,
                            )
                    if name == "ego_view" and hits:
                        ego_px = float(np.ptp(np.array(hits)[:, 1]))
                if visible:
                    self.point, self.seen = self.task.grip(state).copy(), max(visible.values())
                self.diagnostics = dict(
                    image_timestamps=timestamps,
                    ego_height_px=ego_px,
                    wrist_roi=wrist_roi,
                    visible=visible,
                    state_seq=state["seq"],
                    source="simulation_with_camera_rays",
                )
        self.diagnostics["cached"] = not bool(visible)
        return self.point.copy() if self.seen is not None and 0 <= now - self.seen <= self.timeout else None

    def close(self):
        self.client.close()


class BottleTargetObserver:
    """Latest completed camera observation; image work never runs in the caller.

    ZMQ is constructed/used/closed by its owner thread. Camera FK has private
    MjData; it cannot overwrite the controller's IK scratch space. A delayed
    result retains its image timestamp and therefore cannot extend freshness.
    """

    def __init__(self, ik, task, timeout, *, observer_factory=None):
        self.timeout = timeout
        self._lock, self._stop = Lock(), Event()
        self._state = self._point = self._seen = None
        self._diagnostics = self.diagnostics = {}
        if observer_factory is None:
            camera_ik = copy(ik)
            camera_ik.data = ik._mj.MjData(ik.model)
            observer_factory = lambda: AssistedBottleTarget(camera_ik, task, timeout)
        self._factory = observer_factory
        self._thread = Thread(target=self._run, name="bottle-vision", daemon=True)
        self._thread.start()

    def _run(self):
        observer = None
        try:
            observer = self._factory()
            while not self._stop.wait(0.01):
                with self._lock:
                    state = self._state
                if state is None:
                    continue
                started = time.monotonic()
                point = observer.observe(state)
                diagnostics = dict(observer.diagnostics, worker_seconds=time.monotonic() - started)
                with self._lock:
                    self._point, self._seen = point, observer.seen
                    self._diagnostics = diagnostics
        except Exception as exc:
            with self._lock:
                self._point = None
                self._diagnostics = dict(error=f"camera observer: {type(exc).__name__}: {exc}")
        finally:
            if observer is not None:
                observer.close()

    def observe(self, state, *, now=None):
        now = time.time() if now is None else now
        snapshot = deepcopy(state)
        with self._lock:
            self._state = snapshot
            self.diagnostics = deepcopy(self._diagnostics)
            if self._point is not None and self._seen is not None and 0 <= now - self._seen <= self.timeout:
                return self._point.copy()
        return None

    def close(self):
        self._stop.set()
        # Cleanup must not wait indefinitely for a stalled camera SDK call.
        # The daemon owns eventual socket teardown when that call returns.
        self._thread.join(timeout=0.5)
