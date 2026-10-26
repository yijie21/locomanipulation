"""Level L2 of the vision RSI design (docs/handoff/vision-rsi-design-20261023.md section 6): the free objects' poses from
the robot's own cameras. A frame source renders, per camera, a depth image (metres along the view axis) and a per-pixel
body id; the pixels of one object, back-projected, are fitted with its collision shape (ICP from its last estimate,
or from eight headings the first time it is seen). Geometry and masks are exact (what the depth buffer and a stencil
give): what L2 measures is seeing one side, from where the robot stands, with the hand in the way -- not a model.

`RenderedView` answers as TruthView except a free object's pose: refreshed from the cameras when its estimate is older
than `period_s` (simulated time) at a read, kept when too few of its pixels are visible, and -- never seen yet -- where
the map last had it (`prior_bias_m` from its true start position, one horizontal direction per object and round; its
orientation the true start orientation, which also breaks ties between headings the visible side cannot tell apart). The
simulator stands still while the cameras render and the fit runs (design D5: lockstep). `stats()` is what the round
record keeps: per object, reads, refreshes, sightings, the last fit's residual, the estimate's error against the truth
at the end, and every read (time, whether it came from a fresh sighting, the kept estimate or the prior, the visible
pixels, and its position and orientation error against the truth at that moment). A fit whose residual is over
`reject_rms_m`, or whose centre lies farther from the seen points than the shape reaches, is rejected and the last
estimate stays.

`MuJoCoFrames` renders with MuJoCo's own renderer (EGL). With `masks="sam3"` (level L3) the masks come from SAM 3 on
Unreal's RGB of the same moment instead (`ServiceFrames`; locomani.tasks.scene.services): the object is prompted by its
category phrase (services.prompt_of), and in each camera the instance taken is the one at least `sam3_min_score` whose
points land nearest the current estimate, within `assoc_gate_m` (a second TV remote on the same table is not it). Depth
stays MuJoCo's, registered to Unreal's pixels (same camera, 640 x 480; checked 2026-10-23 by overlaying MuJoCo's masks on
Unreal's frames), unless `depth="unreal"`: then Unreal's depth of the visible surfaces, from the same capture as the RGB
(the plugin's "fresh_depth", patches/urlab_fresh_depth.md), so a mask and its depth come from one surface. MuJoCo's depth
shows the physics model's collision shapes (the mug's five boxes have no handle). `shape="visual"` fits the object's
visual meshes, from `visual_scene` (the full scene model the Unreal level was built from; the object bodies' frames are
the physics model's, checked 2026-10-24 on all 64), in place of its collision shape. The read log then also keeps the
chosen masks' IoU with the exact one, whether one was another object, and the best IoU any instance had (`best_iou`:
whether SAM 3 found the object at all)."""

from __future__ import annotations

import functools
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from locomani.tasks.scene.services import SAM3_ADDRESS, UNREAL_ADDRESS, Client, prompt_of, unpack_mask
from locomani.tasks.scene.view import TruthView, geom_rows

GEOM_MESH, GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)


@dataclass(frozen=True)
class RenderConfig:
    cameras: tuple[str, ...] = ("head_camera", "left_wrist", "right_wrist")
    width: int = 320
    height: int = 240
    period_s: float = 0.2  # an estimate older than this at a read is refreshed from the cameras
    min_points: int = 40  # fewer visible pixels than this over all cameras: not seen
    max_range_m: float = 3.0  # depth beyond this is not used
    prior_bias_m: float = 0.05  # never seen yet: the start pose off by this distance (horizontal), the map's memory
    reject_rms_m: float = 0.02  # a fit leaving more than this residual is not taken (the last estimate stays)
    model_points: int = 3000  # surface samples of an object's shape
    masks: str = "exact"  # "exact": the renderer's body ids (L2); "sam3": SAM 3 on Unreal's RGB (L3)
    depth: str = (
        "mujoco"  # "mujoco": MuJoCo's depth (collision shapes); "unreal": Unreal's (visible surfaces), sam3 only
    )
    shape: str = (
        "collision"  # the shape fitted: the physics model's collision geoms, or "visual": visual_scene's meshes
    )
    visual_scene: str = ""  # the scene XML with the objects' visual meshes (shape "visual")
    sam3_min_score: float = 0.3  # an instance below this score is not taken
    assoc_gate_m: float = 0.25  # an instance whose points centre farther than this from the estimate is another object
    unreal_port: int = UNREAL_ADDRESS[1]
    sam3_port: int = SAM3_ADDRESS[1]
    seed: int = 0
    # vision D2 step 2d: a JSON {round seed: {object: [x, y, z] or null}}, the map's object nodes from the round's own
    # mapping tour (scripts/rsi/map_memory.py). Such an object's prior is its node (null: the map found none, so the
    # centre of the surface it stands on); the others keep prior_bias_m
    memory: str = ""

    @classmethod
    def of(cls, spec: dict) -> RenderConfig:
        cfg = cls(**{k: tuple(v) if k == "cameras" else v for k, v in spec.items()})
        if cfg.masks not in ("exact", "sam3"):
            raise ValueError(f"masks {cfg.masks!r}: exact or sam3")
        if cfg.depth not in ("mujoco", "unreal") or (cfg.depth == "unreal" and cfg.masks != "sam3"):
            raise ValueError(f"depth {cfg.depth!r}: mujoco, or unreal with masks sam3 (Unreal's frames)")
        if cfg.shape not in ("collision", "visual") or (cfg.shape == "visual" and not cfg.visual_scene):
            raise ValueError(f"shape {cfg.shape!r}: collision, or visual with a visual_scene")
        if cfg.memory and not Path(cfg.memory).is_file():
            raise ValueError(f"memory {cfg.memory!r}: no such file")
        return cfg


@dataclass
class Frame:
    camera: str
    depth: np.ndarray  # (h, w) metres along the camera's view axis
    body: np.ndarray  # (h, w) body id seen at the pixel, -1 for none
    K: tuple[float, float, float, float]  # fx, fy, cx, cy (pixels)
    pos: np.ndarray  # camera position (world)
    mat: np.ndarray  # camera orientation (world, columns: x right, y up, z backwards -- MuJoCo's convention)
    rgb: np.ndarray | None = None  # (h, w, 3) uint8, when the source renders colour (Unreal)


def intrinsics(fovy_deg: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Square pixels and MuJoCo's vertical field of view."""
    fy = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
    return fy, fy, 0.5 * width, 0.5 * height


def back_project(frame: Frame, mask: np.ndarray, max_range: float = np.inf) -> np.ndarray:
    """World points (n x 3) of the masked pixels with a depth inside (0, max_range)."""
    v, u = np.nonzero(mask)
    z = frame.depth[v, u].astype(float)
    keep = (z > 0) & (z < max_range)
    v, u, z = v[keep], u[keep], z[keep]
    fx, fy, cx, cy = frame.K
    cam = np.stack([(u + 0.5 - cx) / fx * z, -(v + 0.5 - cy) / fy * z, -z], axis=1)
    return cam @ np.asarray(frame.mat, float).T + np.asarray(frame.pos, float)


def surface_points(geoms: list[dict], n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(points, unit normals), `n` each, on the surfaces of collision geoms (scene.view object_geoms rows),
    area-weighted, in the body frame. A normal's sign is arbitrary (the fit uses it as a plane)."""
    tris = []  # (k, 3, 3) triangles in the body frame
    for g in geoms:
        Rg = _quat_mat(g["quat"])
        if g["type"] == "mesh":
            v = np.asarray(g["vert"], float) @ Rg.T + np.asarray(g["pos"], float)
            tris.append(v[np.asarray(g["face"], int)])
        elif g["type"] == "box":
            tris.append(_box_triangles(np.asarray(g["size"], float)) @ Rg.T + np.asarray(g["pos"], float))
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3))
    t = np.concatenate(tris)
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 0
    t, cross, area = t[keep], cross[keep], area[keep]
    i = rng.choice(len(t), size=n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    pts = (1 - r1)[:, None] * t[i, 0] + (r1 * (1 - r2))[:, None] * t[i, 1] + (r1 * r2)[:, None] * t[i, 2]
    return pts, cross[i] / (2 * area[i])[:, None]


STEP_M, STEP_RAD = 0.02, 0.2  # the most one ICP iteration may move: an ill-posed solve (one flat patch) cannot run off


def fit_pose(obs: np.ndarray, model, tree: cKDTree, R0: np.ndarray, t0: np.ndarray, iters: int = 40):
    """Point-to-plane ICP of the observed points onto the model (points, normals): each observed point to its nearest
    model point's tangent plane, the worst 20% of pairs (by distance) trimmed, each iteration's move capped at STEP_M /
    STEP_RAD. Returns (R, t, rms) with world = R @ body + t; rms over the kept pairs' point-to-point distances."""
    pts, nrm = model
    R, t = np.asarray(R0, float), np.asarray(t0, float)
    for _ in range(iters):
        q = (obs - t) @ R  # the observations in the body frame
        d, idx = tree.query(q)
        keep = d <= np.quantile(d, 0.8)
        q, m, n = q[keep], pts[idx[keep]], nrm[idx[keep]]
        A = np.concatenate([np.cross(q, n), n], axis=1)  # moving q by (w x q + v) changes (q - m) . n by A @ [w, v]
        x = np.linalg.lstsq(A, -np.einsum("ij,ij->i", q - m, n), rcond=None)[0]
        w, v = x[:3], x[3:]
        w = w * min(1.0, STEP_RAD / max(float(np.linalg.norm(w)), 1e-12))
        v = v * min(1.0, STEP_M / max(float(np.linalg.norm(v)), 1e-12))
        Rd = _rotvec_mat(w)
        R, t = R @ Rd.T, t - R @ Rd.T @ v  # the body frame moved so that q -> Rd q + v
        if np.linalg.norm(w) < 1e-6 and np.linalg.norm(v) < 1e-6:
            break
    d, _ = tree.query((obs - t) @ R)
    return R, t, float(np.sqrt(np.mean(np.minimum(d, np.quantile(d, 0.8)) ** 2)))


def fit_from_headings(obs: np.ndarray, model, tree: cKDTree, R_prior: np.ndarray, n: int = 8, tie_m: float = 0.001):
    """ICP from `n` headings about the vertical of the prior orientation, centred on the points: the fit with the least
    residual, or -- among fits within `tie_m` of it, which the visible side cannot tell apart (an object's distinguishing
    part facing away) -- the one nearest the prior orientation."""
    fits = []
    for k in range(n):
        a = 2 * np.pi * k / n
        R0 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ R_prior
        fits.append(fit_pose(obs, model, tree, R0, obs.mean(0) - R0 @ model[0].mean(0)))
    least = min(f[2] for f in fits)
    return min((f for f in fits if f[2] <= least + tie_m), key=lambda f: _angle(f[0], R_prior))


class MuJoCoFrames:
    """Depth and body masks of the configured cameras from MuJoCo's renderer, every geom group drawn."""

    def __init__(self, model: mujoco.MjModel, cfg: RenderConfig):
        self.m, self.cfg = model, cfg
        self.r = mujoco.Renderer(model, cfg.height, cfg.width, max_geom=int(model.ngeom) + 1000)  # every geom drawn
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.cams = [(n, model.camera(n).id) for n in cfg.cameras]

    def __call__(self, data: mujoco.MjData) -> list[Frame]:
        out = []
        for name, cid in self.cams:
            self.r.update_scene(data, camera=cid, scene_option=self.opt)
            self.r.enable_depth_rendering()
            depth = self.r.render().copy()
            self.r.disable_depth_rendering()
            self.r.enable_segmentation_rendering()
            seg = self.r.render().copy()
            self.r.disable_segmentation_rendering()
            geom = seg[..., 0]
            is_geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom >= 0)
            body = np.where(is_geom, self.m.geom_bodyid[np.clip(geom, 0, self.m.ngeom - 1)], -1)
            K = intrinsics(float(self.m.cam_fovy[cid]), self.cfg.width, self.cfg.height)
            out.append(Frame(name, depth, body, K, data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()))
        return out


STREAMS = {"head_camera": "ego_view", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}  # Unreal's names


class ServiceFrames:
    """MuJoCoFrames plus Unreal's RGB of the same state (locomani.sim.unreal.rgb_server), pixel for pixel; with
    cfg.depth "unreal", Unreal's depth too (MuJoCo's body ids stay, for the exact-mask diagnostics only)."""

    def __init__(self, model: mujoco.MjModel, cfg: RenderConfig, frames=None, unreal=None):
        self.m, self.cfg = model, cfg
        self.depth = frames if frames is not None else MuJoCoFrames(model, cfg)
        self.unreal = unreal if unreal is not None else Client((UNREAL_ADDRESS[0], int(cfg.unreal_port)))

    def __call__(self, data: mujoco.MjData) -> list[Frame]:
        frames = self.depth(data)
        want = self.cfg.depth == "unreal"
        reply = self.unreal.call(dict(qpos=data.qpos.copy(), depth=want))
        for f in frames:
            rgb = reply["images"][STREAMS[f.camera]]
            if rgb.shape[:2] != f.depth.shape:
                raise RuntimeError(
                    f"{f.camera}: Unreal {rgb.shape[:2]} vs depth {f.depth.shape}: set width 640, height 480"
                )
            f.rgb = rgb
            if want:
                f.depth = np.asarray(reply["depth"][STREAMS[f.camera]], np.float32)
        return frames


@functools.lru_cache(maxsize=2)
def _scene_model(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(path)  # the full apartment model: about 9 s, once per worker


def visual_geoms(scene_xml: str, body_name: str) -> list[dict]:
    """The visual geoms of body `body_name` in the scene `scene_xml`, as TruthView.object_geoms rows."""
    m = _scene_model(str(scene_xml))
    return geom_rows(m, m.body(body_name).id, collision=False)


_FRAMES: dict = {}  # (id(model), RenderConfig) -> frame source: one renderer (and Unreal client) per worker and config


def mujoco_frames(model: mujoco.MjModel, cfg: RenderConfig):
    k = (id(model), cfg)
    if k not in _FRAMES or _FRAMES[k].m is not model:
        _FRAMES[k] = ServiceFrames(model, cfg) if cfg.masks == "sam3" else MuJoCoFrames(model, cfg)
    return _FRAMES[k]


class RenderedView(TruthView):
    """TruthView with the free objects' poses from the cameras (module doc)."""

    __slots__ = ("_cfg", "_round", "_frames", "_shot", "_est", "_model", "_n", "_log", "_sam3", "_seg", "_diag", "_mem")

    def __init__(self, world, cfg: RenderConfig, round_seed: int, frames=None):
        super().__init__(world)
        self._cfg, self._round = cfg, int(round_seed)
        self._frames = frames if frames is not None else mujoco_frames(world.sim.m, cfg)
        self._shot = (None, [])  # (motor tick, frames) of the last render
        self._est: dict = {}  # key -> dict(pos, quat, t, seen, rms)
        self._model: dict = {}  # key -> ((body-frame surface points, normals), the points' tree)
        self._n: dict = {}  # key -> counts
        self._log: dict = {}  # key -> one row per read
        self._sam3 = Client((SAM3_ADDRESS[0], int(cfg.sam3_port))) if cfg.masks == "sam3" else None
        self._seg: dict = {}  # (shot tick, camera, prompt) -> SAM 3 instances [(score, mask)]
        self._diag: dict = {}  # key -> what the last refresh's masks were against the exact ones
        self._mem = json.loads(Path(cfg.memory).read_text()).get(str(self._round), {}) if cfg.memory else {}

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._world.free:
            return self._world.object_pose(key)
        n = self._n.setdefault(key, dict(reads=0, refreshes=0, seen=0, rejected=0))
        n["reads"] += 1
        est, now = self._est.get(key), float(self._world.motor.t)
        if est is None:
            est = self._est[key] = self._prior(key)
        pts = None
        if est["t"] is None or now - est["t"] >= self._cfg.period_s:
            pts = self._refresh(key, est, now)
        if pts is not None and pts < 0:
            src, pts = "rejected", -pts
        else:
            src = "seen" if pts is not None and pts >= self._cfg.min_points else "kept" if est["seen"] else "prior"
        true, true_q = (np.asarray(a, float) for a in self._world.object_pose(key))
        row = dict(t=round(now, 2), src=src, pts=pts, err_cm=round(100 * float(np.linalg.norm(est["pos"] - true)), 2),
                   rot_deg=round(float(np.degrees(_angle(_quat_mat(est["quat"]), _quat_mat(true_q)))), 1))  # fmt: skip
        if pts is not None and self._sam3 is not None:
            row.update(self._diag.pop(key, {}))
        self._log.setdefault(key, []).append(row)
        return est["pos"].copy(), est["quat"].copy()

    def _prior(self, key: str) -> dict:
        pos, quat = self._world.object_pose(key)
        if key in self._mem:  # the map's memory (the orientation stays the start's, as below)
            node = self._mem[key]
            if node is None:
                o = self._world.meta["objects"][key]
                surf = self._world.meta["surfaces"][o["surface"]]
                node = [*surf["xy"], float(surf["top"]) + float(o["box"]["half_extent"][2])]
            return dict(pos=np.asarray(node, float), quat=np.asarray(quat, float), t=None, seen=False, rms=None)
        rng = np.random.default_rng([self._round, int(self._cfg.seed), zlib.crc32(key.encode())])
        a = rng.uniform(0.0, 2 * np.pi)
        off = float(self._cfg.prior_bias_m) * np.array([np.cos(a), np.sin(a), 0.0])
        return dict(pos=np.asarray(pos, float) + off, quat=np.asarray(quat, float), t=None, seen=False, rms=None)

    def _refresh(self, key: str, est: dict, now: float) -> int:
        """Fit the object anew from the cameras if enough of it is visible; returns its visible pixel count, negated
        when the fit was rejected."""
        self._n[key]["refreshes"] += 1
        est["t"] = now
        body = self._world.free[key]
        if self._sam3 is None:
            obs = [back_project(f, f.body == body, self._cfg.max_range_m) for f in self._shoot()]
        else:
            obs = self._sam3_points(key, body, est["pos"])
        obs = np.concatenate(obs) if obs else np.zeros((0, 3))
        if len(obs) < self._cfg.min_points:
            return len(obs)
        model, tree = self._shape(key)
        R_prev = _quat_mat(est["quat"])
        R, t, rms = fit_pose(obs, model, tree, R_prev, est["pos"]) if est["seen"] else (None, None, np.inf)
        if rms > 0.01:  # first sighting, or the last estimate does not explain the points: try every heading
            alt = fit_from_headings(obs, model, tree, R_prev)
            R, t, rms = alt if alt[2] < rms else (R, t, rms)
        reach = float(np.linalg.norm(model[0], axis=1).max())  # the shape's radius about its body origin
        if rms > self._cfg.reject_rms_m or float(np.linalg.norm(t - obs.mean(0))) > reach + 0.02:
            self._n[key]["rejected"] += 1
            return -len(obs)  # seen, but the fit is not believable: the last estimate stays
        est.update(pos=t, quat=_mat_quat(R), seen=True, rms=rms)
        self._n[key]["seen"] += 1
        return len(obs)

    def _sam3_points(self, key: str, body: int, near: np.ndarray) -> list[np.ndarray]:
        """Per camera, the points of the SAM 3 instance of the object's phrase nearest `near` within the gate."""
        cfg, prompt = self._cfg, prompt_of(self._world.meta["objects"][key], key)
        out, ious, wrong, found, oracle = [], [], 0, 0, []
        for f in self._shoot():
            k = (self._shot[0], f.camera, prompt)
            if k not in self._seg:
                res = self._sam3.call(dict(image=f.rgb, prompts=[prompt]))["results"][prompt]
                self._seg[k] = [(score, unpack_mask(m)) for score, m in res]
            found += len(self._seg[k])
            exact = f.body == body
            best = None
            for score, mask in self._seg[k]:
                if exact.any():
                    oracle.append(float((mask & exact).sum() / max((mask | exact).sum(), 1)))
                if score < cfg.sam3_min_score:
                    continue
                p = back_project(f, mask, cfg.max_range_m)
                d = float(np.linalg.norm(p.mean(0) - near)) if len(p) >= 10 else np.inf
                if d <= cfg.assoc_gate_m and (best is None or d < best[0]):
                    best = (d, p, mask)
            if best is not None:
                out.append(best[1])
                ious.append(float((best[2] & exact).sum() / max((best[2] | exact).sum(), 1)))
                wrong += float((best[2] & exact).sum()) < 0.5 * float(best[2].sum())
        self._diag[key] = dict(instances=found, iou=round(float(np.mean(ious)), 3) if ious else None, wrong=wrong,
                               best_iou=round(max(oracle), 3) if oracle else None)  # fmt: skip
        return out

    def _shoot(self) -> list[Frame]:
        k = self._world.motor.k
        if self._shot[0] != k:
            self._shot = (k, self._frames(self._world.sim.d))
            self._seg = {}
        return self._shot[1]

    def _shape(self, key: str):
        if key not in self._model:
            rng = np.random.default_rng([int(self._cfg.seed), zlib.crc32(key.encode())])
            if self._cfg.shape == "visual":
                name = self._world.sim.m.body(self._world.free[key]).name
                geoms = visual_geoms(self._cfg.visual_scene, name)
            else:
                geoms = self.object_geoms(key)
            pts, nrm = surface_points(geoms, int(self._cfg.model_points), rng)
            self._model[key] = ((pts, nrm), cKDTree(pts))
        return self._model[key]

    def stats(self) -> dict:
        """Per object read: reads, refreshes, sightings, the last residual, and the estimate's error now (evaluator side)."""
        out = {}
        for key, est in self._est.items():
            pos, quat = self._world.object_pose(key)
            out[key] = dict(**self._n[key], ever_seen=bool(est["seen"]), rms_mm=None if est["rms"] is None
                            else round(1000 * float(est["rms"]), 2),
                            err_cm=round(100 * float(np.linalg.norm(est["pos"] - np.asarray(pos, float))), 2),
                            log=self._log.get(key, []))  # fmt: skip
        return out


def _box_triangles(half: np.ndarray) -> np.ndarray:
    """The 12 triangles (12 x 3 x 3) of a box of half extents `half`, centred at the origin."""
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    tris = []
    for a, b, k, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        tris += [[c[a], c[b], c[k]], [c[a], c[k], c[d]]]
    return np.array(tris)


def _angle(Ra, Rb) -> float:
    return float(np.arccos(np.clip((np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2, -1.0, 1.0)))


def _rotvec_mat(w) -> np.ndarray:
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def _mat_quat(R) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(R, float).reshape(9))
    return out
