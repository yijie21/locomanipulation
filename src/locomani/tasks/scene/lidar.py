"""The head LiDAR (Livox MID-360) by ray casting in MuJoCo (vision RSI design section 3, D4: simulate it). Mounted where
the lab robot's URDF puts `mid360_link` (assets/unitree_G1_with_wrist_camera/urdf/g1_comp_new_hip_dex1_1.urdf,
`mid360_joint` on torso_link); the mars model carries no site for it, so the pose is torso_link's times that mount.

The field of view and rate are the datasheet's: 360 deg around, -7 to 52 deg in elevation, about 200,000 points a second
in 10 Hz frames; ranges 0.1-40 m with about 2 cm of noise. The real scan is non-repetitive (each frame covers new
directions, the cloud densifies over time); here a low-discrepancy sequence over azimuth and sin(elevation) gives each
frame the next `rays_per_frame` directions. The sensor sits inside torso_link's collision shell (the head), so rays pass
through torso_link (the window); they end on whatever geom they meet first, and returns from the rest of the robot's
body -- an arm in the way -- are dropped (as the real driver's self filter does). Measured 2026-10-23 on the apartment's
physics model (12,420 geoms, the robot at its start): a 20,000-ray frame casts in about 0.48 s, a 2,000-ray frame in
45 ms (mj_multiRay is single-threaded); about 36% of the rays return, median range 3.7 m -- the scene has no ceiling."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

MOUNT_POS = np.array([0.0002835, 0.00003, 0.428434])  # mid360_joint origin xyz in torso_link
MOUNT_PITCH = 0.05112069379091391  # its rpy (0, pitch, 0)
GOLDEN = (np.sqrt(5.0) - 1) / 2  # the R2 sequence's two irrational steps: 1/phi and 1/plastic
PLASTIC = 0.7548776662466927


@dataclass(frozen=True)
class Mid360Config:
    rays_per_frame: int = 20000
    elev_min_deg: float = -7.0
    elev_max_deg: float = 52.0
    min_range_m: float = 0.1
    max_range_m: float = 40.0
    noise_m: float = 0.02
    seed: int = 0
    # geom groups the rays may hit (MuJoCo's geomgroup, six flags), None: all. A full scene model carries visual
    # meshes in groups 0-2 and their collision shapes in group 3: (1, 1, 1, 0, 0, 0) casts on what the cameras see
    geomgroup: tuple[int, ...] | None = None


def directions(n: int, start: int, cfg: Mid360Config) -> np.ndarray:
    """Unit vectors (n x 3, sensor frame: x forward, z up) for scan indices start .. start + n - 1."""
    i = np.arange(start, start + n, dtype=float)
    az = 2 * np.pi * np.mod(0.5 + i * PLASTIC, 1.0)
    s0, s1 = np.sin(np.radians(cfg.elev_min_deg)), np.sin(np.radians(cfg.elev_max_deg))
    sz = s0 + (s1 - s0) * np.mod(0.5 + i * GOLDEN, 1.0)  # uniform in sin(elevation): uniform over the solid angle
    c = np.sqrt(1 - sz**2)
    return np.stack([c * np.cos(az), c * np.sin(az), sz], axis=1)


class Mid360:
    """Frames of the head LiDAR for one model."""

    def __init__(self, model: mujoco.MjModel, cfg: Mid360Config | None = None):
        self.m, self.cfg = model, cfg or Mid360Config()
        self.torso = model.body("torso_link").id
        root = int(model.body_rootid[self.torso])
        self.robot_geom = model.body_rootid[model.geom_bodyid] == root  # geoms of the robot's own tree
        c, s = np.cos(MOUNT_PITCH), np.sin(MOUNT_PITCH)
        self.mount_R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.frame = 0
        self.rng = np.random.default_rng(self.cfg.seed)

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation) of the sensor in the world."""
        Rt = data.xmat[self.torso].reshape(3, 3)
        return data.xpos[self.torso] + Rt @ MOUNT_POS, Rt @ self.mount_R

    def scan(self, data: mujoco.MjData) -> dict:
        """The next frame: world points (k x 3) of the returns, their ranges and geoms, the sensor pose, the ray count
        and the seconds the cast took."""
        n, cfg = int(self.cfg.rays_per_frame), self.cfg
        p, R = self.pose(data)
        vec = directions(n, self.frame * n, cfg) @ R.T
        geom, dist = np.zeros(n, np.int32), np.zeros(n)
        t0 = time.perf_counter()
        group = None if cfg.geomgroup is None else np.asarray(cfg.geomgroup, np.uint8)
        mujoco.mj_multiRay(self.m, data, p, vec.reshape(-1), group, 1, self.torso, geom, dist, None, n, cfg.max_range_m)
        seconds = time.perf_counter() - t0
        self.frame += 1
        hit = (geom >= 0) & (dist >= cfg.min_range_m)
        hit[hit] &= ~self.robot_geom[geom[hit]]
        r = dist[hit] + self.rng.normal(scale=cfg.noise_m, size=int(hit.sum()))
        return dict(points=p + vec[hit] * r[:, None], ranges=r, geoms=geom[hit], origin=p, rotation=R, rays=n,
                    seconds=seconds)  # fmt: skip
