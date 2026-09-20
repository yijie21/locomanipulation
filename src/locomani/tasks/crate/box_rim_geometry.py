"""Validated open-box geometry and deterministic opposite-rim sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

RIMS = ("x-", "x+", "y-", "y+")


def _vector(values, length: int, name: str) -> np.ndarray:
    value = np.array(values, dtype=float, copy=True)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {length} finite values")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class RigidPose:
    """World pose of the box bottom-centre; quaternion order is wxyz."""

    position: np.ndarray
    quaternion_wxyz: np.ndarray

    def __init__(self, position, quaternion_wxyz):
        position = _vector(position, 3, "pose.position")
        quaternion = _vector(quaternion_wxyz, 4, "pose.quaternion_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("pose.quaternion_wxyz must be unit length")
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        if np.linalg.det(rotation) < 1.0 - 1e-8:
            raise ValueError("pose quaternion does not define a proper rotation")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)

    @property
    def rotation(self) -> np.ndarray:
        return Rotation.from_quat(self.quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()

    def transform_points(self, local) -> np.ndarray:
        points = np.asarray(local, dtype=float)
        return points @ self.rotation.T + self.position


@dataclass(frozen=True)
class ForbiddenRimInterval:
    """Closed forbidden interval in metres along a named local rim axis."""

    rim: str
    start_m: float
    end_m: float

    def __post_init__(self):
        if self.rim not in RIMS:
            raise ValueError(f"rim must be one of {RIMS}")
        if not np.isfinite([self.start_m, self.end_m]).all() or self.start_m > self.end_m:
            raise ValueError("forbidden interval needs finite start_m <= end_m")


@dataclass(frozen=True)
class OpenBoxGeometry:
    """Rigid rectangular open box whose local z=0 is the outside bottom."""

    size_m: np.ndarray
    wall_thickness_m: float
    pose: RigidPose
    local_com_m: np.ndarray
    corner_clearance_m: float
    forbidden_intervals: tuple[ForbiddenRimInterval, ...]

    def __init__(
        self,
        size_m,
        wall_thickness_m,
        pose: RigidPose,
        *,
        local_com_m=None,
        corner_clearance_m=0.025,
        forbidden_intervals=(),
    ):
        size = _vector(size_m, 3, "size_m")
        thickness = float(wall_thickness_m)
        clearance = float(corner_clearance_m)
        if np.any(size <= 0):
            raise ValueError("size_m values must be positive")
        if not np.isfinite(thickness) or thickness <= 0 or 2 * thickness >= min(size[:2]) or thickness >= size[2]:
            raise ValueError("wall_thickness_m must leave a positive box interior and wall height")
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("corner_clearance_m must be finite and nonnegative")
        if 2 * clearance >= min(size[:2]):
            raise ValueError("corner_clearance_m leaves no usable rim")
        com = _vector(
            [0.0, 0.0, size[2] / 2] if local_com_m is None else local_com_m,
            3,
            "local_com_m",
        )
        restrictions = tuple(forbidden_intervals)
        if not all(isinstance(item, ForbiddenRimInterval) for item in restrictions):
            raise ValueError("forbidden_intervals must contain ForbiddenRimInterval values")
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "wall_thickness_m", thickness)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "local_com_m", com)
        object.__setattr__(self, "corner_clearance_m", clearance)
        object.__setattr__(self, "forbidden_intervals", restrictions)

    @property
    def com_world(self) -> np.ndarray:
        return self.pose.transform_points(self.local_com_m)

    def rim_extent(self, rim: str) -> tuple[float, float]:
        if rim not in RIMS:
            raise ValueError(f"rim must be one of {RIMS}")
        length = self.size_m[1] if rim.startswith("x") else self.size_m[0]
        half = float(length / 2)
        return -half + self.corner_clearance_m, half - self.corner_clearance_m

    def permits(self, rim: str, coordinate_m: float) -> bool:
        lo, hi = self.rim_extent(rim)
        if coordinate_m < lo - 1e-12 or coordinate_m > hi + 1e-12:
            return False
        return not any(
            interval.rim == rim and interval.start_m - 1e-12 <= coordinate_m <= interval.end_m + 1e-12
            for interval in self.forbidden_intervals
        )

    def permits_span(self, rim: str, coordinate_m: float, half_width_m: float) -> bool:
        if not np.isfinite(half_width_m) or half_width_m < 0:
            raise ValueError("contact half-width must be finite and nonnegative")
        start, end = coordinate_m - half_width_m, coordinate_m + half_width_m
        lo, hi = self.rim_extent(rim)
        if start < lo - 1e-12 or end > hi + 1e-12:
            return False
        return not any(
            interval.rim == rim and start <= interval.end_m + 1e-12 and end >= interval.start_m - 1e-12
            for interval in self.forbidden_intervals
        )

    def rim_point_local(self, rim: str, coordinate_m: float, depth_m: float) -> np.ndarray:
        if not self.permits(rim, coordinate_m):
            raise ValueError(f"coordinate {coordinate_m} is not usable on rim {rim}")
        if not np.isfinite(depth_m) or depth_m < 0 or depth_m >= self.size_m[2]:
            raise ValueError("depth_m must be finite and inside the wall height")
        x_half, y_half = self.size_m[:2] / 2 - self.wall_thickness_m / 2
        if rim == "x-":
            return np.array([-x_half, coordinate_m, self.size_m[2] - depth_m])
        if rim == "x+":
            return np.array([x_half, coordinate_m, self.size_m[2] - depth_m])
        if rim == "y-":
            return np.array([coordinate_m, -y_half, self.size_m[2] - depth_m])
        if rim == "y+":
            return np.array([coordinate_m, y_half, self.size_m[2] - depth_m])
        raise ValueError(f"rim must be one of {RIMS}")


@dataclass(frozen=True)
class RimPairCandidate:
    identifier: str
    rim_pair: str
    coordinate_m: float
    left_offset_m: float
    right_offset_m: float
    depth_m: float
    pitch_deg: float
    points_world: np.ndarray


def sample_opposite_rims(
    box: OpenBoxGeometry,
    *,
    samples_per_rim: int,
    side_offsets_m: tuple[float, ...],
    depths_m: tuple[float, ...],
    palm_pitch_deg: tuple[float, ...],
    contact_half_width_m: float = 0.0,
) -> tuple[RimPairCandidate, ...]:
    """Sample both opposite pairs; returned points are not hand-assigned."""
    if not isinstance(samples_per_rim, int) or samples_per_rim < 1:
        raise ValueError("samples_per_rim must be a positive integer")
    offsets = tuple(float(value) for value in side_offsets_m)
    depths = tuple(float(value) for value in depths_m)
    pitches = tuple(float(value) for value in palm_pitch_deg)
    if not offsets or not depths or not pitches or not np.isfinite([*offsets, *depths, *pitches]).all():
        raise ValueError("offset, depth and pitch candidates must be nonempty and finite")
    candidates = []
    for pair, rims in (("x", ("x-", "x+")), ("y", ("y-", "y+"))):
        lo, hi = box.rim_extent(rims[0])
        lo, hi = lo + contact_half_width_m, hi - contact_half_width_m
        if lo > hi:
            continue
        coordinates_to_sample = [0.5 * (lo + hi)] if samples_per_rim == 1 else np.linspace(lo, hi, samples_per_rim)
        for coordinate in coordinates_to_sample:
            for left_offset in offsets:
                for right_offset in offsets:
                    coordinates = (coordinate + left_offset, coordinate + right_offset)
                    if not all(
                        box.permits_span(rim, value, contact_half_width_m) for rim, value in zip(rims, coordinates)
                    ):
                        continue
                    for depth in depths:
                        local = np.array(
                            [box.rim_point_local(rim, value, depth) for rim, value in zip(rims, coordinates)]
                        )
                        world = box.pose.transform_points(local)
                        world.setflags(write=False)
                        for pitch in pitches:
                            identifier = (
                                f"{pair}:{coordinate:+.4f}:{left_offset:+.4f}:{right_offset:+.4f}:"
                                f"{depth:.4f}:{pitch:+.2f}"
                            )
                            candidates.append(
                                RimPairCandidate(
                                    identifier,
                                    pair,
                                    float(coordinate),
                                    left_offset,
                                    right_offset,
                                    depth,
                                    pitch,
                                    world,
                                )
                            )
    return tuple(candidates)
