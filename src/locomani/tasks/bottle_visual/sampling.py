"""Versioned reset geometry, never used as a student observation or action cue."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LEGACY_PROFILE = "legacy_edges_v1"
DEFAULT_PROFILE = "yplus_center_v1"
PROFILES = (DEFAULT_PROFILE, LEGACY_PROFILE)


def profile_edges(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown scene profile: {profile}")
    return ("y+",) if profile == DEFAULT_PROFILE else ("x-", "y-", "y+")


def validate_edges(profile, edges):
    if not edges or not set(edges) <= set(profile_edges(profile)):
        raise ValueError(f"{profile} requires edges from {profile_edges(profile)}")


def validate_reuse(cohort, profile):
    """Do not mix old multi-edge normal episodes into new recovery datasets."""
    cfg = json.loads((Path(cohort) / "config.json").read_text())
    if cfg.get("arm") != "B" or cfg.get("scene_profile", LEGACY_PROFILE) != profile:
        raise ValueError("Recovery reuse requires a B cohort with the same scene profile")
    status = json.loads((Path(cohort) / "run.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Recovery reuse requires a completed B cohort")


def sample_reset(geometry, sim, rng, edge, stratum, profile, stand_d=None, inboard=None):
    """`geometry` is the geometry module (constants + legacy `sample_scene`); `stand_d` / `inboard` are
    (lo, hi) bands overriding its STAND_D / EDGE_INBOARD (the expert pins stand_d, scene.STRATA sets inboard)."""
    validate_edges(profile, [edge])
    if stratum not in ("high_near", "low_near", "low_deep"):
        raise ValueError(f"Unknown scene stratum: {stratum}")
    runtime = geometry
    stand_d = stand_d or runtime.STAND_D
    if profile == LEGACY_PROFILE:
        return runtime.sample_scene(sim, rng, edge=edge, stand_d=stand_d, inboard=inboard or runtime.EDGE_INBOARD)
    # Preserve the random draw order of the original edge sampler, but
    # centre x on the table rather than the original bottle's off-centre x.
    stand_d = float(rng.uniform(*stand_d))
    inboard = float(rng.uniform(*((0.18, 0.24) if stratum == "low_deep" else (0.12, 0.18))))
    along = float(rng.uniform(-0.08, 0.08))
    surf = sim.smeta["surfaces"]["A"]
    cx, cy = surf["xy"]
    hx, hy = surf["flat_half_extent"]
    if hx <= 0.08 + 0.023 or 2 * hy <= inboard + 0.023:
        raise ValueError("Table is too small for the bottle sampling support")
    bottle = sim.crate0.copy()
    bottle[:2] = cx + along, cy + hy - inboard
    inward, left = np.array([0.0, -1.0]), np.array([1.0, 0.0])
    stand = bottle[:2] - inward * (stand_d - runtime.STAND_BACKSTEP) + left * runtime.STAND_LEFT
    spawn_edge = float(rng.uniform(*runtime.SPAWN_EDGE))
    spawn_lat = float(rng.uniform(-runtime.SPAWN_LAT, runtime.SPAWN_LAT))
    edge_point = bottle[:2] - inward * inboard
    spawn = edge_point - inward * spawn_edge + left * spawn_lat
    toward = bottle[:2] - spawn
    yaw = float(np.arctan2(toward[1], toward[0]) + rng.uniform(-runtime.SPAWN_YAW_ERR, runtime.SPAWN_YAW_ERR))
    meta = dict(
        scene="edge",
        scene_profile=profile,
        edge=edge,
        inboard=inboard,
        along_center_offset=along,
        stand_d=stand_d,
        stand_edge=stand_d - runtime.STAND_BACKSTEP - inboard,
        spawn_edge=spawn_edge,
        spawn_lat=spawn_lat,
        bottle_xy=[round(float(v), 3) for v in bottle[:2]],
        spawn_yaw_deg=float(np.degrees(yaw)),
    )
    return bottle, stand, spawn, yaw, meta


@contextmanager
def reset_sampler(geometry, edge, stratum, profile, stand_d=None, inboard=None):
    """Temporarily make `geometry.sample_scene(sim, rng)` draw this edge / stratum / profile."""
    validate_edges(profile, [edge])
    original = geometry.sample_scene
    geometry.sample_scene = lambda sim, rng: sample_reset(geometry, sim, rng, edge, stratum, profile, stand_d, inboard)
    try:
        yield
    finally:
        geometry.sample_scene = original
