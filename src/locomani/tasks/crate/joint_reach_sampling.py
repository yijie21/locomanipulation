"""Deterministic joint five-dimensional sampling for measured reach trials."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

DIMENSIONS = ("x_m", "y_m", "height_m", "separation_m", "yaw_deg")
MAX_SAMPLE_COUNT = 1_000_000
METHODS = {"sobol", "stratified", "grid", "boundary_refinement"}
CONFIG_FIELDS = {
    "schema_version",
    "method",
    "seed",
    "bounds",
    "count",
    "bodyward_tilt_deg",
    "anchors",
    "anchor_grid",
    "axis_values",
    "observations",
    "neighbor_count",
    "min_normalized_distance",
}


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_bounds(raw: object) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    if not isinstance(raw, dict) or set(raw) != set(DIMENSIONS):
        raise ValueError(f"bounds must contain exactly: {', '.join(DIMENSIONS)}")
    lower = []
    upper = []
    canonical = {}
    for name in DIMENSIONS:
        extent = raw[name]
        if not isinstance(extent, list) or len(extent) != 2:
            raise ValueError(f"bounds.{name} must be [lower, upper]")
        lo = _finite_number(extent[0], f"bounds.{name}[0]")
        hi = _finite_number(extent[1], f"bounds.{name}[1]")
        if not lo < hi:
            raise ValueError(f"bounds.{name} must satisfy lower < upper")
        lower.append(lo)
        upper.append(hi)
        canonical[name] = [lo, hi]
    if lower[3] <= 0.0:
        raise ValueError("bounds.separation_m must remain positive")
    return np.asarray(lower), np.asarray(upper), canonical


def _validate_point(value: object, lower: np.ndarray, upper: np.ndarray, name: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain five finite coordinates") from error
    if point.shape != (5,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain five finite coordinates")
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError(f"{name} lies outside configured bounds")
    return point


def _axis_product(raw: object, lower: np.ndarray, upper: np.ndarray, name: str) -> list[np.ndarray]:
    if raw is None:
        return []
    if not isinstance(raw, dict) or set(raw) != set(DIMENSIONS):
        raise ValueError(f"{name} must contain exactly: {', '.join(DIMENSIONS)}")
    axes: list[list[float]] = []
    product_count = 1
    for dimension_index, dimension in enumerate(DIMENSIONS):
        values = raw[dimension]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}.{dimension} must be a non-empty list")
        axis = [_finite_number(value, f"{name}.{dimension}") for value in values]
        if len(set(axis)) != len(axis):
            raise ValueError(f"{name}.{dimension} values must be unique")
        if any(value < lower[dimension_index] or value > upper[dimension_index] for value in axis):
            raise ValueError(f"{name}.{dimension} contains a value outside bounds")
        axes.append(axis)
        product_count *= len(axis)
        if product_count > MAX_SAMPLE_COUNT:
            raise ValueError(f"{name} expands beyond {MAX_SAMPLE_COUNT} points")
    return [np.asarray(values, dtype=float) for values in itertools.product(*axes)]


def _sample_unit(method: str, count: int, seed: int) -> np.ndarray:
    if method == "sobol":
        sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
        if count > 0 and count & (count - 1) == 0:
            return sampler.random_base2(int(math.log2(count)))
        return sampler.random(count)
    if method == "stratified":
        return qmc.LatinHypercube(d=5, seed=seed).random(count)
    raise ValueError(f"unsupported joint sampling method {method}")


def classify_reach_attempt(result: dict) -> dict:
    """Classify raw evidence without mistaking executor admission for reach motion."""
    if not isinstance(result, dict):
        raise ValueError("reach result must be an object")
    if isinstance(result.get("case"), dict):
        result = result["case"]
    traces = result.get("traces", [])
    if not isinstance(traces, list) or any(not isinstance(trace, dict) for trace in traces):
        raise ValueError("reach result traces must be a list of objects")
    phases = [trace.get("controller_phase") for trace in traces if isinstance(trace.get("controller_phase"), str)]
    reach_traces = [trace for trace in traces if trace.get("controller_phase") == "reach"]
    executor_invoked = result.get("execution_started") is True
    reason = result.get("reason")
    if reach_traces:
        classification = "executed_reach"
    elif executor_invoked and isinstance(reason, str) and reason.startswith("controller_"):
        classification = "controller_admission_rejection"
    elif executor_invoked:
        classification = "executor_invoked_without_reach_evidence"
    else:
        classification = "initialization_or_infrastructure_rejection"
    first_reach = reach_traces[0] if reach_traces else {}
    return {
        "executor_invoked": executor_invoked,
        "reach_commanded": bool(reach_traces),
        "attempt_classification": classification,
        "controller_phases": list(dict.fromkeys(phases)),
        "first_reach_seq": first_reach.get("seq"),
        "first_reach_sim_time_s": first_reach.get("sim_time_s"),
    }


def _observation_label(observation: dict) -> str | None:
    if observation.get("refinement_eligible") is False:
        return None
    reach_commanded = observation.get("reach_commanded")
    if reach_commanded is None and "traces" in observation:
        reach_commanded = classify_reach_attempt(observation)["reach_commanded"]
    if reach_commanded is not True:
        return None
    if observation.get("status") == "success":
        return "success"
    if observation.get("status") == "failed":
        if observation.get("reason") in {"runtime_error", "wall_watchdog_timeout", "simulation_clock_reversed"}:
            return None
        return "failure"
    return None


def _boundary_points(
    config: dict,
    observations: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict]]:
    if not isinstance(observations, dict) or observations.get("schema_version") != 1:
        raise ValueError("observations schema_version must be 1")
    rows = observations.get("observations")
    if not isinstance(rows, list):
        raise ValueError("observations.observations must be a list")
    usable = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"observations[{row_index}] must be an object")
        label = _observation_label(row)
        if label is None:
            continue
        point = _validate_point(row.get("coordinates"), lower, upper, f"observations[{row_index}].coordinates")
        usable.append((row, point, label))
    successes = [value for value in usable if value[2] == "success"]
    failures = [value for value in usable if value[2] == "failure"]
    if not successes or not failures:
        raise ValueError("boundary refinement requires executed success and failure observations")

    scale = upper - lower
    neighbor_count = config.get("neighbor_count", 8)
    if isinstance(neighbor_count, bool) or not isinstance(neighbor_count, int) or neighbor_count < 1:
        raise ValueError("neighbor_count must be a positive integer")
    minimum_distance = _finite_number(config.get("min_normalized_distance", 0.01), "min_normalized_distance")
    if not 0.0 <= minimum_distance < 1.0:
        raise ValueError("min_normalized_distance must be in [0, 1)")

    pairs: dict[tuple[int, int], tuple] = {}
    for success_index, success in enumerate(successes):
        success_normalized = (success[1] - lower) / scale
        ranked = sorted(
            (
                (float(np.linalg.norm(success_normalized - (failure[1] - lower) / scale)), failure_index)
                for failure_index, failure in enumerate(failures)
            ),
            key=lambda value: (value[0], value[1]),
        )
        for distance, failure_index in ranked[:neighbor_count]:
            pairs[(success_index, failure_index)] = (distance, success, failures[failure_index])
    ordered_pairs = sorted(
        pairs.values(),
        key=lambda value: (value[0], str(value[1][0].get("sample_id")), str(value[2][0].get("sample_id"))),
    )

    rng = np.random.default_rng(seed)
    existing = [(value[1] - lower) / scale for value in usable]
    generated_normalized: list[np.ndarray] = []
    points: list[np.ndarray] = []
    provenance: list[dict] = []
    attempts = 0
    max_attempts = max(10_000, count * 500)
    while len(points) < count and attempts < max_attempts:
        distance, success, failure = ordered_pairs[attempts % len(ordered_pairs)]
        success_normalized = (success[1] - lower) / scale
        failure_normalized = (failure[1] - lower) / scale
        midpoint = 0.5 * (success_normalized + failure_normalized)
        if attempts < len(ordered_pairs):
            candidate = midpoint
        else:
            jitter_scale = max(0.005, min(0.08, distance * 0.12))
            candidate = np.clip(midpoint + rng.normal(0.0, jitter_scale, size=5), 0.0, 1.0)
        comparisons = existing + generated_normalized
        nearest = min(float(np.linalg.norm(candidate - value)) for value in comparisons)
        attempts += 1
        if nearest < minimum_distance:
            continue
        generated_normalized.append(candidate)
        points.append(lower + candidate * scale)
        provenance.append(
            {
                "kind": "boundary_refinement",
                "success_sample_id": success[0].get("sample_id"),
                "failure_sample_id": failure[0].get("sample_id"),
                "failure_reason": failure[0].get("reason"),
                "failure_subtype": (
                    "executed_controller_failure"
                    if str(failure[0].get("reason", "")).startswith("controller_")
                    else "executed_reach_failure"
                ),
                "neighbor_distance_normalized": distance,
                "nearest_observed_or_proposed_distance_normalized": nearest,
            }
        )
    if len(points) != count:
        raise ValueError(
            f"could generate only {len(points)} of {count} boundary points at min_normalized_distance={minimum_distance}"
        )
    return points, provenance


def _sample_record(index: int, point: np.ndarray, tilt: float, provenance: dict) -> dict:
    x_m, y_m, height_m, separation_m, yaw_deg = map(float, point)
    yaw_rad = math.radians(yaw_deg)
    line = np.asarray([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
    midpoint = np.asarray([x_m, y_m, height_m])
    left = midpoint + 0.5 * separation_m * line
    right = midpoint - 0.5 * separation_m * line
    coordinates = dict(zip(DIMENSIONS, (x_m, y_m, height_m, separation_m, yaw_deg), strict=True))
    return {
        "index": index,
        "label_zh": f"joint_5d_{index:06d}",
        "parameters": {
            "pair_midpoint_x_m": x_m,
            "pair_midpoint_y_m": y_m,
            "ground_height_z_m": height_m,
            "pair_separation_m": separation_m,
            "pair_line_yaw_deg": yaw_deg,
        },
        "capture_centres_xyz": {"left": left.tolist(), "right": right.tolist()},
        "orientation_endpoints": {},
        "bodyward_tilt_deg": tilt,
        "joint_coordinates": coordinates,
        "sampling_provenance": provenance,
    }


def generate_joint_bank(config: dict, *, observations: dict | None = None) -> dict:
    """Generate a source-compatible bank without classifying unsampled volume."""
    if not isinstance(config, dict):
        raise ValueError("sampling config must be an object")
    unknown = sorted(set(config) - CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"sampling config has unknown fields: {', '.join(unknown)}")
    if config.get("schema_version") != 1:
        raise ValueError("sampling config schema_version must be 1")
    method = config.get("method")
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(METHODS))}")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    lower, upper, bounds = _validate_bounds(config.get("bounds"))
    tilt = _finite_number(config.get("bodyward_tilt_deg", 30.0), "bodyward_tilt_deg")
    if not 0.0 <= tilt <= 45.0:
        raise ValueError("bodyward_tilt_deg must be in [0, 45]")

    points: list[np.ndarray] = []
    provenance: list[dict] = []
    requested_count = config.get("count", 0)
    if isinstance(requested_count, bool) or not isinstance(requested_count, int) or requested_count < 0:
        raise ValueError("count must be a non-negative integer")
    if method in {"sobol", "stratified", "boundary_refinement"} and requested_count < 1:
        raise ValueError(f"count must be positive for {method}")
    if requested_count > MAX_SAMPLE_COUNT:
        raise ValueError(f"count may not exceed {MAX_SAMPLE_COUNT}")

    if method in {"sobol", "stratified"}:
        unit = _sample_unit(method, requested_count, seed)
        points.extend(lower + unit * (upper - lower))
        provenance.extend({"kind": method, "ordinal": index} for index in range(requested_count))
    elif method == "grid":
        grid = _axis_product(config.get("axis_values"), lower, upper, "axis_values")
        points.extend(grid)
        provenance.extend({"kind": "grid", "ordinal": index} for index in range(len(grid)))
    else:
        if observations is None:
            raise ValueError("boundary_refinement requires observations")
        refined, refined_provenance = _boundary_points(config, observations, lower, upper, requested_count, seed)
        points.extend(refined)
        provenance.extend(refined_provenance)

    anchors = config.get("anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("anchors must be a list of five-coordinate points")
    for index, value in enumerate(anchors):
        points.append(_validate_point(value, lower, upper, f"anchors[{index}]"))
        provenance.append({"kind": "anchor", "ordinal": index})
    anchor_grid = _axis_product(config.get("anchor_grid"), lower, upper, "anchor_grid")
    for index, point in enumerate(anchor_grid):
        points.append(point)
        provenance.append({"kind": "anchor_grid", "ordinal": index})

    unique_points: list[np.ndarray] = []
    unique_provenance: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for point, source in zip(points, provenance, strict=True):
        key = tuple(map(float, point))
        if key in seen:
            continue
        seen.add(key)
        unique_points.append(point)
        unique_provenance.append(source)
    if not unique_points:
        raise ValueError("sampling config generated no points")
    if len(unique_points) > MAX_SAMPLE_COUNT:
        raise ValueError(f"generated bank may not exceed {MAX_SAMPLE_COUNT} samples")

    samples = [
        _sample_record(index, point, tilt, source)
        for index, (point, source) in enumerate(zip(unique_points, unique_provenance, strict=True), 1)
    ]
    generation = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "dimensions": list(DIMENSIONS),
        "bounds": bounds,
        "requested_joint_count": requested_count,
        "anchor_count": len(anchors),
        "anchor_grid_count": len(anchor_grid),
        "deduplicated_count": len(points) - len(unique_points),
        "sample_count": len(samples),
        "bodyward_tilt_deg": tilt,
        "config_sha256": _config_hash(config),
        "claim_scope": "points proposed for actual execution; no unsampled cell is classified",
    }
    if observations is not None:
        generation["observations_sha256"] = _config_hash(observations)
        generation["observation_count"] = len(observations.get("observations", []))
    return {
        "schema_version": 1,
        "seed": seed,
        "proposal_status": "joint_5d_trial_points_only",
        "frame": "robot_standing_frame",
        "generation": generation,
        "samples": samples,
    }


def load_observations(path: Path) -> dict:
    return json.loads(path.read_text())


__all__ = [
    "DIMENSIONS",
    "MAX_SAMPLE_COUNT",
    "classify_reach_attempt",
    "generate_joint_bank",
    "load_observations",
]
