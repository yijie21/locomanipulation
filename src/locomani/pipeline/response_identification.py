"""Identify observed joint response to the deploy-selected POSE reference.

The input reference is an asynchronous debug publication.  It identifies the
``current_motion`` frame selected by the deploy, but carries neither the
deploy's publication time nor a controller acknowledgement.  This module pairs
only the nearest preceding agent-receive timestamp with each simulator state,
records its age, and rejects stale, future, non-POSE, or cross-episode data.

The fitted model is deliberately small::

    measured[k] = alpha * measured[k-1]
                  + (1 - alpha) * (reference[k-delay] + bias)

Delay is shared and alpha/bias are per joint.  Fits need excitation and
leave-one-episode-out improvement over a static-bias model.  Rejected fits do
not expose controller parameters.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from locomani.backends.sonic.agent_server import StreamMode
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES, isaac_to_mujoco


class IdentificationInputError(ValueError):
    """Input evidence cannot support response identification."""


@dataclass(frozen=True)
class Episode:
    episode_id: str
    seq: np.ndarray
    simulation_time: np.ndarray
    state_wall: np.ndarray
    reference_received_wall: np.ndarray
    reference_sample_id: np.ndarray
    reference: np.ndarray
    measured: np.ndarray
    pose_send_wall: np.ndarray
    sent_reference: np.ndarray
    sent_frame_index: np.ndarray
    segment_id: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.seq)
        vectors = (
            self.simulation_time,
            self.state_wall,
            self.reference_received_wall,
            self.reference_sample_id,
            self.pose_send_wall,
            self.sent_frame_index,
            self.segment_id,
        )
        if (
            any(len(value) != n for value in vectors)
            or self.reference.shape != (n, 29)
            or self.sent_reference.shape != (n, 29)
            or self.measured.shape
            != (
                n,
                29,
            )
        ):
            raise IdentificationInputError(f"{self.episode_id}: inconsistent episode array shapes")
        if n and (len(np.unique(self.seq)) != n or np.any(np.diff(self.seq) <= 0)):
            raise IdentificationInputError(f"{self.episode_id}: simulator sequence is not unique and increasing")
        if n and (self.segment_id[0] != 0 or np.any(np.diff(self.segment_id) < 0)):
            raise IdentificationInputError(f"{self.episode_id}: segment identity is invalid")
        for segment in _segment_slices(self):
            if segment.stop - segment.start <= 1:
                continue
            if np.any(np.diff(self.seq[segment]) != 1):
                raise IdentificationInputError(f"{self.episode_id}: sequence gap crosses a fitted segment")
            intervals = np.diff(self.simulation_time[segment])
            if np.any(np.abs(intervals - 0.02) > 0.001):
                raise IdentificationInputError(f"{self.episode_id}: fitted segment is not regular 50 Hz physical time")
            if np.any(np.diff(self.reference_sample_id[segment]) < 0):
                raise IdentificationInputError(f"{self.episode_id}: deploy sample identity moved backwards")
            if np.any(np.diff(self.sent_frame_index[segment]) < 0):
                raise IdentificationInputError(f"{self.episode_id}: sent POSE frame index reset inside fitted segment")


def _segment_slices(episode: Episode) -> list[slice]:
    if not len(episode.seq):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(episode.segment_id) != 0) + 1]
    stops = np.r_[starts[1:], len(episode.seq)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _finite_vector(value, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise IdentificationInputError(f"{field} must contain {size} finite values")
    return result


def joint_qpos_indices(scene: Path) -> np.ndarray:
    """Resolve MuJoCo-order body joint qpos addresses from an explicit scene."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise IdentificationInputError("--scene needs the mujoco Python package") from exc

    model = mujoco.MjModel.from_xml_path(str(scene))
    indices = []
    for name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise IdentificationInputError(f"{scene}: missing robot joint {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(indices, dtype=int)


def _trace_by_seq(path: Path, indices: np.ndarray) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as trace:
        missing = {"seq", "qpos"} - set(trace.files)
        if missing:
            raise IdentificationInputError(f"{path}: trace missing {sorted(missing)}")
        seq = np.asarray(trace["seq"], dtype=np.int64)
        qpos = np.asarray(trace["qpos"], dtype=float)
    if seq.ndim != 1 or qpos.ndim != 2 or len(seq) != len(qpos):
        raise IdentificationInputError(f"{path}: invalid trace shapes")
    if len(np.unique(seq)) != len(seq):
        raise IdentificationInputError(f"{path}: duplicate trace sequence crosses reset isolation")
    if indices.shape != (29,) or np.any(indices < 0) or np.any(indices >= qpos.shape[1]):
        raise IdentificationInputError(f"{path}: joint qpos indices do not fit trace width {qpos.shape[1]}")
    return {int(s): qpos[i, indices].copy() for i, s in enumerate(seq)}


def load_episode(
    sidecar_path: Path,
    trace_path: Path,
    indices: np.ndarray,
    *,
    max_reference_age_s: float = 0.08,
    max_pose_send_age_s: float = 0.08,
    joint_match_atol: float = 1e-8,
    episode_id: str | None = None,
) -> Episode:
    """Join one sidecar and one trace strictly within an episode-local seq.

    Reference snapshots are rebuilt by unique sample identity and paired by
    receive wall time with ``bisect_right``. Thus the selected snapshot is
    always the nearest preceding one and can never leak from the future.
    """
    if max_reference_age_s <= 0 or max_pose_send_age_s <= 0:
        raise IdentificationInputError("maximum ages must be positive")
    rows = [json.loads(line) for line in sidecar_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise IdentificationInputError(f"{sidecar_path}: empty sidecar")

    snapshots: dict[int, tuple[float, np.ndarray]] = {}
    for row in rows:
        raw = row.get("deploy_reference")
        if raw is None:
            continue
        if raw.get("source") != "deploy_current_motion_output":
            raise IdentificationInputError(f"{sidecar_path}: unsupported deploy reference source")
        if raw.get("timestamp_semantics") != "agent_receive_wall_no_controller_ack":
            raise IdentificationInputError(f"{sidecar_path}: missing asynchronous receive-time semantics")
        sample_id = int(raw["sample_id"])
        received = float(raw["received_wall"])
        reference = _finite_vector(raw["body_q_target"], 29, "body_q_target")
        if not np.isfinite(received):
            raise IdentificationInputError(f"{sidecar_path}: non-finite reference receive time")
        previous = snapshots.get(sample_id)
        if previous is not None and (previous[0] != received or not np.array_equal(previous[1], reference)):
            raise IdentificationInputError(f"{sidecar_path}: deploy sample identity changed within episode")
        snapshots[sample_id] = received, reference
    if not snapshots:
        raise IdentificationInputError(f"{sidecar_path}: no deploy_reference snapshots; live calibration required")
    ordered = sorted((received, sample_id, reference) for sample_id, (received, reference) in snapshots.items())
    receive_times = [item[0] for item in ordered]
    if len(set(receive_times)) != len(receive_times):
        raise IdentificationInputError(f"{sidecar_path}: ambiguous deploy reference receive timestamps")
    if np.any(np.diff([item[1] for item in ordered]) <= 0):
        raise IdentificationInputError(f"{sidecar_path}: deploy sample identity restarted within episode")

    trace = _trace_by_seq(trace_path, indices)
    samples = []
    seen_seq: set[int] = set()
    for row in rows:
        if row.get("seq") is None or row.get("wall") is None or row.get("t") is None:
            continue
        seq = int(row["seq"])
        if seq in seen_seq:
            raise IdentificationInputError(f"{sidecar_path}: duplicate sidecar sequence crosses reset isolation")
        seen_seq.add(seq)
        measured = trace.get(seq)
        if measured is None:
            continue
        sidecar_joints = _finite_vector(row.get("joints"), 29, "sidecar joints")
        mismatch = float(np.max(np.abs(sidecar_joints - measured)))
        if mismatch > joint_match_atol:
            raise IdentificationInputError(
                f"{sidecar_path}: seq {seq} sidecar/trace joint mismatch {mismatch:.6g} > {joint_match_atol:.6g}"
            )

        command = row.get("reference_command_state") or {}
        pose_send = command.get("pose_send") or {}
        if command.get("control_owner_requested") != "streamed":
            continue
        if command.get("stream_mode_requested") != int(StreamMode.POSE):
            continue
        pose_wall = float(pose_send.get("sent_wall", np.nan))
        state_wall = float(row["wall"])
        if not np.isfinite(pose_wall) or pose_wall > state_wall or state_wall - pose_wall > max_pose_send_age_s:
            continue
        if pose_send.get("timestamp_semantics") != "agent_post_send_wall_no_deploy_ack":
            raise IdentificationInputError(f"{sidecar_path}: POSE send lacks post-send timestamp semantics")
        if pose_send.get("joint_order") != "isaaclab":
            raise IdentificationInputError(f"{sidecar_path}: POSE send joint order is not explicit")
        sent_reference = isaac_to_mujoco(_finite_vector(pose_send.get("joint_pos"), 29, "POSE sent joint_pos"))
        sent_frame_index = int(pose_send["frame_index"])

        index = bisect.bisect_right(receive_times, state_wall) - 1
        if index < 0:
            continue
        received, sample_id, reference = ordered[index]
        age = state_wall - received
        if age < 0:  # guarded by bisect, retained as an invariant check
            raise IdentificationInputError(f"{sidecar_path}: future deploy snapshot selected")
        if age > max_reference_age_s:
            continue
        samples.append(
            (
                seq,
                float(row["t"]),
                state_wall,
                received,
                sample_id,
                reference,
                measured,
                pose_wall,
                sent_reference,
                sent_frame_index,
            )
        )

    if not samples:
        raise IdentificationInputError(f"{sidecar_path}: no fresh streamed-POSE reference/state matches")
    segment_ids = [0]
    for previous, current in zip(samples, samples[1:]):
        contiguous = (
            current[0] == previous[0] + 1
            and abs((current[1] - previous[1]) - 0.02) <= 0.001
            and current[4] >= previous[4]
            and current[9] >= previous[9]
        )
        segment_ids.append(segment_ids[-1] if contiguous else segment_ids[-1] + 1)
    columns = list(zip(*samples, strict=True))
    return Episode(
        episode_id=sidecar_path.stem if episode_id is None else episode_id,
        seq=np.asarray(columns[0], dtype=np.int64),
        simulation_time=np.asarray(columns[1], dtype=float),
        state_wall=np.asarray(columns[2], dtype=float),
        reference_received_wall=np.asarray(columns[3], dtype=float),
        reference_sample_id=np.asarray(columns[4], dtype=np.int64),
        reference=np.stack(columns[5]),
        measured=np.stack(columns[6]),
        pose_send_wall=np.asarray(columns[7], dtype=float),
        sent_reference=np.stack(columns[8]),
        sent_frame_index=np.asarray(columns[9], dtype=np.int64),
        segment_id=np.asarray(segment_ids, dtype=np.int64),
    )


def _fit_at_delay(episodes: Iterable[Episode], delay: int, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs, us = [], [], []
    for episode in episodes:
        for segment in _segment_slices(episode):
            measured, reference = episode.measured[segment], episode.reference[segment]
            if len(measured) <= delay + 1:
                continue
            ys.append(measured[delay + 1 :])
            xs.append(measured[delay:-1])
            us.append(reference[1:-delay] if delay else reference[1:])
    if not ys:
        raise IdentificationInputError("too few samples for delay search")
    y, previous, reference = np.concatenate(ys), np.concatenate(xs), np.concatenate(us)
    alpha = np.zeros(29)
    bias = np.zeros(29)
    for joint in np.flatnonzero(eligible):
        x = previous[:, joint] - reference[:, joint]
        target = y[:, joint] - reference[:, joint]
        centered = x - x.mean()
        denominator = float(centered @ centered)
        a = 0.0 if denominator <= 1e-12 else float(centered @ (target - target.mean()) / denominator)
        a = float(np.clip(a, 0.0, 0.995))
        intercept = float(np.mean(target - a * x))
        alpha[joint] = a
        bias[joint] = intercept / (1.0 - a)
    return alpha, bias


def _fit_static_bias(episodes: Iterable[Episode], eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residuals = [
        episode.measured[segment][1:] - episode.reference[segment][1:]
        for episode in episodes
        for segment in _segment_slices(episode)
        if segment.stop - segment.start > 1
    ]
    if not residuals:
        raise IdentificationInputError("too few samples for static-bias baseline")
    bias = np.zeros(29)
    bias[eligible] = np.concatenate(residuals)[:, eligible].mean(axis=0)
    return np.zeros(29), bias


def _array_errors(
    measured: np.ndarray, reference_input: np.ndarray, delay: int, alpha: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    measured_now = measured[delay + 1 :]
    previous = measured[delay:-1]
    reference = reference_input[1:-delay] if delay else reference_input[1:]
    predicted = alpha * previous + (1.0 - alpha) * (reference + bias)
    return measured_now - predicted, reference


def _errors(episode: Episode, delay: int, alpha: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = [
        _array_errors(episode.measured[segment], episode.reference[segment], delay, alpha, bias)
        for segment in _segment_slices(episode)
        if segment.stop - segment.start > delay + 1
    ]
    if not parts:
        return np.empty((0, 29)), np.empty((0, 29))
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def _score(episodes: Iterable[Episode], delay: int, alpha: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> float:
    errors = [part / scale for episode in episodes if len(part := _errors(episode, delay, alpha, bias)[0])]
    return float(np.mean(np.concatenate(errors) ** 2)) if errors else float("inf")


def _rollout_score(
    episodes: Iterable[Episode],
    delay: int,
    alpha: np.ndarray,
    bias: np.ndarray,
    scale: np.ndarray,
    horizon_steps: int,
) -> float:
    """Fixed-horizon free-run error, re-anchored only at each window start."""
    errors = []
    for episode in episodes:
        for segment in _segment_slices(episode):
            measured, reference_input = episode.measured[segment], episode.reference[segment]
            last_start = len(measured) - horizon_steps
            for start in range(delay, last_start):
                predicted = measured[start].copy()
                for step in range(1, horizon_steps + 1):
                    target = start + step
                    reference = reference_input[target - delay]
                    predicted = alpha * predicted + (1.0 - alpha) * (reference + bias)
                errors.append((measured[start + horizon_steps] - predicted) / scale)
    return float(np.mean(np.asarray(errors) ** 2)) if errors else float("inf")


def _persistence_score(episodes: Iterable[Episode], scale: np.ndarray, horizon_steps: int, *, delay: int) -> float:
    errors = []
    for episode in episodes:
        for segment in _segment_slices(episode):
            measured = episode.measured[segment]
            last_start = len(measured) - horizon_steps
            for start in range(delay, last_start):
                errors.append((measured[start + horizon_steps] - measured[start]) / scale)
    return float(np.mean(np.asarray(errors) ** 2)) if errors else float("inf")


def _axis_identification(eligible: np.ndarray, rejection_reasons: list[str]) -> dict:
    rejected = bool(rejection_reasons)
    return {
        "identified_for_control": [] if rejected else np.flatnonzero(eligible).tolist(),
        "unidentified_for_control": [
            {
                "index": int(i),
                "name": MUJOCO_JOINT_NAMES[i],
                "reasons": (
                    list(rejection_reasons) if eligible[i] else ["insufficient_axis_excitation", *rejection_reasons]
                ),
            }
            for i in range(29)
            if rejected or not eligible[i]
        ],
    }


def identify_response(
    episodes: list[Episode],
    *,
    max_delay_steps: int = 8,
    min_excitation_rad: float = 0.05,
    min_velocity_rms_rad_s: float = 0.02,
    min_eligible_joints: int = 3,
    min_validation_improvement: float = 0.05,
    training_ids: set[str] | None = None,
    heldout_ids: set[str] | None = None,
) -> dict:
    """Fit and validate the response model, or return explicit rejection."""
    reasons = []
    explicit_split = training_ids is not None or heldout_ids is not None
    if explicit_split:
        training_ids = set() if training_ids is None else training_ids
        heldout_ids = set() if heldout_ids is None else heldout_ids
        available = {episode.episode_id for episode in episodes}
        if len(available) != len(episodes):
            reasons.append("loaded_episode_ids_are_not_unique")
        if training_ids & heldout_ids:
            reasons.append("training_and_heldout_episode_ids_overlap")
        if (training_ids | heldout_ids) - available:
            reasons.append("split_references_missing_episode_ids")
        if available - (training_ids | heldout_ids):
            reasons.append("split_does_not_assign_all_loaded_episodes")
        model_training = [episode for episode in episodes if episode.episode_id in training_ids]
        model_heldout = [episode for episode in episodes if episode.episode_id in heldout_ids]
        if len(model_training) < 3 or len(model_heldout) < 2:
            reasons.append("explicit_split_needs_three_training_and_two_heldout_episodes")
    else:
        model_training = episodes
        model_heldout = []
        if len(episodes) < 3:
            reasons.append("need_at_least_three_episodes_for_held_out_validation")
    if max_delay_steps < 0:
        raise ValueError("max_delay_steps must be non-negative")
    if any(
        max((segment.stop - segment.start for segment in _segment_slices(episode)), default=0)
        < max(20, max_delay_steps + 3)
        for episode in episodes
    ):
        reasons.append("insufficient_samples_per_episode")

    references = (
        np.concatenate([episode.reference for episode in model_training]) if model_training else np.empty((0, 29))
    )
    dts = [
        np.diff(ep.simulation_time[segment])
        for ep in episodes
        for segment in _segment_slices(ep)
        if segment.stop - segment.start > 1
    ]
    positive_dt = np.concatenate([dt[dt > 0] for dt in dts]) if dts else np.empty(0)
    dt = float(np.median(positive_dt)) if len(positive_dt) else float("nan")
    episode_dts = np.asarray([np.median(intervals) for intervals in dts if len(intervals)])
    if len(episode_dts) and np.max(np.abs(episode_dts - dt)) > max(1e-6, 0.05 * dt):
        reasons.append("sample_period_differs_across_episodes")
    ranges = np.ptp(references, axis=0) if len(references) else np.zeros(29)
    differences = []
    for episode in model_training:
        for segment in _segment_slices(episode):
            local_dt = np.diff(episode.simulation_time[segment])
            if len(local_dt):
                differences.append(np.diff(episode.reference[segment], axis=0) / local_dt[:, None])
    velocity_rms = np.sqrt(np.mean(np.concatenate(differences) ** 2, axis=0)) if differences else np.zeros(29)
    measured_velocities, measured_oscillations = [], []
    for episode in episodes:
        for segment in _segment_slices(episode):
            local_dt = np.diff(episode.simulation_time[segment])
            if not len(local_dt):
                continue
            velocity = np.diff(episode.measured[segment], axis=0) / local_dt[:, None]
            measured_velocities.append(velocity)
            if len(velocity) >= 5:
                kernel = np.full(5, 0.2)
                trend = np.stack([np.convolve(velocity[:, j], kernel, mode="same") for j in range(29)], axis=1)
                measured_oscillations.append(velocity - trend)
    eligible = (ranges >= min_excitation_rad) & (velocity_rms >= min_velocity_rms_rad_s)
    if int(eligible.sum()) < min_eligible_joints:
        reasons.append("insufficient_reference_excitation")

    report = {
        "episodes": len(episodes),
        "sample_counts": {episode.episode_id: len(episode.seq) for episode in episodes},
        "segments": {
            episode.episode_id: [segment.stop - segment.start for segment in _segment_slices(episode)]
            for episode in episodes
        },
        "timing": {
            "simulation_dt_median_s": dt,
            "reference_timestamp_semantics": "agent_receive_wall_no_controller_ack",
            "pairing": "nearest_preceding_receive_time_no_future_leakage",
            "reference_age_s": _distribution(
                np.concatenate([ep.state_wall - ep.reference_received_wall for ep in episodes])
                if episodes
                else np.empty(0)
            ),
            "pose_send_age_s": _distribution(
                np.concatenate([ep.state_wall - ep.pose_send_wall for ep in episodes]) if episodes else np.empty(0)
            ),
            "sent_to_deploy_selected_joint_gap_rad": _distribution(
                np.concatenate([ep.sent_reference - ep.reference for ep in episodes]) if episodes else np.empty(0)
            ),
            "sent_to_deploy_note": (
                "last preceding agent post-send current sample versus last preceding deploy-selected-frame receive; "
                "asynchronous snapshots, not a same-tick consumption residual"
            ),
        },
        "excitation": {
            "minimum_range_rad": min_excitation_rad,
            "minimum_velocity_rms_rad_s": min_velocity_rms_rad_s,
            "eligible_joint_indices": np.flatnonzero(eligible).tolist(),
            "range_rad": ranges.tolist(),
            "velocity_rms_rad_s": velocity_rms.tolist(),
            "unidentified_joints": [
                {
                    "index": int(i),
                    "name": MUJOCO_JOINT_NAMES[i],
                    "reasons": [
                        *(["range_below_threshold"] if ranges[i] < min_excitation_rad else []),
                        *(["velocity_rms_below_threshold"] if velocity_rms[i] < min_velocity_rms_rad_s else []),
                    ],
                }
                for i in np.flatnonzero(~eligible)
            ],
        },
        "signed_measured_velocity": {
            "mean_rad_s": (
                np.mean(np.concatenate(measured_velocities), axis=0).tolist() if measured_velocities else [0.0] * 29
            ),
            "rms_rad_s": (
                np.sqrt(np.mean(np.concatenate(measured_velocities) ** 2, axis=0)).tolist()
                if measured_velocities
                else [0.0] * 29
            ),
            "five_sample_detrended_rms_rad_s": (
                np.sqrt(np.mean(np.concatenate(measured_oscillations) ** 2, axis=0)).tolist()
                if measured_oscillations
                else [0.0] * 29
            ),
            "note": "signed joint velocities; no rectification or stopping-deceleration inference",
        },
    }
    if reasons:
        report["axis_identification"] = _axis_identification(eligible, reasons)
        return {"identified": False, "rejection_reasons": reasons, **report}

    scale = np.maximum(ranges, min_excitation_rad)
    scale[~eligible] = np.inf
    validation_horizon_steps = max(2, int(round(0.2 / dt)))
    folds = []
    delay_jackknife = []

    def select(training: list[Episode]):
        candidates = []
        for delay in range(max_delay_steps + 1):
            alpha, bias = _fit_at_delay(training, delay, eligible)
            candidates.append(
                (
                    _rollout_score(training, delay, alpha, bias, scale, validation_horizon_steps),
                    delay,
                    alpha,
                    bias,
                )
            )
        return min(candidates, key=lambda item: item[0])

    if explicit_split:
        _, selected_delay, selected_alpha, selected_bias = select(model_training)
        static_alpha, static_bias = _fit_static_bias(model_training, eligible)
        for index in range(len(model_training)):
            jackknife = [episode for i, episode in enumerate(model_training) if i != index]
            delay_jackknife.append(int(select(jackknife)[1]))
        validation_pairs = [(model_training, holdout) for holdout in model_heldout]
    else:
        validation_pairs = [
            ([episode for i, episode in enumerate(episodes) if i != holdout_index], holdout)
            for holdout_index, holdout in enumerate(episodes)
        ]

    for training, holdout in validation_pairs:
        if explicit_split:
            delay, alpha, bias = selected_delay, selected_alpha, selected_bias
        else:
            _, delay, alpha, bias = select(training)
            delay_jackknife.append(int(delay))
        teacher_forced_score = _score([holdout], delay, alpha, bias, scale)
        dynamic_score = _rollout_score([holdout], delay, alpha, bias, scale, validation_horizon_steps)
        if not explicit_split:
            static_alpha, static_bias = _fit_static_bias(training, eligible)
        static_score = _rollout_score([holdout], 0, static_alpha, static_bias, scale, validation_horizon_steps)
        persistence_score = _persistence_score([holdout], scale, validation_horizon_steps, delay=delay)
        static_improvement = 1.0 - dynamic_score / max(static_score, 1e-15)
        persistence_improvement = 1.0 - dynamic_score / max(persistence_score, 1e-15)
        folds.append(
            {
                "holdout": holdout.episode_id,
                "selected_delay_steps": delay,
                "teacher_forced_one_step_normalized_mse_descriptive_only": teacher_forced_score,
                "free_run_horizon_steps": validation_horizon_steps,
                "free_run_horizon_s": validation_horizon_steps * dt,
                "dynamic_free_run_normalized_mse": dynamic_score,
                "static_bias_free_run_normalized_mse": static_score,
                "persistence_free_run_normalized_mse": persistence_score,
                "static_bias_improvement_fraction": static_improvement,
                "persistence_improvement_fraction": persistence_improvement,
            }
        )

    static_improvements = np.asarray([fold["static_bias_improvement_fraction"] for fold in folds])
    persistence_improvements = np.asarray([fold["persistence_improvement_fraction"] for fold in folds])
    delays = np.asarray(delay_jackknife)
    delay_values, delay_counts = np.unique(delays, return_counts=True)
    consensus = float(delay_counts.max() / len(delays))
    if float(np.median(static_improvements)) < min_validation_improvement:
        reasons.append("held_out_static_bias_improvement_below_threshold")
    if float(np.median(persistence_improvements)) < min_validation_improvement:
        reasons.append("held_out_persistence_improvement_below_threshold")
    repeatable = (static_improvements > 0) & (persistence_improvements > 0)
    if int(np.sum(repeatable)) < int(np.ceil(2 * len(folds) / 3)):
        reasons.append("held_out_free_run_improvement_not_repeatable")
    if consensus < 0.5:
        reasons.append("delay_not_consistent_across_held_out_episodes")
    selected_delay = int(select(model_training)[1]) if explicit_split else int(delay_values[np.argmax(delay_counts)])
    if selected_delay == max_delay_steps and max_delay_steps > 0:
        reasons.append("delay_at_search_boundary")

    report["validation"] = {
        "method": "prescribed_episode_holdout" if explicit_split else "leave_one_episode_out",
        "training_episode_ids": [episode.episode_id for episode in model_training],
        "heldout_episode_ids": (
            [episode.episode_id for episode in model_heldout]
            if explicit_split
            else [episode.episode_id for episode in episodes]
        ),
        "delay_jackknife_steps": delay_jackknife,
        "minimum_improvement_fraction": min_validation_improvement,
        "folds": folds,
        "median_static_bias_improvement_fraction": float(np.median(static_improvements)),
        "median_persistence_improvement_fraction": float(np.median(persistence_improvements)),
        "positive_against_both_fold_fraction": float(np.mean(repeatable)),
        "delay_consensus_fraction": consensus,
    }
    if reasons:
        report["axis_identification"] = _axis_identification(eligible, reasons)
        return {"identified": False, "rejection_reasons": reasons, **report}

    alpha, bias = _fit_at_delay(model_training, selected_delay, eligible)
    residuals, delayed_refs = zip(*[_errors(ep, selected_delay, alpha, bias) for ep in model_training], strict=True)
    episode_bias = np.stack([residual.mean(axis=0) for residual in residuals])
    episode_posture = np.stack([reference.mean(axis=0) for reference in delayed_refs])
    posture_correlation = np.zeros(29)
    for joint in np.flatnonzero(eligible):
        if np.std(episode_posture[:, joint]) > 1e-9 and np.std(episode_bias[:, joint]) > 1e-9:
            posture_correlation[joint] = np.corrcoef(episode_posture[:, joint], episode_bias[:, joint])[0, 1]
    phase_correlation = np.zeros(29)
    for joint in np.flatnonzero(eligible):
        velocity_parts, residual_parts = [], []
        for episode in model_training:
            for segment in _segment_slices(episode):
                if segment.stop - segment.start <= selected_delay + 2:
                    continue
                residual, ref = _array_errors(
                    episode.measured[segment], episode.reference[segment], selected_delay, alpha, bias
                )
                velocity_parts.append(np.diff(ref[:, joint]))
                residual_parts.append(residual[1:, joint])
        velocity = np.concatenate(velocity_parts)
        residual = np.concatenate(residual_parts)
        if np.std(velocity) > 1e-9 and np.std(residual) > 1e-9:
            phase_correlation[joint] = np.corrcoef(velocity, residual)[0, 1]

    time_constant = np.zeros(29)
    valid_alpha = eligible & (alpha > 0)
    time_constant[valid_alpha] = -dt / np.log(alpha[valid_alpha])
    report.update(
        identified=True,
        rejection_reasons=[],
        axis_identification=_axis_identification(eligible, []),
        model={
            "equation": "y[k]=alpha*y[k-1]+(1-alpha)*(u[k-delay]+bias)",
            "delay_steps": selected_delay,
            "delay_seconds": selected_delay * dt,
            "joints": [
                {
                    "index": int(i),
                    "name": MUJOCO_JOINT_NAMES[i],
                    "alpha": float(alpha[i]),
                    "time_constant_s": float(time_constant[i]),
                    "static_bias_rad": float(bias[i]),
                }
                for i in np.flatnonzero(eligible)
            ],
            "unidentified_joints": report["excitation"]["unidentified_joints"],
        },
        residual_diagnostics={
            "source": "training episodes only" if explicit_split else "all episodes after leave-one-out validation",
            "response_residual_rad": _distribution(np.concatenate(residuals)[:, eligible]),
            "between_episode_bias_rms_rad": float(np.sqrt(np.mean(episode_bias[:, eligible] ** 2))),
            "posture_bias_correlation": posture_correlation.tolist(),
            "max_abs_posture_bias_correlation": float(np.max(np.abs(posture_correlation[eligible]))),
            "phase_residual_correlation": phase_correlation.tolist(),
            "max_abs_phase_residual_correlation": float(np.max(np.abs(phase_correlation[eligible]))),
        },
    )
    return report


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None, "rms": None}
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def ik_residual_report(manifest_paths: list[Path]) -> dict:
    """Report solver residuals independently from observed tracking response."""
    episodes, hand, posture = [], [], []
    for path in manifest_paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            samples = row.get("execution", {}).get("metrics", {}).get("whole_body", {}).get("samples", [])
            local_hand, local_posture = [], []
            for sample in samples:
                if "hand_error" in sample:
                    local_hand.extend(np.asarray(sample["hand_error"], dtype=float).ravel().tolist())
                if "posture_residual_norm" in sample:
                    local_posture.append(float(sample["posture_residual_norm"]))
            hand.extend(local_hand)
            posture.extend(local_posture)
            episodes.append(
                {
                    "source": f"{path}:{line_number}",
                    "case_id": row.get("case_id"),
                    "hand_error_samples": len(local_hand),
                    "posture_residual_samples": len(local_posture),
                }
            )
    return {
        "provenance": {
            "hand_error": "SupportedMotion solver-reported hand residual; separate from measured tracking response",
            "posture_residual_norm": "SupportedMotion posture objective residual; separate from measured tracking response",
        },
        "episodes": episodes,
        "hand_error": _distribution(np.asarray(hand)),
        "posture_residual_norm": _distribution(np.asarray(posture)),
    }


def _file_provenance(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _json_argument(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_argument(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, help="collection root with collect/, scene/, and ordered rounds")
    parser.add_argument("--sidecar", action="append", type=Path, default=[], help="one per episode, in pair order")
    parser.add_argument("--trace", action="append", type=Path, default=[], help="one per episode, in pair order")
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="optional IK residual manifests")
    parser.add_argument("--case-id", action="append", default=[], help="one per sidecar/trace pair")
    parser.add_argument("--split", type=Path, help="JSON train_case_ids/held_out_case_ids split")
    index = parser.add_mutually_exclusive_group()
    index.add_argument("--scene", type=Path, help="MuJoCo XML used to resolve body joint qpos addresses")
    index.add_argument("--joint-qpos-indices", help="explicit comma-separated 29 qpos addresses")
    parser.add_argument("--max-reference-age", type=float, default=0.08)
    parser.add_argument("--max-pose-send-age", type=float, default=0.08)
    parser.add_argument("--max-delay-steps", type=int, default=8)
    parser.add_argument("--min-excitation-rad", type=float, default=0.05)
    parser.add_argument("--min-velocity-rms", type=float, default=0.02)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.run_root is not None:
        if args.sidecar or args.trace or args.case_id:
            raise SystemExit("--run-root cannot be combined with explicit --sidecar/--trace/--case-id")
        collect = args.run_root / "collect"
        args.sidecar = sorted((collect / "pose_sidecar").glob("*.jsonl"))
        trace_dirs = sorted(collect.glob("state_trace_*"))
        args.trace = [path for directory in trace_dirs for path in sorted(directory.glob("*.npz"))]
        run_manifests = sorted(collect.glob("manifest_*.jsonl"))
        if not args.manifest:
            args.manifest = run_manifests
        args.case_id = [
            json.loads(line)["case_id"]
            for path in run_manifests
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if args.scene is None and args.joint_qpos_indices is None:
            args.scene = args.run_root / "scene" / "scene_locomani.xml"
    if len(args.sidecar) != len(args.trace):
        raise SystemExit("--sidecar and --trace must be repeated equally and are paired in explicit order")
    if args.case_id and len(args.case_id) != len(args.sidecar):
        raise SystemExit("--case-id must be omitted or repeated once per sidecar/trace pair")
    if args.scene:
        indices = joint_qpos_indices(args.scene)
    elif args.joint_qpos_indices:
        indices = np.asarray([int(value) for value in args.joint_qpos_indices.split(",")], dtype=int)
        if indices.shape != (29,):
            raise SystemExit("--joint-qpos-indices must contain exactly 29 values")
    else:
        raise SystemExit("provide --scene or --joint-qpos-indices")

    episodes, input_rejections = [], []
    case_ids = args.case_id or [None] * len(args.sidecar)
    for sidecar, trace, case_id in zip(args.sidecar, args.trace, case_ids, strict=True):
        try:
            episodes.append(
                load_episode(
                    sidecar,
                    trace,
                    indices,
                    max_reference_age_s=args.max_reference_age,
                    max_pose_send_age_s=args.max_pose_send_age,
                    episode_id=case_id,
                )
            )
        except (IdentificationInputError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            input_rejections.append({"sidecar": str(sidecar), "trace": str(trace), "reason": str(exc)})

    split = None if args.split is None else json.loads(args.split.read_text())
    fit = identify_response(
        episodes,
        max_delay_steps=args.max_delay_steps,
        min_excitation_rad=args.min_excitation_rad,
        min_velocity_rms_rad_s=args.min_velocity_rms,
        training_ids=None if split is None else set(split["train_case_ids"]),
        heldout_ids=None if split is None else set(split["held_out_case_ids"]),
    )
    if input_rejections:
        fit = dict(fit)
        fit["identified"] = False
        fit.pop("model", None)
        fit.setdefault("rejection_reasons", []).append("one_or_more_episode_inputs_rejected")
        fit["input_rejections"] = input_rejections
    module = Path(__file__)
    report = {
        "schema_version": 1,
        "reference_claim": "asynchronous deploy-selected-frame publication, not exact controller consumption",
        "response_fit": fit,
        "ik_optimization_residual": ik_residual_report(args.manifest),
        "provenance": {
            "module": _file_provenance(module),
            "sidecars": [_file_provenance(path) for path in args.sidecar],
            "traces": [_file_provenance(path) for path in args.trace],
            "manifests": [_file_provenance(path) for path in args.manifest],
            "scene": None if args.scene is None else _file_provenance(args.scene),
            "split": None if args.split is None else _file_provenance(args.split),
            "arguments": {key: _json_argument(value) for key, value in vars(args).items()},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"response identification report -> {args.out}")
    print("IDENTIFIED" if fit["identified"] else f"REJECTED: {fit.get('rejection_reasons')}")
    return 0 if fit["identified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
