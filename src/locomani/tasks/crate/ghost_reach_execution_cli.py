"""Child process for resumable floor-only dynamic ghost reach probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from locomani.backends.sonic.agent_server import HAND_OPEN
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.backends.sonic.primitives import Primitives
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.box_rim_execution import SonicBoxRimRuntime
from locomani.tasks.crate.expert import prepare_stationary_ready
from locomani.tasks.crate.ghost_reach_execution import (
    ENVELOPE_FIELDS,
    PALM_PITCH_DEG,
    PREVIEW_TILT_DEG,
    TRACKING_FIELDS,
    GhostReachCriteria,
    GhostReachExecutor,
    MeasuredCollisionChecker,
    load_source_cases,
    original_reach_control_config,
    summarize_cases,
)
from locomani.tasks.crate.round_runner import prime

TOP_LEVEL_FIELDS = {"schema_version", "source_targets", "request_identity", "seed", "runtime", "experiment"}
OPTIONAL_TOP_LEVEL_FIELDS = {"envelope"}
RUNTIME_FIELDS = {"backend", "gpu", "instance", "scene_dir", "video_label_prefix"}
EXPERIMENT_FIELDS = set(GhostReachCriteria.__dataclass_fields__)


def _strict_keys(raw: dict, allowed: set[str], required: set[str], name: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(raw) - allowed)
    missing = sorted(required - set(raw))
    if unknown:
        raise ValueError(f"{name} unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_execution_request(path: Path) -> dict:
    request_path = path.resolve()
    raw = json.loads(request_path.read_text())
    _strict_keys(raw, TOP_LEVEL_FIELDS | OPTIONAL_TOP_LEVEL_FIELDS, TOP_LEVEL_FIELDS, "request")
    if raw["schema_version"] != 1:
        raise ValueError("request.schema_version must be 1")
    if not isinstance(raw["request_identity"], str) or not raw["request_identity"].strip():
        raise ValueError("request_identity must be a non-empty string")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ValueError("seed must be an integer")
    runtime = raw["runtime"]
    _strict_keys(runtime, RUNTIME_FIELDS, {"backend", "gpu", "instance", "scene_dir"}, "runtime")
    if not isinstance(runtime["instance"], int) or isinstance(runtime["instance"], bool) or runtime["instance"] < 0:
        raise ValueError("runtime.instance must be a non-negative integer")
    if not isinstance(runtime["gpu"], (str, int)) or isinstance(runtime["gpu"], bool):
        raise ValueError("runtime.gpu must be a string or integer")
    backend = SonicBackendConfig.from_dict(runtime["backend"])
    scene_dir = Path(runtime["scene_dir"])
    if not scene_dir.is_absolute():
        scene_dir = (request_path.parent / scene_dir).resolve()
    else:
        scene_dir = scene_dir.resolve()
    scene_xml = scene_dir / "scene_locomani.xml"
    scene_meta_path = scene_dir / "scene_meta.json"
    if not scene_xml.is_file() or not scene_meta_path.is_file():
        raise ValueError("runtime.scene_dir must contain scene_locomani.xml and scene_meta.json")
    scene_meta = json.loads(scene_meta_path.read_text())
    if scene_meta.get("objects") != {}:
        raise ValueError("ghost reach requires scene_meta.objects={} (no physical target)")
    source_targets = Path(raw["source_targets"])
    if not source_targets.is_absolute():
        source_targets = (request_path.parent / source_targets).resolve()
    else:
        source_targets = source_targets.resolve()
    cases, source_identity = load_source_cases(source_targets)
    experiment = raw["experiment"]
    _strict_keys(experiment, EXPERIMENT_FIELDS, set(), "experiment")
    criteria = GhostReachCriteria(**experiment)
    envelope = raw.get("envelope", {})
    _strict_keys(envelope, set(ENVELOPE_FIELDS | TRACKING_FIELDS), set(), "envelope")
    for name, value in envelope.items():
        if name == "base_anchor":
            continue  # validated by CrateCarryConfig
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"envelope.{name} must be a number")
    control_config = original_reach_control_config(envelope=envelope)  # validates the bounds now
    label_prefix = runtime.get("video_label_prefix", "ghost_reach")
    if (
        not isinstance(label_prefix, str)
        or not label_prefix
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label_prefix)
    ):
        raise ValueError("runtime.video_label_prefix must be a non-empty filename-safe string")
    return {
        "path": request_path,
        "sha256": _sha256(request_path),
        "schema_version": 1,
        "request_identity": raw["request_identity"],
        "seed": raw["seed"],
        "source_targets": source_targets,
        "source_identity": source_identity,
        "cases": cases,
        "criteria": criteria,
        "envelope": dict(envelope),
        "control_config": control_config,
        "runtime": {
            "backend": backend,
            "gpu": str(runtime["gpu"]),
            "instance": runtime["instance"],
            "scene_dir": scene_dir,
            "scene_xml": scene_xml,
            "scene_meta_path": scene_meta_path,
            "scene_meta": scene_meta,
            "video_label_prefix": label_prefix,
        },
    }


def _selected_cases(
    request: dict,
    case_index: int | None = None,
    selection: dict | None = None,
) -> list:
    cases = request["cases"]
    if case_index is not None and selection is not None:
        raise ValueError("case_index and selection are mutually exclusive")
    if selection is not None:
        indices = selection["indices"]
        by_index = {case.index: case for case in cases}
        if not indices:
            raise ValueError("case selection must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("case selection indices must be unique")
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise ValueError(f"source targets do not contain case indices {missing}")
        return [by_index[index] for index in indices]
    if case_index is None:
        return cases
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 1 <= case_index <= len(cases):
        raise ValueError(f"case_index must be an integer from 1 through {len(cases)}")
    selected = [case for case in cases if case.index == case_index]
    if len(selected) != 1:
        raise ValueError(f"source targets do not contain case index {case_index}")
    return selected


def _request_control_config(request: dict):
    """Requests built by hand (tests) may omit the resolved config; rebuild it from the envelope."""
    if "control_config" in request:
        return request["control_config"]
    return original_reach_control_config(envelope=request.get("envelope", {}))


def request_description(
    request: dict,
    output: Path,
    *,
    case_index: int | None = None,
    selection: dict | None = None,
) -> dict:
    runtime = request["runtime"]
    selected = _selected_cases(request, case_index, selection)
    selected_tilts = [case.bodyward_tilt_deg for case in selected]
    one_tilt = len(set(selected_tilts)) == 1
    owner_dir = Path(__file__).resolve().parent
    implementation_files = (
        "ghost_reach_execution_cli.py",
        "ghost_reach_execution.py",
        "coordinated_reach.py",
        "expert.py",
        "bimanual_ik.py",
        "box_rim_execution.py",
        "visual_config.py",
        "carry_config.py",
    )
    return {
        "request": str(request["path"]),
        "request_sha256": request["sha256"],
        "request_identity": request["request_identity"],
        "source_targets": request["source_identity"],
        "scene_dir": str(runtime["scene_dir"]),
        "scene_xml_sha256": _sha256(runtime["scene_xml"]),
        "scene_meta_sha256": _sha256(runtime["scene_meta_path"]),
        "scene_meta": runtime["scene_meta"],
        "instance": runtime["instance"],
        "gpu": runtime["gpu"],
        "output": str(output.resolve()),
        "source_case_count": len(request["cases"]),
        "case_count": len(selected),
        "selected_case_index": case_index,
        "selection": selection
        or {
            "mode": "case_index" if case_index is not None else "all",
            "indices": [case.index for case in selected],
        },
        "criteria": request["criteria"].__dict__,
        "controller": {
            "owner": "CoordinatedReach/WholeBodyIK",
            "resolved_original_reach_control_config": asdict(_request_control_config(request)),
            "envelope": request.get("envelope", {}),
            "resolved_backend_config": runtime["backend"].to_dict(),
            "implementation_sha256": {name: _sha256(owner_dir / name) for name in implementation_files},
            "policy_path": os.environ.get("SONIC_POLICY"),
            "reach_only_extension": True,
            "palm_pitch_deg": 90.0 - selected_tilts[0] if one_tilt else None,
            "preview_bodyward_tilt_deg": selected_tilts[0] if one_tilt else None,
            "default_palm_pitch_deg": PALM_PITCH_DEG,
            "default_bodyward_tilt_deg": PREVIEW_TILT_DEG,
            "per_case_bodyward_tilt_deg": {str(case.index): case.bodyward_tilt_deg for case in selected},
            "tilt_semantics": "fixed for each case; no in-motion tilt strategy",
            "hands": "open",
            "walking_commands": "forbidden",
        },
    }


def resolve_case_selection(
    request: dict,
    *,
    case_index: int | None = None,
    case_range: str | None = None,
    case_indices_path: Path | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict:
    """Resolve one stable subset while retaining global source indices."""
    shard_requested = shard_index is not None or shard_count is not None
    mode_count = sum(value is not None for value in (case_index, case_range, case_indices_path))
    mode_count += int(shard_requested)
    if mode_count > 1:
        raise ValueError("case-index, case-range, case-indices, and shard selection are mutually exclusive")

    all_indices = [case.index for case in request["cases"]]
    if case_index is not None:
        selection = {"mode": "case_index", "case_index": case_index, "indices": [case_index]}
    elif case_range is not None:
        parts = case_range.split(":")
        if len(parts) != 2:
            raise ValueError("case_range must use inclusive START:STOP syntax")
        try:
            start, stop = (int(value) for value in parts)
        except ValueError as error:
            raise ValueError("case_range endpoints must be integers") from error
        if start < 1 or stop < start:
            raise ValueError("case_range must satisfy 1 <= START <= STOP")
        selection = {"mode": "case_range", "start": start, "stop": stop, "indices": list(range(start, stop + 1))}
    elif case_indices_path is not None:
        path = case_indices_path.resolve()
        text = path.read_text()
        try:
            decoded = json.loads(text)
            decoded_from_json = True
        except json.JSONDecodeError:
            decoded = [line.strip() for line in text.splitlines() if line.strip()]
            decoded_from_json = False
        if not isinstance(decoded, list):
            raise ValueError("case_indices file must be a JSON list or contain one integer per line")
        if decoded_from_json and any(not isinstance(value, int) or isinstance(value, bool) for value in decoded):
            raise ValueError("case_indices entries must be integers")
        try:
            indices = [int(value) for value in decoded]
        except (TypeError, ValueError) as error:
            raise ValueError("case_indices entries must be integers") from error
        selection = {
            "mode": "case_indices",
            "path": str(path),
            "sha256": _sha256(path),
            "indices": indices,
        }
    elif shard_requested:
        if shard_index is None or shard_count is None:
            raise ValueError("shard_index and shard_count must be provided together")
        if isinstance(shard_index, bool) or isinstance(shard_count, bool):
            raise ValueError("shard_index and shard_count must be integers")
        if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("sharding requires SHARD_COUNT >= 1 and 0 <= SHARD_INDEX < SHARD_COUNT")
        indices = [index for index in all_indices if (index - 1) % shard_count == shard_index]
        selection = {
            "mode": "shard",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "indices": indices,
        }
    else:
        selection = {"mode": "all", "indices": all_indices}

    _selected_cases(request, selection=selection)
    return selection


class SonicGhostReachRuntime(SonicBoxRimRuntime):
    """POSE-only runtime whose public walk seam fails closed."""

    def __init__(self, primitives, ik, *, planner_arm):
        self.walk_command_count = 0
        super().__init__(
            primitives,
            ik,
            approach_mode="coordinated",
            initial_hands=HAND_OPEN,
            planner_arm=planner_arm,
        )

    def walk_step(self, target_xy, target_yaw, speed) -> None:
        self.walk_command_count += 1
        raise RuntimeError("walking commands are forbidden during ghost reach")

    def safe_stop(self) -> None:
        if self.pose_arm is not None:
            self.pose_arm.hold(0.3, reference_velocity=np.zeros(29, dtype=float))
            self.pose_arm.stop(preserve_reference=True)
            self.pose_arm = None
        else:
            self.stop_walk()


class _PhaseRecorder:
    def __init__(self, primitives, report: dict, reset_labels: dict[str, str] | None = None):
        self.primitives = primitives
        self.report = report
        self.reset_labels = reset_labels or {}
        self.report.setdefault("phase_events", [])
        self.report.setdefault("resets", [])
        self.origin_seq = None
        self.origin_sim_time = None

    def _observation(self) -> dict | None:
        try:
            return self.primitives.obs()
        except Exception:  # noqa: BLE001 - phase metadata is best-effort evidence
            return None

    def record(self, phase: str, details: dict | None = None, *, observation: dict | None = None) -> None:
        details = dict(details or {})
        observation = self._observation() if observation is None else observation
        seq = observation.get("seq") if observation is not None else None
        sim_time = observation.get("t") if observation is not None else None
        if phase == "recording_started":
            self.origin_seq = seq
            self.origin_sim_time = sim_time
            self.report["recording_origin_wall_time_s"] = time.time()
            self.report["recording_origin_seq"] = seq
            self.report["recording_origin_sim_time_s"] = sim_time
        event = {
            "phase": phase,
            "wall_time_s": time.time(),
            "details": details,
        }
        if seq is not None:
            event["seq"] = seq
            if self.origin_seq is not None:
                event["recording_relative_seq"] = int(seq) - int(self.origin_seq)
        if sim_time is not None:
            event["sim_time_s"] = sim_time
        self.report["phase_events"].append(event)
        if phase == "reset_commanded":
            reset_id = details["reset_id"]
            reset_label = details.get("label") or self.reset_labels.get(reset_id, "")
            reset_event = {
                "reset_id": reset_id,
                "label": reset_label,
                "command_wall_time_s": event["wall_time_s"],
            }
            if seq is not None:
                reset_event["command_seq"] = seq
                if self.origin_seq is not None:
                    reset_event["recording_relative_seq"] = int(seq) - int(self.origin_seq)
            if sim_time is not None:
                reset_event["command_sim_time_s"] = sim_time
            self.report["resets"].append(reset_event)


def _initialize_trial(
    primitives,
    reset,
    ik,
    criteria: GhostReachCriteria,
    label: str,
    initialization: dict | None = None,
):
    """Reuse the visual expert's complete whole-body stationary preparation."""
    del criteria  # admission thresholds and timing come from the original visual profile
    initialization = {} if initialization is None else initialization
    reset_labels = {
        "initial_placement": f"{label}_initial_placement",
        "prepared_placement": f"{label}_prepared_placement",
    }
    recorder = _PhaseRecorder(primitives, initialization, reset_labels)

    def placement(reset_id: str) -> None:
        reset.reset(
            robot={"dxy": [0.0, 0.0], "yaw": 0.0},
            label=reset_labels[reset_id],
        )

    config = original_reach_control_config().visual
    planner_arm, preparation = prepare_stationary_ready(
        primitives,
        reset,
        config,
        BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all"),
        placement=placement,
        whole_body=True,
        on_phase=recorder.record,
    )
    for side in ("left", "right"):
        if not np.allclose(planner_arm.hands[side], HAND_OPEN):
            raise RuntimeError(f"prepared PlannerArm {side} hand is not open")
    state = preparation.pop("admission_state")
    ready_reference = np.asarray(preparation.pop("ready_reference"), dtype=float)
    runtime = SonicGhostReachRuntime(primitives, ik, planner_arm=planner_arm)
    if state is None:
        state = runtime.observe()

    origin_seq = initialization.get("recording_origin_seq")
    ready_seq = state.get("seq")
    duration_from_seq = (
        (int(ready_seq) - int(origin_seq)) / 50.0
        if ready_seq is not None and origin_seq is not None and int(ready_seq) >= int(origin_seq)
        else None
    )
    initialization.update(
        {
            **preparation,
            "status": "ready" if preparation["initial_stance_ready"] else "failed",
            "reason": None if preparation["initial_stance_ready"] else "initial_stance_unstable",
            "pins_released_before_quiet_admission": True,
            "first_observed_seq": initialization.get("recording_origin_seq"),
            "ready_seq": state.get("seq"),
            "ready_sim_time_s": state.get("t"),
            "initialization_duration_sim_s": duration_from_seq,
            "initialization_duration_source": "native_seq_50hz" if duration_from_seq is not None else "unavailable",
            "initialization_duration_wall_s": (
                time.time() - float(initialization["recording_origin_wall_time_s"])
                if initialization.get("recording_origin_wall_time_s") is not None
                else None
            ),
            "measured_joints": state["joints"],
            "measured_pelvis": state["pelvis"],
            "ready_reference": ready_reference.tolist(),
        }
    )
    return runtime, state, initialization


def _prime_session(primitives, reset, output: Path, label: str) -> dict:
    report: dict = {
        "status": "running",
        "artifacts": _artifact_paths(output, label),
        "phase_events": [],
        "resets": [],
    }
    recorder = _PhaseRecorder(primitives, report)
    video_started = False
    try:
        reset.start_video(label)
        video_started = True
        recorder.record("recording_started")
        prime(
            primitives,
            reset,
            reset_label=f"{label}_reset",
            on_phase=recorder.record,
        )
        report["status"] = "ready"
    except Exception as error:  # noqa: BLE001 - preserve prime failure evidence
        report.update(
            status="failed",
            reason=str(error),
            error=f"{type(error).__name__}: {error}",
            last_observation=_initialization_observation(primitives),
        )
    finally:
        if video_started:
            recorder.record("recording_ended")
            try:
                reset.end_video()
            except Exception as error:  # noqa: BLE001 - retain primary failure
                report["video_finalization_error"] = f"{type(error).__name__}: {error}"
    return report


def _fresh_observation_after(primitives, previous: dict, *, timeout_s: float = 2.0) -> dict:
    """Wait for a native state published after a queued recording command."""
    previous_seq = previous.get("seq")
    previous_time = previous.get("t")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = primitives.obs()
        seq = state.get("seq")
        sim_time = state.get("t")
        if previous_seq is not None and seq is not None and int(seq) > int(previous_seq):
            return state
        if previous_seq is None and previous_time is not None and sim_time is not None and sim_time > previous_time:
            return state
        time.sleep(0.005)
    raise TimeoutError("native state did not advance across recording command")


def _start_video_confirmed(
    primitives,
    reset,
    label: str,
    report: dict,
    previous: dict,
    raw_video: Path,
    *,
    ready_config=None,
) -> dict:
    """Start a recorder and return a later native state known to be in its span."""
    if ready_config is not None and not _quiet_ready(previous, ready_config):
        raise RuntimeError("quiet ready state was lost before task recording")
    raw_video = Path(raw_video)
    prior_signature = None
    if raw_video.exists():
        stat = raw_video.stat()
        prior_signature = (stat.st_mtime_ns, stat.st_size)
    reset.start_video(label)
    report["recording_start_command_sent"] = True
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if raw_video.exists():
            stat = raw_video.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if stat.st_size > 0 and signature != prior_signature:
                break
        time.sleep(0.01)
    else:
        raise TimeoutError(f"sim recorder did not open {raw_video}")
    visible_state = primitives.obs()
    state = _fresh_observation_after(primitives, visible_state)
    if ready_config is not None and not _quiet_ready(state, ready_config):
        raise RuntimeError("task recording started from an unstable state")
    recorder = _PhaseRecorder(primitives, report)
    recorder.record("recording_started", {"label": label}, observation=state)
    report["recording_start_confirmed_seq"] = state.get("seq")
    report["recording_start_confirmed_sim_time_s"] = state.get("t")
    return state


def _end_video_confirmed(primitives, reset, report: dict) -> dict:
    """Finalise the current recorder before another label may be started."""
    previous = primitives.obs()
    recorder = _PhaseRecorder(primitives, report)
    recorder.origin_seq = report.get("recording_origin_seq")
    recorder.origin_sim_time = report.get("recording_origin_sim_time_s")
    recorder.record("recording_end_requested", observation=previous)
    reset.end_video()
    state = _fresh_observation_after(primitives, previous)
    recorder.record("recording_ended", observation=state)
    report["recording_end_confirmed_seq"] = state.get("seq")
    report["recording_end_confirmed_sim_time_s"] = state.get("t")
    return state


def _quiet_ready(state: dict, config) -> bool:
    _w, x, y, _z = state["pelvis"]["quat"]
    upright = 1 - 2 * (x * x + y * y) >= np.cos(np.radians(config.upright_tolerance_deg))
    return bool(upright and state["pelvis"]["pos"][2] >= 0.72 and np.linalg.norm(state["pelvis"]["vel"]) < 0.06)


def _artifact_paths(output: Path, label: str) -> dict:
    video_dir = output.resolve().parent / "videos"
    return {
        "label": label,
        "raw_video": str(video_dir / f"{label}.mp4"),
        "state_trace": str(video_dir / "trace" / f"{label}_state_trace.npz"),
    }


def _progress_artifacts(output: Path) -> dict[str, Path]:
    output = output.resolve()
    return {
        "case_dir": output.with_name(f"{output.stem}.cases"),
        "checkpoint": output.with_name(f"{output.stem}.checkpoint.json"),
        "progress_jsonl": output.with_name(f"{output.stem}.progress.jsonl"),
        "invocations": output.with_name(f"{output.stem}.invocations.json"),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _resume_identity(description: dict) -> dict:
    return {
        "schema_version": 1,
        "request_sha256": description["request_sha256"],
        "request_identity": description["request_identity"],
        "source_targets": description["source_targets"],
        "scene_xml_sha256": description["scene_xml_sha256"],
        "scene_meta_sha256": description["scene_meta_sha256"],
        "criteria": description["criteria"],
        "controller": description["controller"],
        "selection": description["selection"],
    }


def _case_record_path(artifacts: dict[str, Path], index: int) -> Path:
    return artifacts["case_dir"] / f"{index:06d}.json"


def _compact_case(case: dict) -> dict:
    return {
        "index": case["index"],
        "status": case.get("status"),
        "reason": case.get("reason"),
        "execution_started": bool(case.get("execution_started", False)),
        "sim_duration_s": case.get("sim_duration_s"),
        "max_measured_tool_displacement_m": case.get("max_measured_tool_displacement_m"),
    }


def _load_completed_cases(
    artifacts: dict[str, Path],
    identity: dict,
    selected_indices: list[int],
    *,
    resume: bool,
) -> dict[int, dict]:
    existing_case_files = artifacts["case_dir"].is_dir() and any(artifacts["case_dir"].glob("*.json"))
    any_progress = (
        existing_case_files
        or artifacts["checkpoint"].exists()
        or artifacts["progress_jsonl"].exists()
        or artifacts["invocations"].exists()
    )
    if any_progress and not resume:
        raise ValueError("progress artifacts already exist; pass --resume to retain completed cases")
    if not resume:
        return {}

    if artifacts["checkpoint"].exists():
        checkpoint = json.loads(artifacts["checkpoint"].read_text())
        if checkpoint.get("resume_identity") != identity:
            raise ValueError("checkpoint identity does not match request/source/scene/controller/selection")

    completed: dict[int, dict] = {}
    for index in selected_indices:
        path = _case_record_path(artifacts, index)
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1 or record.get("resume_identity") != identity:
            raise ValueError(f"completed case {index} identity does not match current run")
        case = record.get("case")
        if not isinstance(case, dict) or case.get("index") != index:
            raise ValueError(f"completed case record {index} is malformed")
        completed[index] = case
    return completed


def _begin_recording_invocation(
    artifacts: dict[str, Path],
    identity: dict,
    output: Path,
    label_prefix: str,
    *,
    resume: bool,
) -> tuple[dict, list[dict]]:
    history: list[dict] = []
    if artifacts["invocations"].exists():
        document = json.loads(artifacts["invocations"].read_text())
        if document.get("resume_identity") != identity:
            raise ValueError("recording invocation identity does not match current run")
        history = document.get("invocations", [])
        if not isinstance(history, list):
            raise ValueError("recording invocation history is malformed")
    ordinal = len(history)
    suffix = "" if ordinal == 0 and not resume else f"_resume{ordinal:03d}"
    video_dir = output.resolve().parent / "videos"
    preexisting = (
        sorted(str(path.resolve()) for path in video_dir.glob(f"{label_prefix}*")) if video_dir.exists() else []
    )
    invocation = {
        "ordinal": ordinal,
        "recording_suffix": suffix,
        "started_wall_time_s": time.time(),
        "preexisting_recording_artifacts": preexisting,
        "status": "running",
    }
    history.append(invocation)
    _atomic_json(
        artifacts["invocations"],
        {"schema_version": 1, "resume_identity": identity, "invocations": history},
    )
    return invocation, history


def _update_recording_invocation(
    artifacts: dict[str, Path],
    identity: dict,
    history: list[dict],
    **updates,
) -> None:
    history[-1].update(updates)
    _atomic_json(
        artifacts["invocations"],
        {"schema_version": 1, "resume_identity": identity, "invocations": history},
    )


def _write_checkpoint(
    artifacts: dict[str, Path],
    identity: dict,
    selected_indices: list[int],
    completed: dict[int, dict],
) -> None:
    ordered = [_compact_case(completed[index]) for index in selected_indices if index in completed]
    _atomic_json(
        artifacts["checkpoint"],
        {
            "schema_version": 1,
            "resume_identity": identity,
            "selected_case_count": len(selected_indices),
            "completed_case_count": len(ordered),
            "completed": ordered,
            "summary": summarize_cases(ordered),
        },
    )


def _record_completed_case(
    artifacts: dict[str, Path],
    identity: dict,
    selected_indices: list[int],
    completed: dict[int, dict],
    result: dict,
) -> None:
    index = result["index"]
    if index in completed:
        raise RuntimeError(f"case {index} already has a terminal result")
    result.setdefault("execution_started", False)
    result.setdefault(
        "attempt_classification",
        "executed_reach" if result["execution_started"] else "admission_or_initialization_rejection",
    )
    _atomic_json(
        _case_record_path(artifacts, index),
        {"schema_version": 1, "resume_identity": identity, "case": result},
    )
    progress = {"schema_version": 1, **_compact_case(result), "recorded_wall_time_s": time.time()}
    artifacts["progress_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with artifacts["progress_jsonl"].open("a") as stream:
        stream.write(json.dumps(progress, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    completed[index] = result
    _write_checkpoint(artifacts, identity, selected_indices, completed)


def _initialization_observation(primitives) -> dict | None:
    try:
        state = primitives.obs()
    except Exception:  # noqa: BLE001 - best-effort evidence for primary initialization failure
        return None
    return {
        "seq": state.get("seq"),
        "sim_time_s": state.get("t"),
        "pelvis": state.get("pelvis"),
        "joints": state.get("joints"),
        "fallen": state.get("fallen"),
    }


def run_live(
    request: dict,
    output: Path,
    *,
    case_index: int | None = None,
    selection: dict | None = None,
    resume: bool = False,
) -> dict:
    description = request_description(request, output, case_index=case_index, selection=selection)
    identity = _resume_identity(description)
    selected_cases = _selected_cases(request, case_index, selection)
    selected_indices = [case.index for case in selected_cases]
    progress_artifacts = _progress_artifacts(output)
    completed = _load_completed_cases(
        progress_artifacts,
        identity,
        selected_indices,
        resume=resume,
    )
    cases_out = [completed[index] for index in selected_indices if index in completed]
    _write_checkpoint(progress_artifacts, identity, selected_indices, completed)

    def record_result(result: dict) -> None:
        _record_completed_case(progress_artifacts, identity, selected_indices, completed, result)
        cases_out.append(result)

    if len(completed) == len(selected_indices):
        return {
            "schema_version": 1,
            "request": description,
            "started_from_seed": request["seed"],
            "session_prime": {"status": "resume_complete_no_stack_work"},
            "cases": cases_out,
            "summary": summarize_cases(cases_out),
        }

    recording_invocation, invocation_history = _begin_recording_invocation(
        progress_artifacts,
        identity,
        output,
        request["runtime"]["video_label_prefix"],
        resume=resume,
    )
    recording_prefix = request["runtime"]["video_label_prefix"] + recording_invocation["recording_suffix"]
    case_label_width = max(2, len(str(len(request["cases"]))))
    require_pose_ownership()
    runtime_request = request["runtime"]
    meta = runtime_request["scene_meta"]
    ik = ArmIK(str(runtime_request["scene_xml"]))
    solver = BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all")
    collision_checker = MeasuredCollisionChecker(ArmIK(str(runtime_request["scene_xml"])))
    primitives = Primitives(meta=meta)
    reset = EpisodeControlClient()
    prime_label = f"{recording_prefix}_session_prime"
    session_prime: dict = {}
    try:
        session_prime = _prime_session(primitives, reset, output, prime_label)
        _update_recording_invocation(
            progress_artifacts,
            identity,
            invocation_history,
            session_prime=session_prime,
        )
        if session_prime["status"] != "ready":
            _update_recording_invocation(
                progress_artifacts,
                identity,
                invocation_history,
                status="session_prime_failed",
                ended_wall_time_s=time.time(),
            )
            return {
                "schema_version": 1,
                "request": description,
                "started_from_seed": request["seed"],
                "session_prime": session_prime,
                "cases": cases_out,
                "summary": summarize_cases(cases_out),
            }

        for case in selected_cases:
            if case.index in completed:
                continue
            label = f"{recording_prefix}_{case.index:0{case_label_width}d}"
            artifacts = _artifact_paths(output, label)
            initialization_label = f"{label}_initialization"
            initialization_artifacts = _artifact_paths(output, initialization_label)
            initialization_video_started = False
            task_video_started = False
            execution_started = False
            runtime = None
            initialization: dict = {
                "status": "running",
                "phase_events": [],
                "resets": [],
            }
            try:
                before_initialization = primitives.obs()
                _start_video_confirmed(
                    primitives,
                    reset,
                    initialization_label,
                    initialization,
                    before_initialization,
                    Path(initialization_artifacts["raw_video"]),
                )
                initialization_video_started = True
                runtime, initial_state, initialization = _initialize_trial(
                    primitives,
                    reset,
                    ik,
                    request["criteria"],
                    label,
                    initialization,
                )
            except Exception as error:  # noqa: BLE001 - one bad reset must not skip later samples
                initialization_video_started = bool(initialization.get("recording_start_command_sent"))
                initialization.update(
                    status="failed",
                    reason=str(error),
                    error=f"{type(error).__name__}: {error}",
                    last_observation=_initialization_observation(primitives),
                )
                record_result(
                    {
                        "index": case.index,
                        "label_zh": case.label_zh,
                        "status": "initialization_failed",
                        "reason": str(error),
                        "error": f"{type(error).__name__}: {error}",
                        "initialization": initialization,
                        "initialization_artifacts": initialization_artifacts,
                        "traces": [],
                    }
                )
                try:
                    if runtime is not None:
                        runtime.safe_stop()
                except Exception:
                    pass
                if initialization_video_started:
                    try:
                        _end_video_confirmed(primitives, reset, initialization)
                    except Exception as video_error:
                        initialization["video_finalization_error"] = f"{type(video_error).__name__}: {video_error}"
                continue

            try:
                _end_video_confirmed(primitives, reset, initialization)
                initialization_video_started = False
            except Exception as error:  # noqa: BLE001 - do not overlap initialization and task recorders
                initialization.update(
                    status="failed",
                    reason="initialization_recording_finalization_failed",
                    video_finalization_error=f"{type(error).__name__}: {error}",
                )
                try:
                    runtime.safe_stop()
                except Exception:
                    pass
                record_result(
                    {
                        "index": case.index,
                        "label_zh": case.label_zh,
                        "status": "initialization_failed",
                        "reason": "initialization_recording_finalization_failed",
                        "initialization": initialization,
                        "initialization_artifacts": initialization_artifacts,
                        "traces": [],
                    }
                )
                continue

            if not initialization["initial_stance_ready"]:
                result = {
                    "index": case.index,
                    "label_zh": case.label_zh,
                    "status": "initialization_failed",
                    "reason": "initial_stance_unstable",
                    "initialization": initialization,
                    "initialization_artifacts": initialization_artifacts,
                    "traces": [],
                }
            else:
                task_ready_state = primitives.obs()
                task_precondition_failed = not _quiet_ready(
                    task_ready_state,
                    original_reach_control_config().visual,
                )
                task_boundary = {
                    "initialization_admission_seq": initial_state.get("seq"),
                    "initialization_admission_sim_time_s": initial_state.get("t"),
                    "phase_events": [],
                }
                try:
                    if task_precondition_failed:
                        raise RuntimeError("quiet ready state was lost before task recording")
                    task_initial_state = _start_video_confirmed(
                        primitives,
                        reset,
                        label,
                        task_boundary,
                        task_ready_state,
                        Path(artifacts["raw_video"]),
                        ready_config=original_reach_control_config().visual,
                    )
                    task_video_started = True
                    executor = GhostReachExecutor(
                        solver,
                        runtime,
                        criteria=request["criteria"],
                        control_config=_request_control_config(request),
                        collision_checker=collision_checker,
                    )
                    execution_started = True
                    result = executor.run_case(case, task_initial_state)
                    result["initialization"].update(initialization)
                    result["initialization_artifacts"] = initialization_artifacts
                    result["task_recording_boundary"] = task_boundary
                    result["reach_boundary"] = {
                        "start_seq": task_initial_state.get("seq"),
                        "start_sim_time_s": task_initial_state.get("t"),
                        "end_seq": (result["traces"][-1]["seq"] if result["traces"] else task_initial_state.get("seq")),
                        "end_sim_time_s": result["traces"][-1]["sim_time_s"]
                        if result["traces"]
                        else task_initial_state.get("t"),
                    }
                    result["artifacts"] = artifacts
                except Exception as error:  # noqa: BLE001 - record failure and continue next target
                    task_video_started = bool(task_boundary.get("recording_start_command_sent"))
                    if task_precondition_failed:
                        result = {
                            "index": case.index,
                            "label_zh": case.label_zh,
                            "status": "initialization_failed",
                            "reason": "quiet_ready_state_lost_before_task_recording",
                            "initialization": initialization,
                            "initialization_artifacts": initialization_artifacts,
                            "traces": [],
                        }
                    else:
                        result = {
                            "index": case.index,
                            "label_zh": case.label_zh,
                            "status": "failed",
                            "reason": "runtime_error",
                            "execution_started": execution_started,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                            "initialization": initialization,
                            "initialization_artifacts": initialization_artifacts,
                            "task_recording_boundary": task_boundary,
                            "artifacts": artifacts,
                            "traces": [],
                        }
            try:
                runtime.safe_stop()
            except Exception as stop_error:  # noqa: BLE001
                result["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
            if task_video_started:
                try:
                    _end_video_confirmed(primitives, reset, result["task_recording_boundary"])
                except Exception as video_error:  # noqa: BLE001
                    result["video_finalization_error"] = f"{type(video_error).__name__}: {video_error}"
            record_result(result)
    finally:
        reset.pin_arms()
        reset.close()
        primitives.close()

    cases_out = [completed[index] for index in selected_indices if index in completed]
    _update_recording_invocation(
        progress_artifacts,
        identity,
        invocation_history,
        status="completed",
        ended_wall_time_s=time.time(),
        completed_case_count=len(cases_out),
    )
    return {
        "schema_version": 1,
        "request": description,
        "started_from_seed": request["seed"],
        "session_prime": session_prime,
        "recording_invocation": recording_invocation,
        "cases": cases_out,
        "summary": summarize_cases(cases_out),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--case-index", type=int)
    selection_group.add_argument("--case-range", metavar="START:STOP")
    selection_group.add_argument("--case-indices", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = load_execution_request(args.request)
        selection = resolve_case_selection(
            request,
            case_index=args.case_index,
            case_range=args.case_range,
            case_indices_path=args.case_indices,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        if args.validate_only:
            result = {
                "valid": True,
                **request_description(request, args.output, selection=selection),
            }
        else:
            result = run_live(request, args.output, selection=selection, resume=args.resume)
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - preserve validation/startup failure in requested output
        result = {
            "schema_version": 1,
            "valid": False,
            "status": "setup_failed",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 2
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
