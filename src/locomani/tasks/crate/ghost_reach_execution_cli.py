"""Child process for the floor-only 14-case dynamic ghost reach probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
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
    PALM_PITCH_DEG,
    PREVIEW_TILT_DEG,
    GhostReachCriteria,
    GhostReachExecutor,
    MeasuredCollisionChecker,
    load_source_cases,
    original_reach_control_config,
    summarize_cases,
)
from locomani.tasks.crate.round_runner import prime

TOP_LEVEL_FIELDS = {"schema_version", "source_targets", "request_identity", "seed", "runtime", "experiment"}
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
    _strict_keys(raw, TOP_LEVEL_FIELDS, TOP_LEVEL_FIELDS, "request")
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


def _selected_cases(request: dict, case_index: int | None) -> list:
    cases = request["cases"]
    if case_index is None:
        return cases
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 1 <= case_index <= 14:
        raise ValueError("case_index must be an integer from 1 through 14")
    selected = [case for case in cases if case.index == case_index]
    if len(selected) != 1:
        raise ValueError(f"source targets do not contain case index {case_index}")
    return selected


def request_description(request: dict, output: Path, *, case_index: int | None = None) -> dict:
    runtime = request["runtime"]
    selected = _selected_cases(request, case_index)
    selected_tilts = [case.bodyward_tilt_deg for case in selected]
    one_tilt = len(set(selected_tilts)) == 1
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
        "criteria": request["criteria"].__dict__,
        "controller": {
            "owner": "CoordinatedReach/WholeBodyIK",
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


def run_live(request: dict, output: Path, *, case_index: int | None = None) -> dict:
    require_pose_ownership()
    runtime_request = request["runtime"]
    meta = runtime_request["scene_meta"]
    ik = ArmIK(str(runtime_request["scene_xml"]))
    solver = BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all")
    collision_checker = MeasuredCollisionChecker(ArmIK(str(runtime_request["scene_xml"])))
    primitives = Primitives(meta=meta)
    reset = EpisodeControlClient()
    cases_out: list[dict] = []
    prime_label = f"{runtime_request['video_label_prefix']}_session_prime"
    session_prime: dict = {}
    try:
        session_prime = _prime_session(primitives, reset, output, prime_label)
        if session_prime["status"] != "ready":
            return {
                "schema_version": 1,
                "request": request_description(request, output, case_index=case_index),
                "started_from_seed": request["seed"],
                "session_prime": session_prime,
                "cases": cases_out,
                "summary": summarize_cases(cases_out),
            }

        for case in _selected_cases(request, case_index):
            label = f"{runtime_request['video_label_prefix']}_{case.index:02d}"
            artifacts = _artifact_paths(output, label)
            initialization_label = f"{label}_initialization"
            initialization_artifacts = _artifact_paths(output, initialization_label)
            initialization_video_started = False
            task_video_started = False
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
                cases_out.append(
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
                cases_out.append(
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
                        control_config=original_reach_control_config(),
                        collision_checker=collision_checker,
                    )
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
            cases_out.append(result)
    finally:
        reset.pin_arms()
        reset.close()
        primitives.close()

    return {
        "schema_version": 1,
        "request": request_description(request, output, case_index=case_index),
        "started_from_seed": request["seed"],
        "session_prime": session_prime,
        "cases": cases_out,
        "summary": summarize_cases(cases_out),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(1, 15))
    args = parser.parse_args(argv)
    try:
        request = load_execution_request(args.request)
        if args.validate_only:
            result = {
                "valid": True,
                **request_description(request, args.output, case_index=args.case_index),
            }
        else:
            result = run_live(request, args.output, case_index=args.case_index)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
