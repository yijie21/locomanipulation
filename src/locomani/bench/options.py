"""The switches of a run: which benchmark, which controller, and where the robot's knowledge comes from.

Ported from the bench-options branch (its TODO 49.1, user 2026-11-24) so that both branches switch a run the same way
(selfwm TODO 38.11, user 2026-11-24: the self model on several benchmarks and controllers, truth or perception, stereo
or RGB-D, every combination one set of options). selfwm's differences: the controller axis names the whole-body
controllers this branch runs, and a benchmark's table may couple two axes (spec.py, `couples`).

One selector per mutually exclusive choice (docs/reference/standards.md, "Interfaces and configuration"):

    bench            simple | locomanilab | bigym2 | humanoidarena
    target           truth | perception       where the objects, their parts and the grasp poses come from: the
                                               simulator's state, or the head camera (segmentation + depth + a grasp
                                               planner on the points)
    sensor           none | stereo | rgbd     perception's input: the head's stereo pair turned into depth by a
                                               stereo model, or one RGB image with the simulator's depth image;
                                               `none` exactly when target is truth
    depth_noise      none | realsense         rgbd only: the rendered depth as is, or with a RealSense-like noise model
    self_pose        truth | odometry         where the skill believes the pelvis is: the simulator's, or leg odometry
    controller_root  truth | odometry         the pelvis position ScaleBFM is told (its policy input holds one): the
                                               simulator's, or leg odometry re-anchored at every stand (the loops so far)
    controller       scalebfm | sonic | amo | gr00t_wbc | native
                                             the whole-body controller under our Motor layer (ScaleBFM; SONIC v1.1 in
                                             process; AMO's or NVIDIA's GR00T-WBC legs and waist with our arm solver),
                                             or the benchmark's own controller

Resolution happens once, in this order: the code defaults below, the benchmark's profile
`configs/bench/<bench>.yaml`, then explicit `--bench-set key=value` overrides (an explicit value equal to the
default still counts as explicit). Then the normalisations, each recorded: under target=truth an unset `sensor`
becomes `none`; with a sensor other than rgbd an unset `depth_noise` becomes `none`; an unset axis that the
benchmark's table couples to another's value takes the coupled value. Finally the combination is
checked for consistency and against the benchmark's capability table (spec.py): an unsupported value is an error,
and a `planned` one is an error unless the caller allows planned values (tests, `python -m locomani.bench show`).

The resolved options go into every record (`Resolved.record()`); `signature()` is the short form that keeps runs
with different switches apart (pool / gate signatures, file names).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

from locomani.bench import registry
from locomani.bench.spec import PLANNED
from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1

BENCHES = ("simple", "locomanilab", "bigym2", "humanoidarena")
TARGETS = ("truth", "perception")
SENSORS = ("none", "stereo", "rgbd")
DEPTH_NOISE = ("none", "realsense")
SELF_POSE = ("truth", "odometry")
CONTROLLER_ROOT = ("truth", "odometry")
CONTROLLERS = ("scalebfm", "sonic", "amo", "gr00t_wbc", "native")


@dataclass(frozen=True)
class BenchOptions:
    bench: str = field(
        default="simple", metadata={"choices": BENCHES}
    )  # the benchmark; its profile configs/bench/<bench>.yaml
    target: str = field(
        default="perception", metadata={"choices": TARGETS}
    )  # objects, parts and grasps from the simulator (truth) or the camera (perception)
    sensor: str = field(
        default="stereo", metadata={"choices": SENSORS}
    )  # perception input: stereo pair + stereo model, or one RGB-D camera; none under truth
    depth_noise: str = field(
        default="none", metadata={"choices": DEPTH_NOISE}
    )  # rgbd only: rendered depth as is, or through a RealSense-like noise model
    self_pose: str = field(
        default="odometry", metadata={"choices": SELF_POSE}
    )  # where the skill believes the pelvis is: simulator or leg odometry
    controller_root: str = field(
        default="odometry", metadata={"choices": CONTROLLER_ROOT}
    )  # the pelvis position ScaleBFM is told: the simulator's, or leg odometry re-anchored at each stand
    controller: str = field(
        default="scalebfm", metadata={"choices": CONTROLLERS}
    )  # the whole-body controller under our Motor layer, or the benchmark's own controller

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> BenchOptions:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> BenchOptions:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> BenchOptions:
        return apply_overrides(self, overrides)

    def signature(self) -> str:
        """`simple/perception-stereo/pose-odometry/root-odometry/scalebfm`; rgbd adds its noise (`rgbd-realsense`)."""
        seen = "truth" if self.target == "truth" else f"perception-{self.sensor}"
        if self.sensor == "rgbd" and self.depth_noise != "none":
            seen += f"-{self.depth_noise}"
        return f"{self.bench}/{seen}/pose-{self.self_pose}/root-{self.controller_root}/{self.controller}"


def consistency_errors(o: BenchOptions) -> list[str]:
    """Combinations no benchmark can mean, whatever it supports."""
    errors = []
    if o.target == "truth" and o.sensor != "none":
        errors.append(f"target=truth takes no sensor (got sensor={o.sensor}); perception is what reads a sensor")
    if o.target == "perception" and o.sensor == "none":
        errors.append("target=perception needs sensor=stereo or sensor=rgbd")
    if o.depth_noise != "none" and o.sensor != "rgbd":
        errors.append(f"depth_noise={o.depth_noise} applies to sensor=rgbd only (got sensor={o.sensor})")
    return errors


def applicable(o: BenchOptions) -> list[tuple[str, str]]:
    """The (axis, value) choices a benchmark has to support for `o`: no sensor under target=truth, no depth noise
    without rgbd."""
    out = []
    for axis in ("target", "sensor", "depth_noise", "self_pose", "controller_root", "controller"):
        if (axis == "sensor" and o.sensor == "none") or (axis == "depth_noise" and o.sensor != "rgbd"):
            continue
        out.append((axis, getattr(o, axis)))
    return out


def support_errors(o: BenchOptions, allow_planned: bool = False) -> list[str]:
    """Choices the benchmark's capability table does not have, or has only as planned."""
    spec = registry.spec(o.bench)
    errors = []
    for axis, value in applicable(o):
        status = spec.status(axis, value)
        if status is None:
            errors.append(f"{spec.title} does not support {axis}={value}{spec.why(axis, value)}")
        elif status == PLANNED and not allow_planned:
            errors.append(f"{spec.title}: {axis}={value} is planned, not implemented yet{spec.why(axis, value)}")
    return errors


@dataclass(frozen=True)
class Resolved:
    options: BenchOptions
    profile: str  # the profile file the bench's defaults came from (repo-relative)
    explicit: tuple[str, ...]  # fields set by an override, in the order given
    normalized: tuple[str, ...]  # fields the resolver set (see the module docstring)
    planned: tuple[str, ...]  # "<axis>=<value>" choices that are planned (allowed only when the caller allowed them)

    def record(self) -> dict:
        """What a run stores about its switches."""
        return dict(
            options=self.options.to_dict(),
            signature=self.options.signature(),
            profile=self.profile,
            explicit=list(self.explicit),
            normalized=list(self.normalized),
            planned=list(self.planned),
        )


def _couple(spec, o: BenchOptions, explicit: tuple[str, ...], normalized: list[str]):
    """The table's couplings (spec.py): an unset coupled axis takes the value; an explicit contradiction is an error."""
    errors = []
    for key, (axis, value) in spec.couples.items():
        a, _, v = key.partition("=")
        if getattr(o, a) != v or getattr(o, axis) == value:
            continue
        if axis in explicit:
            errors.append(
                f"{spec.title}: {key} fixes {axis}={value} here (got {axis}={getattr(o, axis)}){spec.why(a, v)}"
            )
        else:
            o, normalized = replace(o, **{axis: value}), normalized + [axis]
    return o, normalized, errors


def _split_bench(overrides: list[str]) -> tuple[str | None, list[str]]:
    bench, rest = None, []
    for kv in overrides:
        key = kv.split("=", 1)[0].strip()
        if key == "bench":
            bench = kv.split("=", 1)[1].strip() if "=" in kv else ""
        else:
            rest.append(kv)
    return bench, rest


def resolve(
    bench: str | None = None, overrides: list[str] | tuple[str, ...] = (), allow_planned: bool = False
) -> Resolved:
    """Code defaults -> configs/bench/<bench>.yaml -> overrides, normalised and checked; raises ValueError."""
    from locomani import profiles  # profiles imports every config module; keep it out of this module's import

    over_bench, rest = _split_bench(list(overrides))
    if bench is not None and over_bench is not None and bench != over_bench:
        raise ValueError(f"two benchmarks given: --bench {bench} and bench={over_bench}")
    name = bench or over_bench or BenchOptions().bench
    if name not in BENCHES:
        raise ValueError(f"unknown benchmark {name!r}; choose one of {BENCHES}")
    base = profiles.load("bench", name)
    if base.bench != name:
        raise ValueError(f"configs/bench/{name}.yaml says bench={base.bench}")
    explicit = tuple(kv.split("=", 1)[0].strip() for kv in rest)
    unknown = [k for k in explicit if k not in BenchOptions.__dataclass_fields__]
    if unknown:
        raise ValueError(f"unknown option(s) {unknown}; options are {list(BenchOptions.__dataclass_fields__)}")
    o = base.with_overrides(rest)
    normalized = []
    if o.target == "truth" and "sensor" not in explicit and o.sensor != "none":
        o, normalized = replace(o, sensor="none"), normalized + ["sensor"]
    if o.sensor != "rgbd" and "depth_noise" not in explicit and o.depth_noise != "none":
        o, normalized = replace(o, depth_noise="none"), normalized + ["depth_noise"]
    spec = registry.spec(name)
    o, normalized, errors = _couple(spec, o, explicit, normalized)
    errors += consistency_errors(o) + support_errors(o, allow_planned)
    if errors:
        raise ValueError(f"{o.signature()}: " + "; ".join(errors))
    planned = tuple(f"{a}={v}" for a, v in applicable(o) if spec.status(a, v) == PLANNED)
    return Resolved(o, f"configs/bench/{name}.yaml", explicit, tuple(normalized), planned)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """--bench NAME and repeatable --bench-set KEY=VALUE, for an entry point that runs on a benchmark."""
    parser.add_argument("--bench", choices=BENCHES, help="the benchmark (its profile configs/bench/<name>.yaml)")
    parser.add_argument(
        "--bench-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one option: target, sensor, depth_noise, self_pose, controller_root, controller",
    )


def from_args(args: argparse.Namespace, allow_planned: bool = False) -> Resolved:
    return resolve(args.bench, args.bench_set, allow_planned)
