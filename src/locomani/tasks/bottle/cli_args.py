"""Resolve bottle gym options without importing or mutating the runtime.

Code defaults < task profile < explicit CLI options. VLA mode retains its own
reach calibration unless --reach-config explicitly selects another one.
Run this module to print the resolved options and their sources without a sim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields

from locomani.tasks.bottle.config import DEFAULT_BOTTLE_TASK, BottleTaskConfig
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig


@dataclass(frozen=True)
class GymOptions:
    """Legacy gym defaults; task profiles supply the collection distribution."""

    rounds: int = 5
    dump_sample: str | None = None
    record: bool = False
    seed: int | None = None
    spawn_off: tuple[float, ...] = (1.0, 3.0)
    spawn_edge: tuple[float, ...] | None = None
    face_object: bool = False
    spawn_lateral: float = 0.35
    spawn_yaw_err: float = 90.0
    edge_jitter: float = 0.10
    stand_retries: int = 1
    until_ok: int | None = None
    one_shot: bool = False
    mid_band: float = 0.30
    targets: str = "pool"
    manifest: str | None = None
    view_gate_x: float = 0.80
    view_gate_y: float = 0.90
    reach_config: str | None = None
    task_config: str | None = None
    # The run's own scene copy. `locomani.paths` resolves LOCOMANI_SIM_DIR at IMPORT, which is
    # correct for a process the stack launched and wrong for one that attaches to a stack and
    # then learns which run it is serving. Passing it explicitly is the rule in AGENTS.md
    # ("parse configuration at the entry; pass explicit values to consumers"); None keeps the
    # imported default for every CLI and diagnostic caller.
    scene_meta: str | None = None
    cases: str | None = None
    write_cases: str | None = None
    sample_only: bool = False
    video_include_init: bool = DEFAULT_BOTTLE_TASK.video_include_init
    print_config: bool = False
    reach_config_obj: ReachConfig = DEFAULT_REACH
    task_config_obj: BottleTaskConfig | None = None
    sources: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def report(self) -> dict:
        options = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"sources", "reach_config_obj", "task_config_obj"}
        }
        options["reach"] = self.reach_config_obj.to_dict()
        return {"options": options, "sources": dict(self.sources)}


def parser() -> argparse.ArgumentParser:
    # Suppression records presence, including an explicit value equal to a default.
    ap = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS, allow_abbrev=False)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--dump-sample", help="write one scene sample for replay")
    ap.add_argument("--record", action="store_true", help="record accepted episodes through the exporter")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--spawn-off", type=float, nargs="+", help="one/two distances behind the stand, metres")
    ap.add_argument("--spawn-edge", type=float, nargs="+", help="one/two distances from the table, metres")
    ap.add_argument("--face-object", action=argparse.BooleanOptionalAction)
    ap.add_argument("--spawn-lateral", type=float, help="lateral spawn jitter, metres")
    ap.add_argument("--spawn-yaw-err", type=float, help="spawn heading jitter, degrees")
    ap.add_argument("--edge-jitter", type=float, help="object jitter along the table edge, metres")
    ap.add_argument("--stand-retries", type=int, choices=(0, 1))
    ap.add_argument("--until-ok", type=int, help="stop after this many successful episodes")
    ap.add_argument("--one-shot", action=argparse.BooleanOptionalAction)
    ap.add_argument("--mid-band", type=float, help="approach deceleration band, metres")
    ap.add_argument("--targets", help="pool, default, or comma-separated object names")
    ap.add_argument("--manifest", help="append one result per attempted round")
    ap.add_argument("--view-gate-x", type=float, help="horizontal ego visibility bound (frame edge=1)")
    ap.add_argument("--view-gate-y", type=float, help="vertical ego visibility bound (frame edge=1)")
    ap.add_argument(
        "--reach-config", help="explicit versioned reach calibration; takes precedence over the task profile"
    )
    ap.add_argument("--task-config", help="task profile: spawn, target, stand retries, expert reach and place settings")
    ap.add_argument("--scene-meta", help="scene_meta.json of the run's own scene copy")
    ap.add_argument("--cases", help="fixed case bank; bypass random sampling")
    ap.add_argument("--write-cases", help="write exact sampled conditions")
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--video-include-init", action=argparse.BooleanOptionalAction)
    ap.add_argument("--print-config", action="store_true", help="report resolved gym inputs and exit before actuation")
    return ap


def resolve_options(
    explicit: dict,
    *,
    profile: BottleTaskConfig | None = None,
    reach: ReachConfig | None = None,
    vla: bool = False,
    video_include_init: bool | None = None,
) -> GymOptions:
    """Merge already-loaded inputs; no file I/O, robot or mutable module state."""
    defaults = GymOptions()
    values = {f.name: getattr(defaults, f.name) for f in fields(defaults) if f.name != "sources"}
    sources = {name: "default" for name in values if not name.endswith("_obj")}
    if video_include_init is not None:
        values["video_include_init"] = video_include_init
        sources["video_include_init"] = "caller"
    profile_path = explicit.get("task_config")
    if profile_path and profile is None:
        raise ValueError("load the task profile before resolving options")
    values["task_config_obj"] = profile
    if profile is not None:
        profile_options = {
            "spawn_edge": profile.spawn.edge_range,
            "spawn_yaw_err": profile.spawn.yaw_err_deg,
            "spawn_lateral": profile.spawn.lateral,
            "edge_jitter": profile.spawn.edge_jitter,
            "face_object": profile.spawn.face_object,
            "targets": profile.target,
            "stand_retries": profile.stand_retries,
            "one_shot": profile.one_shot,
            "video_include_init": profile.video_include_init,
        }
        values.update(profile_options)
        sources.update({name: f"task:{profile_path}" for name in profile_options})
    values.update(explicit)
    sources.update({name: "cli" for name in explicit})
    if "spawn_off" in explicit and "spawn_edge" not in explicit:
        # The runtime chooses edge sampling whenever an edge range is present.
        values["spawn_edge"] = None
        sources["spawn_edge"] = "cli (--spawn-off selects offset sampling)"
    for name in ("spawn_off", "spawn_edge"):
        if values[name] is not None:
            values[name] = tuple(values[name])
            if not 1 <= len(values[name]) <= 2:
                raise ValueError(f"{name} needs one or two distances")
    if values["reach_config"]:
        if reach is None:
            raise ValueError("load the explicit reach calibration before resolving options")
        values["reach_config_obj"] = reach
        sources["reach"] = f"reach:{values['reach_config']}"
    elif profile is not None and not vla:
        values["reach_config_obj"] = profile.reach
        sources["reach"] = f"task:{profile_path}#reach"
    else:
        sources["reach"] = "default (VLA keeps its own reach calibration)" if vla else "default"
    if values["sample_only"] and not values["write_cases"]:
        raise ValueError("--sample-only requires --write-cases")
    if values["cases"] and values["until_ok"]:
        raise ValueError("fixed cases cannot be combined with success-count stopping")
    return GymOptions(**values, sources=sources)


def parse_args(argv=None, *, vla: bool = False, video_include_init: bool | None = None) -> GymOptions:
    ap = parser()
    explicit = vars(ap.parse_args(argv))
    try:
        profile = BottleTaskConfig.load(explicit["task_config"]) if explicit.get("task_config") else None
        reach = ReachConfig.load(explicit["reach_config"]) if explicit.get("reach_config") else None
        return resolve_options(explicit, profile=profile, reach=reach, vla=vla, video_include_init=video_include_init)
    except (ValueError, TypeError, OSError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    print(json.dumps(parse_args().report(), indent=2))
