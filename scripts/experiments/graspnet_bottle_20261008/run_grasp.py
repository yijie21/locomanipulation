"""TODO 21.3.4 / 23.2: one GraspNet object, the robot at the table edge grasping it with `grasp_planned`.

One object per process: locomani modules copy the scene path when imported, so a second scene in the same process is
refused (paths.adopt_sim_dir). For the object: an experiment file (bottle_simple_200.yaml with the object's scene, make_scenes.py) and a one-step
plan [grasp_planned]; scripts/plan_episode.py then starts its own stack, spawns the robot just behind the planned stand
(--spawn-off 0.5, no walk; grasp_planned re-places the settled robot), and runs --rounds rounds, round r with the r-th
of --planners (cycling) on the same object yaw (fixed per object; --yaws N draws one per round), each with its video.

Defaults (2026-10-12): the grasp from the scene's offline cache (--grasp-source cache, grasp_cache.py), our IK planner,
one round. --grasp-source graspnet takes it from graspnet-baseline instead (graspnet_server.py on --graspnet-port);
--planners ik,curobo --rounds 2 is the 21.3.4 planner comparison (curobo_server.py on --curobo-port). With
--fixed-grasp both rounds aim at the one grasp round 0 fixes (grasp_planned.py, "Fixed grasp"). run_one.sh wraps this
with the environment it needs.

    bash scripts/experiments/graspnet_bottle_20261008/run_one.sh 12 1 0 _demo [--grasp-source graspnet]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / "outputs" / "graspnet_bottle_20261008"
BASE_EXPERIMENT = REPO / "configs" / "experiments" / "bottle_simple_200.yaml"


def experiment_for(oid: int) -> Path:
    text = BASE_EXPERIMENT.read_text()
    text = re.sub(r"^name: .*$", f"name: gnb_obj_{oid:03d}", text, count=1, flags=re.M)
    scene = (ROOT / "scenes" / f"obj_{oid:03d}").relative_to(REPO)
    text = re.sub(r"^  dir: outputs/scene/sim_lowcom$", f"  dir: {scene}", text, count=1, flags=re.M)
    if f"dir: {scene}" not in text:
        raise SystemExit("bottle_simple_200.yaml no longer names outputs/scene/sim_lowcom as its scene.dir")
    dest = ROOT / "experiments" / f"gnb_obj_{oid:03d}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", type=int)
    ap.add_argument("--instance", type=int, default=1)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--graspnet-port", type=int, default=5790)
    ap.add_argument("--curobo-port", type=int, default=5791)
    ap.add_argument("--tag", default="", help="suffix of the run names")
    ap.add_argument("--fixed-grasp", action="store_true", help="both rounds aim at the grasp round 0 fixes")
    ap.add_argument(
        "--grasp-source", choices=("cache", "graspnet"), default="cache",
        help="cache: the scene's offline grasp cache (grasp_cache.py, TODO 23.2); graspnet: graspnet-baseline on the "
        "ego depth image (graspnet_server.py on --graspnet-port)",
    )  # fmt: skip
    ap.add_argument("--planners", default="ik", help="round r runs the r-th of these, cycling (ik, curobo)")
    ap.add_argument(
        "--presort",
        choices=("on", "off"),
        default="on",
        help="the IK planner's candidate presort (grasp_planned.PRESORT)",
    )
    ap.add_argument(
        "--arm-pin",
        choices=("off", "arm", "arm+yaw", "arm+waist"),
        default="arm",
        help="TODO 23.4.5: the right arm (and the waist) follow the IK joints through the sim-side pin while the expert executes",
    )
    ap.add_argument(
        "--base-freeze",
        choices=("on", "off"),
        default="off",
        help="TODO 23.4.6.1: freeze the closed-loop IK's virtual base",
    )
    ap.add_argument("--entry-gate", choices=("on", "off"), default="on", help="TODO 23.4.4: grasp_planned.ENTRY_GATE")
    ap.add_argument(
        "--prefer-high", choices=("on", "off"), default="off", help="TODO 23.4.4: grasp_planned.PREFER_HIGH"
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument(
        "--yaws", type=int, default=1,
        help="object yaws drawn per object (seed 1000 + id; the first is the 21.3.4 yaw), used one per round, cycling",
    )  # fmt: skip
    ap.add_argument(
        "--spawn-off",
        type=float,
        default=0.5,
        help="m behind the planned stand; grasp_planned then places the settled robot itself (align_robot_xy). At 0 "
        "the teleport landed 0.11 m off the table edge, at 0.15 a 22 cm box was knocked flat during the spawn",
    )
    args = ap.parse_args()

    import os

    sys.path[:0] = [str(HERE), str(REPO / "src")]
    from locomani.sim.ports import instance_ports

    os.environ["INSTANCE"] = str(args.instance)  # every port default is read at import time: pin the block first
    os.environ.update({k: str(v) for k, v in instance_ports(args.instance).items()})
    from locomani import paths

    oid = args.object
    paths.adopt_sim_dir(str(ROOT / "scenes" / f"obj_{oid:03d}"))  # before any module copies the default scene
    spec = importlib.util.spec_from_file_location("plan_episode", REPO / "scripts" / "plan_episode.py")
    plan_episode = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_episode)
    import grasp_planned

    import locomani.skills.catalogue as catalogue
    from locomani.skills.profiles import load_profiles

    original_build = catalogue.build_catalogue
    plan = ROOT / "experiments" / "plan_grasp_planned.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps([{"primitive": "grasp_planned", "args": {"target": "water_bottle"}}]))
    name = f"gnb_obj_{oid:03d}{args.tag}"
    out_dir = REPO / "outputs" / "runs" / name / "grasp_planned"
    grasp_planned.PLANNER_ORDER = tuple(args.planners.split(","))
    grasp_planned.PRESORT["on"] = args.presort == "on"
    grasp_planned.BASE_FREEZE["on"] = args.base_freeze == "on"
    grasp_planned.ENTRY_GATE["on"] = args.entry_gate == "on"
    grasp_planned.PREFER_HIGH["on"] = args.prefer_high == "on"
    grasp_planned.DECOUPLE["on"] = args.arm_pin != "off"
    grasp_planned.DECOUPLE["waist"] = "yawroll" if args.arm_pin == "arm+yaw" else args.arm_pin == "arm+waist"
    yaws = np.random.default_rng(1000 + oid).uniform(-np.pi, np.pi, size=args.yaws).tolist()
    spawned = {"n": 0}

    def next_yaw(half_extent, psi, rng):
        """One yaw per spawn: with --yaws 1 both rounds meet the object the same way (the comparison), with more each
        round sees the next drawn yaw (the cache acceptance's orientations)."""
        y = yaws[spawned["n"] % len(yaws)]
        spawned["n"] += 1
        return y

    plan_episode.spawn_object_yaw = next_yaw
    yaw = yaws[0]

    def build(profiles=None, **overrides):
        entries = original_build(profiles, **overrides)
        entries["grasp_planned"] = grasp_planned.GraspPlanned(
            profiles or load_profiles(),
            args.graspnet_port,
            args.curobo_port,
            out_dir,
            fixed=args.fixed_grasp,
            source=args.grasp_source,
        )
        return entries

    catalogue.build_catalogue = build
    import locomani.skills.catalogue.bottle as bottle_skills

    stand_plan = bottle_skills.stand_plan
    base_meta = json.loads((REPO / "outputs" / "scene" / "sim_lowcom" / "scene_meta.json").read_text())

    def stand_plan_any_height(meta, target, *args, **kw):
        """The spawn's stand: the bottle line's own plan, or for an object too short for its grasp templates
        ("target height has no comfortable baseline grasp", objects 005 and 009) the plan for the bottle that stood on
        the same spot -- grasp_planned re-places the robot anyway, only the heading and the spawn come from here."""
        try:
            return stand_plan(meta, target, *args, **kw)
        except ValueError:
            return stand_plan(base_meta, target, *args, **kw)

    bottle_skills.stand_plan = stand_plan_any_height
    spawn_single = plan_episode.spawn_single

    def spawn_logged(session, p, reset, meta, target, profile, **kw):
        """The spawn, with the target's tilt logged after it (a box upright in the scene was found lying flat)."""
        ok, why = spawn_single(session, p, reset, meta, target, profile, **kw)
        q = p.obs()["objects"][target]["quat"]
        tilt = float(np.degrees(2 * np.arccos(min(1.0, np.hypot(q[0], q[3])))))
        p.log(
            f"[run_grasp] after the spawn the target is tilted {tilt:.0f} deg at {np.round(p.obs()['objects'][target]['pos'], 3)}"
        )
        return ok, why

    plan_episode.spawn_single = spawn_logged
    argv = [
        str(experiment_for(oid)),
        "--gpu", str(args.gpu),
        "--instance", str(args.instance),
        "--rounds", str(args.rounds),
        "--seed", str(5000 + oid),
        "--spawn-off", str(args.spawn_off),
        "--plan", str(plan),
        "--name", name,
    ]  # fmt: skip
    print(
        f"[run_grasp] object {oid:03d} yaws {np.degrees(yaws).round(0).tolist()} deg, source {args.grasp_source}, "
        f"planners {grasp_planned.PLANNER_ORDER} x {args.rounds} rounds -> outputs/runs/{name}",
        flush=True,
    )
    code = plan_episode.main(argv)
    print(f"[run_grasp] object {oid:03d} done: {code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
