"""grasp_gym.py's argparse CLI, split out of its main() (refactor step 4b,
pure extraction; decision 9's file-size cap). `parse_args()` reads/writes
grasp_gym.VIDEO_INCLUDE_INIT through a deferred import of grasp_gym.py
rather than `global`, because that name's home moved out of this file:
the default value on --video-include-init and the reassignment after
parsing both have to see and touch the SAME module attribute a caller can
externally override (`grasp_gym.VIDEO_INCLUDE_INIT = ...`), which only a
qualified reference guarantees once the code lives in a different module.
The import is deferred (inside the function, not at this file's top level)
because grasp_gym.py imports `parse_args` back -- a module-level import in
the other direction would make the two files a load-order-sensitive
circular import.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """grasp_gym.py's CLI, split out of main() (refactor step 4b, pure
    extraction; decision 9's file-size cap). Deferred grasp_gym import (not
    top-level) because grasp_gym.py imports this function back -- see this
    module's docstring."""
    from locomani.tasks.bottle import grasp_gym

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--dump-sample",
        default=None,
        help="write this round's scene sample to a JSON file so a "
        "replay can reproduce the identical initial condition "
        "(diag_token_replay.py)",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter too, and keep only the episodes that pass every stage predicate",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--spawn-off",
        type=float,
        nargs="+",
        default=[1.0, 3.0],
        metavar="M",
        help="spawn this far behind the stand and walk in. One "
        "value fixes it, two give a per-round range. Long "
        "enough that the approach is a real walk: the arm "
        "rises over the last RAISE_LEAD metres instead of "
        "being held out level the whole way, and the ego "
        "camera picks the object up at 1.77 m "
        "(diag_ego_geom).",
    )
    ap.add_argument(
        "--spawn-edge",
        type=float,
        nargs="+",
        default=None,
        metavar="M",
        help="spawn this far from the TABLE, overriding "
        "--spawn-off. One value fixes it, two give a range. "
        "The table is what a person measures from, and it is "
        "the quantity the clearance floor (EDGE_MIN) is in; "
        "--spawn-off is measured behind the STAND, which "
        "itself sits 0.17-0.32 m from the edge and moves with "
        "the target, so the same --spawn-off is a different "
        "table distance every round. Solved per round by "
        "bisection on the sampled approach line.",
    )
    ap.add_argument(
        "--face-object",
        action="store_true",
        help="wake up pointing at the OBJECT rather than along the "
        "approach line, and take --spawn-yaw-err as jitter "
        "around that. The two coincide only for a spawn dead "
        "on the line: with a lateral offset, and especially "
        "with a close --spawn-edge, facing the approach "
        "heading leaves the object well off centre.",
    )
    ap.add_argument(
        "--spawn-lateral",
        type=float,
        default=0.35,
        metavar="M",
        help="spawn up to this far to either side of the approach "
        "line. 0 puts every episode dead on it, which is a "
        "correction the model then never sees.",
    )
    ap.add_argument(
        "--spawn-yaw-err",
        type=float,
        default=90.0,
        metavar="DEG",
        help="spawn facing up to this far off the approach "
        "heading, and turn to it before walking. The turn "
        "happens AT THE SPAWN, metres from the table: near "
        "the object a heading correction is a 110-160 deg "
        "pirouette that walks the pelvis into the edge.",
    )
    ap.add_argument(
        "--edge-jitter",
        type=float,
        default=0.10,
        metavar="M",
        help="how far the target may slide ALONG its edge, either "
        "way, from its nominal spot. The default 0.10 puts "
        "it in a 20 cm segment of a 100 cm edge, which is "
        "enough variation for the grasp but lets the "
        "walk-up be learnt as 'go to the +y edge, reach a "
        "little left' rather than as locating the object. "
        "Widen it when the batch has few targets and the "
        "approach is what has to generalise. The rejection "
        "sampler validates the stand and both walk legs "
        "against the table, so a value too large costs "
        "retries, not bad samples.",
    )
    ap.add_argument(
        "--stand-retries",
        type=int,
        choices=(0, 1),
        default=1,
        help="reset retries after a stand-stage failure; use 0 for fixed-case acceptance",
    )
    ap.add_argument(
        "--until-ok",
        type=int,
        default=None,
        metavar="N",
        help="stop as soon as N rounds have SUCCEEDED instead of "
        "running all --rounds. --rounds is then the cap on "
        "how many attempts that may take, so a run that "
        "never reaches N still terminates.",
    )
    ap.add_argument(
        "--one-shot",
        action="store_true",
        help="discard any round whose first approach parks out "
        "of reach instead of walking in again (flow200: "
        "the repair re-walk is recorded hesitation, and "
        "the dataset doctrine is one continuous arrival "
        "or nothing).",
    )
    ap.add_argument(
        "--mid-band",
        type=float,
        default=0.30,
        metavar="M",
        help="distance before plan_r where the approach drops "
        "from COARSE to the MID deceleration speed "
        "(default 0.30, the calibrated brake; flow200 "
        "runs 0.20 for a shorter crawl-in at a small "
        "carry-risk cost)",
    )
    ap.add_argument(
        "--targets",
        default="pool",
        help="which objects to draw the round's target from: "
        "'pool' (every entry in the scene's target_pool), "
        "'default' (the single scene target, the old "
        "behaviour), or a comma-separated list of names",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="append one JSON line per round: the target, the "
        "instruction, and whether the episode was kept. "
        "convert_to_hf_format.py joins on this to give each "
        "episode its own task string, which the exporter "
        "cannot do -- its --task-prompt is per SESSION.",
    )
    # 0.8 / 0.9, the values both gates were hardcoded to before they became
    # parameters. Making them settable is right; moving the DEFAULT is a
    # different change: it decides which rounds enter the dataset, so the
    # 191-episode walk200 batch and the next one would be selected by
    # different rules. The horizontal bound is the stricter of the two on
    # purpose -- yaw scatter moves the object across the frame fast.
    ap.add_argument(
        "--view-gate-x",
        type=float,
        default=0.80,
        metavar="U",
        help="ego-frame visibility gate, horizontal "
        "(|u_x| must stay under this; frame edge is 1.0). "
        "Camera geometry, not spawn distance.",
    )
    ap.add_argument(
        "--view-gate-y",
        type=float,
        default=0.90,
        metavar="U",
        help="ego-frame visibility gate, vertical "
        "(|u_y| must stay under this; frame edge is 1.0). "
        "Camera geometry, not spawn distance.",
    )
    ap.add_argument("--reach-config", help="versioned JSON tabletop reach calibration")
    ap.add_argument(
        "--task-config",
        help="BottleTaskConfig JSON: the whole profile. Its `reach` block becomes the reach "
        "calibration, its `spawn` block the spawn distribution, plus `target`, `stand_retries`, "
        "`goal` and `place`. Anything given explicitly on the command line wins over the file.",
    )
    ap.add_argument("--cases", help="fixed JSON case bank; bypass random sampling")
    ap.add_argument("--write-cases", help="write exact sampled conditions as JSON (exclusive output)")
    ap.add_argument("--sample-only", action="store_true", help="generate cases without starting the robot")
    ap.add_argument(
        "--video-include-init",
        action=argparse.BooleanOptionalAction,
        default=grasp_gym.VIDEO_INCLUDE_INIT,
        help="include the ready-posture prelude in diagnostic video "
        "(BottleTaskConfig.video_include_init, was SIM_VIDEO_INCLUDE_INIT; "
        "grasp_gym runs as its own subprocess from run_learn_pilot.py's "
        "collect stage, so this is a CLI flag rather than only a "
        "module-level override)",
    )
    args = ap.parse_args()
    if args.sample_only and not args.write_cases:
        ap.error("--sample-only requires --write-cases")
    if args.cases and args.until_ok:
        ap.error("fixed cases cannot be combined with success-count stopping")
    grasp_gym.VIDEO_INCLUDE_INIT = args.video_include_init
    if args.task_config:
        _apply_task_config(ap, args)
    return args


def _apply_task_config(ap, args) -> None:
    """Fill in from the task profile whatever the command line left at its default.

    Until 2026-09-10 this flag read only `goal` and `place`, so naming a
    collection profile got the profile's pick-and-place knobs and NOT its reach
    calibration or its spawn distribution -- the run silently fell back to
    `ReachConfig()` (whose ready_grasp_recovery=False cannot grasp) and to
    grasp_gym's own +-90 deg spawn defaults. Both failures surface stages later
    and look like the robot's fault: "ready_grasp_recovery=False is retired"
    after a successful walk, or a whole batch of "[view] object not in ego view
    at spawn". The correct invocation existed only inside
    run_learn_pilot.py's collect stage; now it is what the flag does.

    An explicit flag still wins, so run_learn_pilot's own expansion is a no-op
    against the same profile rather than a conflict.
    """
    # Deferred, like parse_args' own grasp_gym import: grasp_gym imports this
    # module back, so a module-level import would be circular.
    from locomani.tasks.bottle import grasp_gym
    from locomani.tasks.bottle.config import BottleTaskConfig

    cfg = BottleTaskConfig.load(args.task_config)
    applied = []
    skipped = []
    if not args.reach_config:
        # The reach block moves the SAMPLED STAND, not just the grasp:
        # ready_grasp_policy.sample_stand reads max_drop / backshift /
        # ready_grasp_recovery. Substituting it into a closed-loop eval would
        # silently make that run a different arm from every number already in
        # docs/ledger/bottle.md, which were measured on DEFAULT_REACH via
        # scripts/run_eval_arm.sh. No eval passes --task-config today; this
        # keeps it that way by construction rather than by luck (session cf,
        # 2026-09-10). An eval that WANTS the collection calibration must say
        # so with an explicit --reach-config, and record the protocol change.
        if grasp_gym.VLA_POLICY is None:
            args.reach_config_obj = cfg.reach
            applied.append("reach")
        else:
            skipped.append("reach (VLA round: an eval keeps its own calibration)")
    if args.spawn_edge == ap.get_default("spawn_edge"):
        args.spawn_edge = list(cfg.spawn.edge_range)
        applied.append("spawn_edge")
    if args.spawn_yaw_err == ap.get_default("spawn_yaw_err"):
        args.spawn_yaw_err = cfg.spawn.yaw_err_deg
        applied.append("spawn_yaw_err")
    if args.spawn_lateral == ap.get_default("spawn_lateral"):
        args.spawn_lateral = cfg.spawn.lateral
        applied.append("spawn_lateral")
    if args.edge_jitter == ap.get_default("edge_jitter"):
        args.edge_jitter = cfg.spawn.edge_jitter
        applied.append("edge_jitter")
    if not args.face_object and cfg.spawn.face_object:
        args.face_object = True
        applied.append("face_object")
    if args.targets == ap.get_default("targets"):
        args.targets = cfg.target
        applied.append("targets")
    if args.stand_retries == ap.get_default("stand_retries"):
        args.stand_retries = cfg.stand_retries
        applied.append("stand_retries")
    # Provenance: a log that does not say which profile produced a distribution
    # cannot be compared with one that used another.
    note = f"[gym] task-config {args.task_config}: applied {', '.join(applied) or 'nothing (all overridden)'}"
    if skipped:
        note += f"; skipped {', '.join(skipped)}"
    print(note)
