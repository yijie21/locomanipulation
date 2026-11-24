"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md §2, D6-D8, D11): the scene RSI inner loop on one task
family -- the warehouse loop (scalebfm_warehouse_rsi_20261014/loop.py) with skills as candidates. One iteration:
Reviewer (claude -p) -> Prober (the hypotheses' parameter overrides on the diagnostic pool) -> Manager (claude -p) ->
Engineers (claude -p, three in parallel by default: code in the skills package and/or overrides) -> Composer (claude -p:
one candidate from their drafts) -> Gate (gate.py) -> ledger. State lives in files under
<sandbox>/outputs/scene_rsi_20261016/<family>/loop, so the driver stops and resumes at any iteration.

Runs inside a sandbox checkout (sandbox.py). The roles see the sandbox, rules.md and the plan; the acceptance and sealed
pools and the gate live in the hidden tree (`..._hidden/`), and every tool call in a role's transcript is grepped for
that name and for paths outside the sandbox (the warehouse loop's audit, own tool-results excepted; since TODO 32.13 only
the calls' inputs are read, not the role's thinking, text or StructuredOutput answer). A hit voids what the role
produced and the loop goes on (TODO 36.7, user 2026-10-22): a Reviewer's, Manager's or Composer's (or the only
Engineer's) voids the iteration, one Engineer's of K only its draft; two voided iterations in a row stop the loop.

The Engineer may change only the sandbox's skills package (src/locomani/tasks/scene/skills/); anything else it touches
voids the call and is restored. On a rejection the package goes back to the incumbent's snapshot.

Stops (D8 and the warehouse budgets): the incumbent's acceptance successes >= STOP_RATE of the pool, this run's gate
calls >= 12, wall clock >= MAX_WALL_H, a Manager `stop` (model cost has no cap since the user lifted it on 2026-10-19;
RSI_COST_CAP_USD sets one), MAX_VOID_STREAK iterations in a row voided by the audit (an abnormal stop); and
(D13.4) the incumbent wins >= STOP_RATE of the acceptance starts that no task-wide park covers, when those are at
least UNPARKED_MIN of the pool. Below that share the gate judges on an acceptance pool drawn afresh from the unparked
tasks (`accept_pool`, rule 26).

The candidate pool (TODO 33.8, D7, user 2026-10-22): the gate (rule 2: b - c >= 3 and c <= b / 3, no p) leaves the
best POOL_KEEP rejected candidates with b > c in loop/candidates_pool.json, cleared when the incumbent changes. The
Manager may give `base: <hash>` (the Engineers start from that candidate, the gate still compares with the incumbent);
the Composer may merge a pool candidate's edits or adopt one as it stands (`adopted_pool`), e.g. one an older gate rule
rejected. TODO 33.14 (D11): the Manager context leads with `recurring` (mechanisms proposed in two or more iterations,
never built, from its `repeats` links), and `reopen` is not offered after MAX_REOPEN_RUN reopens in a row. TODO 33.13
(rules.md 30): a draft whose skills package holds a string constant equal to an object's name is void.

TODO 33.15-33.17 (user 2026-10-22: "把 33.15 到 33.18 先做了吧，开始下一轮的 RSI"; rules.md 31-33, version 15): round
records carry a failure class (scene_round.py, failure_class.py); the gate reports its tally per class; the failure cards
lead with the classes. After the probe each hypothesis gets a `status` (probe_status). A mechanism proposed in two or more
iterations, never built and not refuted takes one Engineer slot (force_build, a `forced` ledger row) unless the
Manager's `skip` names one of SKIP_CHECKS as failing. After a rejection the Manager reads `repair_notes`: the failure
classes, the gate's per-class tally and CLASS_SUGGEST. `run --stop-rate R` sets the 70 % stop for a run continued from
a strong incumbent.

Infeasibility (plan §8, D13): a Reviewer's claim that some starts cannot be done with the present actions must give the
actions it assumed, a hard constraint as a check file (claim_check.py), exits, what would refute it, and 1-3 attempts.
The driver runs the check and the attempts (closed loop); both must hold to park the starts. Parked starts keep
counting as failures everywhere; each iteration's incumbent diagnostic rounds reopen any park whose start it finishes
(the claim is recorded as wrong). The parks' exits, grouped by capability, are the Manager's gap candidates; its
`gap_proposals` (skills too big for one Engineer call) go to gaps.jsonl for the outer loop. An exit that fits one
Engineer call (TODO 27.8.7) is the Manager's `reopen`: the Engineer builds it, the candidate runs on that park's
diagnostic starts only (no gate call), and a finished start lifts the park and sends the candidate to the gate as usual.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/rsi/loop.py run pick_single > /dev/null 2>&1 &
    $PY scripts/rsi/loop.py status pick_single
    $PY scripts/rsi/loop.py run pick_single --stub --max-iters 1   # plumbing, no model calls
"""

from __future__ import annotations

import argparse
import calendar
import collections
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]
import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import design as DS  # noqa: E402
import diag_pool as DP  # noqa: E402
import family as F  # noqa: E402
import records as RC  # noqa: E402
import segments as SG  # noqa: E402
from gate import GATE_RULE, candidate_hash, drop_limit, overrides  # noqa: E402

from locomani import paths  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"
PLAN = REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md"
RULES = HERE / "rules.md"
PROMPTS = HERE / "prompts"  # configure() resets SKILLS, RULES, PLAN, PROMPTS and CAPS from backend.layout(family)
CAPS = dict(BK.SCENE_CAPS)
STUB_FILE = (
    "__init__.py"  # the file a stub draft appends its comment to; a backend whose __init__.py is frozen names another
)
MODEL = os.environ.get("RSI_MODEL", "claude-opus-5-5")
COST_CAP = float(os.environ["RSI_COST_CAP_USD"]) if "RSI_COST_CAP_USD" in os.environ else None  # user 10-19: no cap
WALL_DEFAULT_H = 6.0
MAX_WALL_H = float(os.environ.get("RSI_MAX_WALL_H", WALL_DEFAULT_H))  # cmd_run resolves it again (resolve_settings)
MAX_GATE_CALLS = 12
MAX_VOID_STREAK = 2  # TODO 36.7 (user 2026-10-22): voided iterations in a row that stop the loop
HISTORY_FULL = 2  # TODO 36.7: the Manager reads the last 2 iterations' ledger rows in full, older ones one line each
MAX_HYPOTHESES = 4
POOL_KEEP = 2  # TODO 33.8 (D7, user 2026-10-22): rejected candidates kept for the Manager's `base` and the Composer
MAX_REOPEN_RUN = 2  # TODO 33.14 (D11): Manager `reopen` decisions in a row at most
MAX_ATTEMPTS = 3  # closed-loop attempts per infeasibility claim
STOP_RATE_DEFAULT = 0.70  # D8: the inner loop stops when the incumbent wins this share of the acceptance pool
STOP_RATE = STOP_RATE_DEFAULT  # this run's (`run --stop-rate`, resolve_settings; TODO 33, user 2026-10-22)
# TODO 37.3 (D, user 2026-10-23): minutes a straggler Engineer gets once ceil(K/2) drafts are in; 0 waits for ever.
# Run C's slowest Engineer took 22-37 min against a median draft of about 14, waiting on its own one-process rounds.
ENGINEER_LATE_MIN_DEFAULT = 10.0
ENGINEER_LATE_MIN = ENGINEER_LATE_MIN_DEFAULT  # this run's (`run --engineer-late-min`, resolve_settings)
# TODO 37.4 (E, user 2026-10-23): after the gate rejects the composed candidate, the best draft it has not judged goes
# to the gate in the same iteration (one more gate call, 3-5 min, instead of another 60-80 min iteration)
SECOND_SHOT_DEFAULT = False  # 2026-11-20: off; `run --second-shot` still asks for it
SECOND_SHOT = SECOND_SHOT_DEFAULT  # this run's (`run --second-shot / --no-second-shot`, resolve_settings)
REPAIR = (
    False  # run --repair: a draft that broke a start the incumbent finishes gets one more turn (rule 38, TODO 44.2)
)
# TODO 45.7 (user 2026-11-09): RSI_SHADOW=<shadow.py of the imagined Motor> scores each iteration's drafts for the hidden
# tree and nothing else. An environment variable taken out of the environment, not a flag: meta.json is what roles read,
# and a role's subprocess inherits the environment.
SHADOW = os.environ.pop("RSI_SHADOW", None)
SHADOW_PROCS = 6
SHADOW_TIMEOUT_S = 900
# TODO 45.7.2 (user 2026-11-09): RSI_IMAGINE=<composer_view.py of the imagined Motor> gives the Composer the imagined
# result on starts no real round runs on. Off, the Composer's prompt and context are what they were, byte for byte.
IMAGINE = os.environ.pop("RSI_IMAGINE", None)
IMAGINE_TIMEOUT_S = 3600  # about 23 minutes with 6 processes, 40 with 3
REPAIR_MIN = 20.0  # minutes of that turn; its own clock: --engineer-late-min is for a draft still missing, and a
# replay's second repair lost 13 minutes of work to it with its first draft already in
# TODO 33.23 (plan section 3.8, D21; user 2026-10-23): the diagnostic pool is re-drawn when the incumbent fails fewer
# than DIAG_MIN_FAILS of its starts (run C's incumbent was 30/30 on it while 19 of 90 hidden-pool starts failed, so the
# roles saw no failure at all): fresh starts from the family's diagnostic seed range, the incumbent runs on them, the
# failing ones replace passing starts until DIAG_TARGET_FAILS fail; at most DIAG_DRAW_MAX fresh starts a redraw; 0 = off
DIAG_MIN_FAILS_DEFAULT, DIAG_TARGET_FAILS_DEFAULT, DIAG_DRAW_MAX_DEFAULT = DP.MIN_FAILS_DEFAULT, DP.TARGET_FAILS_DEFAULT, DP.DRAW_MAX_DEFAULT  # fmt: skip
DIAG_MIN_FAILS, DIAG_TARGET_FAILS, DIAG_DRAW_MAX = DIAG_MIN_FAILS_DEFAULT, DIAG_TARGET_FAILS_DEFAULT, DIAG_DRAW_MAX_DEFAULT  # fmt: skip
SKIP_CHECKS = ("interface_kept", "safety_checks_kept", "changed_code_reached", "evidence_not_refuting")  # TODO 33.17
STATUSES = ("hypothesis", "verified", "refuted", "mixed", "untested")  # TODO 33.17: a hypothesis after its probe
# TODO 33.17 (plan section 3.4, from DexAgent's diagnose_stage3_failure): what a failure class's next step usually is,
# shown to the Manager after a rejection (repair_notes); the class comes from scripts/rsi/failure_class.py (33.15)
CLASS_SUGGEST = {
    "too_wide": "the object does not fit the open jaw on the side presented: scripts/rsi/graspability.py (rotate the "
    "approach to a side that fits, a rim pinch, a top pinch)",
    "side_reach": "the hand did not get to the grasp: a nearer stance, or re-approach through stage() (rules.md 31)",
    "pinch_reach": "the top pinch's wrist attitude was not delivered: a twist the arm delivers, a lower pre-grasp",
    "walk_off": "the walk stopped off the stance: re-walk through stage() with stance_check, or a stance with more "
    "clearance",
    "knocked": "the object moved before the grip: a raised-hand approach, a path clear of it",
    "slip": "gripped, then lost: grip_check / lift_check and a re-grip through stage()",
    "fell": "the robot fell: a controller or motor-layer limit, an escalation (rules.md 20), not a skill fix",
}
SCENE_CLASS_SUGGEST = CLASS_SUGGEST  # configure() swaps in a second backend's own table (layout()['class_suggest'])
ATTRIBUTE_CLASSES = ("fell_walk",)  # TODO 33.20: failure classes the driver attributes to a layer (isolate.CLASSES)
ATTRIBUTE_MIN = 2  # ... once the incumbent fails at least this many diagnostic rounds of the class
UNPARKED_MIN = 0.5  # D13.4's line counts only when the starts outside the parked tasks are this share of the pool
REDRAW_SEED = 5000  # a redrawn acceptance pool (rule 26, redraw_accept) takes the family's acceptance seed + this
# (user 2026-10-18: r1_in_single stopped at 14/16 on the acceptance pool and was 9/20 on the sealed one)
PROCS = int(os.environ.get("RSI_PROCS", "12"))  # simulation processes; the runbook's machine: 12 over 2 GPUs
GPUS = os.environ.get("RSI_GPUS", "0,1")  # CUDA devices the shards rotate over (one GPU: RSI_GPUS=0 RSI_PROCS=6)
ORIGIN = (REPO / "third_party").resolve().parent  # a sandbox links third_party to the origin checkout
SCRATCH_ROOT = Path(paths.TMP)  # <LOCOMANI_WORKSPACE>/tmp: the roles' scratch directory and the leak check's "scratch"
ROLE_GRACE_S = 120  # after a role answers, its leftover processes get this long before the driver kills them
SKILLS_NOTE = (
    "the incumbent's skills; `pick.py` chooses the edge, the grasp yaw and the stance, `grasp_program.py` is the "
    "ported closed-loop hand program"
)
ISOLATION = (
    "\n\nThis checkout ({REPO}) and your scratch directory ({TMP}) are the whole world for this run. Do not read "
    "or list anything outside them, not even their parent directories (an `ls /workspace/tmp/` voided a Reviewer on "
    "2026-10-20): no other checkout of this project, nothing under ~/.claude, no path through third_party/.. "
    "A role call that does is void.\n"
)
FAMILY = ""
LOOP = HIDDEN = STATE = GATE_DIR = Path()
# role -> claude --effort level. Pinned: role calls used to inherit the operator's settings.json (effortLevel xhigh for
# Opus 5.5 on this machine, so every recorded run is xhigh), and another operator's settings would change the loop.
# run --effort engineer=medium overrides one role (TODO 32.8). Reviewer high since TODO 37.5 (2026-10-23: 16 replays on
# 4 states, model time 15.6 vs 24.7 min, wall 33 vs 57, the same mechanisms found, coverage and probe gains not lower;
# `--effort reviewer=xhigh` restores it); the Engineers stay xhigh (32.8).
EFFORT: dict = dict(reviewer="high", manager="xhigh", engineer="xhigh", composer="xhigh")
STOP_AFTER = None  # run --stop-after probe: a replay of the Reviewer and the Prober only (TODO 32.6)
SCOPED = False  # run --scoped-review: two Reviewers, procedure and object/scene, two hypotheses each (TODO 32.11)
REVIEWERS = 1  # run --reviewers N: N Reviewers on the same inputs, no scopes, MAX_HYPOTHESES each (TODO 43.2: not the
# default, the gate gave 34.25 / 40 against 35.75 with one, and a cause hidden from one Reviewer was hidden from both)
CANDIDATES = 3  # run --candidates K: K Engineers in parallel and a Composer (TODO 32.9, Harness2's compose step); 3 by
# default since TODO 32.15 (user 2026-10-22: "设置成默认的吧"), --candidates 1 is the one-Engineer loop, no Composer
LIBRARY_NOTE = (  # TODO 32.10: shown when the sandbox holds the experience library
    "The experience library, scripts/rsi/experience/INDEX.md: what earlier families' loops found, one entry per family "
    "(per change a verdict -- helped, hurt, no observed effect, confounded --, at most one lesson with its cue, and the "
    "code that run ended with in final.patch). Read the index; open an entry when its cue matches a failure here. An "
    "entry's numbers come from other objects and surfaces: they are hypotheses for this family, to check on these "
    "rounds, and where this family's rounds disagree, trust the rounds.\n"
)
SCOPES = {  # TODO 32.11 (Harness2's scoped proposals): each scoped Reviewer gives at most two hypotheses in its scope
    "P": "YOUR SCOPE: HOW THE ROBOT DOES IT. You own the skill's procedure -- the order and timing of walk, stance, reach, "
    "grip, lift and hold, the tolerances and when a step counts as done, mode switches, approach paths, retries and "
    "recovery -- the part that should hold for any object. Trace a failing round's steps, fix the earliest step where "
    "it diverges from a success. Do not propose object-specific grasp choices; another Reviewer owns them.",
    "O": "YOUR SCOPE: WHAT THE OBJECTS AND THE SCENE REQUIRE. You own what this family's objects and surfaces demand -- "
    "which side and orientation a grasp can take given the object's size and shape, the jaw's width, the surface's "
    "height, slabs above, neighbours and edges -- written as rules over the geometry, not lists of objects. Do not "
    "propose changes to the step sequence or its timing; another Reviewer owns them.",
    # TODO 33.3 (plan section 4): the third direction, a different way of doing it
    "S": "YOUR SCOPE: A DIFFERENT WAY TO DO IT. You own how a person would change the approach when the current one "
    "cannot work: stand elsewhere (the other side, a nearer edge, the side), handle the object first (pull it closer, "
    "push it to an edge, turn it, tilt it, move a neighbour away), use both hands, switch hands. First show with the "
    "body model that the current way falls outside what the body delivers, then that the new way falls inside it; "
    "what a parameter cannot test goes in `needs_code`. Do not tune the current procedure's tolerances or timing, or "
    "the current approach's grasp side; the other Reviewers own them.",
}
SCOPED_NOTE = (  # TODO 33.3: what the Manager reads when the scoped Reviewers ran
    "Hypotheses come from three scoped Reviewers: P (the procedure), O (what the objects and the scene require) and S "
    "(a different way of doing it: another stance, handling the object first, two hands, the other hand). S hypotheses "
    "mostly need code, so their probes are often empty: judge them by the body model's prediction their evidence "
    "cites, not by a missing probe result.\n"
)
TWIN_NOTE = (  # TODO 43.2: what the Manager reads when several Reviewers ran
    "Hypotheses come from several Reviewers who worked independently on the same inputs (the first letter of an id says "
    "which). The same mechanism from two of them is agreement, not two mechanisms: give it one slot, and spend the "
    "other slots on what only one of them found when its evidence holds.\n"
)
CANDIDATES_NOTE = (  # TODO 32.9: what the Manager reads when an iteration builds several candidates
    "This iteration builds up to {K} candidates at once: with `engineer`, list up to {K} distinct hypothesis ids in "
    "`hypotheses`, best first (a repeated id is dropped, never run twice; a free slot is filled by the driver with another "
    "hypothesis that is not refuted and whose tasks do not overlap the ones assigned, or left empty; `hypothesis` is the "
    "first). `tune` lists hypotheses whose best probed set (`best_set`, net at least 1, no `needs_code`) the driver builds "
    "into a candidate itself, no Engineer; with `tune` given, `hypotheses` may be empty. {K} Engineers build the rest in "
    "parallel from the incumbent, each runs on the diagnostic starts its scope names (and a few others), and a Composer "
    "merges what those rounds support into one candidate for the gate, adopts one, or keeps the incumbent.\n"
)
PATCHED = re.compile(r"^\+\+\+ \S+/skills/(\S+)", re.M)  # the files a candidate patch changes

CLAIM_SCHEMA = {  # D13.2: an infeasibility claim
    "type": "object",
    "required": ["id", "task", "kind", "starts", "actions_assumed", "bound", "exits", "refuted_by", "attempts"],
    "properties": {
        "id": {"type": "string"},
        "task": {"type": "string"},
        "kind": {"enum": ["current_skills", "physical"]},
        "starts": {"type": "array", "items": {"type": "string"}},
        "actions_assumed": {"type": "string"},
        "bound": {
            "type": "object",
            "required": ["statement", "check"],
            "properties": {"statement": {"type": "string"}, "check": {"type": "string"}},
        },
        "exits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["capability", "how"],
                "properties": {"capability": {"type": "string"}, "how": {"type": "string"}},
            },
        },
        "no_exit_reason": {"type": "string"},
        "refuted_by": {"type": "string"},
        "attempts": {
            "type": "array",
            "maxItems": MAX_ATTEMPTS,
            "items": {
                "type": "object",
                "required": ["name", "set"],
                "properties": {"name": {"type": "string"}, "set": {"type": "object"}},
            },
        },
    },
}
HYPOTHESES_SCHEMA = {
    "type": "object",
    "required": ["hypotheses", "notes"],
    "properties": {
        "notes": {"type": "string"},
        "infeasible": {"type": "array", "items": CLAIM_SCHEMA},
        "hypotheses": {
            "type": "array",
            "maxItems": MAX_HYPOTHESES,
            "items": {
                "type": "object",
                "required": ["id", "claim", "instead", "evidence", "delta", "predicted_fixed", "predicted_broken"],
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "instead": {"type": "string"},  # TODO 33.17: what to change, in one sentence
                    "status": {"enum": list(STATUSES)},  # set by the driver after the probe (probe_status)
                    "evidence": {"type": "string"},
                    "delta": {"type": "object", "properties": {"set": {"type": "object"}}, "required": ["set"]},
                    "needs_code": {"type": "string"},  # the probe never runs it; its `delta` is a hint for the Engineer
                    # change 1: the model names the knobs, the driver sweeps them (sweep_specs, probe)
                    "sweep": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "required": ["key", "values"],
                            "properties": {
                                "key": {"type": "string"},
                                "values": {"type": "array", "minItems": 2, "maxItems": 5},
                            },
                        },
                    },
                    # TODO 33.15.1: a measurable claim the driver checks on the record library when the probe cannot
                    "measure": {
                        "type": "object",
                        "required": ["feature", "op", "value", "expect"],
                        "properties": {
                            "feature": {"type": "string"},
                            "op": {"enum": sorted(RC.OPS)},
                            "value": {"type": ["number", "string"]},
                            "expect": {"enum": ["fails", "succeeds"]},
                            "tasks": {"type": "array", "items": {"type": "string"}},
                            "scope": {"enum": ["incumbent", "all"]},
                        },
                    },
                    # TODO 33.19: a design probe -- one factor at a few levels, forced on the same diagnostic starts
                    "design": {
                        "type": "object",
                        "required": ["factor", "levels"],
                        "properties": {
                            "factor": {"enum": list(DS.FACTORS)},
                            "levels": {"type": "array", "items": {"type": "number"}},
                            "starts": {"type": "integer"},
                            "tasks": {"type": "array", "items": {"type": "string"}},
                            "name": {"type": "string"},
                        },
                    },
                    "predicted_fixed": {"type": "array", "items": {"type": "string"}},
                    "predicted_broken": {"type": "array", "items": {"type": "string"}},
                    "tasks": {"type": "array", "items": {"type": "string"}},  # a slow bed: where the probe runs
                },
            },
        },
    },
}
MANAGER_SCHEMA = {
    "type": "object",
    "required": ["decision", "reason", "summary_zh"],
    "properties": {
        "decision": {"enum": ["engineer", "reopen", "review", "stop"]},
        "hypothesis": {"type": "string"},
        "hypotheses": {"type": "array", "items": {"type": "string"}},  # TODO 32.9: one per Engineer, best first
        "reopen_claim": {"type": "string"},  # 27.8.7: the park whose exit the Engineer builds
        "base": {"type": "string"},  # TODO 33.8: a candidate_pool hash the Engineers start from
        # change 1: hypotheses whose best probed set the driver builds into a candidate itself (kind "tuned")
        "tune": {"type": "array", "items": {"type": "string"}},
        "skip": {  # TODO 33.17: a forced hypothesis not built, with the check that fails ({"H2": "interface_kept"})
            "type": "object",
            "additionalProperties": {"enum": list(SKIP_CHECKS)},
        },
        "repeats": {  # TODO 33.14: this iteration's hypotheses that are earlier iterations' mechanisms ("NN:id")
            "type": "array",
            "items": {
                "type": "object",
                "required": ["now", "same_as"],
                "properties": {"now": {"type": "string"}, "same_as": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "focus": {"type": "string"},
        "reason": {"type": "string"},
        "escalation": {"type": "string"},
        "infeasible": {"type": "string"},
        "gap_proposals": {  # D13.6: skills too big for one Engineer call, for the outer loop
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "capability", "why", "tasks"],
                "properties": {
                    "name": {"type": "string"},
                    "capability": {"type": "string"},
                    "why": {"type": "string"},
                    "tasks": {"type": "array", "items": {"type": "string"}},
                    "est_cost": {"type": "string"},
                },
            },
        },
        "summary_zh": {"type": "string"},
    },
}
COMPOSER_SCHEMA = {  # TODO 32.9
    "type": "object",
    "required": ["decision", "notes"],
    "properties": {
        "decision": {"enum": ["merge", "adopt", "keep"]},
        "adopted": {"type": "integer"},
        "adopted_pool": {"type": "string"},  # TODO 33.8: adopt a candidate_pool entry (its hash) as it stands
        "set": {"type": "object"},
        "mechanism": {"type": "string"},
        "notes": {"type": "string"},
    },
}

ENGINEER_SCHEMA = {
    "type": "object",
    "required": ["mechanism", "files_changed", "set", "tests", "scope"],
    "properties": {
        "mechanism": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "set": {"type": "object"},
        "tests": {"type": "string"},
        # change 1: knobs the draft adds or uses; the driver sweeps them on the draft's starts (tune_draft)
        "tunable": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["key", "values"],
                "properties": {
                    "key": {"type": "string"},
                    "values": {"type": "array", "minItems": 2, "maxItems": 5},
                },
            },
        },
        "scope": {  # rule 38: what the draft says it changes; the driver measures what it did change (footprint)
            "type": "object",
            "required": ["when", "starts"],
            "properties": {
                "when": {"type": "string"},
                "starts": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def configure(family: str) -> None:
    global FAMILY, LOOP, HIDDEN, STATE, GATE_DIR, PLAN, SKILLS, RULES, PROMPTS, CAPS, STUB_FILE, CLASS_SUGGEST
    FAMILY = family
    LOOP = REPO / "outputs" / "scene_rsi_20261016" / family
    HIDDEN = REPO / "outputs" / "scene_rsi_20261016_hidden" / family  # acceptance / sealed pools and the gate
    STATE, GATE_DIR = LOOP / "loop", HIDDEN / "gate"
    spec = family_spec()
    PLAN = REPO / spec["plan"] if spec.get("plan") else REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md"
    CLASS_SUGGEST = SCENE_CLASS_SUGGEST
    if BK.of(spec):  # a second backend has its own package, rules and prompts; a scene family keeps the constants
        lay = BK.layout(spec)
        SKILLS, RULES, PROMPTS, CAPS = lay["skills"], lay["rules"], lay["prompts"], lay["caps"]
        PLAN = lay.get("plan") or PLAN
        STUB_FILE = lay.get("stub_file") or STUB_FILE
        CLASS_SUGGEST = lay.get("class_suggest") or SCENE_CLASS_SUGGEST


def family_spec() -> dict:
    p = HERE / "families" / f"{FAMILY}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def prompt_vars() -> dict:
    """What a family sets in the role prompts (TODO 31): the plan the roles read, a note on the seed's skills, and the
    pool sizes. A family without them gets the text the prompts had before."""
    spec = family_spec()
    pools = spec.get("pools") or {}
    n_diag = (pools.get("diag") or {}).get("n", 50)
    if cap("check_half"):  # TODO 48.17: the roles see the half the check pool leaves (half of each task, rounded up)
        n_diag -= n_diag // 2
    return dict(
        PLAN=PLAN,
        SKILLS_NOTE=spec.get("skills_note") or SKILLS_NOTE,
        N_DIAG=n_diag,
        N_ACCEPT=(pools.get("accept") or {}).get("n", 60),
        CANDIDATES_NOTE=(CANDIDATES_NOTE.replace("{K}", str(CANDIDATES)) if CANDIDATES > 1 else "")
        + (SCOPED_NOTE if SCOPED else "")
        + (TWIN_NOTE if REVIEWERS > 1 else ""),
        LIBRARY=LIBRARY_NOTE if (HERE / "experience" / "INDEX.md").exists() else "",
    )


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def ledger(entry: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": now(), **entry}, ensure_ascii=False, default=str) + "\n")


def entries() -> list[dict]:
    p = STATE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _live_pool() -> tuple[Path, list[str], list[str]]:
    """(the redrawn acceptance pool's path, the tasks it holds, the task-wide parked tasks) -- the path is the
    family's own pool when the parks leave at least UNPARKED_MIN of it (or nothing)."""
    base = HIDDEN / "pool_accept"
    labels = B.pool_labels(base) if base.exists() else []
    parked = sorted({p["task"] for p in jload(STATE / "parked.json", {}).values() if p["scope"] == "task"})
    live = sorted({task_of(x) for x in labels} - set(parked))
    keep = [x for x in labels if task_of(x) in live]
    if not parked or not keep or len(keep) >= UNPARKED_MIN * len(labels):
        return base, live, parked
    return HIDDEN / f"pool_accept_{hashlib.sha256(json.dumps(live).encode()).hexdigest()[:8]}", live, parked


def accept_pool() -> Path:
    """The acceptance pool the gate judges on: the family's, or the one `redraw_accept` drew for the present parks."""
    pool, _, _ = _live_pool()
    return pool if pool.exists() else HIDDEN / "pool_accept"


def redraw_accept() -> None:
    """Rule 26 (user 2026-10-19, plan §12): once task-wide parks cover more than 1 - UNPARKED_MIN of the acceptance
    pool, a pool of the same size drawn afresh (seeds + REDRAW_SEED) from the tasks they leave -- r2_in_single had 36
    of its 60 starts on parked tasks, and its candidates were 3-5 net wins better four times, never significant. A
    new set of parked tasks draws a new pool; the gate runs the incumbent on starts it has not seen. Parked starts
    still count in the sealed and diagnostic pools."""
    pool, live, parked = _live_pool()
    if pool.exists() or BK.of(FAMILY):  # a backend's pools are named by ids, not by a seed a redraw could offset
        return
    fam = F.load(FAMILY)
    fam["tasks"], fam["rows"] = live, {t: fam["rows"][t] for t in live}
    spec = fam["pools"]["accept"]
    fam["pools"]["accept"] = dict(spec, seed=int(spec["seed"]) + REDRAW_SEED)
    res = F.draw(fam, "accept", pool)
    ledger(
        dict(step="redraw_accept", tasks=live, parked_tasks=parked, n=res["written"], unavailable=res["unavailable"])
    )


def incumbent_ok_here(es: list[dict]) -> int | None:
    """TODO 36 (review 2026-10-22, L12): the incumbent's successes on the CURRENT acceptance pool, from the latest gate
    call that was judged on it (rows carry `pool`): the candidate's count if that call accepted, the incumbent's if it
    rejected. None when no such call exists (a rule-26 redraw since, or rows from before `pool` was recorded) -- the
    caller then keeps the last accepted count."""
    pool = str(accept_pool())
    for e in reversed(es):
        r = e.get("result")
        if e.get("step") == "gate" and not e.get("void") and e.get("pool") == pool and r:
            return r["candidate_ok"] if e.get("accepted") else r["incumbent_ok"]
    return None


def run_cost(es: list[dict], meta: dict | None) -> float:
    """TODO 36 (review 2026-10-22, L6): cost of THIS run -- ledger rows from meta['started'] on; rows a replay copied
    from its source run (older `t`) are not this run's spending."""
    t0 = (meta or {}).get("started") or ""
    return sum(e.get("cost_usd", 0.0) for e in es if str(e.get("t", "")) >= t0)


def run_gates(es: list[dict], meta: dict | None) -> list[dict]:
    """TODO 33.18.2: the gate calls of THIS run, not void -- rows from meta['started'] on, like run_cost; a replay.py
    setup copies the source run's gate rows (run C started with 9 of 12 spent and stopped after 3)."""
    t0 = (meta or {}).get("started") or ""
    return [e for e in es if e.get("step") == "gate" and not e.get("void") and str(e.get("t", "")) >= t0]


def voided_in_a_row(es: list[dict]) -> list[int]:
    """The latest iterations, oldest first, that the audit voided (`void_iter` rows), counted back one by one from the
    newest iteration in the ledger until one that was not voided or has no rows. An iteration a re-probe (`void`) or
    re-audit (`reaudit`) row takes up again after its void_iter row is not void any more."""
    status: dict[int, bool] = {}
    for e in es:
        if "iter" not in e or not isinstance(e["iter"], int):
            continue
        n = e["iter"]
        if e.get("step") == "void_iter":
            status[n] = True
        elif e.get("step") in ("void", "reaudit"):
            status[n] = False
        else:
            status.setdefault(n, False)
    out, n = [], max(status, default=-1)
    while status.get(n):
        out.append(n)
        n -= 1
    return out[::-1]


def stop_reason(meta: dict) -> str | None:
    es = entries()
    voided = voided_in_a_row(es)
    if len(voided) >= MAX_VOID_STREAK:  # TODO 36.7: an abnormal stop (is_abnormal)
        return f"audit: iterations {', '.join(map(str, voided))} voided in a row (roles named off-limits paths)"
    accepted = [e for e in es if e.get("step") == "gate" and e.get("accepted") and not e.get("void")]
    n_accept = len(B.pool_labels(accept_pool()))
    here = incumbent_ok_here(es)
    if here is not None:
        if here >= STOP_RATE * n_accept:
            return f"the incumbent wins {here}/{n_accept} >= {STOP_RATE:.0%}: sealed set"
    elif accepted and accepted[-1]["result"]["candidate_ok"] >= STOP_RATE * n_accept:
        return f"the incumbent wins {accepted[-1]['result']['candidate_ok']}/{n_accept} >= {STOP_RATE:.0%}: sealed set"
    used_up = check_stop(es, meta)  # TODO 48.17 check_half
    if used_up:
        return used_up
    gates = run_gates(es, meta)
    if len(gates) >= MAX_GATE_CALLS:
        return f"gate calls {len(gates)} >= {MAX_GATE_CALLS}"
    if wall_h(meta) >= MAX_WALL_H:
        return f"wall clock >= {MAX_WALL_H} h"
    cost = run_cost(es, meta)
    if COST_CAP is not None and cost >= COST_CAP:
        return f"model cost {cost:.0f} >= {COST_CAP:.0f} USD"
    m = [e for e in es if e.get("step") == "manager" and not e.get("void")]  # a voided Manager decided nothing
    if m and m[-1].get("decision") == "stop":
        return "manager: stop"
    inc = jload(GATE_DIR / "incumbent.json")
    v = unparked_view(Path(inc["rounds"])) if inc else None
    if v and v["n"] >= UNPARKED_MIN * v["all_n"] and v["ok"] >= STOP_RATE * v["n"]:
        return (f"the incumbent wins {v['ok']}/{v['n']} >= {STOP_RATE:.0%} of the acceptance starts outside the parked "
                f"tasks {v['parked_tasks']} ({v['all_ok']}/{v['all_n']} of all): sealed set")  # fmt: skip
    return None


# -- infeasibility claims and parks (D13, plan §8) -----------------------------------------------------------------


def task_of(label: str) -> str:
    return label.rsplit("_s", 1)[0]


def claim_problems(c: dict, diag: list[str]) -> list[str]:
    """What a claim lacks (D13.2): the actions it assumed, a hard constraint code can recompute (a check file), exits
    or -- for a physical claim -- why there are none, what would refute it, 1..MAX_ATTEMPTS attempts. Any problem
    and the claim is refused."""
    out = []
    if not c.get("starts"):
        out.append("no starts")
    bad = [x for x in c.get("starts") or [] if x not in diag or task_of(x) != c.get("task")]
    if bad:
        out.append(f"not diagnostic starts of {c.get('task')}: {bad}")
    if len((c.get("actions_assumed") or "").strip()) < 10:
        out.append("actions_assumed is empty")
    b = c.get("bound") or {}
    if not (b.get("statement") or "").strip():
        out.append("bound.statement is empty")
    chk = Path(b.get("check") or "")
    if chk.suffix != ".py" or not chk.is_file():
        out.append(f"bound.check {b.get('check')!r} is not a Python file")
    exits = [e for e in c.get("exits") or [] if (e.get("capability") or "").strip() and (e.get("how") or "").strip()]
    if not exits and not (c.get("kind") == "physical" and (c.get("no_exit_reason") or "").strip()):
        out.append("no exits (only a physical claim may give no_exit_reason instead)")
    if not (c.get("refuted_by") or "").strip():
        out.append("refuted_by is empty")
    n = len([a for a in c.get("attempts") or [] if a.get("set")])
    if not 1 <= n <= MAX_ATTEMPTS:
        out.append(f"{n} attempts with a non-empty set (need 1-{MAX_ATTEMPTS})")
    return out


def claim_id(c: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(c.get("id") or "claim"))[:60]


def verify_claims(claims: list[dict], inc: dict, it_dir: Path) -> list[dict]:
    """D13.3: a claim parks its starts only if its hard constraint holds on every one of them (claim_check.py, run by
    the driver on the family's diagnostic starts of that task) and none of its attempts -- the incumbent plus the
    attempt's overrides, closed loop on the claimed starts -- succeeds."""
    diag = B.pool_labels(LOOP / "pool_diag")
    meta = REPO / F.load(FAMILY)["scene"] / "scene_meta.json"
    res, jobs, owner = [], [], []
    for c in claims:
        r = dict(id=claim_id(c), task=c.get("task"), kind=c.get("kind"), problems=claim_problems(c, diag))
        res.append(r)
        if r["problems"]:
            continue
        d = it_dir / "claims" / r["id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(c["bound"]["check"], d / "check.py")  # the version that was judged
        task_starts = [x for x in diag if task_of(x) == c["task"]]
        cmd = [str(SIM_PY), str(HERE / "claim_check.py"), str(d / "check.py"), str(meta),
               *[str(LOOP / "pool_diag" / f"{x}.json") for x in task_starts]]  # fmt: skip
        try:
            out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=300,
                                 env=dict(os.environ, PYTHONPATH=str(REPO / "src")))  # fmt: skip
            r["bound"] = json.loads(out.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            r["bound"] = {}
            r["problems"].append(f"bound.check did not run: {type(e).__name__}")
            continue
        for k, a in enumerate([a for a in c["attempts"] if a.get("set")][:MAX_ATTEMPTS]):
            jobs.append(B.Job(Path(inc["dir"]) / "skills", d / f"attempt_{k}", {**inc["set"], **a["set"]},
                              tuple(c["starts"])))  # fmt: skip
            owner.append((r, a.get("name") or f"attempt {k}"))
    if jobs:
        B.run_jobs(jobs, FAMILY, LOOP / "pool_diag", PROCS, it_dir / "prober.log", GPUS)
    for (r, name), job in zip(owner, jobs):
        out = B.outcomes(job.out)
        whys = collections.Counter((x.get("plan_why") or x["why"] or "?")[:60] for x in out.values() if not x["ok"])
        r.setdefault("attempts", {})[name] = dict(n=len(out), ok=sorted(k for k, x in out.items() if x["ok"]),
                                                  missing=sorted(set(job.labels) - set(out)), why=dict(whys))  # fmt: skip
    for r, c in zip(res, claims):
        if r["problems"]:
            r["parked"] = False
            continue
        r["bound_ok"] = all(r["bound"].get(x, {}).get("holds") for x in c["starts"])
        atts = list(r.get("attempts", {}).values())
        r["loop_ok"] = bool(atts) and all(not a["ok"] and not a["missing"] for a in atts)
        r["parked"] = r["bound_ok"] and r["loop_ok"]
        r["scope"] = "task" if set(c["starts"]) == {x for x in diag if task_of(x) == c["task"]} else "starts"
    return res


def park(results: list[dict], claims: list[dict], n: int) -> None:
    """Record the verified claims as parks (a claim with the same id replaces the old one) and every other claim as
    refused (a part missing) or unproven (the check or the attempts said otherwise)."""
    parked = jload(STATE / "parked.json", {})
    for r, c in zip(results, claims):
        if r.get("parked"):
            parked[r["id"]] = dict(iter=n, task=c["task"], kind=c["kind"], scope=r["scope"], starts=sorted(c["starts"]),
                                   statement=c["bound"]["statement"], actions_assumed=c["actions_assumed"],
                                   exits=c.get("exits") or [], no_exit_reason=c.get("no_exit_reason"),
                                   refuted_by=c["refuted_by"], user_confirmed=False, since=now())  # fmt: skip
            ledger(dict(step="park", iter=n, claim=r["id"], task=c["task"], kind=c["kind"], scope=r["scope"],
                        starts=sorted(c["starts"])))  # fmt: skip
        else:
            ledger(dict(step="claim_refused" if r["problems"] else "claim_unproven", iter=n, claim=r["id"],
                        task=r.get("task"), problems=r["problems"], bound_ok=r.get("bound_ok"),
                        loop_ok=r.get("loop_ok"), attempts=r.get("attempts")))  # fmt: skip
    jdump(STATE / "parked.json", parked)


def parked_labels(parked: dict, labels) -> dict:
    """{label: claim id} for the labels a park covers (a task-wide park: every start of its task)."""
    out = {}
    for cid, p in parked.items():
        for x in labels:
            if (p["scope"] == "task" and task_of(x) == p["task"]) or x in p["starts"]:
                out[x] = cid
    return out


def unpark(base: Path, n: int) -> list[str]:
    """D13.5: a parked start the incumbent now finishes reopens its claim, recorded as wrong. `base` holds the
    incumbent's diagnostic rounds, run at the start of every iteration (so after every accepted skill)."""
    parked = jload(STATE / "parked.json", {})
    out = B.outcomes(base)
    wins: dict = {}
    for x, cid in parked_labels(parked, out).items():
        if out[x]["ok"]:
            wins.setdefault(cid, []).append(x)
    for cid, won in sorted(wins.items()):
        p = parked.pop(cid)
        ledger(dict(step="unpark", iter=n, claim=cid, task=p["task"], won=sorted(won), claim_wrong=True))
    if wins:
        jdump(STATE / "parked.json", parked)
    return sorted(wins)


def reopen_test(n: int, it_dir: Path, cand: Path, cid: str) -> bool:
    """TODO 27.8.7: a candidate built for park `cid`'s exit runs on that park's diagnostic starts only, before any gate
    call. Returns whether it finished any of them; None finished: the candidate is dropped. The park stays either way
    (TODO 36 (review 2026-10-22): 32.12's three reopen candidates all finished a start, all were rejected, and the
    parks were gone): `settle_reopen` lifts it only once the gate has accepted this candidate."""
    parked = jload(STATE / "parked.json", {})
    p = parked.get(cid)
    out_dir = it_dir / "reopen" / cid
    if p is not None:
        B.run_jobs([B.Job(cand / "skills", out_dir, overrides(cand), tuple(p["starts"]))], FAMILY, LOOP / "pool_diag",
                   PROCS, it_dir / "prober.log", GPUS)  # fmt: skip
    out = B.outcomes(out_dir) if p is not None else {}
    won = reopen_won(it_dir, cid)
    whys = collections.Counter((x.get("plan_why") or x["why"] or "?")[:60] for x in out.values() if not x["ok"])
    ledger(dict(step="reopen_test", iter=n, claim=cid, n=len(out), ok=won, why=dict(whys),
                error=None if p is not None else "no such park"))  # fmt: skip
    return bool(won)


def reopen_won(it_dir: Path, cid: str) -> list[str]:
    """The park's starts the reopen candidate finished (reopen_test's rounds)."""
    return sorted(k for k, x in B.outcomes(it_dir / "reopen" / cid).items() if x["ok"])


def settle_reopen(n: int, cid: str, won: list[str], accepted: bool) -> None:
    """The gate's verdict on a reopen candidate: accepted -> the park lifts (`unpark`, by reopen; the claim was about
    the present skills and stands, so it is not recorded as wrong); rejected or void -> `reopen_rejected`, the park
    stays (TODO 36 (review 2026-10-22))."""
    parked = jload(STATE / "parked.json", {})
    p = parked.get(cid)
    task = p["task"] if p else None
    if accepted and p is not None:
        parked.pop(cid)
        jdump(STATE / "parked.json", parked)
        ledger(dict(step="unpark", iter=n, claim=cid, task=task, won=won, claim_wrong=False, by="reopen"))
    elif not accepted:
        ledger(dict(step="reopen_rejected", iter=n, claim=cid, task=task, won=won))


def unparked_view(rounds: Path) -> dict | None:
    """D13.4's second number on the acceptance pool: successes on the starts outside the task-wide parks (a park of
    some starts names diagnostic starts and says nothing about a hidden pool). None without such parks or rounds."""
    tasks = sorted({p["task"] for p in jload(STATE / "parked.json", {}).values() if p["scope"] == "task"})
    labels = B.pool_labels(accept_pool())
    out = {k: v for k, v in B.outcomes(rounds).items() if v.get("rule", 1) == B.SUCCESS_RULE}
    if not tasks or not labels or any(x not in out for x in labels):
        return None
    keep = [x for x in labels if task_of(x) not in tasks]
    return dict(ok=sum(out[x]["ok"] for x in keep), n=len(keep), parked_tasks=tasks,
                all_ok=sum(out[x]["ok"] for x in labels), all_n=len(labels))  # fmt: skip


def gap_candidates() -> dict:
    """D13.7: the parks' exits by capability, most tasks first -- what each would reopen."""
    out: dict = {}
    for cid, p in jload(STATE / "parked.json", {}).items():
        for e in p.get("exits") or []:
            g = out.setdefault(e["capability"].strip().lower(), dict(tasks=set(), starts=0, claims=[], how=[]))
            g["tasks"].add(p["task"])
            g["starts"] += len(p["starts"])
            g["claims"].append(cid)
            g["how"].append(e["how"][:300])
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]["tasks"]), -kv[1]["starts"], kv[0]))
    return {k: dict(v, tasks=sorted(v["tasks"])) for k, v in ranked}


# -- model calls (the warehouse loop's) ----------------------------------------------------------------------------


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


class RoleFailed(RuntimeError):
    pass


class RoleLate(RoleFailed):
    """TODO 37.3: the driver stopped the role (an Engineer past --engineer-late-min after the other drafts were in)."""


def role_tmp() -> Path:
    return SCRATCH_ROOT / f"rsi_scene_{REPO.name}"


def role_dir(role: str) -> Path:
    """TODO 36 (review 2026-10-22, L5): a role that runs beside others (engineer_2, reviewer_P) gets its own scratch
    directory under role_tmp(), so same-named scripts and output directories do not collide."""
    return role_tmp() / role if "_" in role else role_tmp()


_HID = dict(n=0, mode=None)  # reference-counted lock on the hidden tree (L9)
_HID_LOCK = threading.Lock()


def hidden_mode_file() -> Path:
    return STATE / "hidden_mode.txt"


@contextlib.contextmanager
def hidden_locked():
    """TODO 36 (review 2026-10-22, L9): while any role runs, the family's hidden directory has mode 000, so a `grep -r
    outputs/` or `ls -R` cannot read pools or verdicts (the audit sees only tool inputs, never what a command's output
    shows). Parallel roles share one lock; the original mode is kept in STATE so cmd_run can restore it after a
    crash. The gate and the hidden pools are used between role calls only."""
    with _HID_LOCK:
        if _HID["n"] == 0 and hidden_lockable():
            _HID["mode"] = stat.S_IMODE(HIDDEN.stat().st_mode)
            hidden_mode_file().write_text(oct(_HID["mode"]))
            HIDDEN.chmod(0)
        _HID["n"] += 1
    try:
        yield
    finally:
        with _HID_LOCK:
            _HID["n"] -= 1
            if _HID["n"] == 0 and _HID["mode"] is not None:
                HIDDEN.chmod(_HID["mode"])
                hidden_mode_file().unlink(missing_ok=True)
                _HID["mode"] = None


def hidden_lockable() -> bool:
    """Only a family directory inside ..._hidden/ is ever locked: a test once ran a role call with HIDDEN left at
    Path() -- the working directory -- and the repository root went to mode 000 (2026-10-22)."""
    return HIDDEN.parent.name == "scene_rsi_20261016_hidden" and HIDDEN.is_dir() and not HIDDEN.is_symlink()


def unlock_hidden() -> None:
    """A crash while a role ran leaves the hidden tree at mode 000: put it back (cmd_run, at start)."""
    if not hidden_lockable():
        return
    f = hidden_mode_file()
    if f.exists():
        HIDDEN.chmod(int(f.read_text().strip() or "755", 8))
        f.unlink()
    elif stat.S_IMODE(HIDDEN.stat().st_mode) & 0o700 == 0:
        HIDDEN.chmod(0o755)
    if hidden_parent_file().exists():  # a crash while the speculative Reviewer ran (parent_locked)
        HIDDEN.parent.chmod(int(hidden_parent_file().read_text().strip() or "755", 8))
        hidden_parent_file().unlink()
    if private_gate_file().exists():  # ... and the gate's private copy it left
        p = Path(private_gate_file().read_text().strip())
        if p.parent == HIDDEN.parent and re.fullmatch(r"\.[0-9a-f]{24}", p.name):
            shutil.rmtree(p, ignore_errors=True)
        private_gate_file().unlink()
    _HID.update(n=0, mode=None)


@contextlib.contextmanager
def rounds_locked(base: Path):
    """Rule 37: the incumbent's diagnostic rounds have mode 000 while the Reviewer runs, the failure cards are what it
    gets of them. In the replays of 2026-11-05 this cost as good a diagnosis as the digest with the rounds open and about
    20 % less (TODO 43). The Prober and the later roles read the rounds again."""
    mode = stat.S_IMODE(base.stat().st_mode)
    base.chmod(0)
    try:
        yield
    finally:
        base.chmod(mode)


def unlock_rounds() -> None:
    """A crash while the Reviewer ran leaves the rounds at mode 000: put them back (at the start of a command)."""
    for d in (LOOP / "diag").glob("incumbent_*"):
        if stat.S_IMODE(d.stat().st_mode) & 0o700 == 0:
            d.chmod(0o755)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


_GIT_LOCK = threading.Lock()


def undo_commits(head0: str) -> dict | None:
    """TODO 36 (review 2026-10-22, L10): a role that committed in the sandbox would hide its edits from changed_files()
    (git status against HEAD). HEAD moved: reset --soft to where it was, then unstage, so the edits are working-tree
    changes the stray / skills logic sees (a staged file would survive `git checkout -- f`). Returns the note for the
    role's ledger row."""
    with _GIT_LOCK:
        head1 = git_head()
        if not head0 or head1 == head0:
            return None
        subprocess.run(["git", "reset", "--soft", head0], cwd=REPO, capture_output=True)
        subprocess.run(["git", "reset", "-q"], cwd=REPO, capture_output=True)
    return dict(was=head0, moved_to=head1)


def claude(
    role: str, prompt: str, schema: dict, it_dir: Path, tools: str, stub=None, stop: threading.Event | None = None,
    resume: str | None = None,
) -> tuple[dict, dict]:  # fmt: skip
    """One role call. `stop` (TODO 37.3): an event the driver sets to give the role up; run_role then kills the
    session and raises RoleLate. `resume` (rule 38): one more turn in that finished session, under the same role, so
    it keeps what it read and its try.py budget; its prompt and raw answer go to <role>_repair_*."""
    if resume is None:
        prompt += fill(ISOLATION, REPO=REPO, TMP=role_dir(role))
    name = f"{role}_repair" if resume else role
    (it_dir / f"{name}_prompt.md").write_text(prompt)
    if stub is not None:
        return stub, dict(cost_usd=0.0, session=None, seconds=0.0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    role_dir(role).mkdir(parents=True, exist_ok=True)
    # TODO 36 (review 2026-10-22): no memory file from the sandbox's ancestors (the 23 KB machine guide in /workspace)
    # or the operator's home reaches a role; tested with one session
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", TMPDIR=str(role_dir(role)),
               # TODO 37.1: try.py's round budget is per role and iteration (rules.md 18)
               RSI_TRY_LEDGER=str(role_dir(role) / f"try_{it_dir.name}.jsonl"))  # fmt: skip
    cmd = [
        "claude", "-p", "--model", MODEL, "--output-format", "json", "--json-schema", json.dumps(schema),
        "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config",
        "--permission-mode", "bypassPermissions", "--disallowedTools", tools,
        *(["--effort", EFFORT.get(role) or EFFORT[role.split("_")[0]]] if (role in EFFORT or role.split("_")[0] in EFFORT) else []),
        *(["--add-dir", *allowed_dirs()] if allowed_dirs() else []),
        *(["--max-budget-usd", str(max(1.0, COST_CAP / 8))] if COST_CAP is not None else []),
    ]  # fmt: skip
    session = resume or str(uuid.uuid4())
    how = ["--resume", session] if resume else ["--session-id", session]
    before = transcript(session)
    seen = before.read_text(errors="replace").count(ANSWERED) if before is not None else 0
    t0 = time.time()
    head0 = git_head()
    with hidden_locked():
        stdout, stderr, rc, killed = run_role([*cmd, *how], prompt, env, session, stop=stop, seen=seen)
    committed = undo_commits(head0)
    (it_dir / f"{name}_raw.json").write_text(stdout or stderr)
    try:
        out = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RoleFailed(f"{role}: no JSON from claude -p (rc {rc}): {(stderr or stdout)[-300:]}") from e
    if out.get("is_error"):
        raise RoleFailed(f"{role}: {out.get('subtype')} {str(out.get('result'))[-300:]}")
    acct = dict(cost_usd=float(out.get("total_cost_usd") or 0.0), session=out.get("session_id"),
                seconds=round(time.time() - t0), usage=session_usage(out), killed_leftovers=killed,
                **({"committed": committed} if committed else {}))  # fmt: skip
    so = out.get("structured_output") or structured_from_transcript(acct["session"])
    if so is None:
        raise RoleFailed(f"{role}: no structured output; result: {str(out.get('result'))[-300:]}")
    return so, acct


ANSWERED = "Structured output provided successfully"  # in a session's transcript once it has answered


def run_role(
    cmd: list[str], prompt: str, env: dict, session: str, grace: float = ROLE_GRACE_S, poll: float = 10.0,
    stop: threading.Event | None = None, seen: int = 0,
) -> tuple[str, str, int, list[str]]:  # fmt: skip
    """Run a role's `claude -p` to its end. claude -p does not exit while a background shell it started still runs,
    and a role once left `until ! pgrep -f "<dir>"; do sleep 10; done`, which matches itself and never ends (iteration
    2, 2026-10-16: the driver stalled until a human killed it). So once the transcript shows the structured answer,
    the role's leftover processes get `grace` seconds, then the driver kills them (listed in the ledger). `stop` set
    (TODO 37.3): the session and everything under it are killed and RoleLate is raised. `seen`: the answers already in
    the transcript when a finished session takes another turn (rule 38)."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=REPO, env=env)  # fmt: skip
    t0, answered, killed, feed = time.time(), None, [], prompt
    while True:
        try:
            stdout, stderr = proc.communicate(input=feed, timeout=poll)
            return stdout, stderr, proc.returncode, killed
        except subprocess.TimeoutExpired:
            feed = None
        if time.time() - t0 > 4 * 3600:
            proc.kill()
            stdout, stderr = proc.communicate()
            return stdout, stderr or "timed out after 4 h", proc.returncode, killed
        if stop is not None and stop.is_set():
            for pid in descendants(proc.pid):
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
            proc.kill()
            proc.communicate()
            raise RoleLate(f"stopped by the driver {int(time.time() - t0)} s in")
        if answered is None:
            p = transcript(session)
            if p is not None and p.read_text(errors="replace").count(ANSWERED) > seen:
                answered = time.time()
        elif time.time() - answered > grace:
            for pid in descendants(proc.pid):
                try:
                    args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    os.kill(pid, 15)
                    killed.append(args.strip()[:200])
                except OSError:
                    pass
            answered = time.time()


def descendants(pid: int) -> list[int]:
    """Every process below pid, children first."""
    kids = collections.defaultdict(list)
    for line in subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout.splitlines():
        p, pp = (int(x) for x in line.split())
        kids[pp].append(p)
    out, todo = [], [pid]
    while todo:
        for c in kids[todo.pop(0)]:
            out.append(c)
            todo.append(c)
    return out


def transcript(session: str | None) -> Path | None:
    if not session:
        return None
    p = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(REPO)) / f"{session}.jsonl"
    return p if p.exists() else None


USAGE_KEYS = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
              "cache_read_input_tokens": "cacheReadInputTokens", "cache_creation_input_tokens": "cacheCreationInputTokens"}  # fmt: skip


def session_usage(out: dict) -> dict | None:
    """The whole session's tokens, from `claude -p`'s `modelUsage` (per model, like `total_cost_usd`). Its `usage` covers
    the last turn only: a role whose background command's notification, or a follow-up prompt, opened another turn
    was booked at ~700 output tokens for a 76-minute session that wrote 133,483 (TODO 37.5.1)."""
    models = out.get("modelUsage") or {}
    if not models:
        return out.get("usage")
    return {k: sum(int(m.get(v) or 0) for m in models.values()) for k, v in USAGE_KEYS.items()}


def structured_from_transcript(session: str | None) -> dict | None:
    p = transcript(session)
    so = None
    for line in p.open() if p else ():
        if '"StructuredOutput"' not in line:
            continue
        for c in (json.loads(line).get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "StructuredOutput":
                so = c.get("input")
    return so


def scene_links() -> list[Path]:
    """Where the sandbox's scene links point in the origin (sandbox.py LINKS / scene_exclude): the whole
    outputs/scene_discovery, or that scene's linked entries and its physics copy. Empty outside a sandbox."""
    root = REPO / "outputs" / "scene_discovery"
    if ORIGIN == REPO or not root.exists():
        return []
    if root.is_symlink():
        return [root.resolve()]
    return sorted(
        {p.resolve() for d in root.iterdir() for p in ([d] if d.is_symlink() else d.iterdir()) if p.is_symlink()}
    )


def allowed_dirs() -> list[str]:
    """`--add-dir` for a role's `claude -p`: the directories the scene links resolve to. Claude Code 2.1 refuses Read /
    Grep through a symlink that leaves the session's working directories even under bypassPermissions ("The file is a
    symlink into <origin>, which is outside the allowed working directories"; pick-repro iteration 0, 2026-10-23), so
    without this a Reviewer cannot open scene_meta.json and, reporting why, names the origin path and voids its round."""
    return sorted({str(p if p.is_dir() else p.parent) for p in scene_links()})


BARE_END = (
    r"""(?:[\s"'`):;,|&>\\]|$)"""  # what may follow a path named bare: a space, quote, separator, an escape, the end
)


def leak_patterns() -> dict[str, str]:
    pats = {"gate": re.escape(HIDDEN.parent.name)}
    pats["kit"] = r"rsi_kit|scene_rsi_20261016/kit/"  # kit.py: every family's pools and outcomes (2026-10-20)
    if ORIGIN != REPO:
        # the origin checkout, except what the sandbox links into it: third_party, the scene's linked entries, and the
        # scene directories themselves named bare -- a role that resolves a link reaches the origin's scene directory,
        # and three drafts were voided for naming it (user 2026-10-24: 越界检查要放行原仓库场景目录本身); an entry
        # the sandbox does not link (scene_exclude), or a glob over the directory, is still flagged
        links = [p for p in scene_links() if p.is_relative_to(ORIGIN)]
        linked = ["/" + re.escape(str(p.relative_to(ORIGIN))) + r"(?:/|\b)" for p in links]
        scene_root = ORIGIN / "outputs" / "scene_discovery"
        dirs = sorted({p.parent for p in links if p.parent != scene_root and p.parent.is_relative_to(scene_root)})
        bare = ["/" + re.escape(str(d.relative_to(ORIGIN))) + rf"/?(?={BARE_END})" for d in dirs]
        pats["origin"] = re.escape(str(ORIGIN)) + rf"(?!/third_party{''.join('|' + x for x in linked + bare)})"
        pats["climb"] = r"third_party/\.\."
        # Claude Code's own home (other sessions' transcripts, settings), not a `.claude/` inside the checkout: the
        # iteration-5 Engineer only looked for the sandbox's project settings and that voided its round (2026-10-23)
        pats["claude_home"] = re.escape(str(Path.home() / ".claude")) + r"(?:/|\b)|~/\.claude"
        pats["sibling"] = re.escape(str(REPO.parent)) + "/" + rf"(?!{re.escape(REPO.name)}(?:/|\b))"
        pats["scratch"] = re.escape(str(SCRATCH_ROOT)) + "/" + rf"(?!{re.escape(role_tmp().name)}(?:/|\b))"
    if be := BK.of(FAMILY):  # the backend's own forbidden paths (layout()["leak"]: regexes, e.g. released episodes)
        pats.update({f"backend_{i}": rx for i, rx in enumerate(BK.layout(FAMILY).get("leak") or [])})
    return pats


def own_link_targets() -> list[str]:
    """The real paths the sandbox's own symlinks point to: third_party, the scene entries under outputs/scene_discovery
    (three levels) and the *_physics links at its root -- places a role may legitimately walk into and `..` out of."""
    links = [REPO / "third_party", *REPO.glob("*_physics")]
    base = REPO / "outputs" / "scene_discovery"
    todo = [(base, 0)] if base.is_dir() else []
    while todo:
        d, k = todo.pop()
        for e in os.scandir(d):
            if e.is_symlink():
                links.append(Path(e.path))
            elif e.is_dir() and k < 2:
                todo.append((Path(e.path), k + 1))
    return sorted({os.path.realpath(x) for x in links if x.is_symlink()})


def reaches(path: str) -> bool:
    """A resolved path that touches the file system outside: it exists, or the directory it would be created in does
    (writing a new file there is leaving the sandbox too). A glob or a brace reaches the directory it is cut in at its
    first wildcard, so `../*/x` lists the sandbox's parent. A fragment that resolves into a directory that does not
    exist reaches nothing (TODO 37.9.2)."""
    m = re.search(r"[*?\[{]", path)
    if m:
        return os.path.isdir(os.path.dirname(path[: m.start()]) or "/")
    return os.path.lexists(path) or os.path.isdir(os.path.dirname(path) or "/")


def climbs_out(text: str, targets: list[str], cwd: dict | None = None) -> bool:
    """TODO 36 (review 2026-10-22, L8; review_outer finding 2): a path-like token in a tool input with a `..` segment,
    resolved (symlinks first, as the shell does) from the directory the command runs in, that lands outside the sandbox
    and outside the targets of its own links. `ls outputs/scene_discovery/<scene>/mesh/../../../` listed the origin's
    outputs/. The directory is the sandbox root, or the last `cd` before the token in this command or in an earlier one
    (`cwd`, which transcript_leaks carries from line to line; a shell that left the sandbox is back at its root on the
    next call). Resolving everything from the root voided a review that ran `cd <loop>/pool && diff a
    ../../../../../src/.../pick.py`, a path inside the sandbox (2026-10-22, option B's iteration 6). The role's own
    scratch tree (role_tmp) counts as inside, and a bare `..` counts only as a `cd` target: `from .. import hull` in a
    script an Engineer wrote is no path (iteration 7 voided two drafts for that and for `cd <scratch>/engineer_2 && ln
    -s ../x`); `transcript_leaks` also leaves file contents out of this check (`climb_view`). A climb counts only when it
    reaches something (`reaches`): run D's iteration-19 Engineer built a path inside the sandbox with
    `p.replace('iter_19/cand_2/smoke_final', '../diag/candidate_bafa768225dd')`, and that fragment, resolved from the
    root into a directory that does not exist, voided the draft (2026-10-24, TODO 37.9.2)."""
    repo = os.path.realpath(REPO)
    inside = lambda path, root: path == root or path.startswith(root.rstrip("/") + "/")  # noqa: E731
    roots = [repo, os.path.realpath(role_tmp()), *targets]
    state = cwd if cwd is not None else {}
    base = state.get("dir", repo)
    toks = re.findall(r"[^\s\"'`;|&<>()$\\,]+", text.replace("\\n", " ").replace("\\t", " "))
    out = False
    for i, tok in enumerate(toks):
        raw = tok if tok.startswith("/") else os.path.join(base, tok)
        head, last = os.path.split(raw.rstrip("/") or "/")
        # the directories' links, not the last name's: the venv's `python` is a link out to the base interpreter, and
        # `<diag dir>/../../../../../third_party/.../.venv_sim/bin/python` voided run C's iteration-10 draft (TODO 37.9.2)
        path = os.path.realpath(raw) if last in ("", ".", "..") else os.path.join(os.path.realpath(head), last)
        is_cd = i and toks[i - 1] == "cd"
        if ".." in tok.split("/") and not tok.startswith("~") and (tok != ".." or is_cd):
            if not any(inside(path, r) for r in roots) and reaches(path):
                out = True
        if is_cd and not tok.startswith(("~", "-")):
            base = path
    state["dir"] = base if inside(base, repo) else repo
    return out


def transcript_leaks(session: str | None) -> list[str]:
    p = transcript(session)
    if p is None:
        return []
    own = re.compile(r"[^\"\s]*\.claude/projects/[^\"\s/]+/" + re.escape(str(session)) + r"/tool-results/[^\"\s]*")
    pats, found = leak_patterns(), set()
    targets = own_link_targets() if ORIGIN != REPO else None
    cwd = {}
    with p.open() as fh:
        for line in fh:
            line = own.sub("", audited(line))
            found |= {k for k, pat in pats.items() if re.search(pat, line)}
            if targets is not None and climbs_out(climb_view(line), targets, cwd):
                found.add("origin")
    return sorted(found)


CONTENT_KEYS = ("content", "new_string", "old_string", "edits", "new_source")  # what a tool writes, not where


def climb_view(audited_line: str) -> str:
    """The tool inputs `climbs_out` reads: the audited calls without the contents a Write / Edit puts into a file (the
    off-limits patterns still read those: code that names a forbidden path would open it when it runs)."""
    try:
        calls = json.loads(audited_line)
    except ValueError:
        return audited_line
    if not isinstance(calls, list):
        return audited_line
    strip = lambda d: {k: v for k, v in d.items() if k not in CONTENT_KEYS} if isinstance(d, dict) else d  # noqa: E731
    return json.dumps([[c[0], strip(c[1]), strip(c[2])] if isinstance(c, list) and len(c) == 3 else c for c in calls])


def audited(line: str) -> str:
    """What the audit reads of a transcript line: the inputs of its tool calls, with the line's `wireToolInputs` copy
    of them, but not the role's answer (its StructuredOutput call) and not its thinking or text. Opening a file or
    running a command takes a tool call; an answer or a thought may name an off-limits tree to say it was not used
    (TODO 32.13, user 2026-10-22: 32.12's iteration 4 Reviewer and two Engineers were voided for that; the old grep
    read whole lines, and every block of a reply that ended in a tool call carries "tool_use" as its stop_reason).
    A line that does not parse is read whole."""
    if '"tool_use"' not in line:
        return ""
    try:
        msg = json.loads(line)
        content = msg["message"]["content"]
    except (ValueError, KeyError, TypeError):
        return line
    if not isinstance(content, list):
        return line
    wire = msg.get("wireToolInputs") if isinstance(msg.get("wireToolInputs"), dict) else {}
    calls = [[c.get("name"), c.get("input"), wire.get(c.get("id"))] for c in content
             if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") != "StructuredOutput"]  # fmt: skip
    return json.dumps(calls, ensure_ascii=False) if calls else ""


def reaudit(n: int) -> None:
    """Iteration n's review, if the audit voided it, read again under the present audit before its hypotheses are
    re-probed (TODO 32.13): a hit, or a transcript gone, refuses the re-probe."""
    rev = [e for e in entries() if e.get("step") == "review" and e.get("iter") == n]
    if not rev or not rev[-1].get("void"):
        return
    sessions = rev[-1].get("sessions") or [rev[-1]["session"]]
    if any(transcript(s) is None for s in sessions):
        raise SystemExit(f"iteration {n}'s review transcript is gone: it cannot be audited again")
    leaks = sorted({x for s in sessions for x in transcript_leaks(s)})
    if leaks:
        raise SystemExit(f"iteration {n}'s review still names off-limits paths ({', '.join(leaks)})")
    ledger(dict(step="reaudit", iter=n, sessions=sessions, was=rev[-1].get("leaks"), leaks=[]))


# -- steps ------------------------------------------------------------------------------------------------------------


def incumbent() -> dict:
    """The gate's incumbent copied into the loop's tree (roles must not open the gate's files)."""
    inc = jload(GATE_DIR / "incumbent.json")
    src = Path(inc["dir"]) if inc else None
    if src is None:  # before the first gate call: the sandbox's own skills
        src = STATE / "seed_candidate"
        if not src.exists():
            shutil.copytree(SKILLS, src / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (src / "candidate.json").write_text('{"set": {}}')
    h = candidate_hash(src)
    dst = LOOP / "incumbents" / h
    if not dst.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dict(dir=str(dst), hash=h, set=overrides(dst), since_gate_call=(inc or {}).get("since"))


def diag_incumbent(inc: dict, it_dir: Path) -> Path:
    d = LOOP / "diag" / f"incumbent_{inc['hash']}"
    side = LOOP / "diag" / f"candidate_{inc['hash']}"
    if not d.exists() and side.exists():
        side.rename(d)
    B.run_jobs([B.Job(Path(inc["dir"]) / "skills", d, inc["set"], None, True)], FAMILY, LOOP / "pool_diag", PROCS,
               it_dir / "prober.log", GPUS)  # fmt: skip
    return d


# -- TODO 48.17 (user 2026-11-23, "按 1->4 改"): four changes from the ledgers of simple_q, lml_q, lml_fridge and
# lml_drawer, each a family cap (backend FALLBACK_CAPS: on for the SIMPLE and LocoManiLab backends; a scene family
# has none of them and runs as before).
#   check_half: half of each task's diagnostic starts is kept from the roles; a candidate goes to the gate only when it
#     wins on that half too, and `check_stop` candidates in a row that do not stop the run (the visible half is used
#     up: late candidates gained 17-35 visible starts net and nothing at the gate).
#   stage1_first: the candidate's diagnostic rounds and the early review start only once the gate's stage 1 passed (a
#     call stage 1 rejected cost 1.3-1.6 h, nearly all of it for them).
#   probe_parallel: the Prober runs beside the Engineers, not before the Manager (8.4 h of simple_q's 28; accepted
#     versions came mostly from hypotheses the probe had called untested or refuted).
#   role_rerun: a Reviewer's, Manager's or Composer's first leak in an iteration discards its output and the role runs
#     once more; a second voids the iteration (two of three voided iterations were a role grepping for the hidden
#     tree's name to make sure its package did not use it, with the tree locked; USD 85 of 484).


def cap(key: str, default=False):
    """A family cap (backend.layout()["caps"]); `default` when the family's caps do not hold it."""
    return CAPS.get(key, default) if isinstance(CAPS, dict) else default


def check_pool() -> Path:
    """check_half: the diagnostic starts the roles do not see (rules 47), in the hidden tree."""
    return HIDDEN / "pool_check"


def check_rounds(kind: str, h: str) -> Path:
    """check_half: rounds on the check pool, `incumbent_<hash>` or `candidate_<hash>` as under LOOP/diag."""
    return HIDDEN / "check" / f"{kind}_{h}"


def check_labels_of(labels: list[str]) -> list[str]:
    """Half (rounded down) of each task's starts, ordered by a hash of the family and the label: the same split every
    time, and no start is picked for what its rounds showed."""
    by: dict[str, list[str]] = {}
    for x in labels:
        by.setdefault(task_of(x), []).append(x)
    out: list[str] = []
    for xs in by.values():
        xs = sorted(xs, key=lambda x: hashlib.sha1(f"{FAMILY}:{x}".encode()).hexdigest())
        out += xs[: len(xs) // 2]
    return sorted(out)


def split_check_half() -> dict | None:
    """check_half, once per loop at the start of a run: half of each task's starts move from LOOP/pool_diag to the
    hidden pool_check, and every file of those starts under LOOP/diag/<dir> (rounds, _old_ rounds, traces) to
    HIDDEN/check/<dir>, so an incumbent's or a pooled candidate's check rounds are there already. Rounds of those
    starts in earlier iteration directories stay where they are: the half was visible until now. Returns the ledger
    row, or None when the family has no check half or the pool is split already."""
    if not cap("check_half") or check_pool().exists():
        return None
    pool = LOOP / "pool_diag"
    chk = check_labels_of(B.pool_labels(pool))
    dst = check_pool()
    dst.mkdir(parents=True)
    for x in chk:
        shutil.move(str(pool / f"{x}.json"), str(dst / f"{x}.json"))
    names = set(chk)
    moved = 0
    for d in sorted((LOOP / "diag").glob("*")):
        if not d.is_dir():
            continue
        for f in list(d.iterdir()):
            stem = f.name[5:] if f.name.startswith("_old_") else f.name
            if stem.split(".", 1)[0] in names:
                out = HIDDEN / "check" / d.name
                out.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(out / f.name))
                moved += 1
    row = dict(step="check_split", n_build=len(B.pool_labels(pool)), n_check=len(chk), files_moved=moved)
    jdump(dst / "_split.json", dict(row, labels=chk, t=now()))
    ledger(row)
    print(f"check half: {len(chk)} of {len(chk) + row['n_build']} diagnostic starts kept from the roles")
    return row


def check_candidate(n: int, it_dir: Path, cand: Path) -> dict:
    """check_half: the candidate and the incumbent on the check pool, start by start. The incumbent's rounds run once
    per incumbent (an accepted candidate's are taken over); the per-start lists stay in the hidden tree, the ledger
    row and the iteration directory get counts only."""
    inc = incumbent()
    base = check_rounds("incumbent", inc["hash"])
    side = check_rounds("candidate", inc["hash"])
    if not base.exists() and side.exists():
        side.rename(base)
    h = candidate_hash(cand)
    d = check_rounds("candidate", h)
    log = HIDDEN / "check" / "run.log"
    B.run_jobs([B.Job(Path(inc["dir"]) / "skills", base, inc["set"], None, True),
                B.Job(cand / "skills", d, overrides(cand), None, True)], FAMILY, check_pool(), PROCS, log, GPUS)  # fmt: skip
    f = flips(base, d)
    out = dict(n=f["n"], fixed=len(f["fixed"]), broken=len(f["broken"]), net=len(f["fixed"]) - len(f["broken"]),
               ok=f["ok"], incumbent_ok=f["incumbent_ok"])  # fmt: skip
    k = cap("task_regress", None)  # TODO 48.19.1: the gate's per-task rule, on the check half too (counts by task only)
    if k:
        net = collections.Counter()
        for x in f["fixed"]:
            net[task_of(x)] += 1
        for x in f["broken"]:
            net[task_of(x)] -= 1
        out["task_regress"] = {t: v for t, v in sorted(net.items()) if -v >= int(k)}
    with (HIDDEN / "check" / "calls.jsonl").open("a") as fh:
        fh.write(
            json.dumps(dict(out, t=now(), iter=n, hash=h, fixed_starts=f["fixed"], broken_starts=f["broken"])) + "\n"
        )
    return out


def check_stop(es: list[dict], meta: dict | None) -> str | None:
    """check_half: the last `check_stop` candidates of this run all failed the check half."""
    k = int(cap("check_stop", 0) or 0)
    t0 = (meta or {}).get("started") or ""
    rows = [e for e in es if e.get("step") == "check" and str(e.get("t", "")) >= t0]
    if k and len(rows) >= k and not any(r.get("sent") for r in rows[-k:]):
        nets = ", ".join(str(r.get("net")) for r in rows[-k:])
        return (f"diagnostic pool used up: the last {k} candidates did not win on its hidden half (net {nets}; "
                f"rules 47)")  # fmt: skip
    return None


def stage1_passed(gate_dir: Path, h: str, t0: float) -> bool:
    """stage1_first: gate.py wrote this call's stage-1 file (its work directory is <call>_<hash>)."""
    return any(p.stat().st_mtime >= t0 - 1 for p in gate_dir.glob(f"*_{h}/stage1_passed.json"))


def rerun_note(role: str, leaks: list[str]) -> str:
    """role_rerun: what the second attempt is told."""
    return (f"\n\nNOTE: an earlier attempt of this {role} in this iteration was discarded because its tool calls named "
            f"off-limits paths ({', '.join(leaks)}). Do not name the hidden tree or those paths in any command, not even "
            f"to exclude them or to check that the package does not use them: the driver checks that.\n")  # fmt: skip


def discard_attempt(n: int, role: str, acct: dict, leaks: list[str], *paths: Path) -> None:
    """role_rerun: the first attempt's files leave the visible tree (hide_void), its own scratch directory is emptied
    (an unsuffixed role shares role_tmp() with the others and their try.py ledgers: that one stays), and a `rerun`
    row keeps its cost (run_cost sums the rows)."""
    hide_void(n, *paths)
    d = role_dir(role)
    if d.is_dir() and d != role_tmp():
        shutil.rmtree(d, ignore_errors=True)
    ledger(dict(step="rerun", iter=n, role=role, leaks=leaks, cost_usd=acct.get("cost_usd", 0.0),
                session=acct.get("session"), seconds=acct.get("seconds")))  # fmt: skip
    print(f"iteration {n}: the {role} named off-limits paths ({', '.join(leaks)}); it runs once more")


def attribute(inc: dict, base: Path, n: int, it_dir: Path, stub: bool) -> dict | None:
    """TODO 33.20 (plan section 1.4, D20): whose failure is it. When the incumbent fails ATTRIBUTE_MIN or more
    diagnostic rounds of a class in ATTRIBUTE_CLASSES, each one's failing command is run again alone from its start
    (scripts/rsi/isolate.py; cached per incumbent in diag/attribution_<hash>, which the roles may read). `motor` -- the
    command alone fails the same way -- is the motor layer's or the controller's limit: an escalation row
    (loop/escalations.jsonl, rules.md 20) and a body-model constraint (body/constraints.jsonl); a skill may route around
    it, not fix it. `skill` -- alone it does not fail -- is the skill's. The summary goes to the Reviewer and the
    Manager."""
    if stub or BK.of(FAMILY):  # isolate.py replays a scene round command by command; a backend has no such replay
        return None
    fails = [r for r in B.outcomes(base).values() if not r.get("ok")]
    count = collections.Counter(r.get("failure_class") for r in fails)
    classes = [c for c in ATTRIBUTE_CLASSES if count[c] >= ATTRIBUTE_MIN]
    if not classes:
        return None
    out = LOOP / "diag" / f"attribution_{inc['hash']}"
    cmd = [str(SIM_PY), str(HERE / "isolate.py"), FAMILY, "--pool", str(LOOP / "pool_diag"), "--rounds", str(base),
           "--out", str(out), *[x for c in classes for x in ("--class", c)]]  # fmt: skip
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0",
               CUDA_VISIBLE_DEVICES=GPUS.split(",")[0])  # fmt: skip
    with (it_dir / "isolate.log").open("w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        ledger(dict(step="attribution", iter=n, error=f"isolate.py exit {rc}"))
        return None
    rows = [jload(p) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    motor = [r for r in rows if r.get("verdict") == "motor"]
    record_escalations(inc, n, motor)
    summary = dict(
        classes=classes,
        rounds=[
            dict(
                label=r["label"],
                failure_class=r.get("original", {}).get("failure_class"),
                verdict=r.get("verdict"),
                command=(r.get("call") or {}).get("name"),
                alone=r.get("isolated", {}).get("call_why"),
            )
            for r in rows
        ],  # fmt: skip
        note="motor: the command alone, from the round's start, fails the same way -- the motor layer's or the "
        "controller's limit (rules.md 20): route around it (another first move, another stance), do not try to fix the "
        "layer below; skill: alone it does not fail, the skill's earlier steps cause it",
        dir=str(out),
    )
    ledger(dict(step="attribution", iter=n, classes=classes, motor=[r["label"] for r in motor],
                skill=[r["label"] for r in rows if r.get("verdict") == "skill"]))  # fmt: skip
    return summary


def record_escalations(inc: dict, n: int, motor: list[dict]) -> None:
    """TODO 33.20: each `motor` attribution once per incumbent, as an escalation row (loop/escalations.jsonl, for the
    people, rules.md 20) and a body-model constraint (body/constraints.jsonl, for the roles)."""
    esc_f, con_f = STATE / "escalations.jsonl", LOOP.parent / "body" / "constraints.jsonl"
    old = [json.loads(x) for x in esc_f.read_text().splitlines() if x.strip()] if esc_f.exists() else []
    known = {(e.get("incumbent"), e.get("label")) for e in old}
    for r in motor:
        if (inc["hash"], r["label"]) in known:
            continue
        call = r.get("call") or {}
        what = f"{call.get('name')}({', '.join(json.dumps(a) for a in call.get('args', []))})"
        row = dict(t=now(), iter=n, incumbent=inc["hash"], label=r["label"], task=r.get("task"),
                   failure_class=r["original"]["failure_class"], command=what, alone=r["isolated"]["call_why"],
                   rule=20, layer="motor layer / controller")  # fmt: skip
        with esc_f.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        con_f.parent.mkdir(parents=True, exist_ok=True)
        with con_f.open("a") as fh:
            fh.write(json.dumps(dict(row, kind="isolated_failure")) + "\n")


def refresh_diag_pool(inc: dict, base: Path, n: int | None) -> dict | None:
    """TODO 33.23: after the incumbent has run on the diagnostic pool (`base`), re-draw the pool when it fails fewer than
    DIAG_MIN_FAILS starts (diag_pool.refresh: fresh starts from the diagnostic seed range, the incumbent closed loop on
    them, the failing ones swapped in for passing starts until DIAG_TARGET_FAILS fail). Parked starts and the tasks
    parked as a whole stay. The candidate pool's entries then run on the starts they have not seen, so pool_view's
    flips stay comparable. Returns the ledger row, or None when nothing changed."""
    if DIAG_MIN_FAILS <= 0 or not CAPS["diag_redraw"]:  # CAPS: backend.layout(family)["caps"]; scene: all on
        return None
    parked = jload(STATE / "parked.json", {})
    frozen = {x for p in parked.values() for x in p.get("starts") or []}
    tasks = {p["task"] for p in parked.values() if p.get("scope") == "task"}
    scratch = LOOP / "diag_redraw" / (f"iter_{n:02d}" if n is not None else now().replace(":", ""))

    def run(pool_dir: Path, out: Path) -> None:
        B.run_jobs([B.Job(Path(inc["dir"]) / "skills", out, inc["set"], None, True)], FAMILY, pool_dir, PROCS,
                   scratch / "redraw.log", GPUS)  # fmt: skip

    row = DP.refresh(F.load(FAMILY), LOOP / "pool_diag", base, run, scratch, tasks, frozen, DIAG_MIN_FAILS,
                     DIAG_TARGET_FAILS, DIAG_DRAW_MAX, draw=F.draw)  # fmt: skip
    if row is None:
        return None
    ledger(dict(row, iter=n, incumbent=inc["hash"]))
    print(f"diag pool redrawn: {row['fails_before']} -> {row['fails_after']} failing of {row['n']} "
          f"(+{len(row['added'])} -{len(row['retired'])}, {row['drawn']} fresh starts run)")  # fmt: skip
    jobs = [B.Job(Path(e["dir"]) / "skills", LOOP / "diag" / f"candidate_{e['hash']}", overrides(Path(e["dir"])), None, True)
            for e in pool_load() if (LOOP / "diag" / f"candidate_{e['hash']}").exists() and Path(e["dir"]).exists()]  # fmt: skip
    if jobs and row["added"]:
        B.run_jobs(jobs, FAMILY, LOOP / "pool_diag", PROCS, scratch / "redraw.log", GPUS)
    return row


def summarize(d: Path) -> dict:
    """Per task and per failure: what the Reviewer reads first (the round files hold the rest)."""
    out = B.outcomes(d)
    per = collections.defaultdict(lambda: [0, 0])
    why = collections.Counter()
    for r in out.values():
        per[r["task"]][0] += r["ok"]
        per[r["task"]][1] += 1
        if not r["ok"]:
            why[(r.get("plan_why") or r["why"])[:70]] += 1
    return dict(ok=sum(r["ok"] for r in out.values()), n=len(out), per_task={k: f"{a}/{b}" for k, (a, b) in per.items()},
                failures=dict(why.most_common()))  # fmt: skip


# -- program search for parameters, drafts that run where they act (changes 1 and 4) ---------------------------------


def sweep_specs(h: dict) -> list[dict]:
    """A hypothesis's `sweep` made safe: at most 2 keys, each 2-5 distinct values."""
    out: list[dict] = []
    for it in (h.get("sweep") or [])[:2]:
        if not isinstance(it, dict) or not isinstance(it.get("key"), str) or it["key"] in [o["key"] for o in out]:
            continue
        vals: list = []
        for v in it.get("values") or []:
            if v not in vals:
                vals.append(v)
        if len(vals) >= 2:
            out.append(dict(key=it["key"], values=vals[:5]))
    return out


def sweep_row(key: str, value, before: dict, after: dict) -> dict:
    """One swept value against the incumbent, over the starts both ran: fixed (incumbent failed, now ok), broken
    (incumbent ok, now failed), net = fixed - broken."""
    fixed = sum(1 for k in after if after[k] and not before.get(k))
    broken = sum(1 for k in after if before.get(k) and not after[k])
    return dict(key=key, value=value, n=len(after), fixed=fixed, broken=broken, net=fixed - broken)


def best_row(rows: list[dict]) -> int:
    """Index of the best swept value: largest net, then fewer broken, then the earlier in the list."""
    return min(range(len(rows)), key=lambda i: (-rows[i]["net"], rows[i]["broken"], i))


def subsample_starts(labels, before: dict, m: int) -> list[str]:
    """m starts, deterministically: the incumbent's failures first up to ceil(2m/3) in label order, then its successes in
    label order, more failures when the successes run out. Label order in the result."""
    labels = sorted(labels)
    if m >= len(labels):
        return labels
    lose = [x for x in labels if not before.get(x)]
    win = [x for x in labels if before.get(x)]
    f = lose[: math.ceil(2 * m / 3)]
    w = win[: m - len(f)]
    if len(f) + len(w) < m:
        f = lose[: m - len(w)]
    return sorted(f + w)


class SweepBudget:
    """Rounds an iteration's program sweeps (the probe's and the Engineers' knobs) may run in all (caps sweep_rounds)."""

    def __init__(self, total: int):
        self.total, self.spent, self._lock = int(total), 0, threading.Lock()

    def reserve(self, want: int) -> int:
        with self._lock:
            got = max(0, min(int(want), self.total - self.spent))
            self.spent += got
            return got

    @property
    def left(self) -> int:
        return max(0, self.total - self.spent)


SWEEP: SweepBudget | None = None


def sweep_budget() -> SweepBudget:
    global SWEEP
    if SWEEP is None:
        SWEEP = SweepBudget(CAPS.get("sweep_rounds", 1200))
    return SWEEP


def _ok_of(d: Path) -> dict:
    return {k: bool(r["ok"]) for k, r in B.outcomes(d).items()}


def sweep_mechanism(best_set: dict, keys: list[str]) -> str:
    return "program sweep: " + ", ".join(f"{k}={best_set[k]}" for k in keys if k in best_set)


def draft_labels(pool: list[str], declared, inc_ok: dict, sentinels: int) -> tuple[tuple | None, dict]:
    """Change 4: the starts a draft's diagnostic run covers. Declared starts ["all"], none, or none of the pool's: the whole
    pool (None, as before). Otherwise every pool start of the tasks its declared starts belong to, plus up to `sentinels`
    starts of each other task -- the incumbent's successes first, then its failures, in label order. Returns the labels
    and `ran` = {n, tasks, sentinels (the labels)}."""
    pool = sorted(pool)
    tasks_all = sorted({task_of(x) for x in pool})
    whole = dict(n=len(pool), tasks=tasks_all, sentinels=[])
    decl = [x for x in (declared or []) if isinstance(x, str)]
    if not decl or "all" in decl:
        return None, whole
    own = ({task_of(x) for x in decl} | {x for x in decl if x in tasks_all}) & set(tasks_all)
    if not own:
        return None, whole
    sent: list[str] = []
    for t in tasks_all:
        if t in own:
            continue
        starts = [x for x in pool if task_of(x) == t]
        sent += ([x for x in starts if inc_ok.get(x)] + [x for x in starts if not inc_ok.get(x)])[:sentinels]
    labels = sorted([x for x in pool if task_of(x) in own] + sent)
    return tuple(labels), dict(n=len(labels), tasks=sorted(own), sentinels=sent)


def draft_run(eng: dict, inc: dict) -> tuple[tuple | None, dict]:
    """draft_labels for a draft's declared scope, on this loop's diagnostic pool and the incumbent's rounds."""
    ok = _ok_of(LOOP / "diag" / f"incumbent_{inc['hash']}")
    return draft_labels(B.pool_labels(LOOP / "pool_diag"), ((eng or {}).get("scope") or {}).get("starts"), ok,
                        int(CAPS.get("draft_sentinels", 3)))  # fmt: skip


def seed_rounds(src: Path, dst: Path, labels=None) -> int:
    """Round files (and their .npz) of `src` copied into `dst`, those not there yet; `labels` limits them."""
    n = 0
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        if f.name.startswith("_") or (labels is not None and f.stem not in labels) or (dst / f.name).exists():
            continue
        for g in (f, f.with_suffix(".npz")):
            if g.exists():
                shutil.copy2(g, dst / g.name)
        n += 1
    return n


def probe_labels(h: dict) -> tuple | None:
    """The diagnostic starts a hypothesis's probe runs; None = the whole pool, as for every scene family. A backend
    whose rounds cost minutes (caps["probe_scope"]) probes a hypothesis only on the tasks it is about: its `tasks`,
    else the tasks of its predicted starts; when it names none, the whole pool. The drafts built later still run on
    every start, so a regression elsewhere is seen there and at the gate."""
    if not CAPS.get("probe_scope"):
        return None
    labels = B.pool_labels(LOOP / "pool_diag")
    task_of = lambda x: x.rsplit("_s", 1)[0]  # noqa: E731
    known = sorted({task_of(x) for x in labels})
    named = {t for t in h.get("tasks") or [] if t in known}
    if not named:
        named = {task_of(x) for x in (h.get("predicted_fixed") or []) + (h.get("predicted_broken") or [])} & set(known)
    return tuple(x for x in labels if task_of(x) in named) if named else None


def probe(hyps: list[dict], inc: dict, base: Path, it_dir: Path) -> list[dict]:
    """Every hypothesis on the diagnostic starts (probe_labels). One with `needs_code` is not run: `untested`. One with
    `sweep` (change 1) runs one job per value of its first key, then one per value of its second with the first at its
    best, each on the same starts (subsampled when values x starts exceed its share of caps sweep_rounds); its row
    describes the best set (`best_set`, `sweep`). Any other runs once with its delta."""
    global SWEEP
    SWEEP = SweepBudget(CAPS.get("sweep_rounds", 1200))
    skills = Path(inc["dir"]) / "skills"
    scope = {h["id"]: probe_labels(h) for h in hyps}
    before = _ok_of(base)
    pool = B.pool_labels(LOOP / "pool_diag")
    runnable = [h for h in hyps if not h.get("needs_code")]
    share = SWEEP.total // max(1, sum(1 for h in runnable if sweep_specs(h)))
    plan: dict = {}
    for h in runnable:
        specs, labels, sub = sweep_specs(h), scope[h["id"]], False
        if specs:
            starts = sorted(labels if labels is not None else pool)
            n_jobs = sum(len(s["values"]) for s in specs)
            if n_jobs * len(starts) > share:
                starts, sub = subsample_starts(starts, before, max(1, share // n_jobs)), True
                labels = tuple(starts)
            SWEEP.reserve(n_jobs * len(starts))
        plan[h["id"]] = dict(labels=labels, specs=specs, sub=sub)
    d1 = lambda h, i: it_dir / "probe" / (f"{h['id']}__s1_{i}" if plan[h["id"]]["specs"] else h["id"])  # noqa: E731
    jobs, failed = [], None
    for h in runnable:
        p, fixed_set = plan[h["id"]], {**inc["set"], **h["delta"].get("set", {})}
        if p["specs"]:
            jobs += [B.Job(skills, d1(h, i), {**fixed_set, p["specs"][0]["key"]: v}, p["labels"])
                     for i, v in enumerate(p["specs"][0]["values"])]  # fmt: skip
        else:
            jobs.append(B.Job(skills, d1(h, 0), fixed_set, p["labels"]))
    try:
        B.run_jobs(jobs, FAMILY, LOOP / "pool_diag", PROCS, it_dir / "prober.log", GPUS)
    except Exception as e:
        failed = f"{type(e).__name__}: {e}"
    final: dict = {}  # id -> (rows, best_set, best_dir, second key dropped)
    jobs2, meta2 = [], {}
    for h in runnable:
        p = plan[h["id"]]
        if not p["specs"] or failed:
            continue
        fixed_set = {**inc["set"], **h["delta"].get("set", {})}
        s1 = p["specs"][0]
        rows1 = [sweep_row(s1["key"], v, before, _ok_of(d1(h, i))) for i, v in enumerate(s1["values"])]
        live = [i for i, r in enumerate(rows1) if r["n"]]
        if not live:
            continue
        b1 = live[best_row([rows1[i] for i in live])]
        set1 = {**h["delta"].get("set", {}), s1["key"]: s1["values"][b1]}
        final[h["id"]] = (rows1, set1, d1(h, b1), False)
        if len(p["specs"]) == 2:
            s2 = p["specs"][1]
            meta2[h["id"]] = (rows1, b1, set1)
            jobs2 += [B.Job(skills, it_dir / "probe" / f"{h['id']}__s2_{i}", {**fixed_set, **set1, s2["key"]: v}, p["labels"])
                      for i, v in enumerate(s2["values"])]  # fmt: skip
    if jobs2:
        try:
            B.run_jobs(jobs2, FAMILY, LOOP / "pool_diag", PROCS, it_dir / "prober.log", GPUS)
        except Exception as e:
            failed = failed or f"{type(e).__name__}: {e}"
        for h in runnable:
            if h["id"] not in meta2:
                continue
            rows1, b1, set1 = meta2[h["id"]]
            s2 = plan[h["id"]]["specs"][1]
            rows2 = [sweep_row(s2["key"], v, before, _ok_of(it_dir / "probe" / f"{h['id']}__s2_{i}"))
                     for i, v in enumerate(s2["values"])]  # fmt: skip
            live = [i for i, r in enumerate(rows2) if r["n"]]
            b2 = live[best_row([rows2[i] for i in live])] if live else None
            if b2 is not None and rows2[b2]["net"] >= rows1[b1]["net"]:  # a second key that only hurts is left out
                final[h["id"]] = (rows1 + rows2, {**set1, s2["key"]: s2["values"][b2]},
                                  it_dir / "probe" / f"{h['id']}__s2_{b2}", False)  # fmt: skip
            else:
                final[h["id"]] = (rows1 + rows2, set1, final[h["id"]][2], True)
    res = []
    for h in hyps:
        if h.get("needs_code"):
            res.append(dict(id=h["id"], status="untested", reason="needs code"))
            continue
        p = plan[h["id"]]
        if p["specs"]:
            rows, best_set, rdir, dropped = final.get(h["id"], ([], None, None, False))
        else:
            rows, best_set, rdir, dropped = None, dict(h["delta"].get("set", {})), d1(h, 0), False
        after = _ok_of(rdir) if rdir is not None else {}
        if failed or not after:
            res.append(dict(id=h["id"], error=failed or "no rounds ran (an unknown parameter? see prober.log)"))
            continue
        fixed = sorted(k for k in after if after[k] and not before.get(k))
        broken = sorted(k for k in after if before.get(k) and not after[k])
        pf, pb = set(h["predicted_fixed"]), set(h["predicted_broken"])
        res.append(dict(id=h["id"], n=len(after), ok_before=sum(before.values()), ok_after=sum(after.values()),
                        fixed=fixed, broken=broken, predicted_fixed_hit=len(pf & set(fixed)), predicted_fixed_n=len(pf),
                        predicted_broken_hit=len(pb & set(broken)), predicted_broken_n=len(pb),
                        summary=summarize(rdir), best_set=best_set, best_dir=str(rdir)))  # fmt: skip
        if rows is not None:
            res[-1].update(sweep=rows, **({"second_key_dropped": True} if dropped else {}))
        if p["sub"]:
            res[-1].update(subsampled=True, starts=sorted(p["labels"]))
        if scope[h["id"]] is not None or p["sub"]:  # both counts are over the starts it ran, and it says which
            res[-1].update(ok_before=sum(bool(before.get(k)) for k in after))
        if scope[h["id"]] is not None:
            res[-1].update(scope=sorted({k.rsplit("_s", 1)[0] for k in scope[h["id"]]}))
    return res


def parameters_of(inc: dict) -> dict:
    """The incumbent's overridable parameters (skills.parameters) with its overrides applied."""
    if be := BK.of(FAMILY):  # a second backend reads its package's parameters from the source text
        try:
            pkg = be.load_skills(Path(inc["dir"]) / "skills", "scene_skills", FAMILY)
            pkg.apply_set(pkg, inc["set"])
            return json.loads(json.dumps(pkg.parameters(pkg), default=str))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[-500:]}
    r = subprocess.run([str(SIM_PY), "-c", (
        "import json,sys; sys.path[:0]=[sys.argv[1], 'src']; from scene_round import load_skills; "
        "p=load_skills(__import__('pathlib').Path(sys.argv[2])); p.apply_set(p, json.loads(sys.argv[3])); "
        "print(json.dumps(p.parameters(p), default=str))"), str(HERE), str(Path(inc["dir"]) / "skills"),
        json.dumps(inc["set"])], cwd=REPO, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))  # fmt: skip
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}


def _cut(x, k: int) -> str:
    t = " ".join(str(x if x is not None else "").split())
    return t if len(t) <= k else t[: k - 3] + "..."


def iteration_line(n: int, rows: list[dict]) -> str:
    """Iteration n's ledger rows as one line (TODO 36.7): the hypotheses (from iter_NN/hypotheses.json), the probes,
    the claims' parks and reopens, the Manager's decision, each Engineer's draft, the Composer and the gate."""
    hyps = (jload(STATE / f"iter_{n:02d}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
    parts = [f"iteration {n}"]
    for e in rows:
        st = e.get("step")
        if st == "review":
            if e.get("void"):
                parts.append(f"review void ({', '.join(e.get('leaks') or [])})")
            else:
                parts.append(f"review: {e.get('n_hypotheses', 0)} hypotheses, {e.get('n_claims', 0)} claims")
                hs = [f"{h['id']} {_cut(h.get('claim'), 140)}" for h in hyps if isinstance(h, dict) and "id" in h]
                if hs:
                    parts.append("; ".join(hs))
        elif st == "probe":
            parts.append("probe " + ", ".join(
                f"{r['id']} {r.get('ok_before')}->{r.get('ok_after')}/{r.get('n')} (predicted fixed "
                f"{r.get('predicted_fixed_hit', 0)}/{r.get('predicted_fixed_n', 0)}{', ' + r['status'] if r.get('status') else ''})"
                for r in e.get("results") or [] if isinstance(r, dict) and "id" in r))  # fmt: skip
        elif st == "manager":
            what = e.get("hypotheses") or e.get("reopen_claim") or e.get("hypothesis") or ""
            what = ",".join(what) if isinstance(what, list) else what
            what += f" on pool {e['base']}" if e.get("base") else ""
            parts.append(f"manager {e.get('decision')} {what}{' (void)' if e.get('void') else ''}: "
                         f"{_cut(e.get('reason'), 200)}")  # fmt: skip
        elif st == "engineer":
            k = f" {e['candidate']}" if e.get("candidate") is not None else ""
            how = ("late (stopped)" if e.get("late") else "void" if e.get("void")
                   else f"duplicate of {e['duplicate_of']}" if e.get("duplicate_of") else "ok")  # fmt: skip
            how += f", error {_cut(e['error'], 80)}" if e.get("error") else ""
            how += f", stray {e['stray']}" if e.get("stray") else ""
            parts.append(f"engineer{k} {e.get('hypothesis') or ''}: {how} -- {_cut(e.get('mechanism'), 120)}")
        elif st == "compose":
            parts.append(f"composer {e.get('decision')}{' ' + str(e['adopted']) if e.get('adopted') else ''}"
                         f"{' (void)' if e.get('void') else ''}: {_cut(e.get('reason') or e.get('notes'), 120)}")  # fmt: skip
        elif st == "gate":
            r = e.get("result") or {}
            v = "void" if e.get("void") else "accepted" if e.get("accepted") else "rejected"
            bc = f" b {r['b']} c {r['c']} p {r['p_one_sided']:.3g}" if "b" in r and "p_one_sided" in r else ""
            parts.append(f"gate {v}{bc}{' ' + e['hash'] if e.get('hash') else ''}: {_cut(e.get('reason'), 100)}")
        elif st == "park":
            parts.append(
                f"park {e.get('claim')} {e.get('task')} ({e.get('scope')}, {len(e.get('starts') or [])} starts)"
            )
        elif st == "unpark":
            parts.append(f"unpark {e.get('claim')} {e.get('task')} by {e.get('by')}"
                         f"{' (claim wrong)' if e.get('claim_wrong') else ''}")  # fmt: skip
        elif st == "reopen_test":
            parts.append(f"reopen test {e.get('claim')}: {len(e.get('ok') or [])}/{e.get('n')}")
        elif st == "pool":
            if e.get("cleared"):
                parts.append(f"pool cleared ({', '.join(e['cleared'])})")
            else:
                parts.append(f"pool +{e.get('added')} kept {', '.join(e.get('kept') or [])}")
        elif st == "diag_pool":
            parts.append(f"diag pool redrawn: {e.get('fails_before')} -> {e.get('fails_after')} failing of {e.get('n')} "
                         f"(+{len(e.get('added') or [])} -{len(e.get('retired') or [])}, {e.get('drawn')} fresh starts run)")  # fmt: skip
        elif st == "forced":
            parts.append(f"forced build {e.get('hypothesis')} (mechanism {', '.join(e.get('mechanism') or [])}, "
                         f"replaced {e.get('replaced')})")  # fmt: skip
        elif st == "forced_skip":
            parts.append(f"forced build of {e.get('hypothesis')} skipped: {e.get('check')}")
        elif st == "second_shot":
            parts.append(f"second shot: draft {e.get('draft')} {e.get('hypothesis') or ''} (fixed {e.get('fixed')} "
                         f"broken {e.get('broken')}) after {e.get('after')}")  # fmt: skip
        elif st == "void_iter":
            parts.append(f"VOID: the {e.get('role')} named off-limits paths ({', '.join(e.get('leaks') or [])})")
        else:
            parts.append(f"{st}: {_cut(e.get('reason') or e.get('error') or '', 120)}".rstrip(": "))
    return " | ".join(parts)


def compact_history(es: list[dict], full: int = HISTORY_FULL) -> list:
    """TODO 36.7 (user 2026-10-22): the ledger for the Manager -- the rows from the last `full` iterations' first row on
    as they are (without `usage`), everything before as one line per iteration (iteration_line) and one per row
    outside an iteration (a stop, a resume). 32.12's iteration 5 Manager read 96 KB, 70% of it history."""
    rows = [{k: v for k, v in e.items() if k != "usage"} for e in es]
    iters = sorted({e["iter"] for e in rows if isinstance(e.get("iter"), int)})
    keep = set(iters[-full:]) if full > 0 else set()
    cut = next((i for i, e in enumerate(rows) if e.get("iter") in keep), len(rows))
    out, seen = [], {}
    for e in rows[:cut]:
        n = e.get("iter")
        if isinstance(n, int):
            seen.setdefault(n, []).append(e)
            if len(seen[n]) == 1:
                out.append(n)  # placeholder: the line goes where the iteration's first row was
        else:
            out.append(_cut(f"{e.get('t', '')} {e.get('step')}: " + json.dumps(
                {k: v for k, v in e.items() if k not in ("t", "step")}, ensure_ascii=False), 240))  # fmt: skip
    return [iteration_line(x, seen[x]) if isinstance(x, int) else x for x in out] + rows[cut:]


def context(inc: dict, extra: dict, compact: bool = True) -> dict:
    """The role's context file. `compact` (TODO 36.7 for the Manager, 37.2 for every role): the last HISTORY_FULL
    iterations' ledger rows in full, older iterations one line each (compact_history); `mechanisms` is the index of
    every hypothesis so far (mechanism_table)."""
    es = entries()
    meta = jload(STATE / "meta.json", {})
    # TODO 36 (review 2026-10-22, L6) and 33.18.2: this run's cost and gate calls only (not rows a replay copied), and
    # the wall clock the Manager used to compute from meta.json by hand
    wall = dict(wall_h=round(wall_h(meta), 2), max_wall_h=MAX_WALL_H) if "started_s" in meta else {}
    return dict(
        family=FAMILY,
        incumbent=inc,
        budget=dict(
            gate_calls=len(run_gates(es, meta)),
            gate_calls_max=MAX_GATE_CALLS,
            cost_usd=round(run_cost(es, meta), 2),
            cost_cap_usd=COST_CAP,
            stop_rate=STOP_RATE,
            **wall,
        ),  # fmt: skip
        history=compact_history(es) if compact else [{k: v for k, v in e.items() if k != "usage"} for e in es],
        mechanisms=mechanism_table(es),
        # TODO 33.8: rejections under the present gate rule; one an older rule rejected may be judged again
        rejected_hashes=sorted(
            {
                e["hash"]
                for e in es
                if e.get("step") == "gate"
                and not e.get("accepted")
                and not e.get("void")
                and e.get("hash")
                and e.get("rule", 1) == GATE_RULE
            }
        ),
        **extra,
    )


def put_skills(src: Path) -> None:
    """The sandbox's skills package := `src` (the incumbent's snapshot)."""
    if be := BK.of(FAMILY):  # a second backend refuses to write inside its seed package
        be.guard_write(FAMILY, SKILLS)
    shutil.rmtree(SKILLS)
    shutil.copytree(src, SKILLS, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def changed_files() -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=REPO, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        f = line[3:].split(" -> ")[-1].strip('"')
        if not f.startswith(("outputs/", "__pycache__")) and "__pycache__" not in f:
            out.append(f)
    return sorted(out)


def snapshot(n: int) -> None:
    """The loop's mutable state as iteration n starts, so replay.py can restart a copy of the loop there (TODO 32.5):
    the parks, the Manager's focus, the gap files, meta, the ledgers' lengths and the gate's incumbent. In the hidden
    tree: the gate's incumbent names hidden paths, and the roles must not be led to them."""
    d = HIDDEN / "state_before" / f"iter_{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("parked.json", "focus.json", "gap_candidates.json", "gaps.jsonl", "meta.json", "interventions.jsonl"):
        if (STATE / f).exists():
            shutil.copy2(STATE / f, d / f)
    if (GATE_DIR / "incumbent.json").exists():
        shutil.copy2(GATE_DIR / "incumbent.json", d / "gate_incumbent.json")
    n_lines = lambda f: len(f.read_text().splitlines()) if f.exists() else 0  # noqa: E731
    jdump(
        d / "lengths.json", dict(ledger=n_lines(STATE / "ledger.jsonl"), gate_ledger=n_lines(GATE_DIR / "ledger.jsonl"))
    )


def reviewer_cards(n: int, base: Path, it_dir: Path, log: bool = True) -> Path:
    """The failure cards of the incumbent's diagnostic rounds (cards.py, rule 37), in iter_NN/cards."""
    out = it_dir / "cards"
    if be := BK.of(FAMILY):  # the backend writes its own failure cards (cards.py needs a scene file)
        out.mkdir(parents=True, exist_ok=True)
        be.cards(list(B.outcomes(base).values()), out)
        if log:
            ledger(cards_row(n, out))
        return out / "cards.md"
    r = subprocess.run([str(SIM_PY), str(HERE / "cards.py"), str(base), str(REPO / family_spec()["scene"]), "--out", str(out)],
                       capture_output=True, text=True)  # fmt: skip
    (it_dir / "cards.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"cards.py failed ({r.returncode}): {r.stderr[-800:]}")
    if log:
        ledger(cards_row(n, out))
    return out / "cards.md"


# -- the Reviewer's working files for the Engineers (change 2) ------------------------------------------------------------

NOTE_EXT = (".py", ".sh", ".md", ".txt", ".json", ".csv", ".tsv")
NOTE_MAX_FILE = 256 * 1024
NOTE_MAX_ALL = 4 * 1024 * 1024


def review_sources() -> list[tuple[str, Path, bool]]:
    """Where the Reviewer(s) of this run keep working files: (tag, directory, recurse). One Reviewer shares role_tmp() with
    every other role, so only its top-level files count; joint Reviewers each have their own directory."""
    if SCOPED:
        tags = list(SCOPES)
    elif REVIEWERS > 1:
        tags = [chr(65 + k) for k in range(REVIEWERS)]
    else:
        return [("", role_tmp(), False)]
    return [(t, role_dir(f"reviewer_{t}"), True) for t in tags]


def copy_review_notes(it_dir: Path, sources: list[tuple[str, Path, bool]], since: float) -> dict:
    """Change 2: the Reviewer's working files, modified since the review started, into it_dir/review_notes with an INDEX.md.
    Extensions NOTE_EXT, each at most NOTE_MAX_FILE bytes, NOTE_MAX_ALL in all (the largest are left out first)."""
    dst = it_dir / "review_notes"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    found: list[tuple[str, Path, int]] = []
    for tag, d, deep in sources:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*") if deep else d.glob("*")):
            try:
                if f.is_symlink() or not f.is_file() or f.suffix.lower() not in NOTE_EXT or "__pycache__" in f.parts:
                    continue
                st = f.stat()
            except OSError:
                continue
            if st.st_mtime >= since and st.st_size <= NOTE_MAX_FILE:
                rel = f.relative_to(d).as_posix()
                found.append((f"{tag}/{rel}" if tag else rel, f, st.st_size))
    total, skipped = sum(x[2] for x in found), []
    for item in sorted(found, key=lambda x: (-x[2], x[0])):
        if total <= NOTE_MAX_ALL:
            break
        found.remove(item)
        total -= item[2]
        skipped.append(item[0])
    lines = ["# The Reviewer's working files", "", "Files the Reviewer wrote while it worked, copied unchanged.", ""]
    for rel, f, size in sorted(found):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
        lines.append(f"- `{rel}` ({size} bytes)")
    if not found:
        lines.append("(none)")
    if skipped:
        lines += ["", "Left out, the total would pass 4 MB: " + ", ".join(sorted(skipped))]
    (dst / "INDEX.md").write_text("\n".join(lines) + "\n")
    return dict(files=len(found), bytes=total, skipped=len(skipped))


def review_notes_dir(it_dir: Path) -> Path:
    """it_dir/review_notes, made (with an INDEX saying there are none) when this iteration has no copy."""
    d = it_dir / "review_notes"
    if not d.exists():
        d.mkdir(parents=True)
        (d / "INDEX.md").write_text("# The Reviewer's working files\n\n(none)\n")
    return d


# -- the next review while the gate runs (change 5) ----------------------------------------------------------------------

TEXT_EXT = (".json", ".jsonl", ".txt", ".md")


def rewrite_paths(root: Path, old: str, new: str) -> None:
    """Every `old` in the text files (TEXT_EXT) under `root` becomes `new`."""
    for f in root.rglob("*"):
        if f.suffix in TEXT_EXT and f.is_file() and not f.is_symlink():
            t = f.read_text(errors="surrogateescape")
            if old in t:
                f.write_text(t.replace(old, new), errors="surrogateescape")


def _digest(f: Path) -> str:
    return hashlib.sha1(f.read_bytes()).hexdigest()


def tree_digest(root: Path) -> dict[str, str]:
    return {f.relative_to(root).as_posix(): _digest(f) for f in sorted(root.rglob("*")) if f.is_file()}


def private_gate_copy(hidden: Path, pool: Path) -> tuple[Path, dict]:
    """A private copy of what gate.py reads and writes -- the gate directory, the acceptance pool, and the seeds and
    practice files the backends read (RSI_HIDDEN_FAMILY) -- in a dot directory with a random name beside `hidden`, with
    every absolute path inside it rewritten from `hidden` to the copy. Returns the copy and its manifest (digests of the
    gate directory as it was left)."""
    p = hidden.parent / ("." + secrets.token_hex(12))
    p.mkdir(mode=0o700)
    if (hidden / "gate").is_dir():
        shutil.copytree(hidden / "gate", p / "gate", symlinks=True)
    else:
        (p / "gate").mkdir()
    shutil.copytree(pool, p / pool.name)
    for name in ("seeds.json", "practice"):
        if (hidden / name).is_dir():
            shutil.copytree(hidden / name, p / name)
        elif (hidden / name).is_file():
            shutil.copy2(hidden / name, p / name)
    rewrite_paths(p, str(hidden), str(p))
    return p, tree_digest(p / "gate")


def copy_back_gate(p: Path, hidden: Path, manifest: dict) -> list[str]:
    """The gate's new and changed files from the private copy into hidden/gate (paths rewritten back to `hidden`), and the
    removals it made (a retired round is renamed). Returns what was written."""
    src, dst, written = p / "gate", hidden / "gate", []
    now_ = tree_digest(src)
    for rel, dig in now_.items():
        if manifest.get(rel) == dig:
            continue
        f, g = src / rel, dst / rel
        g.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT_EXT:
            g.write_text(f.read_text(errors="surrogateescape").replace(str(p), str(hidden)), errors="surrogateescape")
        else:
            shutil.copy2(f, g)
        written.append(rel)
    for rel in manifest:
        if rel not in now_ and (dst / rel).exists():
            (dst / rel).unlink()
    return written


def hidden_parent_file() -> Path:
    return STATE / "hidden_parent_mode.txt"


def private_gate_file() -> Path:
    return STATE / "hidden_private_gate.txt"


@contextlib.contextmanager
def parent_locked():
    """While the speculative Reviewer runs, the hidden tree's parent is 0o311: its children are reachable by exact name (the
    gate's private copy is) but nothing lists it. The original mode is kept in STATE for unlock_hidden."""
    if not hidden_lockable():
        yield
        return
    par = HIDDEN.parent
    mode = stat.S_IMODE(par.stat().st_mode)
    hidden_parent_file().write_text(oct(mode))
    par.chmod(0o311)
    try:
        yield
    finally:
        par.chmod(mode)
        hidden_parent_file().unlink(missing_ok=True)


def spec_enabled(stub: bool) -> bool:
    """caps spec_review on, a backend family without claim checks or diagnostic redraws (the speculative path skips
    attribution, the redraw and unpark, which are no-ops for it), and not a plain stub (the stub's gate is a dry run)."""
    return bool(CAPS.get("spec_review") and BK.of(FAMILY) and not CAPS.get("claims") and not CAPS.get("diag_redraw")
                and (not stub or os.environ.get("RSI_STUB_REAL_GATE") == "1"))  # fmt: skip


def spec_worth() -> bool:
    """A speculative review is only worth its cost when the run goes on after this gate call."""
    meta = jload(STATE / "meta.json", {})
    try:
        return stop_reason(meta) is None and len(run_gates(entries(), meta)) + 1 < MAX_GATE_CALLS
    except KeyError:
        return True


def spec_decision(spec: dict | None, incumbent_hash: str) -> str | None:
    """iterate(n+1): "adopt" a speculative review made for this very incumbent, "discard" any other, None when there is none."""
    if not spec:
        return None
    return "adopt" if spec.get("for_incumbent") == incumbent_hash else "discard"


def iteration_dirs() -> list[Path]:
    """The iteration directories (iter_NN): not a speculative review's that no iteration took up (spec.json still there)
    nor the renamed discards."""
    return [
        d for d in sorted(STATE.glob("iter_*")) if re.fullmatch(r"iter_\d+", d.name) and not (d / "spec.json").exists()
    ]


def discard_spec(n: int, it_dir: Path, spec: dict, why: str, inc_hash: str) -> Path:
    """The speculative review of iteration n was made for another incumbent: its directory is kept under a new name and the
    ledger says what it cost."""
    gone = it_dir.with_name(f"{it_dir.name}_spec_discarded_{time.strftime('%Y%m%dT%H%M%S')}")
    it_dir.rename(gone)
    acct = spec.get("account") or {}
    ledger(dict(step="spec_discarded", iter=n, why=why, for_incumbent=spec.get("for_incumbent"), incumbent=inc_hash,
                from_iter=spec.get("from_iter"), kept=str(gone), cost_usd=acct.get("cost_usd", 0.0),
                seconds=acct.get("seconds"), leaks=spec.get("leaks")))  # fmt: skip
    it_dir.mkdir(parents=True, exist_ok=True)
    return gone


def cards_row(n: int, out: Path) -> dict:
    imgs = list((out / "img").glob("*.png"))
    return dict(step="cards", iter=n, cards_bytes=(out / "cards.md").stat().st_size, images=len(imgs),
                image_bytes=sum(p.stat().st_size for p in imgs))  # fmt: skip


def review_common(n: int, inc: dict, base: Path, cards: Path, attribution: dict | None) -> tuple[dict, dict]:
    """What the Reviewer's prompt and the later roles' prompts are filled from: (`common` variables, context `extra`)."""
    common = dict(RULES=RULES, **prompt_vars(), LOOP=LOOP, SIM_PY=SIM_PY, DIAG_INC=base, SKILLS=SKILLS, FAMILY=FAMILY,
                  FAMILY_FILE=HERE / "families" / f"{FAMILY}.json", POOL=LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=role_tmp(), CARDS=cards)  # fmt: skip
    extra = dict(diag_summary=summarize(base), parameters=parameters_of(inc), focus=jload(STATE / "focus.json", {}),
                 parked=jload(STATE / "parked.json", {}),
                 segments=SG.segment_table(B.outcomes(base).values()))  # fmt: skip  # TODO 48.17.6 (rules 48)
    if attribution:
        extra["attribution"] = attribution
    prev = STATE / f"iter_{n - 1:02d}" / "designs.json"  # TODO 33.19: last iteration's design probes
    if prev.exists():
        extra["designs_last"] = jload(prev)
    return common, extra


def run_review(n: int, it_dir: Path, inc: dict, base: Path, common: dict, extra: dict, stub: bool, note: str = ""):
    """The Reviewer(s) of iteration n on the incumbent `inc` and its rounds `base`: (hypotheses, accounting, sessions, leaks).
    Writes hypotheses.json and review_notes; the ledger row is the caller's. `note`: appended to the prompt (a rerun)."""
    ctx = it_dir / "reviewer_context.json"
    jdump(ctx, context(inc, extra))
    prompt = fill((PROMPTS / "reviewer.md").read_text(), CONTEXT=ctx, **common) + note
    stub_h = dict(notes="stub", hypotheses=[dict(id="H1", claim="stub", instead="stub", evidence="stub",
                  delta={"set": {"PLAN_HELD": "pick_simple"} if "PLAN_HELD" in extra["parameters"] else {}},
                  predicted_fixed=[], predicted_broken=[])])  # fmt: skip
    t0 = time.time()
    with rounds_locked(base):
        if SCOPED:
            hyps, acct = scoped_review(prompt, it_dir, stub_h if stub else None)
        elif REVIEWERS > 1:
            hyps, acct = twin_review(prompt, it_dir, stub_h if stub else None)
        else:
            hyps, acct = claude("reviewer", prompt, HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent",
                                stub_h if stub else None)  # fmt: skip
    jdump(it_dir / "hypotheses.json", hyps)
    sessions = acct.pop("sessions", None) or [acct["session"]]
    leaks = sorted({x for sess in sessions for x in transcript_leaks(sess)})
    acct["review_notes"] = copy_review_notes(it_dir, review_sources(), t0) if not leaks else None
    common["REVIEW_NOTES"] = review_notes_dir(it_dir)
    return hyps, acct, sessions, leaks


def review_row(n: int, hyps: dict, acct: dict, sessions: list, leaks: list, **more) -> None:
    n_claims = len(hyps["infeasible"]) if isinstance(hyps.get("infeasible"), list) else 0
    ledger(dict(step="review", iter=n, n_hypotheses=len(hyps["hypotheses"]), n_claims=n_claims, void=bool(leaks),
                leaks=leaks, **acct, **({"sessions": sessions} if len(sessions) > 1 else {}), **more))  # fmt: skip


def speculative_review(n: int, cand: Path, pre: Path, stub: bool, call_no: int, prev_dir: str) -> dict:
    """Change 5: what iterate(n + 1) does up to and including the review, for the candidate of iteration n as the incumbent,
    on the candidate's diagnostic rounds `pre`, written to iter_{n+1} with spec.json. No ledger row. Skipped here because
    a backend family skips them anyway: the incumbent's rounds run (they are `pre`), the diagnostic redraw, unpark and
    the attribution."""
    it2 = STATE / f"iter_{n + 1:02d}"
    if it2.exists():
        it2.rename(it2.with_name(f"{it2.name}_spec_stale_{time.strftime('%Y%m%dT%H%M%S')}"))
    it2.mkdir(parents=True)
    h = candidate_hash(cand)
    dst = LOOP / "incumbents" / h
    if not dst.exists():
        shutil.copytree(cand, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    inc = dict(dir=str(dst), hash=h, set=overrides(dst), since_gate_call=call_no)
    put_skills(dst / "skills")
    try:
        cards = reviewer_cards(n + 1, pre, it2, log=False)
        common, extra = review_common(n + 1, inc, pre, cards, None)
        hyps, acct, sessions, leaks = run_review(n + 1, it2, inc, pre, common, extra, stub)
    finally:
        put_skills(Path(prev_dir) / "skills")
    spec = dict(for_incumbent=h, from_iter=n, account=dict(acct, sessions=sessions), leaks=leaks)
    jdump(it2 / "spec.json", spec)
    return spec


def iterate(n: int, meta: dict, stub: bool) -> str | None:
    it_dir = STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    snapshot(n)
    inc = incumbent()
    spec = jload(it_dir / "spec.json")  # change 5: a review made while the gate of iteration n - 1 ran
    verdict = spec_decision(spec, inc["hash"])
    why_discard = "the candidate it was made for did not become the incumbent"
    if verdict == "adopt" and spec.get("leaks") and cap("role_rerun") and not stub:  # TODO 48.17: review again
        verdict, why_discard = (
            "discard",
            f"its Reviewer named off-limits paths ({', '.join(spec['leaks'])}): reviewed again",
        )
    if verdict == "discard":
        gone = discard_spec(n, it_dir, spec, why_discard, inc["hash"])
        if spec.get("leaks"):  # a leaking review's files leave the visible tree
            hide_void(n, gone)
    put_skills(Path(inc["dir"]) / "skills")
    base_changes = changed_files()
    base = diag_incumbent(inc, it_dir)
    refresh_diag_pool(inc, base, n)
    unpark(base, n)
    attribution = attribute(inc, base, n, it_dir, stub)  # TODO 33.20
    if verdict == "adopt":  # the review and its cards are taken as they are; the ledger rows are written now
        ledger(cards_row(n, it_dir / "cards"))
        common, extra = review_common(n, inc, base, it_dir / "cards" / "cards.md", attribution)
        common["REVIEW_NOTES"] = review_notes_dir(it_dir)
        hyps = jload(it_dir / "hypotheses.json")
        acct = dict(spec["account"])
        sessions = acct.pop("sessions", None) or [acct["session"]]
        leaks = spec.get("leaks") or []
        review_row(n, hyps, acct, sessions, leaks, speculative=True)
        (it_dir / "spec.json").rename(it_dir / "spec_adopted.json")
    else:
        cards = reviewer_cards(n, base, it_dir)
        common, extra = review_common(n, inc, base, cards, attribution)
        hyps, acct, sessions, leaks = run_review(n, it_dir, inc, base, common, extra, stub)
        if leaks and cap("role_rerun") and not stub:  # TODO 48.17: the first leak discards the review, one more call
            discard_attempt(n, "reviewer", acct, leaks, it_dir / "hypotheses.json")
            for d in role_tmp().glob("reviewer_*"):  # the scratch of several Reviewers (reviewer_A, ...)
                shutil.rmtree(d, ignore_errors=True)
            hyps, acct, sessions, leaks = run_review(n, it_dir, inc, base, common, extra, stub,
                                                     note=rerun_note("Reviewer", leaks))  # fmt: skip
        review_row(n, hyps, acct, sessions, leaks)
    if leaks:
        void_iteration(n, "reviewer", leaks)
        return None
    return after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def joint_review(prompts: dict, it_dir: Path, stub: dict | None, per: int) -> tuple[dict, dict]:
    """Several Reviewers in parallel, one per prompt ({tag: prompt}); at most `per` hypotheses of each are kept. Their
    hypotheses and claims are merged with short ids, the tag and the position (AH1, BH2: a Reviewer's own id may be
    anything, and the Manager has to repeat these), the Reviewer's own in `own_id`; their notes are joined. The
    accounting sums cost and keeps the longer wall time; `session` is the first one's (the leak check reads them all)."""
    out: dict = {}

    def one(tag):
        try:
            out[tag] = claude(
                f"reviewer_{tag}", prompts[tag], HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent", stub
            )
        except RoleFailed as e:
            out[tag] = e

    ts = [threading.Thread(target=one, args=(t,)) for t in prompts]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for tag in prompts:  # a failed call reaches cmd_run as the RoleFailed it raised, not a KeyError
        if isinstance(out.get(tag), RoleFailed):
            raise out[tag]
    hyps = dict(notes="", hypotheses=[], infeasible=[])
    for tag in prompts:
        h, _ = out[tag]
        hyps["notes"] += f"[{tag}] {h.get('notes', '')}\n"
        for i, x in enumerate(h["hypotheses"][:per], 1):
            hyps["hypotheses"].append(dict(x, id=f"{tag}H{i}", own_id=x["id"], scope=tag))
        hyps["infeasible"] += [
            dict(c, id=f"{tag}{c.get('id')}") for c in h.get("infeasible") or [] if isinstance(c, dict)
        ]
    accts = [out[t][1] for t in prompts]
    acct = dict(cost_usd=sum(a["cost_usd"] for a in accts), session=accts[0]["session"],
                seconds=max(a["seconds"] for a in accts), sessions=[a["session"] for a in accts])  # fmt: skip
    if any(a.get("committed") for a in accts):
        acct["committed"] = [a["committed"] for a in accts if a.get("committed")]
    return hyps, acct


def scoped_review(prompt: str, it_dir: Path, stub: dict | None) -> tuple[dict, dict]:
    """Reviewers held to a scope each (SCOPES) and two hypotheses."""
    text = prompt.replace("propose at most 4 interventions", "propose at most 2 interventions, within your scope")
    return joint_review({t: f"{text}\n\n{SCOPES[t]}\n" for t in SCOPES}, it_dir, stub, 2)


def twin_review(prompt: str, it_dir: Path, stub: dict | None) -> tuple[dict, dict]:
    """TODO 43.2: REVIEWERS Reviewers on the same prompt. Repeats of one iteration differ more than the Reviewer's
    inputs ever moved the result (reviewer_card/REPORT_ab.md), so the independent readings are the point."""
    return joint_review({chr(65 + k): prompt for k in range(REVIEWERS)}, it_dir, stub, MAX_HYPOTHESES)


def hypothesis_cap() -> int:
    """How many hypotheses the Prober runs: MAX_HYPOTHESES a Reviewer, or two per scope with the scoped Reviewers
    (TODO 33.3: three scopes, six, the plan's 4 -> 6; the probes are cheap since the speed-up)."""
    return 2 * len(SCOPES) if SCOPED else REVIEWERS * MAX_HYPOTHESES


def probe_now(n: int, hyps: dict, inc: dict, base: Path, it_dir: Path, parallel: bool = False) -> list[dict]:
    """The Prober on the iteration's hypotheses, their statuses, probe.json and the ledger's probe row."""
    pr = probe(hyps["hypotheses"][: hypothesis_cap()], inc, base, it_dir)
    set_statuses(hyps, pr)  # TODO 33.17
    records_statuses(hyps, pr, base)  # TODO 33.15.1
    jdump(it_dir / "hypotheses.json", hyps)
    jdump(it_dir / "probe.json", pr)
    ledger(dict(step="probe", iter=n, results=[{k: v for k, v in p.items() if k != "summary"} for p in pr],
                **({"parallel": True} if parallel else {})))  # fmt: skip
    return pr


PROBE_PENDING = ("runs beside the Engineers this iteration (rules 33, TODO 48.17): no result yet; the Composer gets "
                 "them, and the program builds the tuned drafts from them")  # fmt: skip


def after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub) -> str | None:
    """Prober -> Manager -> Engineer -> Gate for a reviewed iteration (also `run --reprobe N`). With probe_parallel
    (TODO 48.17) the Prober runs beside the Engineers instead (multi_engineer_step); a Manager decision that builds
    no drafts gets the probe before it is carried out."""
    parallel = bool(cap("probe_parallel")) and CANDIDATES > 1 and STOP_AFTER != "probe"
    pr = None if parallel else probe_now(n, hyps, inc, base, it_dir)
    claims = [c for c in hyps.get("infeasible") or [] if isinstance(c, dict)]  # older hypotheses carry a string
    if not CAPS["claims"]:  # a backend without claim checks: an infeasibility claim parks nothing
        claims = []
    cr = verify_claims(claims, inc, it_dir) if claims else []
    jdump(it_dir / "claims.json", cr)
    park(cr, claims, n)
    if STOP_AFTER == "probe":
        return "replay: stopped after the probe (--stop-after probe)"
    designs = run_designs(hyps, inc, it_dir, n)  # TODO 33.19
    gaps = gap_candidates()
    jdump(STATE / "gap_candidates.json", gaps)
    ctx = it_dir / "manager_context.json"
    es = entries()
    may_reopen = reopen_allowed(es)
    lead = dict(recurring=recurring(es))  # TODO 33.14: first in the context
    if extra.get("attribution"):  # TODO 33.20: which recurring failures are the motor layer's
        lead["attribution"] = extra["attribution"]
    if designs:  # TODO 33.19: how the factors the Reviewer asked about change the outcome (not evidence)
        lead["designs"] = designs
    notes = repair_notes(es, base, n)  # TODO 33.17: after a rejection
    if notes:
        lead["repair_notes"] = notes
    if not may_reopen:
        lead["reopen_blocked"] = (f"the last {MAX_REOPEN_RUN} iterations were reopens (rules.md 28, version 14): build a "
                                  "hypothesis this iteration")  # fmt: skip
    body = context(inc, dict(extra, hypotheses=hyps, probe=pr if pr is not None else PROBE_PENDING, claims=cr,
                             parked=jload(STATE / "parked.json", {}),
                             gap_candidates=gaps, candidate_pool=pool_view(inc, base)), compact=True)  # fmt: skip
    jdump(ctx, dict(lead, **body))
    prompt = fill((PROMPTS / "manager.md").read_text(), CONTEXT=ctx, **common)
    stub_m = dict(decision="engineer", hypothesis="H1", reason="stub", summary_zh="桩测试")
    ids = [h["id"] for h in hyps["hypotheses"]]
    dec, acct = claude("manager", prompt, manager_schema(may_reopen, ids), it_dir, "Edit,Write,NotebookEdit,Agent,Bash",
                       stub_m if stub else None)  # fmt: skip
    leaks = transcript_leaks(acct["session"])
    if leaks and cap("role_rerun") and not stub:  # TODO 48.17: the first leak discards the decision, one more call
        discard_attempt(n, "manager", acct, leaks)
        dec, acct = claude("manager", prompt + rerun_note("Manager", leaks), manager_schema(may_reopen, ids), it_dir,
                           "Edit,Write,NotebookEdit,Agent,Bash")  # fmt: skip
        leaks = transcript_leaks(acct["session"])
    if pr is None and dec.get("tune"):  # probe_parallel: no probe result to tune from; the program picks the drafts
        dec = dict(dec, tune_ignored=dec["tune"], tune=[])
    jdump(it_dir / "manager.json", dec)
    tune, assign = dec.get("tune") or [], {}
    if (
        dec["decision"] == "engineer" and CANDIDATES > 1
    ):  # change 3: the Engineers' ids as multi_engineer_step takes them
        ids_, dups = assign_ids(dec, hyps["hypotheses"], CANDIDATES, tune)
        assign = dict(assigned=ids_, dropped_duplicates=dups)
    elif tune:  # change 1: tuned drafts need the Composer's candidates (decision engineer, K > 1)
        assign = dict(tune_ignored=True)
    ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), tune=tune or None, **assign,
                hypotheses=dec.get("hypotheses") if CANDIDATES > 1 else None, reopen_claim=dec.get("reopen_claim"),
                base=dec.get("base"), repeats=dec.get("repeats"), skip=dec.get("skip"),
                reason=dec["reason"], summary_zh=dec["summary_zh"], escalation=dec.get("escalation"), infeasible=dec.get("infeasible"),
                gap_proposals=dec.get("gap_proposals"), leaks=leaks, void=bool(leaks), **acct))  # fmt: skip
    if dec.get("gap_proposals") and not leaks:
        with (STATE / "gaps.jsonl").open("a") as fh:
            for g in dec["gap_proposals"]:
                fh.write(json.dumps(dict(t=now(), iter=n, **g), ensure_ascii=False) + "\n")
    if leaks:
        void_iteration(n, "manager", leaks)
        return None
    if dec["decision"] == "reopen" and not may_reopen:  # the schema left it out; an invalid decision ends the iteration
        ledger(dict(step="manager_invalid", iter=n, reason=f"reopen after {MAX_REOPEN_RUN} reopens in a row"))
        return None
    if dec["decision"] == "stop":
        return "manager: stop"
    if dec["decision"] == "review":
        jdump(STATE / "focus.json", dict(iter=n, focus=dec.get("focus", ""), reason=dec["reason"]))
        return None
    dec = force_build(n, dec, hyps, it_dir)  # TODO 33.17 (rules.md 33)
    if pr is None and not (CANDIDATES > 1 and dec["decision"] == "engineer"):  # a reopen: no Engineers to run beside
        pr = probe_now(n, hyps, inc, base, it_dir)
    return engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)


def base_of(n: int, dec: dict) -> dict | None:
    """The Manager's `base` (TODO 33.8): its candidate_pool entry, or None (no base given; an unknown hash is recorded
    and the Engineers start from the incumbent)."""
    h = dec.get("base")
    if not h:
        return None
    e = pool_entry(h)
    if e is None:
        ledger(dict(step="base_unknown", iter=n, base=h, pool=[x["hash"] for x in pool_load()]))
    return e


def base_brief(e: dict | None) -> dict | None:
    """What an Engineer or the Composer reads of the base: the gate still compares the result with the incumbent."""
    if e is None:
        return None
    return dict(hash=e["hash"], dir=e["dir"], set=overrides(Path(e["dir"])), gate=e.get("gate"),
                older_rule=e.get("rule", 1) != GATE_RULE)  # fmt: skip


def engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub) -> str | None:
    """The Manager's decision carried out: the chosen hypothesis (or the reopened park) to the Engineer, then the
    gate (after_engineer)."""
    if CANDIDATES > 1 and dec["decision"] == "engineer":
        return multi_engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)
    reopen = dec.get("reopen_claim") if dec["decision"] == "reopen" else None
    park_ = jload(STATE / "parked.json", {}).get(reopen) if reopen else None
    if reopen and park_ is None:
        ledger(dict(step="reopen_test", iter=n, claim=reopen, n=0, ok=[], why={}, error="no such park"))
        return None
    chosen = next((h for h in hyps["hypotheses"] if h["id"] == dec.get("hypothesis")), None)
    bentry = base_of(n, dec)  # TODO 33.8: the working copy starts from a pool candidate instead of the incumbent
    if bentry:
        put_skills(Path(bentry["dir"]) / "skills")
    ctx = it_dir / "engineer_context.json"
    claims_of = STATE / f"iter_{int(park_['iter']):02d}" / "claims.json" if reopen else None
    tried = [r for r in jload(claims_of, []) if r.get("id") == reopen] if reopen else []
    jdump(ctx, context(inc, dict(hypothesis=None if reopen else chosen, manager=dec,
                                 probe=[] if reopen else [p for p in pr if p.get("id") == dec.get("hypothesis")],
                                 park=dict(park_, id=reopen, checked=tried) if reopen else None,
                                 base=base_brief(bentry))))  # fmt: skip
    prompt = fill((PROMPTS / "engineer.md").read_text(), CONTEXT=ctx, ITER=it_dir, **common)
    start_set = overrides(Path(bentry["dir"])) if bentry else inc["set"]
    stub_e = dict(mechanism="stub", files_changed=[], set={**start_set, **(chosen or {}).get("delta", {}).get("set", {})},
                  tests="none", scope=dict(when="stub", starts=["all"]))  # fmt: skip
    eng, acct = claude("engineer", prompt, ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_e if stub else None)
    return after_engineer(n, it_dir, inc, eng, acct, base_changes, stub, reopen)


def after_engineer(
    n: int, it_dir: Path, inc: dict, eng: dict, acct: dict, base_changes: list, stub: bool, reopen: str | None = None
) -> str | None:
    rel = SKILLS.relative_to(REPO).as_posix() + "/"
    now_changed = changed_files()
    changed = [f for f in now_changed if f not in base_changes or f.startswith(rel)]
    stray = [f for f in now_changed if f not in base_changes and not f.startswith(rel)]
    leaks = transcript_leaks(acct["session"])
    ledger(dict(step="engineer", iter=n, changed=changed, stray=stray, mechanism=eng["mechanism"], set=eng["set"],
                scope=eng.get("scope"), void=bool(leaks), leaks=leaks, **acct))  # fmt: skip
    cand = it_dir / "candidate"
    if cand.exists():
        shutil.rmtree(cand)
    shutil.copytree(SKILLS, cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
    patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                            str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
    (it_dir / "code.patch").write_text(patch)
    for f in stray:  # back as they were
        if (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=REPO, capture_output=True)
        elif (REPO / f).exists():
            (REPO / f).unlink()
    put_skills(Path(inc["dir"]) / "skills")
    if leaks or stray:
        hide_void(n, cand, it_dir / "code.patch")
    if leaks:
        void_iteration(n, "engineer", leaks)
        return None
    if stray:
        ledger(
            dict(
                step="gate",
                iter=n,
                accepted=False,
                void=True,
                reason=f"engineer changed files outside the skills package: {stray}",
            )
        )
        return None
    named = named_objects(cand / "skills")  # TODO 33.13 (rules.md 30)
    if named:
        named_void(n, named, "the candidate")
        return None
    if reopen and not reopen_test(n, it_dir, cand, reopen):
        return None
    return gate_candidate(n, it_dir, cand, stub, reopen=(reopen, reopen_won(it_dir, reopen)) if reopen else None)


def hide_void(n: int, *paths: Path) -> None:
    """TODO 36 (review 2026-10-22, L5): a voided draft (leak or stray) leaves the visible loop tree for
    HIDDEN/<family>/void/iter_NN/, so the Composer and later roles cannot read code written after the audit flagged
    its role. (HIDDEN is unlocked between role calls.)"""
    dst = HIDDEN / "void" / f"iter_{n:02d}"
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():
            if (dst / p.name).exists():
                shutil.rmtree(dst / p.name) if (dst / p.name).is_dir() else (dst / p.name).unlink()
            shutil.move(str(p), str(dst / p.name))


def void_iteration(n: int, role: str, leaks: list[str]) -> None:
    """TODO 36.7 (user 2026-10-22, "按你的建议的来"): a leak voids what the leaking role produced and the loop goes
    on. A Reviewer's, Manager's or Composer's leak -- or the only Engineer's (K = 1, a reopen) -- voids iteration n:
    the incumbent stays and the next iteration starts; MAX_VOID_STREAK of them in a row stop the loop (stop_reason).
    One Engineer's of K voids only its draft (multi_engineer_step). Until then any of these stopped the loop."""
    ledger(dict(step="void_iter", iter=n, role=role, leaks=leaks))
    print(f"iteration {n} void: the {role} named off-limits paths ({', '.join(leaks)})")


def gate_lines() -> list[str]:
    p = GATE_DIR / "ledger.jsonl"
    return p.read_text().splitlines() if p.exists() else []


def gate_candidate(
    n: int, it_dir: Path, cand: Path, stub: bool, reopen: tuple | None = None, shot: int = 1
) -> str | None:
    """`cand` to the gate on the acceptance pool, its diagnostic rounds beside it (the next incumbent's, if accepted).
    `reopen` = (park id, starts won): the park lifts only if this gate call accepts (settle_reopen). `shot` 2: the
    iteration's second gate call (second_shot, TODO 37.4), marked on its ledger row.
    TODO 36 (review 2026-10-22): the verdict is the ledger line gate.py wrote for THIS candidate; no new line, a line
    for another hash, or an exit code other than 0 / 1 is a gate error and stops the run (the last line used to be
    taken as the verdict, so a crashed gate recorded the previous call's).
    TODO 48.17: with check_half the candidate first runs on the hidden half of the diagnostic starts and goes to the gate
    only when it wins there by `check_min_net`; with stage1_first its diagnostic rounds and the early review start
    once gate.py has passed stage 1."""
    if cap("check_half") and not stub and check_pool().exists():
        chk = check_candidate(n, it_dir, cand)
        sent = chk["net"] >= int(cap("check_min_net", 1)) and not chk.get("task_regress")
        jdump(it_dir / "check.json", dict(chk, sent=sent))
        ledger(dict(step="check", iter=n, hash=candidate_hash(cand), sent=sent, **chk))
        if not sent:
            print(f"iteration {n}: the candidate wins {chk['fixed']} and loses {chk['broken']} of the {chk['n']} hidden "
                  f"diagnostic starts (tasks that lose {cap('task_regress', None)} or more net: "
                  f"{chk.get('task_regress') or 'none'}): not sent to the gate")  # fmt: skip
            return None
    redraw_accept()
    before = len(gate_lines())
    pool = accept_pool()
    dry = stub and not (
        BK.of(FAMILY) and os.environ.get("RSI_STUB_REAL_GATE") == "1"
    )  # a fake bed can run the real gate
    spec_on = shot == 1 and spec_enabled(stub) and spec_worth()
    gate_dir, pool_arg, env, priv = GATE_DIR, pool, dict(os.environ, PYTHONPATH="src"), None
    if spec_on:  # change 5: the gate on a private copy, so the hidden tree can be locked for the next Reviewer
        priv = private_gate_copy(HIDDEN, pool)
        private_gate_file().write_text(str(priv[0]))
        gate_dir, pool_arg = priv[0] / "gate", priv[0] / pool.name
        env["RSI_HIDDEN_FAMILY"] = str(priv[0])
    gate_args = [str(SIM_PY), str(HERE / "gate.py"), FAMILY, "--gate", str(gate_dir), "--pool", str(pool_arg),
                 "--candidate", str(cand), "--gpus", GPUS]  # fmt: skip
    if dry:
        gate_args += ["--dry-run"]
    share = max(1, PROCS * 5 // 12)
    gate_args += ["--procs", str(PROCS - share if not stub else PROCS)]
    stop, side = threading.Event(), None
    pre = LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
    gate_done, box, spec_thread = threading.Event(), {}, None
    call_no = len([x for x in gate_lines() if not json.loads(x).get("dry_run")]) + 1
    prev_dir = incumbent()["dir"]

    def start_side() -> None:
        """The candidate's diagnostic rounds beside the gate (the next incumbent's, if accepted), and the early review."""
        nonlocal side, spec_thread
        if not stub or spec_on:
            side = threading.Thread(target=B.run_jobs, args=([B.Job(cand / "skills", pre, overrides(cand), None, True)],
                                    FAMILY, LOOP / "pool_diag", share, it_dir / "prober.log", GPUS),
                                    kwargs=dict(stop=stop), daemon=True)  # fmt: skip
            side.start()
        if spec_on:
            spec_thread = threading.Thread(target=run_spec, daemon=True)
            spec_thread.start()

    if spec_on:

        def run_spec():
            try:
                while side.is_alive():  # the candidate's rounds must be complete, and the gate still running
                    if gate_done.is_set():
                        return
                    time.sleep(1.0)
                if gate_done.is_set() or len(B.outcomes(pre)) < len(B.pool_labels(LOOP / "pool_diag")):
                    return
                with parent_locked():
                    box["spec"] = speculative_review(n, cand, pre, stub, call_no, prev_dir)
            except BaseException as e:  # noqa: BLE001 -- recorded below; the gate's result does not depend on it
                box["error"] = f"{type(e).__name__}: {e}"[:400]

    first = bool(cap("stage1_first")) and not dry  # TODO 48.17: the side work waits for stage 1
    if not first:
        start_side()
    t0, proc = time.time(), None
    with (it_dir / "gate.log").open("w") as glog:
        try:
            proc = subprocess.Popen(gate_args, cwd=REPO, stdout=glog, stderr=subprocess.STDOUT, text=True, env=env)
            if first:
                while proc.poll() is None and not stage1_passed(gate_dir, candidate_hash(cand), t0):
                    time.sleep(5.0)
                if proc.poll() is None:  # stage 1 passed and the gate runs on
                    start_side()
            rc = proc.wait()
        except BaseException:
            gate_done.set()
            stop.set()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            raise
    gate_done.set()
    if (
        priv is not None
    ):  # the speculative Reviewer (if stage 1 let it start) is waited for; then the gate's copy is back
        if spec_thread is not None:
            spec_thread.join()
        copy_back_gate(priv[0], HIDDEN, priv[1])
        shutil.rmtree(priv[0], ignore_errors=True)
        private_gate_file().unlink(missing_ok=True)
        it2 = STATE / f"iter_{n + 1:02d}"
        if "error" in box or (it2.exists() and not (it2 / "spec.json").exists()):
            if it2.exists():
                it2.rename(it2.with_name(f"{it2.name}_spec_failed_{time.strftime('%Y%m%dT%H%M%S')}"))
            if "error" in box:
                ledger(dict(step="spec_failed", iter=n, for_iter=n + 1, error=box["error"]))
    new = gate_lines()[before:]
    try:
        verdict = json.loads(new[-1]) if new else None
    except ValueError:
        verdict = None
    if rc not in (0, 1) or verdict is None or verdict.get("hash") != candidate_hash(cand):
        if side is not None:
            stop.set()
            side.join()
        ledger(dict(step="gate_error", iter=n, rc=rc, log=str(it_dir / "gate.log"),
                    new_lines=len(new)))  # fmt: skip
        return f"gate error at iteration {n} (see gate.log)"
    jdump(it_dir / "gate.json", verdict)
    void = bool(dry or verdict.get("void"))  # a void verdict (incomplete pairing) is no rejection and no gate call
    accepted = bool(verdict["accepted"]) and not void
    if side is not None:
        if not accepted:
            stop.set()
        side.join()
    rounds = GATE_DIR / f"{verdict.get('call', 0):02d}_{verdict['hash']}" / "candidate"
    ledger(dict(step="gate", iter=n, accepted=accepted, reason=verdict["reason"], hash=verdict["hash"],
                result=verdict.get("result"), stage1=verdict.get("stage1"), unparked=unparked_view(rounds),
                pool=str(accept_pool()), void=void, rule=verdict.get("rule", 1),
                **({"shot": shot} if shot > 1 else {})))  # fmt: skip
    if not stub:
        pool_update(n, cand, verdict, accepted)
    if reopen:
        settle_reopen(n, reopen[0], reopen[1], accepted)
    return None


def revert_strays(base_changes: list) -> list[str]:
    """Files a role changed outside the skills working copies, put back as they were; returns them."""
    stray = [f for f in changed_files() if f not in base_changes]
    for f in stray:
        if (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=REPO, capture_output=True)
        elif (REPO / f).exists():
            (REPO / f).unlink()
    return stray


def flips(inc_dir: Path, cand_dir: Path) -> dict:
    """A candidate's diagnostic rounds against the incumbent's, start by start."""
    a, b = B.outcomes(inc_dir), B.outcomes(cand_dir)
    both = sorted(set(a) & set(b))
    return dict(ok=sum(bool(b[x]["ok"]) for x in both), incumbent_ok=sum(bool(a[x]["ok"]) for x in both), n=len(both),
                fixed=[x for x in both if b[x]["ok"] and not a[x]["ok"]],
                broken=[x for x in both if a[x]["ok"] and not b[x]["ok"]])  # fmt: skip


# -- the candidate pool (TODO 33.8, D7), recurring hypotheses and the reopen limit (TODO 33.14, D11), names (33.13) --


def pool_path() -> Path:
    return STATE / "candidates_pool.json"


def pool_load() -> list[dict]:
    return jload(pool_path(), []) or []


def pool_ok(r: dict | None) -> bool:
    """A rejected full-pool tally the pool keeps: more wins than losses, falls and drops within the gate's limits."""
    if not r or "b" not in r:
        return False
    falls_ok = r["candidate_fell"] <= r["incumbent_fell"] + 1
    return r["b"] > r["c"] and falls_ok and r["candidate_dropped"] <= drop_limit(r)


def iter_of(path: str) -> int | None:
    """The iteration a candidate directory belongs to (loop/iter_NN/...)."""
    for part in Path(path).parts:
        if part.startswith("iter_") and part[5:].isdigit():
            return int(part[5:])
    return None


GATE_KEYS = ("n", "candidate_ok", "incumbent_ok", "b", "c", "p_one_sided", "candidate_fell", "incumbent_fell",
             "candidate_dropped", "incumbent_dropped")  # fmt: skip


def pool_add(src: Path, verdict: dict, n: int | None) -> dict:
    """`src` (skills + candidate.json, as the gate judged it) copied to STATE/pool/<hash>; its pool entry."""
    h = verdict["hash"]
    d = STATE / "pool" / h
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    r = verdict["result"]
    return dict(hash=h, iter=n, call=verdict.get("call"), rule=verdict.get("rule", 1), dir=str(d),
                gate={k: r[k] for k in GATE_KEYS if k in r}, reason=verdict.get("reason"))  # fmt: skip


def pool_rank(es: list[dict]) -> list[dict]:
    """The best POOL_KEEP by b - c, the most recent gate call first on ties."""
    return sorted(es, key=lambda e: (-(e["gate"]["b"] - e["gate"]["c"]), -(e.get("call") or 0)))[:POOL_KEEP]


def pool_update(n: int, cand: Path, verdict: dict, accepted: bool) -> None:
    """After a gate call: an accepted candidate clears the pool (the new incumbent's rejects start a new one; the old
    entries stay in the ledger and under STATE/pool); a rejected one with b > c and no fall or drop excess joins it."""
    if accepted:
        old = pool_load()
        jdump(pool_path(), [])
        if old:
            ledger(dict(step="pool", iter=n, cleared=[e["hash"] for e in old], reason="the incumbent changed"))
        return
    if verdict.get("void") or not pool_ok(verdict.get("result")):
        return
    kept = pool_rank([e for e in pool_load() if e["hash"] != verdict["hash"]] + [pool_add(cand, verdict, n)])
    jdump(pool_path(), kept)
    ledger(dict(step="pool", iter=n, added=verdict["hash"], kept=[e["hash"] for e in kept]))


def rebuild_pool() -> list[dict]:
    """The pool from the gate's ledger, for state from before the pool existed (a replay.py setup of an older run): the
    rejected calls since the present incumbent was accepted (incumbent.json `since`), copied from the gate's own
    snapshots."""
    inc = jload(GATE_DIR / "incumbent.json") or {}
    since = int(inc.get("since") or 0)
    found = []
    for line in gate_lines():
        v = json.loads(line)
        if v.get("dry_run") or v.get("void") or v.get("accepted") or int(v.get("call") or 0) <= since:
            continue
        src = GATE_DIR / f"{int(v['call']):02d}_{v['hash']}" / "cand"
        if pool_ok(v.get("result")) and src.exists():
            found.append(pool_add(src, v, iter_of(v.get("candidate", ""))))
    kept = pool_rank(found)
    jdump(pool_path(), kept)
    if kept:
        ledger(dict(step="pool", rebuilt=[e["hash"] for e in kept], since_gate_call=since))
    return kept


def pool_entry(h: str | None) -> dict | None:
    return next((e for e in pool_load() if h and e["hash"] == h), None)


def pool_view(inc: dict, diag_inc: Path | None) -> list[dict]:
    """The pool as a role reads it: per entry the gate's tally, whether an older gate rule rejected it (`older_rule`:
    it may be judged again as it stands), its directory and overrides, its diff against the incumbent, and its flips
    against the incumbent's diagnostic rounds where it ran on them."""
    out = []
    for e in pool_load():
        d = Path(e["dir"])
        patch = STATE / "pool" / f"{e['hash']}.patch"
        patch.write_text(subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc",
                                         str(Path(inc["dir"]) / "skills"), str(d / "skills")],
                                        capture_output=True, text=True).stdout)  # fmt: skip
        v = dict(e, older_rule=e.get("rule", 1) != GATE_RULE, set=overrides(d), patch=str(patch))
        diag = LOOP / "diag" / f"candidate_{e['hash']}"
        if diag_inc is not None and diag.exists():
            v["diag"] = dict(flips(diag_inc, diag), rounds=str(diag))
        out.append(v)
    return out


def hyp_key(n: int, hid: str) -> str:
    return f"{int(n):02d}:{hid}"


def norm_key(x: str) -> str | None:
    """'1:H2', '01:H2', 'iter 1:H2', '1/H2' -> '01:H2'."""
    m = re.match(r"^\s*(?:iter(?:ation)?[_ ]?)?(\d+)\s*[:/ ]\s*(\S+)\s*$", str(x))
    return hyp_key(int(m.group(1)), m.group(2)) if m else None


def mechanism_links(es: list[dict]) -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """The Manager's `repeats` links as groups of hypothesis keys (`NN:id`), the keys an Engineer built, and each
    key's group root (keys never linked are their own group). Shared by recurring() and mechanism_table()."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    built = set()
    for e in es:
        if not isinstance(e.get("iter"), int) or e.get("void"):
            continue
        n = e["iter"]
        if e.get("step") == "manager":
            for r in e.get("repeats") or []:
                a = norm_key(f"{n}:{r.get('now', '')}") if isinstance(r, dict) else None
                for b in (r.get("same_as") or []) if a else []:
                    b = norm_key(b)
                    if b:
                        parent[find(a)] = find(b)
            if e.get("decision") == "engineer":
                built |= {hyp_key(n, h) for h in (e.get("hypotheses") or [e.get("hypothesis")]) if h}
        elif e.get("step") == "engineer" and e.get("hypothesis"):
            built.add(hyp_key(n, e["hypothesis"]))
    groups = collections.defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return dict(groups), built, {x: find(x) for x in list(parent)}


def recurring(es: list[dict]) -> list[dict]:
    """TODO 33.14 (D11, user 2026-10-22): mechanisms Reviewers proposed in two or more iterations that no Engineer was
    given. The Manager links a hypothesis to earlier iterations' (`repeats`); linked ids are one mechanism; one given
    to an Engineer (a Manager `engineer` decision, an Engineer row) counts as built. 32.12 proposed the dish soap's
    narrow side in all five iterations and built it in none."""
    groups, built, _ = mechanism_links(es)
    out = []
    for members in groups.values():
        iters = {m.split(":", 1)[0] for m in members}
        if len(iters) < 2 or members & built:
            continue
        latest = max(members)
        it, hid = latest.split(":", 1)
        hs = (jload(STATE / f"iter_{it}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
        claim = next((h.get("claim") for h in hs if isinstance(h, dict) and h.get("id") == hid), "")
        out.append(dict(proposed=len(iters), ids=sorted(members), latest=latest, claim=_cut(claim, 300)))
    return sorted(out, key=lambda g: (-g["proposed"], g["latest"]))


def mechanism_table(es: list[dict]) -> list[dict]:
    """TODO 37.2 (user 2026-10-23): one row per hypothesis so far -- the index of what was tried, for every role. Run
    C's Reviewer read the whole ledger (203-255 KB of `history`, 170-210k tokens of context per turn) and paged
    through it with scripts; the Manager alone had the compacted history (36.7). Per row: the iteration and id, the
    claim and `instead`, the probe's status and counts, the Manager's `repeats` link (`same_as`) or `skip`, what was
    built from it (each draft's diagnostic flips, void, duplicate, late), the Composer's and the gate's verdicts when
    its draft went on, and `open`: nothing of its mechanism (its linked group) was ever built and the probe did not
    refute it. Hypotheses only; parks and reopens stay in `history`."""
    groups, built, root = mechanism_links(es)
    iters = sorted({e["iter"] for e in es if isinstance(e.get("iter"), int)})
    out = []
    for n in iters:
        rows = [e for e in es if e.get("iter") == n]
        hyps = (jload(STATE / f"iter_{n:02d}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
        cands = {c.get("number"): c for c in (jload(STATE / f"iter_{n:02d}" / "candidates.json", []) or [])
                 if isinstance(c, dict)}  # fmt: skip
        probe = {r.get("id"): r for e in rows if e.get("step") == "probe"
                 for r in e.get("results") or [] if isinstance(r, dict)}  # fmt: skip
        man = next((e for e in reversed(rows) if e.get("step") == "manager" and not e.get("void")), None)
        single = man.get("hypothesis") if man and man.get("decision") == "engineer" else None
        eng = [e for e in rows if e.get("step") == "engineer"]
        comp = next((e for e in rows if e.get("step") == "compose"), None)
        gates = [e for e in rows if e.get("step") == "gate" and not e.get("void")]
        shot2 = next((e for e in rows if e.get("step") == "second_shot"), None)  # TODO 37.4
        repeats = {r.get("now"): r.get("same_as") for r in (man or {}).get("repeats") or [] if isinstance(r, dict)}
        skip = (man or {}).get("skip") or {}
        for h in hyps:
            if not isinstance(h, dict) or "id" not in h:
                continue
            hid = h["id"]
            key = hyp_key(n, hid)
            row = dict(iter=n, id=hid, claim=_cut(h.get("claim"), 160), instead=_cut(h.get("instead"), 100),
                       status=h.get("status") or probe_status(h, probe.get(hid)))  # fmt: skip
            p = probe.get(hid)
            if p and "fixed" in p:
                row["probe"] = (f"{p.get('ok_before')}->{p.get('ok_after')}/{p.get('n')}, fixed {len(p['fixed'])} "
                                f"broken {len(p.get('broken') or [])}")  # fmt: skip
            elif p and p.get("error"):
                row["probe"] = "error: " + _cut(p["error"], 80)
            if repeats.get(hid):
                row["same_as"] = repeats[hid]
            if skip.get(hid):
                row["skip"] = skip[hid]
            mine = [e for e in eng if e.get("hypothesis") == hid or (e.get("hypothesis") is None and hid == single)]
            drafts = []
            for e in mine:
                k = e.get("candidate")
                c = cands.get(k) if k is not None else None
                if e.get("late"):
                    d = "late: stopped before it answered"
                elif e.get("void"):
                    d = "void" + (f" ({_cut(e.get('error'), 60)})" if e.get("error") else "")
                elif e.get("duplicate_of"):
                    d = f"duplicate of draft {e['duplicate_of']}"
                elif c and "fixed" in c:
                    d = f"fixed {len(c['fixed'])} broken {len(c['broken'])}"
                else:
                    d = "built"
                drafts.append((f"draft {k}: " if k is not None else "") + d)
            if drafts:
                row["built"] = "; ".join(drafts)
            ks = {e.get("candidate") for e in mine}
            went_on = bool(mine) and (comp is None or comp.get("decision") == "merge"
                                      or (comp.get("decision") == "adopt" and comp.get("adopted") in ks))  # fmt: skip
            if comp and mine:
                row["composer"] = ("merge" if comp.get("decision") == "merge" else
                                   "adopted" if went_on else "kept the incumbent" if comp.get("decision") == "keep"
                                   else "not adopted")  # fmt: skip
            second = shot2 is not None and shot2.get("draft") in ks and len(gates) > 1  # TODO 37.4
            gate = gates[-1] if second else (gates[0] if gates else None)
            if gate and (went_on or second):
                r = gate.get("result") or {}
                row["gate"] = (
                    ("accepted" if gate.get("accepted") else "rejected")
                    + (f" b {r['b']} c {r['c']}" if "b" in r and "c" in r else "")
                    + (" (second shot)" if second else "")
                )
            members = groups.get(root.get(key, key), {key})
            row["open"] = not (members & built) and row["status"] != "refuted"
            out.append(row)
    return out


def probe_status(h: dict, r: dict | None) -> str:
    """TODO 33.17 (plan section 3.4, D14): a hypothesis after its probe. `verified`: the probe fixed a start and broke
    none; `refuted`: it fixed none; `mixed`: it fixed some and broke some. One with no parameter delta (its test needs
    code, `needs_code`) or whose probe failed to run stays `hypothesis`: nothing was measured."""
    if r and r.get("status") == "untested":  # `needs_code`: the probe did not run it
        return "untested"
    if not ((h.get("delta") or {}).get("set") or sweep_specs(h)) or not r or "error" in r or "fixed" not in r:
        return "hypothesis"
    fixed, broken = len(r.get("fixed") or []), len(r.get("broken") or [])
    if fixed == 0:
        return "refuted"
    return "verified" if broken == 0 else "mixed"


def set_statuses(hyps: dict, pr: list[dict]) -> None:
    """Every hypothesis's `status` from its probe result (probe_status), on the hypothesis and on its probe row."""
    by = {r.get("id"): r for r in pr}
    for h in hyps.get("hypotheses") or []:
        if isinstance(h, dict) and "id" in h:
            h["status"] = probe_status(h, by.get(h["id"]))
            if h["id"] in by:
                by[h["id"]]["status"] = h["status"]


def records_statuses(hyps: dict, pr: list[dict], base: Path) -> None:
    """TODO 33.15.1 (plan section 3.6): a hypothesis the probe left at `hypothesis` (its test needs code) that names a
    `measure` gets its status from the record library (records.judge on the rounds the roles may see, this machine,
    the incumbent's version by default): verified, refuted, or still a hypothesis when there are too few rounds. The
    verdict and its counts go on the hypothesis and on its probe row (`records`), so statuses() and 33.17 read it."""
    want = [h for h in hyps.get("hypotheses") or []
            if isinstance(h, dict) and isinstance(h.get("measure"), dict)
            and h.get("status") in ("hypothesis", "untested")]  # fmt: skip
    if not want:
        return
    rows, version, host = RC.load(REPO, SCRATCH_ROOT), RC.version_of(base), RC.machine()
    by = {r.get("id"): r for r in pr}
    for h in want:
        q = RC.judge(rows, h["measure"], version, host)
        if h.get("status") == "untested" and q["verdict"] == "hypothesis":  # too few rounds: still not tested
            continue
        h["status"], h["records"] = q["verdict"], q
        if h["id"] in by:
            by[h["id"]].update(status=q["verdict"], records=q)


def run_designs(hyps: dict, inc: dict, it_dir: Path, n: int) -> list[dict]:
    """TODO 33.19 (plan section 1.3, D16): the hypotheses' design probes (design.py), at most DS.MAX_PROBES an
    iteration and DS.MAX_ROUNDS rounds each, under the incumbent on the diagnostic pool's starts. Not evidence (rule 7):
    the per-level outcome goes to this iteration's Manager (`lead`), the next Reviewer (`designs_last`), the ledger, and
    this run's measured curves (body/curves_run.jsonl), each tagged with the machine, the controller and its scope --
    `skill` for a parameter of the incumbent, `start` for a start factor (TODO 33.1.7). Not body/curves.jsonl: that is
    the snapshot the sandbox was built with, the one skills read at run time (body.curves, body.best_level), fixed for
    the whole run so a skill behaves the same in every round of it; scripts/rsi/body_curves.py gathers a finished
    run's curves into the origin's."""
    if not CAPS["design_probes"]:  # a backend without design probes (design.py varies scene start factors)
        return []
    specs = [(h["id"], h["design"]) for h in hyps.get("hypotheses") or []
             if isinstance(h, dict) and isinstance(h.get("design"), dict)]  # fmt: skip
    if not specs:
        return []
    tasks_all, params, out = list(F.load(FAMILY)["tasks"]), parameters_of(inc), []
    for k, (hid, spec) in enumerate(specs):
        why = (
            "over the limit of design probes per iteration" if k >= DS.MAX_PROBES else DS.check(spec, tasks_all, params)
        )
        if why:
            out.append(dict(hypothesis=hid, factor=spec.get("factor"), error=why))
            continue
        res = DS.run(spec, FAMILY, LOOP / "pool_diag", Path(inc["dir"]) / "skills", inc["set"], it_dir / f"design_{k}",
                     PROCS, GPUS)  # fmt: skip
        out.append(dict(hypothesis=hid, **res))
    jdump(it_dir / "designs.json", out)
    ledger(dict(step="design", iter=n, results=[{k: v for k, v in r.items() if k != "dir"} for r in out]))
    curves = LOOP.parent / "body" / "curves_run.jsonl"
    curves.parent.mkdir(parents=True, exist_ok=True)
    with curves.open("a") as fh:
        for r in out:
            if "levels" in r:
                scope = {"param": "skill", "primitive": "body"}.get(r.get("factor"), "start")  # 33.19.1: body alone
                tag = dict(machine=RC.machine(), scope=scope)
                fh.write(json.dumps(dict(t=now(), iter=n, incumbent=inc["hash"], family=FAMILY, **tag, **r)) + "\n")
    return out


def statuses(es: list[dict]) -> dict[str, str]:
    """`NN:id` -> status, from the ledger's probe rows (rows from before TODO 33.17 carry none)."""
    out = {}
    for e in es:
        if e.get("step") == "probe" and isinstance(e.get("iter"), int):
            for r in e.get("results") or []:
                if isinstance(r, dict) and r.get("id") and r.get("status"):
                    out[hyp_key(e["iter"], r["id"])] = r["status"]
    return out


def assign_ids(dec: dict, hyps: list[dict], k: int, tune=()) -> tuple[list[str], int]:
    """Change 3: the Engineers' hypothesis ids -- the Manager's `hypotheses` (else `hypothesis`) with duplicates removed,
    order kept, at most k. Fewer than k: topped up from the other hypotheses in Reviewer order that are not refuted, not
    in `tune`, and whose `tasks` do not overlap the tasks already assigned; none qualifies, fewer Engineers. No
    hypotheses at all and `tune` set: no Engineer (the tuned drafts go to the Composer alone). Returns (ids, duplicates
    dropped)."""
    known = {h["id"]: h for h in hyps if isinstance(h, dict) and "id" in h}
    tune = [t for t in dict.fromkeys(tune or []) if t in known]
    raw = [h for h in (dec.get("hypotheses") or []) if h in known] or [h for h in [dec.get("hypothesis")] if h in known]
    if not raw:
        if tune:
            return [], 0
        raw = [next(iter(known))] if known else []
    ids = list(dict.fromkeys(raw))
    dropped = len(raw) - len(ids)
    ids = ids[:k]
    taken = {t for i in ids for t in known[i].get("tasks") or []}
    for hid, h in known.items():
        if len(ids) >= k:
            break
        if hid in ids or hid in tune or h.get("status") == "refuted":
            continue
        tasks = set(h.get("tasks") or [])
        if tasks & taken:
            continue
        ids.append(hid)
        taken |= tasks
    return ids, dropped


def force_build(n: int, dec: dict, hyps: dict, it_dir: Path) -> dict:
    """TODO 33.17 (plan section 3.4, D14; user 2026-10-22: "把 33.15 到 33.18 先做了吧"): a mechanism Reviewers proposed
    in two or more iterations (recurring, with this Manager's `repeats` in the ledger already), never given to an
    Engineer, with a member among this iteration's hypotheses, and not refuted (its latest probed member's status) takes
    one Engineer slot -- the last one -- unless the Manager's `skip` names one of SKIP_CHECKS for it. The first such
    mechanism only (the most proposed). Returns the decision the Engineers carry out; the Manager's own stays in
    manager.json. 32.12 proposed the dish soap's narrow side five times and built it none; option B built it in its
    eighth iteration, two after the prompt rule (33.14) asked for it."""
    if dec.get("decision") != "engineer":
        return dec
    es = entries()
    st = statuses(es)
    known = [h["id"] for h in hyps.get("hypotheses") or [] if isinstance(h, dict) and "id" in h]
    for g in recurring(es):
        mine = [m for m in g["ids"] if m.startswith(f"{n:02d}:") and m.split(":", 1)[1] in known]
        if not mine:
            continue
        probed = [m for m in sorted(g["ids"]) if st.get(m) and st[m] not in ("hypothesis", "untested")]
        if probed and st[probed[-1]] == "refuted":
            continue
        hid = mine[0].split(":", 1)[1]
        skip = dec.get("skip") or {}
        check = skip.get(hid) or skip.get(mine[0])
        if check in SKIP_CHECKS:
            ledger(dict(step="forced_skip", iter=n, hypothesis=hid, mechanism=g["ids"], check=check))
            return dec
        if CANDIDATES > 1:
            ids, _ = assign_ids(
                dec, hyps.get("hypotheses") or [], CANDIDATES, [*(dec.get("tune") or []), hid]
            )  # hid keeps its slot
            replaced = None
            if hid not in ids:  # the last slot, or a free one
                replaced = ids[-1] if len(ids) >= CANDIDATES else None
                ids = [*ids[: CANDIDATES - 1], hid] if len(ids) >= CANDIDATES else [*ids, hid]
            out = dict(dec, hypotheses=ids)
        else:
            replaced = dec.get("hypothesis")
            out = dict(dec, hypothesis=hid)
        ledger(dict(step="forced", iter=n, hypothesis=hid, replaced=replaced, mechanism=g["ids"],
                    proposed=g["proposed"], skip_given=check))  # fmt: skip
        jdump(it_dir / "manager_forced.json", out)
        return out
    return dec


def repair_notes(es: list[dict], diag: Path, n: int) -> dict | None:
    """TODO 33.17 (plan sections 3.4, 3.5): when the last iteration before `n` ended with a gate rejection or the
    Composer kept the incumbent, what the Manager reads first after `recurring`: the incumbent's diagnostic failures by
    class (`failure_class`, 33.15), the rejected candidate's tally by class from the gate (wins under the class the
    incumbent failed with, losses under the class the candidate failed with), and the standing suggestion per class
    (CLASS_SUGGEST). DexAgent attaches its diagnosis to every rejection the same way."""
    prev = max((e["iter"] for e in es if isinstance(e.get("iter"), int) and e["iter"] < n), default=None)
    if prev is None:
        return None
    rows = [e for e in es if e.get("iter") == prev]
    gate = next((e for e in reversed(rows) if e.get("step") == "gate" and not e.get("void")), None)
    kept = any(e.get("step") == "compose" and e.get("decision") == "keep" for e in rows)
    if gate and not gate.get("accepted"):
        r = gate.get("result") or gate.get("stage1") or {}
        trigger = f"iteration {prev}: the gate rejected ({_cut(gate.get('reason'), 80)})"
    elif kept and not gate:
        r, trigger = {}, f"iteration {prev}: the Composer kept the incumbent"
    else:
        return None
    classes = collections.Counter(
        (x.get("failure_class") or "unknown") for x in B.outcomes(diag).values() if not x.get("ok")
    )  # fmt: skip
    per = r.get("per_class") or {}
    names = sorted(set(classes) | set(per), key=lambda k: (-classes.get(k, 0), k))
    return dict(trigger=trigger, diag_failures_by_class=dict(classes.most_common()), gate_by_class=per,
                suggestions={k: CLASS_SUGGEST[k] for k in names if k in CLASS_SUGGEST})  # fmt: skip


def reopen_allowed(es: list[dict]) -> bool:
    """TODO 33.14 (D11): no more than MAX_REOPEN_RUN Manager `reopen` decisions in a row (the last valid decision per
    iteration). 32.12's iterations 3-5 were all reopens of one park each."""
    last = {}
    for e in es:
        if e.get("step") == "manager" and not e.get("void") and isinstance(e.get("iter"), int):
            last[e["iter"]] = e.get("decision")
    recent = [last[k] for k in sorted(last)][-MAX_REOPEN_RUN:]
    return not (len(recent) == MAX_REOPEN_RUN and all(d == "reopen" for d in recent))


def manager_schema(allow_reopen: bool, ids: list | None = None) -> dict:
    """`ids`: this iteration's hypotheses, the only ids the Manager can name. multi_engineer_step keeps the ids it knows,
    so one the Manager shortened was dropped without a word: a replay gave its three Engineers one hypothesis."""
    sch = json.loads(json.dumps(MANAGER_SCHEMA))
    if not allow_reopen:
        sch["properties"]["decision"]["enum"] = [d for d in sch["properties"]["decision"]["enum"] if d != "reopen"]
    if ids:
        sch["properties"]["hypothesis"]["enum"] = list(ids)
        sch["properties"]["hypotheses"]["items"]["enum"] = list(ids)  # may be empty when `tune` names the drafts
        sch["properties"]["tune"]["items"]["enum"] = list(ids)
    return sch


_NAMES: dict = {}


def object_names() -> set[str]:
    """Names a skills package may not branch on (rules.md 30, TODO 33.13, D10): every scene object's key, body and ADT
    instance name, and the key's stem (Mug_2 -> Mug) unless the stem is also a semantic name (prototype, category:
    Orange_A -> Orange stays allowed). Semantic fields (category, prototype) are features a flow may use."""
    scene = family_spec().get("scene")
    if scene not in _NAMES:
        objs = (jload(REPO / scene / "scene_meta.json", {}) or {}).get("objects") or {} if scene else {}
        names, semantic = set(), set()
        for k, o in objs.items():
            sem = o.get("semantic") or {}
            adt = sem.get("adt") or {}
            semantic |= {str(x).lower() for x in (adt.get("prototype_name"), adt.get("category")) if x}
            names |= {k, o.get("body") or "", adt.get("instance_name") or ""}
        stems = {re.sub(r"_(\d+|[A-Z])$", "", k) for k in objs}
        names |= {x for x in stems if len(x) >= 3 and x.lower() not in semantic}
        _NAMES[scene] = {x for x in names if x}
    return _NAMES[scene]


def named_objects(skills: Path, names: set[str] | None = None) -> list[str]:
    """String constants in the skills package (docstrings aside, f-string parts included) equal to an object name
    (object_names): `file:line 'name'`. A hit voids the draft (rules.md 30): a flow chosen by name cannot be reused for
    a similar object."""
    import ast

    names = object_names() if names is None else names
    hits = []
    for f in sorted(Path(skills).rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except (SyntaxError, ValueError):
            continue  # the gate's static step reports a package that does not load
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docs.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if node.value in names:
                    hits.append(f"{f.relative_to(skills).as_posix()}:{node.lineno} {node.value!r}")
    return hits


def named_void(n: int, named: list[str], what: str) -> None:
    ledger(dict(step="gate", iter=n, accepted=False, void=True, named_objects=named,
                reason=f"{what} branches on object names (rules.md 30): {', '.join(named[:6])}"))  # fmt: skip


def early_diag(k: int, w: Path, eng: dict, inc: dict, it_dir: Path) -> None:
    """TODO 37.4 (E): a draft's diagnostic rounds start the moment its Engineer answers, in its own thread, instead of
    after the slowest Engineer; the rounds go to the same hash-keyed directory the main step uses, so that later call
    finds them done. A failure here only costs the head start (the main step runs what is missing)."""
    try:
        cand = it_dir / f"candidate_{k}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(w / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
        h = candidate_hash(cand)
        if h == inc["hash"] or named_objects(cand / "skills"):
            return
        labels, _ = draft_run(eng, inc)  # change 4: the draft's tasks and a few sentinels, or the whole pool
        B.run_jobs([B.Job(cand / "skills", LOOP / "diag" / f"candidate_{h}", overrides(cand), labels, True)], FAMILY,
                   LOOP / "pool_diag", PROCS, it_dir / f"diag_{k}.log", GPUS)  # fmt: skip
    except Exception as e:  # noqa: BLE001 -- the main step repeats the run
        print(f"early diag of draft {k}: {type(e).__name__}: {e}", flush=True)


def round_path(r: dict) -> list:
    """What a round did, step by step. Rounds are deterministic, so two rounds of a start with different paths mean
    the code reached this one."""
    return [[s.get("step"), s.get("ok"), s.get("t0"), s.get("t1")] for s in r.get("steps") or []]


def same_round(a: dict, b: dict) -> bool:
    return (
        round_path(a) == round_path(b) and bool(a.get("ok")) == bool(b.get("ok")) and a.get("t_end") == b.get("t_end")
    )


def footprint(ref: Path, cand: Path, scope: dict | None) -> dict:
    """Rule 38: the diagnostic rounds a draft changed against the rounds it started from (`ref`), measured, beside what
    its Engineer said: `outside` changed and not declared, `missed` declared and left as they were. Only starts both ran
    count; `not_run` are the pool starts the draft did not run (change 4)."""
    a, b = B.outcomes(ref), B.outcomes(cand)
    both = sorted(set(a) & set(b))
    changed = [x for x in both if not same_round(a[x], b[x])]
    declared = list((scope or {}).get("starts") or [])
    return dict(n=len(both), changed=changed, declared=declared, not_run=sorted(set(a) - set(b)),
                outside=[] if "all" in declared else [x for x in changed if x not in declared],
                missed=[x for x in declared if x in both and x not in changed])  # fmt: skip


def segments_touched(ref: Path, cand: Path, changed: list[str], table: dict) -> dict:
    """TODO 48.17.6 (rules 48): where each round a draft changed first left the rounds it started from, by the kind
    of that step in its task (segments.touched): a change that starts in a stable step is a risk the Composer weighs."""
    a, b = B.outcomes(ref), B.outcomes(cand)
    where = {}
    for x in changed:
        if x in a and x in b:
            d = first_difference(a[x], b[x])
            s = d["incumbent"] or d["draft"]
            if s:
                where[x] = (a[x].get("task") or task_of(x), s["step"])
    return SG.touched(where, table)


def first_difference(a: dict, b: dict) -> dict:
    """Where round `b` left round `a`'s path: the first step that differs, as each round has it."""
    pa, pb = round_path(a), round_path(b)
    k = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
    keys = ("step", "ok", "t0", "t1")
    return dict(index=k, incumbent=dict(zip(keys, pa[k])) if k < len(pa) else None,
                draft=dict(zip(keys, pb[k])) if k < len(pb) else None)  # fmt: skip


def repair_brief(inc_rounds: Path, cand_rounds: Path, was: dict, fp: dict) -> dict:
    """What a repair turn reads of the draft's run on all the diagnostic starts."""
    a, b = B.outcomes(inc_rounds), B.outcomes(cand_rounds)
    broken = [dict(label=x, why=b[x].get("why"), failure_class=b[x].get("failure_class"),
                   first_difference=first_difference(a[x], b[x])) for x in was["broken"]]  # fmt: skip
    return dict(draft_rounds=str(cand_rounds), incumbent_rounds=str(inc_rounds), ok=was["ok"], incumbent_ok=was["incumbent_ok"],
                n=was["n"], fixed=was["fixed"], broken=broken, changed=fp["changed"], outside=fp["outside"], missed=fp["missed"])  # fmt: skip


def repair_draft(
    n: int, k: int, w: Path, res: tuple[dict, dict], inc: dict, ref: Path, it_dir: Path, common: dict
) -> tuple[dict, dict]:  # fmt: skip
    """Rule 38 (TODO 44.2, `run --repair`): an Engineer checks its draft on the starts it means to fix, so it may never
    see a success it broke; only the Composer does, and the Composer cannot ask for a repair. A draft that broke one of
    the incumbent's successes gets one more turn in its own session with that run's result. The repair stands when it
    breaks fewer starts without losing net ones; otherwise the first draft goes on (its rounds are kept, hash-keyed)."""
    eng, acct = res
    cand = it_dir / f"candidate_{k}"
    if not cand.exists():  # early_diag failed before it copied the draft
        return res
    inc_rounds = LOOP / "diag" / f"incumbent_{inc['hash']}"
    rounds = LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
    if not B.outcomes(rounds):
        return res
    was = flips(inc_rounds, rounds)
    if not was["broken"]:
        return res
    first = w / "skills_first"
    shutil.copytree(w / "skills", first, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    brief = w / "repair_brief.json"
    jdump(brief, repair_brief(inc_rounds, rounds, was, footprint(ref, rounds, eng.get("scope"))))
    prompt = fill((PROMPTS / "repair.md").read_text(), BRIEF=brief, ITER=w, **dict(common, SKILLS=w / "skills"))
    row = dict(
        step="repair",
        iter=n,
        candidate=k,
        session=acct["session"],
        before=dict(fixed=len(was["fixed"]), broken=was["broken"]),
    )
    stop = threading.Event()
    timer = threading.Timer(REPAIR_MIN * 60, stop.set)
    timer.start()
    try:
        eng2, acct2 = claude(f"engineer_{k}", prompt, ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent", stop=stop,
                             resume=acct["session"])  # fmt: skip
    except RoleFailed as e:
        timer.cancel()
        shutil.rmtree(w / "skills")
        shutil.copytree(first, w / "skills")
        early_diag(k, w, eng, inc, it_dir)
        ledger(dict(row, kept="first", error=str(e)[:300], late=isinstance(e, RoleLate)))
        return res
    timer.cancel()
    early_diag(k, w, eng2, inc, it_dir)
    rounds2 = LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
    now_ = flips(inc_rounds, rounds2) if B.outcomes(rounds2) else None
    net = lambda f: len(f["fixed"]) - len(f["broken"])  # noqa: E731
    better = now_ is not None and len(now_["broken"]) < len(was["broken"]) and net(now_) >= net(was)
    total = dict(acct, cost_usd=acct["cost_usd"] + acct2["cost_usd"], seconds=acct["seconds"] + acct2["seconds"])
    ledger(dict(row, kept="repair" if better else "first", cost_usd=acct2["cost_usd"], seconds=acct2["seconds"],
                after=dict(fixed=len(now_["fixed"]), broken=now_["broken"]) if now_ else None))  # fmt: skip
    if better:
        return eng2, total
    shutil.rmtree(w / "skills")
    shutil.copytree(first, w / "skills")
    early_diag(k, w, eng, inc, it_dir)  # candidate_k back to the first draft; its rounds are done
    return eng, total


def second_shot_pick(cands: list[dict], gated_hash: str, rejected: set) -> dict | None:
    """TODO 37.4 (E): the draft to send after a rejection -- one the gate has not judged (not the rejected hash, not
    rejected before under the present rule) with more diagnostic starts fixed than broken; the largest net first,
    fewer broken on ties, then the lower draft number."""
    ok = [c for c in cands if c.get("hash") not in rejected and c.get("hash") != gated_hash and "fixed" in c
          and len(c["fixed"]) > len(c.get("broken") or [])]  # fmt: skip
    return min(ok, key=lambda c: (-(len(c["fixed"]) - len(c["broken"])), len(c["broken"]), c["number"])) if ok else None


def second_shot(n: int, it_dir: Path, cands: list[dict], gated_hash: str, stub: bool) -> str | None:
    """TODO 37.4 (E, user 2026-10-23): when the gate rejected the composed candidate (a real verdict, not a void call),
    the best remaining draft goes to the gate in the same iteration: a second shot costs the gate's 3-5 minutes
    instead of another 60-80 minute iteration (run C: iterations 10 and 12 ended in rejections with two unsent drafts
    each). It is a gate call of the budget, so none is taken when the budget is spent. The first shot's verdict and
    log stay as gate_shot1.json / gate_shot1.log."""
    verdict = jload(it_dir / "gate.json")
    if not verdict or verdict.get("void") or verdict.get("accepted"):
        return None
    es = entries()
    if len(run_gates(es, jload(STATE / "meta.json", {}))) >= MAX_GATE_CALLS:
        return None
    rejected = {e["hash"] for e in es if e.get("step") == "gate" and not e.get("accepted") and not e.get("void")
                and e.get("hash") and e.get("rule", 1) == GATE_RULE}  # fmt: skip
    pick = second_shot_pick(cands, gated_hash, rejected)
    if pick is None:
        return None
    for f in ("gate.json", "gate.log"):
        if (it_dir / f).exists():
            shutil.copy(it_dir / f, it_dir / f.replace(".", "_shot1."))
    ledger(dict(step="second_shot", iter=n, draft=pick["number"], hypothesis=pick.get("hypothesis"), hash=pick["hash"],
                fixed=len(pick["fixed"]), broken=len(pick["broken"]), after=gated_hash))  # fmt: skip
    return gate_candidate(n, it_dir, Path(pick["dir"]), stub, shot=2)


def shadow_drafts(n: int, cands: list[dict], inc: dict) -> None:
    """TODO 45.7: the imagined score of this iteration's drafts and incumbent, before their real rounds, written under
    HIDDEN (mode 000 while any role runs), neither in the ledger nor in a context. Nothing is returned on purpose, and a
    failure or a timeout is one line in the log: a measurement must not be able to stop or steer an iteration."""
    if not SHADOW:
        return
    out = HIDDEN / "shadow" / f"iter_{n:02d}"
    t0, error = time.time(), None
    try:
        out.mkdir(parents=True, exist_ok=True)
        rows = [
            dict(number=c["number"], hash=c["hash"], dir=c["dir"], set=overrides(Path(c["dir"])), incumbent=False)
            for c in cands
        ]
        rows.append(dict(number="incumbent", hash=inc["hash"], dir=inc["dir"], set=inc["set"], incumbent=True))
        jdump(out / "candidates.json", rows)
        r = subprocess.run([str(B.SIM_PY), SHADOW, "--family", FAMILY, "--pool", str((LOOP / "pool_diag").resolve()),
                            "--candidates", str((out / "candidates.json").resolve()), "--out", str(out.resolve()),
                            "--procs", str(min(PROCS, SHADOW_PROCS))], cwd=Path(SHADOW).resolve().parent, capture_output=True,
                           text=True, timeout=SHADOW_TIMEOUT_S,
                           env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})  # fmt: skip
        (out / "shadow.log").write_text(r.stdout + r.stderr)
        error = None if r.returncode == 0 else f"exit {r.returncode}"
    except Exception as e:  # noqa: BLE001
        error = repr(e)[:300]
    print(f"shadow iter {n}: {'done' if error is None else 'failed, ' + error} in {time.time() - t0:.0f} s", flush=True)


def imagined_view(n: int, it_dir: Path, cands: list[dict], inc: dict, base: Path) -> dict | None:
    """TODO 45.7.2: what the imagined Motor says about the drafts on starts outside the diagnostic pool, for the Composer's
    context. None when it is off or fails (one line in the log): the iteration goes on with the real rounds alone."""
    if not IMAGINE:
        return None
    out = it_dir / "imagined"
    t0, view, error = time.time(), None, None
    try:
        out.mkdir(parents=True, exist_ok=True)
        rows = [dict(number=c["number"], hash=c["hash"], dir=c["dir"], set=overrides(Path(c["dir"])), incumbent=False,
                     diag=c["diag"]) for c in cands]  # fmt: skip
        rows.append(
            dict(number="incumbent", hash=inc["hash"], dir=inc["dir"], set=inc["set"], incumbent=True, diag=str(base))
        )
        jdump(out / "candidates.json", rows)
        forbid = [str(p.resolve()) for p in (HIDDEN / "pool_accept", HIDDEN / "pool_sealed") if p.exists()]
        r = subprocess.run([str(B.SIM_PY), IMAGINE, "--family", FAMILY, "--diag-pool", str((LOOP / "pool_diag").resolve()),
                            "--extra-pool", str((LOOP / "pool_imagine").resolve()), "--forbid", *forbid,
                            "--candidates", str((out / "candidates.json").resolve()), "--out", str(out.resolve()),
                            "--procs", str(min(PROCS, SHADOW_PROCS))], cwd=Path(IMAGINE).resolve().parent, capture_output=True,
                           text=True, timeout=IMAGINE_TIMEOUT_S,
                           env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})  # fmt: skip
        (out / "imagined.log").write_text(r.stdout + r.stderr)
        if r.returncode == 0:
            view = json.loads((out / "imagined.json").read_text())
        else:
            error = f"exit {r.returncode}"
    except Exception as e:  # noqa: BLE001
        error = repr(e)[:300]
    print(
        f"imagined iter {n}: {'done' if error is None else 'failed, ' + error} in {time.time() - t0:.0f} s", flush=True
    )
    return view


MIN_SWEEP_STARTS = 3  # an Engineer-declared knob is not swept on fewer starts than this


def unknown_keys(skills: Path, set_: dict, items: list[dict]) -> list[str]:
    """The tunable keys the draft's package does not accept (its loader's apply_set refuses them)."""
    if be := BK.of(FAMILY):
        try:
            pkg = be.load_skills(skills, "tunable_check", FAMILY)
        except Exception:  # noqa: BLE001 -- a package that does not load tunes nothing
            return [it["key"] for it in items]
        bad = []
        for it in items:
            try:
                pkg.apply_set(pkg, {**set_, it["key"]: it["values"][0]})
            except Exception:  # noqa: BLE001
                bad.append(it["key"])
        return bad
    code = ("import json,sys; sys.path[:0]=[sys.argv[1], 'src']; from scene_round import load_skills; "
            "p=load_skills(__import__('pathlib').Path(sys.argv[2])); s=json.loads(sys.argv[3]); bad=[]\n"
            "for it in json.loads(sys.argv[4]):\n"
            "    try: p.apply_set(p, {**s, it['key']: it['values'][0]})\n"
            "    except Exception: bad.append(it['key'])\n"
            "print(json.dumps(bad))")  # fmt: skip
    r = subprocess.run([str(SIM_PY), "-c", code, str(HERE), str(skills), json.dumps(set_), json.dumps(items)], cwd=REPO,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))  # fmt: skip
    return (
        json.loads(r.stdout.strip().splitlines()[-1])
        if r.returncode == 0 and r.stdout.strip()
        else [it["key"] for it in items]
    )


def tunables_of(eng: dict) -> list[dict]:
    """An Engineer's `tunable` made safe: at most 3 distinct keys, each with 2-5 distinct values (as a sweep)."""
    out: dict = {}
    for it in (eng.get("tunable") or [])[:3]:
        for spec in sweep_specs(dict(sweep=[it])) if isinstance(it, dict) else []:
            out.setdefault(spec["key"], spec)
    return list(out.values())


def tune_draft(k: int, w: Path, eng: dict, inc: dict, it_dir: Path) -> dict:
    """Change 1 (Engineer-declared knobs): after the draft's own diagnostic run, sweep its `tunable` keys one at a time on
    the draft's scope starts (sentinels excluded), each against the incumbent as in the probe; a key's best value joins the
    draft's set when it beats the draft as it stands (net, then fewer broken). Unknown keys are dropped and recorded. The
    draft's candidate dir, hash and rounds then follow the tuned set (the best variant's rounds are copied, its sentinels
    run by early_diag). Returns the draft's answer with `tuned`, `tunable_sweep` and `tunable_dropped` added."""
    items = tunables_of(eng)
    cand = it_dir / f"candidate_{k}"
    if not items or not cand.exists():
        return eng
    set0 = dict(eng.get("set") or {})
    bad = set(unknown_keys(cand / "skills", set0, items))
    kept = [it for it in items if it["key"] not in bad]
    out = dict(eng, tuned={}, tunable_sweep=[], tunable_dropped=sorted(bad))
    labels, ran = draft_run(eng, inc)
    pool = B.pool_labels(LOOP / "pool_diag")
    starts = sorted(set(labels if labels is not None else pool) - set(ran["sentinels"]))
    before = _ok_of(LOOP / "diag" / f"incumbent_{inc['hash']}")
    cur_dir = LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
    cur_ok = _ok_of(cur_dir)
    starts = [x for x in starts if x in cur_ok and x in before]
    n_vals = sum(len(it["values"]) for it in kept)
    if not kept or not starts or n_vals == 0:
        return out
    m = min(len(starts), sweep_budget().left // n_vals)
    if m < MIN_SWEEP_STARTS:
        out["tunable_sweep"] = [dict(skipped="sweep budget spent", left=sweep_budget().left)]
        return out
    sweep_budget().reserve(n_vals * m)
    sub = subsample_starts(starts, before, m)
    cur_row = sweep_row("", None, before, {x: cur_ok[x] for x in sub})
    cur_set, best_dir, table = dict(set0), None, []
    for ki, it in enumerate(kept):
        dirs = [it_dir / f"tune_{k}" / f"{ki}_{i}" for i in range(len(it["values"]))]
        try:
            B.run_jobs([B.Job(cand / "skills", d, {**cur_set, it["key"]: v}, tuple(sub))
                        for d, v in zip(dirs, it["values"])], FAMILY, LOOP / "pool_diag", PROCS,
                       it_dir / f"tune_{k}.log", GPUS)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            table.append(dict(key=it["key"], error=f"{type(e).__name__}: {e}"[:200]))
            continue
        rows = [sweep_row(it["key"], v, before, _ok_of(d)) for d, v in zip(dirs, it["values"])]
        live = [i for i, r in enumerate(rows) if r["n"]]
        table.append(
            dict(key=it["key"], baseline=cur_row, rows=rows, starts=len(sub), subsampled=len(sub) < len(starts))
        )
        if not live:
            continue
        b = live[best_row([rows[i] for i in live])]
        if (rows[b]["net"], -rows[b]["broken"]) > (cur_row["net"], -cur_row["broken"]):
            cur_set[it["key"]] = it["values"][b]
            cur_row, best_dir = rows[b], dirs[b]
            table[-1]["adopted"] = it["values"][b]
    out["tunable_sweep"] = table
    out["tuned"] = {key: v for key, v in cur_set.items() if set0.get(key) != v}
    if out["tuned"]:
        out["set"] = cur_set
        (cand / "candidate.json").write_text(json.dumps({"set": cur_set}, indent=1))
        if best_dir is not None:
            seed_rounds(best_dir, LOOP / "diag" / f"candidate_{candidate_hash(cand)}", sub)
        early_diag(k, w, out, inc, it_dir)  # the draft's other starts and its sentinels, under the tuned hash
    return out


def tuned_drafts(n, it_dir, inc, known, pr, tune, src, bentry, number0, taken, jobs) -> list[dict]:
    """Change 1: a candidate the driver builds from a probed hypothesis -- the incumbent's skills (or the Manager's base)
    with the incumbent's overrides plus the probe's best set. Its diagnostic rounds are the probe rounds of the best value,
    copied under the candidate's hash (same skills, same set), and the other starts of change 4's labels run as jobs.
    Needs a probe net of at least 1 and no `needs_code`."""
    by = {p.get("id"): p for p in pr}
    out: list[dict] = []
    pool = B.pool_labels(LOOP / "pool_diag")
    ok_inc = _ok_of(LOOP / "diag" / f"incumbent_{inc['hash']}")
    for hid in tune:
        p, row = by.get(hid) or {}, dict(step="tuned", iter=n, hypothesis=hid)
        net = len(p.get("fixed") or []) - len(p.get("broken") or [])
        why = ("needs_code" if known[hid].get("needs_code") else "no probe result" if "fixed" not in p
               else f"probe net {net} < 1" if net < 1 else None)  # fmt: skip
        if why:
            ledger(dict(row, skipped=why))
            continue
        best_set = dict(p.get("best_set") or {})
        num = number0 + len(out)
        cand = it_dir / f"tuned_{hid}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(src / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": {**overrides(src), **best_set}}, indent=1))
        h = candidate_hash(cand)
        if h == inc["hash"] or h in taken:
            ledger(dict(row, skipped="the incumbent" if h == inc["hash"] else f"same as draft {taken[h]['number']}",
                        best_set=best_set, hash=h))  # fmt: skip
            continue
        declared = list(probe_labels(known[hid]) or ["all"])
        labels, ran = draft_labels(pool, declared, ok_inc, int(CAPS.get("draft_sentinels", 3)))
        diag = LOOP / "diag" / f"candidate_{h}"
        if bentry is None and p.get("best_dir") and Path(p["best_dir"]).is_dir():
            seed_rounds(Path(p["best_dir"]), diag, labels)
        (it_dir / f"code_{num}.patch").write_text("")
        c = dict(number=num, kind="tuned", hypothesis=hid, mechanism=sweep_mechanism(best_set, list(best_set)),
                 scope=dict(when="the probe's starts", starts=declared), dir=str(cand), patch=str(it_dir / f"code_{num}.patch"),
                 hash=h, diag=str(diag), ran=ran, best_set=best_set)  # fmt: skip
        out.append(c)
        taken[h] = c
        jobs.append(B.Job(cand / "skills", diag, overrides(cand), labels, True))
        ledger(dict(row, candidate=num, best_set=best_set, hash=h, net=net))
    return out


def wait_drafts(threads: dict, stops: dict, poll: float = 2.0, sessions: dict | None = None) -> list:
    """TODO 37.3 (D, user 2026-10-23): wait for the Engineers; once ceil(K/2) Engineers' SESSIONS have ended, the rest get
    ENGINEER_LATE_MIN minutes counted from that moment, then the stop events of the sessions still running are set
    (run_role kills the session, RoleLate) and the iteration goes on with the drafts it has. `sessions` ({k: time the
    Engineer's session ended}, filled by the callers' threads) keeps the draft's own diagnostic rounds and knob sweep, which
    a thread also runs after its session, out of the deadline; a thread that ends without an entry counts from its end.
    The late hypothesis counts as not built, so rule 33 builds it first next time. A straggler is stopped rather than left
    running: the Engineers share the sandbox, and a file it changed outside its copy later could not be told from the
    Composer's (rule 28). Returns the Engineers stopped."""
    need = (len(threads) + 1) // 2
    sessions = {} if sessions is None else sessions
    late: list = []
    while True:
        for k, t in threads.items():
            if k not in sessions and not t.is_alive():
                sessions[k] = time.time()
        alive = [k for k, t in threads.items() if t.is_alive()]
        if not alive:
            break
        running = [k for k in alive if k not in sessions]
        if ENGINEER_LATE_MIN > 0 and need <= len(sessions) < len(threads) and running and not late:
            if time.time() - sorted(sessions.values())[need - 1] > ENGINEER_LATE_MIN * 60:
                for k in running:
                    stops[k].set()
                late = running
        time.sleep(poll)
    for t in threads.values():
        t.join()
    return late


def multi_engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub) -> str | None:
    """TODO 32.9 (Harness2's contrastive compose step): CANDIDATES Engineers build the Manager's hypotheses in parallel,
    each in its own copy of the incumbent's skills package (`cand_k/skills`); every valid candidate runs on the
    diagnostic pool; the Composer merges what those rounds support into one candidate, adopts one, or keeps the
    incumbent; only that one candidate reaches the gate, so the gate's test stays one test per call."""
    known = {h["id"]: h for h in hyps["hypotheses"]}
    tune = [t for t in dict.fromkeys(dec.get("tune") or []) if t in known]  # change 1: tuned drafts, no Engineer
    ids, _ = assign_ids(dec, hyps["hypotheses"], CANDIDATES, tune)  # change 3: no duplicate assignments
    bentry = base_of(n, dec)  # TODO 33.8: every working copy starts from that pool candidate instead of the incumbent
    src = Path(bentry["dir"]) if bentry else Path(inc["dir"])
    parallel, probe_box = pr is None, {}
    if parallel:  # TODO 48.17 probe_parallel: the Prober beside the Engineers; its rounds share the queue with theirs

        def run_probe():
            try:
                probe_box["pr"] = probe_now(n, hyps, inc, LOOP / "diag" / f"incumbent_{inc['hash']}", it_dir, True)
            except Exception as e:  # noqa: BLE001 -- the drafts go on without the probe; recorded below
                probe_box["error"] = f"{type(e).__name__}: {e}"[:400]

        probe_thread = threading.Thread(target=run_probe, daemon=True)
        probe_thread.start()
    calls = []
    for k, hid in enumerate(ids, 1):
        w = it_dir / f"cand_{k}"
        if w.exists():
            shutil.rmtree(w)
        shutil.copytree(src / "skills", w / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        ctx = w / "engineer_context.json"
        jdump(ctx, context(inc, dict(hypothesis=known[hid], manager=dec,
                                     probe=PROBE_PENDING if parallel else [p for p in pr if p.get("id") == hid],
                                     park=None, draft=dict(number=k, of=len(ids)), base=base_brief(bentry))))  # fmt: skip
        prompt = fill(
            (PROMPTS / "engineer.md").read_text(),
            CONTEXT=ctx,
            ITER=w,
            **dict(common, SKILLS=w / "skills", TMP=role_dir(f"engineer_{k}")),
        )
        stub_e = dict(mechanism=f"stub {k}", files_changed=[], set={**overrides(src)}, tests="none",
                      scope=dict(when="stub", starts=["all"]))  # fmt: skip
        if stub:  # a stub draft must differ from the incumbent, or the plumbing after it never runs
            with (w / "skills" / STUB_FILE).open("a") as fh:
                fh.write(f"\n# stub draft {k}\n")
        calls.append((k, hid, w, prompt, stub_e))
    out: dict = {}

    stops = {k: threading.Event() for k, *_ in calls}
    sess_end: dict = {}  # k -> when Engineer k's session ended (wait_drafts' deadline counts from there)
    inc_rounds = LOOP / "diag" / f"incumbent_{inc['hash']}"
    pooled = LOOP / "diag" / f"candidate_{bentry['hash']}" if bentry else None
    ref = pooled if pooled is not None and pooled.is_dir() else inc_rounds  # the rounds the drafts start from

    def one(k, w, prompt, stub_e):
        try:
            out[k] = claude(f"engineer_{k}", prompt, ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent",
                            stub_e if stub else None, stop=stops[k])  # fmt: skip
        except RoleFailed as e:
            out[k] = e
            return
        finally:
            sess_end[k] = time.time()
        if not stub:
            early_diag(k, w, out[k][0], inc, it_dir)
            if REPAIR:
                out[k] = repair_draft(n, k, w, out[k], inc, ref, it_dir, common)
            try:  # change 1: the draft's declared knobs, swept on its own starts
                out[k] = (tune_draft(k, w, out[k][0], inc, it_dir), out[k][1])
            except Exception as e:  # noqa: BLE001 -- the draft stands as it was
                print(f"tunables of draft {k}: {type(e).__name__}: {e}", flush=True)

    threads = {k: threading.Thread(target=one, args=(k, w, pr_, se)) for k, _, w, pr_, se in calls}
    for t in threads.values():
        t.start()
    wait_drafts(threads, stops, sessions=sess_end)
    stray = revert_strays(base_changes)
    put_skills(Path(inc["dir"]) / "skills")
    cands, jobs, first, leaked = [], [], {}, {}  # first: candidate hash -> the draft kept (identical drafts run once)
    for k, hid, w, _, _ in calls:
        res = out.get(k)
        if isinstance(res, Exception) or res is None:
            ledger(dict(step="engineer", iter=n, candidate=k, hypothesis=hid, void=True, late=isinstance(res, RoleLate),
                        error=str(res)[:300]))  # fmt: skip
            continue
        eng, acct = res
        leaks = transcript_leaks(acct["session"])
        cand = it_dir / f"candidate_{k}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(w / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
        patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                                str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
        (it_dir / f"code_{k}.patch").write_text(patch)
        h = candidate_hash(cand)
        named = named_objects(cand / "skills")  # TODO 33.13 (rules.md 30): a draft that branches on names is void
        void = bool(leaks) or bool(stray) or h == inc["hash"] or bool(named)
        dup = first.get(h) if not void else None  # TODO 36 (review 2026-10-22, L5): the first of identical drafts stays
        labels, ran = draft_run(eng, inc)  # change 4: where the draft's diagnostic rounds ran
        ledger(dict(step="engineer", iter=n, candidate=k, hypothesis=hid, changed=sorted(set(PATCHED.findall(patch))),
                    stray=stray, mechanism=eng["mechanism"], set=eng["set"], scope=eng.get("scope"), void=void, leaks=leaks,
                    ran=ran, **({"tuned": eng["tuned"], "tunable_sweep": eng["tunable_sweep"],
                                 "tunable_dropped": eng["tunable_dropped"]} if "tuned" in eng else {}),
                    **({"named_objects": named} if named else {}), **({"base": bentry["hash"]} if bentry else {}),
                    **({"duplicate_of": dup["number"]} if dup else {}), **acct))  # fmt: skip
        if leaks:
            leaked[k] = leaks
        if leaks or stray:  # voided drafts leave the visible tree
            hide_void(n, w, cand, it_dir / f"code_{k}.patch")
        if dup:
            dup.setdefault("duplicate_drafts", []).append(k)
        elif not void:
            cands.append(dict(number=k, kind="engineer", hypothesis=hid, mechanism=eng["mechanism"], scope=eng.get("scope"),
                              dir=str(cand), patch=str(it_dir / f"code_{k}.patch"), hash=h, ran=ran,
                              diag=str(LOOP / "diag" / f"candidate_{h}"), **({"tuned": eng["tuned"]} if "tuned" in eng else {})))  # fmt: skip
            first[h] = cands[-1]
            jobs.append(B.Job(cand / "skills", LOOP / "diag" / f"candidate_{candidate_hash(cand)}", overrides(cand),
                              labels, True))  # fmt: skip
    if parallel:  # the probe's results, and the tuned drafts the program picks from them (the Manager had none)
        probe_thread.join()
        pr = probe_box.get("pr") or []
        if "error" in probe_box:
            ledger(dict(step="probe_failed", iter=n, error=probe_box["error"]))
        nets = {p.get("id"): len(p.get("fixed") or []) - len(p.get("broken") or []) for p in pr if p.get("best_set")}
        tune = sorted((h for h in nets if h in known and h not in ids and nets[h] >= 1), key=lambda h: -nets[h])
        tune = tune[: int(cap("tune_max", 2))]
    cands += tuned_drafts(n, it_dir, inc, known, pr, tune, src, bentry, len(calls) + 1, first, jobs)  # change 1
    if not cands:
        ledger(dict(step="compose", iter=n, decision="none", reason="no valid candidate", stray=stray))
        if leaked and len(leaked) == len(calls):  # every draft voided by its Engineer's leak: the iteration is void
            void_iteration(n, "engineers", sorted({x for v in leaked.values() for x in v}))
        return None
    shadow_drafts(n, cands, inc)
    if not stub:
        B.run_jobs(jobs, FAMILY, LOOP / "pool_diag", PROCS, it_dir / "prober.log", GPUS)
    base = LOOP / "diag" / f"incumbent_{inc['hash']}"
    seg_table = SG.segment_table(B.outcomes(base).values())  # TODO 48.17.6 (rules 48)
    for c in cands:
        c.update(flips(base, Path(c["diag"])), footprint=footprint(ref, Path(c["diag"]), c["scope"]))
        c["segments_touched"] = segments_touched(ref, Path(c["diag"]), c["footprint"]["changed"], seg_table)
    jdump(it_dir / "candidates.json", cands)
    imagined = imagined_view(n, it_dir, cands, inc, base)
    # the Composer
    wc = it_dir / "compose" / "skills"
    if wc.parent.exists():
        shutil.rmtree(wc.parent)
    shutil.copytree(src / "skills", wc, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ctx = it_dir / "composer_context.json"
    jdump(ctx, context(inc, dict(hypotheses=[known[h] for h in dict.fromkeys([*ids, *tune])], candidates=cands,
                                 incumbent_diag=str(base), manager=dec, base=base_brief(bentry),
                                 candidate_pool=pool_view(inc, base), **({"imagined": imagined} if imagined else {}),
                                 **({"probe": pr} if parallel else {}))))  # fmt: skip
    tpl = (PROMPTS / "composer.md").read_text()
    if imagined:
        tpl += "\n" + (PROMPTS / "composer_imagined.md").read_text()
    prompt = fill(tpl, CONTEXT=ctx, ITER=it_dir, **dict(common, SKILLS=wc))
    stub_c = dict(decision="adopt", adopted=cands[0]["number"], notes="stub")
    comp, acct = claude("composer", prompt, COMPOSER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_c if stub else None)
    leaks = transcript_leaks(acct["session"])
    if leaks and cap("role_rerun") and not stub:  # TODO 48.17: the first leak discards the composition, one more call
        revert_strays(base_changes)
        put_skills(Path(inc["dir"]) / "skills")
        discard_attempt(n, "composer", acct, leaks, it_dir / "compose")
        shutil.copytree(src / "skills", wc, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        comp, acct = claude("composer", prompt + rerun_note("Composer", leaks), COMPOSER_SCHEMA, it_dir,
                            "NotebookEdit,Agent")  # fmt: skip
        leaks = transcript_leaks(acct["session"])
    stray = revert_strays(base_changes)
    put_skills(Path(inc["dir"]) / "skills")
    jdump(it_dir / "composer.json", comp)
    ledger(dict(step="compose", iter=n, decision=comp["decision"], adopted=comp.get("adopted"), notes=comp["notes"],
                mechanism=comp.get("mechanism", ""), set=comp.get("set"), stray=stray, void=bool(leaks or stray),
                leaks=leaks, **acct))  # fmt: skip
    if leaks or stray:  # its working copy leaves the visible tree as a voided draft does
        hide_void(n, it_dir / "compose")
        if leaks:
            void_iteration(n, "composer", leaks)
        return None
    if comp["decision"] == "keep":
        return None
    if comp["decision"] == "adopt" and comp.get("adopted_pool"):  # TODO 33.8: a pool candidate as it stands
        pe = pool_entry(comp["adopted_pool"])
        if pe is None:
            ledger(dict(step="gate", iter=n, accepted=False, void=True,
                        reason=f"adopted pool candidate {comp['adopted_pool']} is not in the pool"))  # fmt: skip
            return None
        cand = Path(pe["dir"])
    elif comp["decision"] == "adopt":
        pick = next((c for c in cands if c["number"] == comp.get("adopted")), None)
        if pick is None:
            ledger(
                dict(
                    step="gate",
                    iter=n,
                    accepted=False,
                    void=True,
                    reason=f"adopted candidate {comp.get('adopted')} is not a valid one",
                )
            )
            return None
        cand = Path(pick["dir"])
    else:
        cand = it_dir / "candidate"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(wc, cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": comp.get("set") or {}}, indent=1))
        patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                                str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
        (it_dir / "code.patch").write_text(patch)
        if candidate_hash(cand) == inc["hash"]:
            ledger(
                dict(step="gate", iter=n, accepted=False, void=True, reason="the composed candidate is the incumbent")
            )
            return None
    named = named_objects(cand / "skills")  # TODO 33.13 (rules.md 30)
    if named:
        named_void(n, named, "the composed candidate")
        return None
    why = gate_candidate(n, it_dir, cand, stub)
    if why is not None or stub or not SECOND_SHOT:
        return why
    return second_shot(n, it_dir, cands, candidate_hash(cand), stub)  # TODO 37.4


def engineer_again(n: int, src: Path, stub: bool) -> str | None:
    """Iteration n as a replay of another run's iteration `src` from its Manager's decision on (TODO 32.8): that
    iteration's hypotheses, probes, claims and Manager decision are taken as they were, and only the Engineer and the
    gate run again -- so a change to the Engineer's call is compared on the same decision."""
    it_dir = STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    snapshot(n)
    for f in ("hypotheses.json", "probe.json", "claims.json", "manager.json"):
        if (src / f).exists():
            shutil.copy2(src / f, it_dir / f)
    if (src / "review_notes").is_dir() and not (it_dir / "review_notes").exists():
        shutil.copytree(src / "review_notes", it_dir / "review_notes")
    hyps, pr, dec = jload(it_dir / "hypotheses.json"), jload(it_dir / "probe.json", []), jload(it_dir / "manager.json")
    if hyps is None or dec is None:
        raise SystemExit(f"{src} has no hypotheses or Manager decision")
    if dec["decision"] not in ("engineer", "reopen"):
        raise SystemExit(f"{src}: the Manager decided {dec['decision']!r}; nothing for the Engineer")
    claims = [c for c in hyps.get("infeasible") or [] if isinstance(c, dict)]
    if claims and (it_dir / "claims.json").exists():  # that iteration's parks, as its after_review recorded them
        park(jload(it_dir / "claims.json"), claims, n)
    inc = incumbent()
    put_skills(Path(inc["dir"]) / "skills")
    base_changes = changed_files()
    base = diag_incumbent(inc, it_dir)
    ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), reason=dec["reason"],
                summary_zh=dec.get("summary_zh", ""), replayed_from=str(src), cost_usd=0.0, seconds=0))  # fmt: skip
    common = dict(RULES=RULES, **prompt_vars(), LOOP=LOOP, SIM_PY=SIM_PY, DIAG_INC=base, SKILLS=SKILLS, FAMILY=FAMILY,
                  FAMILY_FILE=HERE / "families" / f"{FAMILY}.json", POOL=LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=role_tmp(), REVIEW_NOTES=review_notes_dir(it_dir))  # fmt: skip
    return engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)


def reprobe(n: int, stub: bool) -> str | None:
    """Re-run iteration n's probes on its Reviewer's unchanged hypotheses and continue from the Manager: for probes
    lost to the harness or never run because a driver bug stopped the loop (the hypotheses never saw a probe result,
    so they stand), or after a review the audit voided and a fixed audit passes (`reaudit`). The earlier Manager call
    is voided in the ledger; the human fix that made this necessary goes in interventions.jsonl."""
    it_dir = STATE / f"iter_{n:02d}"
    hyps = jload(it_dir / "hypotheses.json")
    if hyps is None:
        raise SystemExit(f"{it_dir} has no hypotheses")
    reaudit(n)
    managed = (it_dir / "manager.json").exists()
    for f in ("probe", "manager.json", "probe.json"):
        p = it_dir / f
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.rename(p.with_name(p.name + ".void"))
    if managed:  # the voided Manager's focus; before a Manager ran, focus.json is the one this Reviewer was given
        (STATE / "focus.json").unlink(missing_ok=True)
    ledger(dict(step="void", iter=n, reason="re-probing the same hypotheses after a driver fix (interventions.jsonl)"))
    inc = incumbent()
    put_skills(Path(inc["dir"]) / "skills")
    base_changes = changed_files()
    base = diag_incumbent(inc, it_dir)
    refresh_diag_pool(inc, base, n)
    unpark(base, n)
    attribution = attribute(inc, base, n, it_dir, stub)  # TODO 33.20
    common = dict(RULES=RULES, **prompt_vars(), LOOP=LOOP, SIM_PY=SIM_PY, DIAG_INC=base, SKILLS=SKILLS, FAMILY=FAMILY,
                  FAMILY_FILE=HERE / "families" / f"{FAMILY}.json", POOL=LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=role_tmp(), REVIEW_NOTES=review_notes_dir(it_dir))  # fmt: skip
    extra = dict(diag_summary=summarize(base), parameters=parameters_of(inc), focus=jload(STATE / "focus.json", {}),
                 parked=jload(STATE / "parked.json", {}),
                 segments=SG.segment_table(B.outcomes(base).values()))  # fmt: skip  # TODO 48.17.6 (rules 48)
    if attribution:
        extra["attribution"] = attribution
    return after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def wall_h(meta: dict) -> float:
    """Hours the run has been running: since its start, less the time it stood between a stop and a resume."""
    return (time.time() - meta["started_s"] - meta.get("paused_s", 0.0)) / 3600


BOOKKEEPING_STEPS = (
    "stop",
    "settings",
    "diag_pool",
    "sealed",
)  # rows a stopped run's ledger can gain without an iteration


def resume(meta: dict) -> None:
    """A run started again after its own stop: the hours it stood stopped do not count against MAX_WALL_H (32.12
    stood stopped for a day while the audit was fixed, TODO 32.13). The stop may be followed by bookkeeping rows --
    a standalone `loop.py redraw` (33.23), a `settings` row, a wall-clock `stop` the resume itself wrote -- so the
    pause runs from the first row of that trailing run (run C, 2026-10-23: a redraw row after the stop hid a 12 h
    pause and the resumed run stopped at once on the wall clock); a `resume` row ends the walk back, its gap is
    already counted."""
    es = entries()
    tail = []
    for e in reversed(es):
        if e.get("step") not in BOOKKEEPING_STEPS:
            break
        tail.append(e)
    if not any(e.get("step") == "stop" for e in tail):
        return
    t = calendar.timegm(time.strptime(tail[-1]["t"], "%Y-%m-%dT%H:%M:%SZ"))
    if t < int(meta["started_s"]):  # a stop from before this run started (a replay's copied ledger); t is whole seconds
        return
    gap = time.time() - t
    meta["paused_s"] = meta.get("paused_s", 0.0) + gap
    ledger(dict(step="resume", stopped_h=round(gap / 3600, 2), wall_h=round(wall_h(meta), 2)))


EFFORT_ROLE = re.compile(r"^(reviewer(_[PO])?|manager|engineer(_\d+)?|composer)$")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def parse_effort(items: list[str]) -> dict:
    """TODO 36 (review 2026-10-22, L7): `--effort ROLE=LEVEL`; an unknown role (`enginer=medium` used to add an unused
    key silently) or level is refused with the valid ones listed."""
    out = {}
    for x in items:
        role, _, level = x.partition("=")
        if not EFFORT_ROLE.match(role):
            raise SystemExit(f"--effort: unknown role {role!r}; valid: reviewer (reviewer_P, reviewer_O), manager, "
                             f"engineer (engineer_1, engineer_2, ...), composer")  # fmt: skip
        if level not in EFFORT_LEVELS:
            raise SystemExit(f"--effort {x}: level must be one of {', '.join(EFFORT_LEVELS)}")
        out[role] = level
    return out


def resolve_settings(a, meta: dict, resuming: bool) -> dict:
    """TODO 36 (review 2026-10-22, L3): a run's candidates, scoped_review, max_wall_h and stop_rate are what the command
    line says, else -- when the run is resumed (meta.json has `started`) -- what meta.json recorded, else the module
    default. A resume used to rewrite them (32.12 needed RSI_MAX_WALL_H=5 again; an older K = 1 run resumed as K = 3).
    `stop_rate` (`--stop-rate`, user 2026-10-22): grasp_atomic continued from an incumbent at 29/40 on the acceptance
    pool, which the default 70 % stop would end at once."""

    def pick(cli, key, default):
        return cli if cli is not None else (meta[key] if resuming and key in meta else default)

    env_wall = os.environ.get("RSI_MAX_WALL_H")
    return dict(
        candidates=max(1, int(pick(getattr(a, "candidates", None), "candidates", CANDIDATES))),
        scoped_review=bool(pick(True if getattr(a, "scoped_review", None) else None, "scoped_review", SCOPED)),
        reviewers=max(1, int(pick(getattr(a, "reviewers", None), "reviewers", REVIEWERS))),
        repair=bool(pick(True if getattr(a, "repair", None) else None, "repair", REPAIR)),
        max_wall_h=float(env_wall) if env_wall else float(pick(None, "max_wall_h", WALL_DEFAULT_H)),
        stop_rate=float(pick(getattr(a, "stop_rate", None), "stop_rate", STOP_RATE_DEFAULT)),
        engineer_late_min=float(
            pick(getattr(a, "engineer_late_min", None), "engineer_late_min", ENGINEER_LATE_MIN_DEFAULT)
        ),  # fmt: skip
        second_shot=bool(pick(getattr(a, "second_shot", None), "second_shot", SECOND_SHOT_DEFAULT)),
        diag_min_fails=int(pick(getattr(a, "diag_min_fails", None), "diag_min_fails", DIAG_MIN_FAILS_DEFAULT)),
        diag_target_fails=int(
            pick(getattr(a, "diag_target_fails", None), "diag_target_fails", DIAG_TARGET_FAILS_DEFAULT)
        ),  # fmt: skip
        diag_draw_max=int(pick(getattr(a, "diag_draw_max", None), "diag_draw_max", DIAG_DRAW_MAX_DEFAULT)),
    )


def settings_changes(cfg: dict, meta: dict, resuming: bool) -> dict:
    """What a resume changes in the run's settings (a `settings` ledger row). A key an older run did not record compares
    with the default it ran under (stop_rate: 0.70 before 2026-10-22)."""
    if not resuming:
        return {}
    was = {k: meta.get(k, STOP_RATE_DEFAULT if k == "stop_rate" else None) for k in cfg}
    return {k: dict(old=was[k], new=v) for k, v in cfg.items() if was[k] is not None and was[k] != v}


def is_abnormal(why: str) -> bool:
    """A stop that is a failure, not a finish: a role call that failed, a gate error, the audit voiding
    MAX_VOID_STREAK iterations in a row (TODO 36.7; "named off-limits" is the reason older runs stopped with)."""
    return why.startswith(("role call failed", "gate error", "audit:")) or "named off-limits" in why


class Terminated(BaseException):
    pass


def cmd_run(a) -> int:
    """TODO 36 (review 2026-10-22, L4): exit 0 for a normal stop, 3 for an abnormal one (see is_abnormal) or a crash.
    An uncaught exception or SIGTERM writes a `stop` row first, so resume() counts the downtime as a pause."""
    STATE.mkdir(parents=True, exist_ok=True)
    efforts = parse_effort(getattr(a, "effort", None) or [])
    unlock_hidden()
    unlock_rounds()

    def on_term(*_):
        raise Terminated()

    old = None
    try:
        old = signal.signal(signal.SIGTERM, on_term)
    except ValueError:  # not the main thread
        pass
    try:
        return _run(a, efforts)
    except Terminated:
        ledger(dict(step="stop", reason="terminated"))
        print("stop: terminated")
        return 3
    except (Exception, KeyboardInterrupt) as e:
        ledger(dict(step="stop", reason=f"crashed: {e!r}"[:500]))
        traceback.print_exc()
        return 3
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        unlock_hidden()


def _run(a, efforts: dict) -> int:
    global STOP_AFTER, CANDIDATES, SCOPED, REVIEWERS, REPAIR, MAX_WALL_H, STOP_RATE, ENGINEER_LATE_MIN, SECOND_SHOT
    global DIAG_MIN_FAILS, DIAG_TARGET_FAILS, DIAG_DRAW_MAX
    prior = jload(STATE / "meta.json")
    resuming = bool(prior and "started" in prior)
    meta = prior or dict(started=now(), started_s=time.time(), cost_cap_usd=COST_CAP, model=MODEL, family=FAMILY)
    resume(meta)
    cfg = resolve_settings(a, meta, resuming)
    CANDIDATES, SCOPED, MAX_WALL_H = cfg["candidates"], cfg["scoped_review"], cfg["max_wall_h"]
    REVIEWERS, REPAIR = cfg["reviewers"], cfg["repair"]
    if SCOPED and REVIEWERS > 1:
        raise SystemExit("--scoped-review and --reviewers are two ways of running several Reviewers: choose one")
    STOP_RATE, ENGINEER_LATE_MIN, SECOND_SHOT = cfg["stop_rate"], cfg["engineer_late_min"], cfg["second_shot"]
    DIAG_MIN_FAILS, DIAG_TARGET_FAILS = cfg["diag_min_fails"], cfg["diag_target_fails"]
    DIAG_DRAW_MAX = cfg["diag_draw_max"]
    if not cfg["stop_rate"] > 0:
        raise SystemExit(f"--stop-rate must be > 0 (got {cfg['stop_rate']})")
    changed = settings_changes(cfg, meta, resuming)
    if changed:
        ledger(dict(step="settings", **changed))
    meta.update(cost_cap_usd=COST_CAP, model=MODEL, **cfg)  # this run's limits (roles read meta.json)
    jdump(STATE / "meta.json", meta)
    if not (LOOP / "pool_diag").exists() or not accept_pool().exists():
        raise SystemExit("draw the pools first (family.py draw ... diag / accept)")
    if not pool_path().exists():  # TODO 33.8: state from before the candidate pool (a replay of an older run)
        rebuild_pool()
    split_check_half()  # TODO 48.17 check_half: once per loop
    n = len(iteration_dirs())  # change 5: not a speculative review no iteration has taken up
    iters = a.max_iters
    STOP_AFTER = getattr(a, "stop_after", None)
    EFFORT.update(efforts)

    def finish(why: str) -> int:
        ledger(dict(step="stop", reason=why))
        print("stop:", why)
        write_hit_rate()
        if stop_reason(meta) is not None:  # a stop rule ended the run (not a failed role, a crash or --max-iters)
            run_sealed(why, a.stub)
        return 3 if is_abnormal(why) else 0

    if getattr(a, "engineer_from", None) is not None:  # one iteration: the Engineer and the gate on a recorded decision
        try:
            why = stop_reason(meta) or engineer_again(n, a.engineer_from.resolve(), a.stub)
        except RoleFailed as e:
            why = f"role call failed: {e}"
        return finish(why or "replay: engineer and gate done")
    if a.reprobe is not None:  # the re-probed iteration counts as one of --max-iters
        try:
            why = reprobe(a.reprobe, a.stub)
        except RoleFailed as e:
            why = f"role call failed: {e}"
        if why:
            return finish(why)
        iters -= 1
    for _ in range(iters):
        why = stop_reason(meta)
        if why is None:
            try:
                why = iterate(n, meta, a.stub)
            except RoleFailed as e:
                why = f"role call failed: {e}"
            n += 1
            write_hit_rate()
        if why:
            return finish(why)
    return 0


def write_hit_rate() -> None:
    """TODO 48.19.3: the roles' hit rates (hit_rate.py) to loop/hit_rate.json; never stops the run."""
    try:
        import hit_rate

        jdump(STATE / "hit_rate.json", hit_rate.compute(STATE))
    except Exception as e:  # bookkeeping only
        print(f"hit rate not written: {e!r}"[:300])


def run_sealed(why: str, stub: bool) -> None:
    """TODO 48.19.2 (user 2026-11-23, "三条都加上"; rules 49): a run a stop rule ends runs the sealed evaluation once for
    its final incumbent (sealed.py: the final incumbent and the seed on the sealed pool, in the gate's environment, and
    its confirmation against the seed). Its numbers stay in the hidden tree (<hidden>/sealed/summary.json, history.jsonl);
    the ledger records only that it ran. Off for a stub run, an abnormal stop, a family without caps sealed_at_stop,
    a run with nothing accepted, and a final incumbent the sealed history already holds."""
    if stub or is_abnormal(why) or not cap("sealed_at_stop") or os.environ.get("RSI_NO_SEALED") == "1":
        return
    inc = jload(GATE_DIR / "incumbent.json")
    if not inc or Path(inc["dir"]).name == "incumbent_seed":
        return
    final = Path(inc["dir"]).parent.name  # <call>_<hash>
    hist = HIDDEN / "sealed" / "history.jsonl"
    if hist.exists() and any(json.loads(x).get("final") == final for x in hist.read_text().splitlines() if x.strip()):
        return
    (HIDDEN / "sealed").mkdir(parents=True, exist_ok=True)
    log = HIDDEN / "sealed" / f"run_{final}.log"
    print(f"sealed evaluation of {final} (log in the hidden tree)", flush=True)
    t0 = now()
    with log.open("a") as fh:
        rc = subprocess.run([str(SIM_PY), str(HERE / "sealed.py"), FAMILY, "--procs", str(PROCS), "--gpus", GPUS],
                            cwd=REPO, env=dict(os.environ, PYTHONPATH="src"), stdout=fh, stderr=subprocess.STDOUT,
                            check=False).returncode  # fmt: skip
    ledger(dict(step="sealed", final=final, started=t0, rc=rc))


def cmd_redraw(a) -> int:
    """TODO 33.23: re-draw the diagnostic pool for the present incumbent outside an iteration (first use: run C's
    incumbent, 30/30 on its pool, before that run continues). The incumbent's rounds and the redraw's log go with the
    last iteration's directory; the ledger row carries that iteration's number. Same --diag-* settings as `run`."""
    global DIAG_MIN_FAILS, DIAG_TARGET_FAILS, DIAG_DRAW_MAX
    meta = jload(STATE / "meta.json", {})
    cfg = resolve_settings(a, meta, bool(meta and "started" in meta))
    DIAG_MIN_FAILS, DIAG_TARGET_FAILS, DIAG_DRAW_MAX = cfg["diag_min_fails"], cfg["diag_target_fails"], cfg["diag_draw_max"]  # fmt: skip
    unlock_hidden()
    unlock_rounds()
    inc = incumbent()
    n = max([e["iter"] for e in entries() if isinstance(e.get("iter"), int)] or [0])
    it_dir = STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    base = diag_incumbent(inc, it_dir)
    row = refresh_diag_pool(inc, base, n)
    print(json.dumps(row or dict(step="diag_pool", incumbent=inc["hash"], skipped="the pool keeps enough failing starts"),
                     ensure_ascii=False)[:1200])  # fmt: skip
    return 0


def cmd_status(_a) -> int:
    es = entries()
    meta = jload(STATE / "meta.json", {})
    print(f"family {FAMILY}; started {meta.get('started')}; entries {len(es)}; "
          f"cost {sum(e.get('cost_usd', 0) for e in es):.2f} USD")  # fmt: skip
    for e in es[-12:]:
        print(json.dumps({k: v for k, v in e.items() if k not in ("usage", "results")}, ensure_ascii=False)[:300])
    parked = jload(STATE / "parked.json", {})
    for cid, p in parked.items():
        print(f"parked {cid}: {p['task']} ({p['scope']}, {len(p['starts'])} diag starts, {p['kind']})")
    inc = jload(GATE_DIR / "incumbent.json")
    v = unparked_view(Path(inc["rounds"])) if inc else None
    if v:
        print(f"acceptance: {v['all_ok']}/{v['all_n']} of all starts, {v['ok']}/{v['n']} outside {v['parked_tasks']}")
    try:  # TODO 48.19.3
        import hit_rate

        print(hit_rate.line(hit_rate.compute(STATE)))
    except Exception as e:
        print(f"hit rate: {e!r}"[:200])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("family")
    r.add_argument("--max-iters", type=int, default=100)
    r.add_argument("--stub", action="store_true", help="canned role outputs, gate --dry-run: plumbing only")
    r.add_argument(
        "--reprobe", type=int, default=None, help="re-probe iteration N's hypotheses first (probes lost or not run)"
    )
    r.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Engineers per iteration, merged by a Composer (TODO 32.9; default 3 since TODO 32.15; 1: no Composer; "
        "a resumed run keeps its recorded value unless given)",
    )
    r.add_argument("--scoped-review", action="store_true", default=None, help="two scoped Reviewers (TODO 32.11)")
    r.add_argument("--reviewers", type=int, help="N independent Reviewers on the same inputs (TODO 43.2; default 1)")
    r.add_argument("--repair", action="store_true", default=None, help="a draft that broke a start the incumbent "
                   "finishes gets one more turn (TODO 44.2, rule 38; off by default)")  # fmt: skip
    r.add_argument("--stop-rate", type=float, default=None, help="stop when the incumbent wins this share of the "
                   "acceptance pool (default 0.70; a resumed run keeps its recorded value unless given)")  # fmt: skip
    r.add_argument("--engineer-late-min", type=float, default=None, help="minutes a straggler Engineer gets once half "
                   "the drafts are in, then it is stopped (default 10; 0 waits; a resumed run keeps its value; TODO 37.3)")  # fmt: skip
    r.add_argument("--second-shot", action=argparse.BooleanOptionalAction, default=None, help="after the gate rejects "
                   "the composed candidate, gate the best unsent draft in the same iteration (default on; TODO 37.4)")  # fmt: skip
    for flag, key, text in (("--diag-min-fails", "diag_min_fails", "re-draw the diagnostic pool when the incumbent fails "
                             "fewer than this many of its starts (default 10; 0 never; TODO 33.23)"),
                            ("--diag-target-fails", "diag_target_fails", "... until this many fail (default 15)"),
                            ("--diag-draw-max", "diag_draw_max", "fresh starts a redraw may draw and run (default 90)")):  # fmt: skip
        r.add_argument(flag, dest=key, type=int, default=None, help=text)
    r.add_argument("--stop-after", choices=("probe",), help="replays: end the iteration after the Prober (TODO 32.6)")
    r.add_argument("--effort", action="append", default=[], metavar="ROLE=LEVEL",
                   help="claude --effort for a role (reviewer, manager, engineer, composer), e.g. engineer=medium (TODO 32.8)")  # fmt: skip
    r.add_argument("--engineer-from", type=Path, help="replays: the next iteration takes this recorded iteration's "
                   "hypotheses, probes and Manager decision and runs only the Engineer and the gate (TODO 32.8)")  # fmt: skip
    s = sub.add_parser("status")
    s.add_argument("family")
    d = sub.add_parser("redraw", help="re-draw the diagnostic pool for the present incumbent now (TODO 33.23)")
    d.add_argument("family")
    for flag, key in (("--diag-min-fails", "diag_min_fails"), ("--diag-target-fails", "diag_target_fails"),
                      ("--diag-draw-max", "diag_draw_max")):  # fmt: skip
        d.add_argument(flag, dest=key, type=int, default=None)
    a = ap.parse_args(argv)
    configure(a.family)
    return cmd_run(a) if a.cmd == "run" else cmd_redraw(a) if a.cmd == "redraw" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
