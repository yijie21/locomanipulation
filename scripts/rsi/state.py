"""The loop driver's shared state: the run's paths, caps, limits and the plug-in hooks. `configure(family)` sets them; every other module reads them as `S.NAME` (rebinding one is a write to S)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import backend as BK  # noqa: E402
import plugin  # noqa: E402

from locomani import paths  # noqa: E402

S = sys.modules[__name__]
HERE = Path(__file__).resolve().parent


REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]


SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"


PLAN = REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md"
RULES = HERE / "rules_scene.md"  # the scene appendix; a benchmark family's is its notes file (rules_notes/)


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
STOP_RATE_DEFAULT = 0.70  # D8: the inner loop stops when the incumbent wins this share of the acceptance pool


STOP_RATE = STOP_RATE_DEFAULT  # this run's (`run --stop-rate`, resolve_settings; TODO 33, user 2026-10-22)


# TODO 37.3 (D, user 2026-10-23): minutes a straggler Engineer gets once ceil(K/2) drafts are in; 0 waits for ever.
# Run C's slowest Engineer took 22-37 min against a median draft of about 14, waiting on its own one-process rounds.
ENGINEER_LATE_MIN_DEFAULT = 10.0
ENGINEER_LATE_MIN = ENGINEER_LATE_MIN_DEFAULT  # this run's (`run --engineer-late-min`, resolve_settings)


# TODO 33.23 (plan section 3.8, D21; user 2026-10-23): the diagnostic pool is re-drawn when the incumbent fails fewer
# than DIAG_MIN_FAILS of its starts (run C's incumbent was 30/30 on it while 19 of 90 hidden-pool starts failed, so the
# roles saw no failure at all): fresh starts from the family's diagnostic seed range, the incumbent runs on them, the
# failing ones replace passing starts until DIAG_TARGET_FAILS fail; at most DIAG_DRAW_MAX fresh starts a redraw; 0 = off
DIAG_MIN_FAILS_DEFAULT, DIAG_TARGET_FAILS_DEFAULT, DIAG_DRAW_MAX_DEFAULT = 10, 15, 90  # fmt: skip  # = scene/diag_pool.py MIN_FAILS_DEFAULT, TARGET_FAILS_DEFAULT, DRAW_MAX_DEFAULT
DIAG_MIN_FAILS, DIAG_TARGET_FAILS, DIAG_DRAW_MAX = DIAG_MIN_FAILS_DEFAULT, DIAG_TARGET_FAILS_DEFAULT, DIAG_DRAW_MAX_DEFAULT  # fmt: skip


STATUSES = ("hypothesis", "verified", "refuted", "mixed", "untested")  # TODO 33.17: a hypothesis after its probe
CLASS_SUGGEST: dict = {}  # the failure-class hints the Manager reads after a rejection; the backend's layout() or the scene plug-in sets them


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
H = plugin.NULL  # the plug-in hooks (plugin.py): the scene line's part of the loop, or no-ops for a benchmark family
FAMILY = ""
LOOP = HIDDEN = STATE = GATE_DIR = Path()


# role -> claude --effort level. Pinned: role calls used to inherit the operator's settings.json (effortLevel xhigh for
# Opus 5.5 on this machine, so every recorded run is xhigh), and another operator's settings would change the loop.
# run --effort engineer=medium overrides one role (TODO 32.8). Reviewer high since TODO 37.5 (2026-10-23: 16 replays on
# 4 states, model time 15.6 vs 24.7 min, wall 33 vs 57, the same mechanisms found, coverage and probe gains not lower;
# `--effort reviewer=xhigh` restores it); the Engineers stay xhigh (32.8).
EFFORT: dict = dict(reviewer="high", manager="xhigh", engineer="xhigh", composer="xhigh")
STOP_AFTER = None  # run --stop-after probe: a replay of the Reviewer and the Prober only (TODO 32.6)


CANDIDATES = 3  # run --candidates K: K Engineers in parallel and a Composer (TODO 32.9, Harness2's compose step); 3 by


# default since TODO 32.15 (user 2026-10-22: "设置成默认的吧"), --candidates 1 is the one-Engineer loop, no Composer
LIBRARY_NOTE = (  # TODO 32.10: shown when the sandbox holds the experience library
    "The experience library, scripts/rsi/experience/INDEX.md: what earlier families' loops found, one entry per family "
    "(per change a verdict -- helped, hurt, no observed effect, confounded --, at most one lesson with its cue, and the "
    "code that run ended with in final.patch). Read the index; open an entry when its cue matches a failure here. An "
    "entry's numbers come from other objects and surfaces: they are hypotheses for this family, to check on these "
    "rounds, and where this family's rounds disagree, trust the rounds.\n"
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


def configure(family: str) -> None:
    S.FAMILY = family
    S.LOOP = REPO / "outputs" / "scene_rsi_20261016" / family
    S.HIDDEN = REPO / "outputs" / "scene_rsi_20261016_hidden" / family  # acceptance / sealed pools and the gate
    S.STATE, S.GATE_DIR = S.LOOP / "loop", S.HIDDEN / "gate"
    spec = family_spec()
    S.PLAN = REPO / spec["plan"] if spec.get("plan") else REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md"
    S.H = plugin.load(scene=BK.of(spec) is None)  # the scene plug-in only for a family without a backend
    S.CLASS_SUGGEST = S.H.CLASS_SUGGEST
    if BK.of(spec):  # a second backend has its own package, rules and prompts; a scene family keeps the constants
        lay = BK.layout(spec)
        S.SKILLS, S.RULES, S.PROMPTS, S.CAPS = lay["skills"], lay["rules"], lay["prompts"], lay["caps"]
        S.PLAN = lay.get("plan") or S.PLAN
        S.STUB_FILE = lay.get("stub_file") or S.STUB_FILE
        S.CLASS_SUGGEST = lay.get("class_suggest") or {}


def family_spec() -> dict:
    p = HERE / "families" / f"{S.FAMILY}.json"
    return json.loads(p.read_text()) if p.exists() else {}


SCENE_NOTE = (
    "in a simulated apartment"  # the prompts' opening line; a family on another scene sets `scene_note` (TODO 38.10)
)


def cap(key: str, default=False):
    """A family cap (backend.layout()["caps"]); `default` when the family's caps do not hold it."""
    return S.CAPS.get(key, default) if isinstance(S.CAPS, dict) else default


SWEEP = None


def resolve_settings(a, meta: dict, resuming: bool) -> dict:
    """TODO 36 (review 2026-10-22, L3): a run's candidates, max_wall_h and stop_rate are what the command
    line says, else -- when the run is resumed (meta.json has `started`) -- what meta.json recorded, else the module
    default. A resume used to rewrite them (32.12 needed RSI_MAX_WALL_H=5 again; an older K = 1 run resumed as K = 3).
    `stop_rate` (`--stop-rate`, user 2026-10-22): grasp_atomic continued from an incumbent at 29/40 on the acceptance
    pool, which the default 70 % stop would end at once."""

    def pick(cli, key, default):
        return cli if cli is not None else (meta[key] if resuming and key in meta else default)

    env_wall = os.environ.get("RSI_MAX_WALL_H")
    return dict(
        candidates=max(1, int(pick(getattr(a, "candidates", None), "candidates", S.CANDIDATES))),
        max_wall_h=float(env_wall) if env_wall else float(pick(None, "max_wall_h", S.WALL_DEFAULT_H)),
        stop_rate=float(pick(getattr(a, "stop_rate", None), "stop_rate", S.STOP_RATE_DEFAULT)),
        engineer_late_min=float(
            pick(getattr(a, "engineer_late_min", None), "engineer_late_min", S.ENGINEER_LATE_MIN_DEFAULT)
        ),  # fmt: skip
        diag_min_fails=int(pick(getattr(a, "diag_min_fails", None), "diag_min_fails", S.DIAG_MIN_FAILS_DEFAULT)),
        diag_target_fails=int(
            pick(getattr(a, "diag_target_fails", None), "diag_target_fails", S.DIAG_TARGET_FAILS_DEFAULT)
        ),  # fmt: skip
        diag_draw_max=int(pick(getattr(a, "diag_draw_max", None), "diag_draw_max", S.DIAG_DRAW_MAX_DEFAULT)),
    )
