"""A config knob the eval path declares must actually reach the eval client.

The sonic clients read their tunables as LIVE module globals on `vla_round`
(`vr.OBS_DUMP`, `vr.SNAP`, ...) -- deliberately, so a caller can override one by
assignment; see vla_obs.py's docstring. The consequence is that a knob is only
live if `eval_closed_loop.py` assigns it. `backend.obs_dump` never was: its only
other consumer is `stack.py`, which exports it as `TF_OBS_DUMP` into the STACK's
environment, a different process from the eval client. So `--set
backend.obs_dump=DIR` was accepted, echoed into the config snapshot, and wrote
nothing -- found on 2026-09-11 when a diagnosis run produced zero dumps.

`grip_delay`, `replace` and `lockstep.no_seed` were inert the same way. Their
config defaults equal the module defaults, so no recorded result is affected.

A "is the field name mentioned anywhere" check does NOT catch this -- obs_dump
was mentioned, in the wrong process. The invariant that does: every global the
backend reads live AND that a config field exists for must be assigned by the
eval entry point.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SRC = Path(__file__).resolve().parents[1] / "src/locomani"
SONIC = SRC / "backends/sonic"

# Globals the backend reads live but that are NOT config-backed: physical
# thresholds and gym constants, owned by the module and not meant to be set per
# eval. Anything not listed here needs a config field and an assignment.
NOT_CONFIG_BACKED = {"CONST", "LIFT_M", "NEAR_M", "STALL_ABORT_M"}


def live_globals() -> set[str]:
    """Every `vr.<GLOBAL>` the sonic backend resolves at call time."""
    found: set[str] = set()
    for path in SONIC.glob("*.py"):
        found |= set(re.findall(r"\bvr\.([A-Z_][A-Z0-9_]*)\b", path.read_text()))
    return found


def assigned_by_eval() -> set[str]:
    text = (SRC / "pipeline/eval_closed_loop.py").read_text()
    return set(re.findall(r"\bvla_round\.([A-Z_][A-Z0-9_]*)\s*=", text))


def test_every_config_backed_live_global_is_assigned_by_the_eval_entry_point():
    unassigned = live_globals() - assigned_by_eval() - NOT_CONFIG_BACKED
    assert not unassigned, (
        f"{sorted(unassigned)} are read live by the sonic backend but never assigned in "
        "eval_closed_loop.py -- a --set for them is silently inert. Either assign them from "
        "the config, or add them to NOT_CONFIG_BACKED with a reason if they are module constants."
    )


def test_obs_dump_specifically_reaches_the_client():
    """The one that was actually broken, named so a revert is unambiguous."""
    text = (SRC / "pipeline/eval_closed_loop.py").read_text()
    assert "vla_round.OBS_DUMP = cfg.backend.obs_dump" in text, (
        "without this line --set backend.obs_dump=DIR writes nothing and reports no error"
    )


def test_stack_env_alone_does_not_count_as_wiring():
    """stack.py's TF_OBS_DUMP goes to the stack, not to this process.

    Pinned so nobody 'fixes' a future occurrence by adding an env export and
    believing the knob is live.
    """
    stack = (SRC / "pipeline/stack.py").read_text()
    assert "TF_OBS_DUMP" in stack, "stack.py still exports it for the stack's own processes"
    obs = (SONIC / "vla_obs.py").read_text()
    assert "vr.OBS_DUMP" in obs and "os.environ" not in obs.split("def get_chunk")[1], (
        "the dump is gated by the module global, not by the environment; if that ever changes, "
        "this test and eval_closed_loop's assignment both need revisiting"
    )
