"""The seam between the loop's core and the scene line (TODO 54 phase 3).

The core (loop, cascade, drafts, roles, ledger, audit, stop, state) knows one hook object, `state.H`. A benchmark family
(a family file with a `backend` key) gets `NULL`: every hook is a no-op or the empty answer, and nothing under
scripts/rsi/scene/ is imported. A family without a backend is a scene family: `load(scene=True)` puts scripts/rsi/scene
on the path and imports scene/loop_scene.py, which implements the same names (the Manager flow, infeasibility claims and
parks, reopens, design probes, the diagnostic redraw, attribution, the replays).

A hook added here must be added to scene/loop_scene.py too; tests/test_rsi_loop_e2e.py checks that both have the same
names."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE_DIR = HERE / "scene"


class Null:
    """What a benchmark family's loop calls where the scene line does something."""

    scene = False
    CLASS_SUGGEST: dict = {}
    UNPARKED_MIN = 0.0

    # --- before the Reviewer (iterate): the diagnostic redraw, the lifted parks, the attribution of failures to a layer
    def before_review(self, inc, base, n, it_dir, stub) -> dict | None:
        return None

    # --- after the Reviewer: the scene line's Prober -> Manager -> Engineer flow (a benchmark family's program_step runs)
    def after_review(self, n, it_dir, inc, base, base_changes, hyps, common, extra, stub) -> str | None:
        raise SystemExit(
            "the Manager flow belongs to the scene line (caps manager 'llm' on a family without a backend)"
        )

    # --- the Reviewer's schema fields that only a scene family has: {"top": {...}, "hypothesis": {...}}
    def schema_fields(self, cap) -> dict:
        return {}

    # --- the gate and the stop rules
    def accept_pool(self) -> Path | None:
        return None

    def stop_reason(self, es: list[dict]) -> str | None:
        return None

    def redraw_accept(self) -> None:
        return None

    def unparked_view(self, rounds: Path) -> dict | None:
        return None

    def settle_reopen(self, n, park_id, won, accepted) -> None:
        return None

    def reopen_test(self, n, it_dir, cand, reopen) -> bool:
        return True

    def reopen_won(self, it_dir, reopen) -> list:
        return []

    # --- the Engineers' drafts (a scene family also supplies parameters_of, reviewer_cards and unknown_keys: its
    # skills are loaded in a process of their own, a backend's are read by the backend)
    def named_objects(self, skills: Path) -> list[str]:
        return []

    def tuned_drafts(self, n, it_dir, inc, known, pr, tune, src, bentry, number0, taken, jobs) -> list[dict]:
        return []

    # --- commands that only exist for the scene line
    def engineer_again(self, n, src, stub) -> str | None:
        raise SystemExit("--engineer-from replays a Manager decision: a scene family's command")

    def reprobe(self, n, stub) -> str | None:
        raise SystemExit("--reprobe replays the Prober: a scene family's command")

    def cmd_redraw(self, a) -> int:
        raise SystemExit("redraw is the scene line's diagnostic-pool redraw (caps diag_redraw)")


NULL = Null()


def load(scene: bool):
    """The hooks for a family: NULL, or the scene plug-in's module (imported now, not before)."""
    if not scene:
        return NULL
    for p in (str(HERE), str(SCENE_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("loop_scene")
