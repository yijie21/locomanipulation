"""Executes an agent's plan one primitive at a time (design doc §7).

Every call goes through the same gate: the name must be in the catalogue, the args must be ones the
primitive accepts, the session's arm owner must be one the primitive starts from, and the primitive's
own `preconditions` must be empty. Only then does `run` execute. A rejected call and a completed call
are both one entry in `session.trace` and one JSON line in the trace file; the agent reads that entry.
A primitive that reports success but leaves the session inconsistent is a bug, so that raises.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path

from locomani.skills.contract import Outcome, Primitive, Violation, _plain
from locomani.skills.session import Session, invariant_violations, snapshot


class Executor:
    def __init__(self, session: Session, catalogue: Mapping[str, Primitive], trace_path: Path | None = None):
        self.session, self.catalogue = session, dict(catalogue)
        self.trace_path = Path(trace_path) if trace_path is not None else None

    def step(self, name: str, args: dict | None = None) -> dict:
        """Validate, run and trace one call. Returns the trace entry (`rejected` or `outcome`)."""
        args = dict(args or {})
        entry = {"primitive": name, "args": _plain(args)}
        rejected = self._gate(name, args)
        if rejected:
            entry["rejected"] = [v.to_dict() for v in rejected]
            return self._trace(entry)
        prim = self.catalogue[name]
        owner_before = self.session.arm_owner
        t0 = self.session.clock()
        outcome = prim.run(self.session, **args)
        entry["elapsed_s"] = float(self.session.clock() - t0)
        if outcome.ok:
            yields = prim.yields
            allowed = {owner_before} if yields is None else set(yields) if isinstance(yields, frozenset) else {yields}
            if self.session.arm_owner not in allowed:
                declared = ", ".join(sorted(o.value for o in allowed))
                raise RuntimeError(f"{name}: ok but arm_owner is {self.session.arm_owner.value}, declared {declared}")
            broken = invariant_violations(self.session)
            if broken:
                raise RuntimeError(f"{name}: ok but the session breaks invariants: {broken}")
        entry["outcome"] = outcome.to_dict()
        entry["session"] = snapshot(self.session)
        return self._trace(entry)

    def _gate(self, name: str, args: dict) -> list[Violation]:
        prim = self.catalogue.get(name)
        if prim is None:
            return [Violation("unknown_primitive", name, note=f"known: {sorted(self.catalogue)}")]
        accepted = _accepted_args(prim)
        unknown = sorted(set(args) - accepted)
        if unknown:
            return [Violation("unknown_arg", ", ".join(unknown), note=f"{name} accepts {sorted(accepted)}")]
        if self.session.arm_owner not in prim.requires:
            allowed = sorted(o.value for o in prim.requires)
            return [Violation("arm_owner", self.session.arm_owner.value, note=f"{name} starts from {allowed}")]
        return list(prim.preconditions(self.session, **args))

    def run_plan(self, steps: list[dict]) -> list[dict]:
        """`steps` = [{"primitive": name, "args": {...}}, ...]; stops at the first rejected or failed step."""
        entries = []
        for step in steps:
            entry = self.step(step["primitive"], step.get("args"))
            entries.append(entry)
            if "rejected" in entry or not entry["outcome"]["ok"]:
                break
        return entries

    def abort(self, reason: str) -> dict:
        return self._trace({"primitive": "abort", "reason": reason, "session": snapshot(self.session)})

    def _trace(self, entry: dict) -> dict:
        self.session.trace.append(entry)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry


def _accepted_args(prim: Primitive) -> set[str]:
    params = inspect.signature(prim.run).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return set()  # `**args` primitives (placeholders) accept nothing until they declare a signature
    return {n for n in params if n != "session"}


def _jsonable(value):
    """Kept for callers that want the JSON-safe copy without importing contract internals."""
    return _plain(value)
