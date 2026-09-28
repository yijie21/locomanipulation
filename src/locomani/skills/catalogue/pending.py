"""Catalogue entries whose implementation is still inside the experts or does not exist.

They keep the contract surface complete (a plan can name them, the executor rejects them with a
machine-readable `unimplemented` violation) and document what each one wraps once extracted.
See docs/handoff/primitive-contracts-design-20260928.md §5.
"""

from __future__ import annotations

from locomani.skills.contract import Outcome, Violation

PENDING: dict[str, tuple] = {
    # name: (requires, yields, status, note). Empty since 2026-09-28: put_down_bimanual got a first
    # implementation (catalogue/crate_put_down.py). The machinery stays for the next declared-but-unbuilt entry.
}


class Unimplemented:
    """A declared primitive without a runnable body; every call is rejected before `run`."""

    def __init__(self, name: str, requires, yields, status: str, note: str):
        self.name, self.requires, self.yields, self.status, self.note = name, requires, yields, status, note

    def preconditions(self, session, **args) -> list[Violation]:
        return [Violation("unimplemented", self.name, note=f"{self.status}: {self.note}")]

    def run(self, session, **args) -> Outcome:
        raise RuntimeError(f"{self.name} is {self.status}; the executor must reject it before run()")


def pending_entries() -> dict[str, Unimplemented]:
    return {name: Unimplemented(name, *spec) for name, spec in PENDING.items()}
