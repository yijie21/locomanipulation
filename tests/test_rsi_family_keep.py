"""A family whose sandbox keeps only listed paths (`sandbox.keep`, sandbox.py) must keep every scripts/rsi module the
loop imports and every role command the rules and prompts name, and its `closure` must reach every src module those
kept scripts import -- else a sandbox built from the family file fails at the loop's first import (2026-10-23: the
grasp_atomic keep list lacked diag_pool.py, try.py and prep.py; run C's sandbox only worked because they were copied
in by hand, and the Reviewer-effort replays built from the file died on `import diag_pool`)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RSI = ROOT / "scripts" / "rsi"
SCENE = RSI / "scene"
SRC = ROOT / "src"
ENTRIES = ("loop", "try", "sealed", "gate", "batch")  # what the driver and gate run (scene families add the plug-in)


def imported(f: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def src_file(mod: str) -> Path | None:
    p = SRC.joinpath(*mod.split("."))
    return next((q for q in (p.with_suffix(".py"), p / "__init__.py") if q.exists()), None)


def is_scene(family: str) -> bool:
    return not json.loads((RSI / "families" / f"{family}.json").read_text()).get("backend")


def where(m: str) -> Path | None:
    return next((p for p in (RSI / f"{m}.py", SCENE / f"{m}.py") if p.exists()), None)


def named_commands() -> set[str]:
    files = [
        RSI / "rules_core.md",
        RSI / "rules_scene.md",
        *sorted((RSI / "rules_notes").glob("*.md")),
        *sorted((RSI / "prompts").rglob("*.md")),
    ]
    text = "".join(f.read_text() for f in files)
    return {m for m in re.findall(r"scripts/rsi/(?:scene/)?([a-z_]+)\.py", text) if where(m)}


def scripts_needed(scene: bool) -> set[str]:
    """The scripts/rsi modules (and, for a scene family, scripts/rsi/scene ones) the loop and the roles run. A benchmark
    family's loop never imports the plug-in: its lazy imports of scene modules are not followed."""
    local = {p.stem for p in RSI.glob("*.py")} | ({p.stem for p in SCENE.glob("*.py")} if scene else set())
    seen, todo = set(), {*ENTRIES, *(["scene_round", "family"] if scene else []), *named_commands()}
    while todo:
        m = todo.pop()
        if m in seen or m not in local:
            continue
        seen.add(m)
        todo |= {x.split(".")[0] for x in imported(where(m))}
    return seen


def src_reached(entries: list[str]) -> set[str]:
    """The locomani modules an AST walk reaches from the closure entries (a stand-in for `-X importtime`)."""
    seen, todo = set(), set(entries)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        f = where(m) or src_file(m)
        if f is None:
            continue
        for x in imported(f):
            if x.startswith("locomani"):
                parts = x.split(".")
                todo |= {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
            elif where(x.split(".")[0]):
                todo.add(x.split(".")[0])
    return seen


FAMILIES = sorted(p.stem for p in (RSI / "families").glob("*.json") if "sandbox" in json.loads(p.read_text()))


@pytest.mark.parametrize("family", FAMILIES)
def test_the_keep_list_holds_what_the_loop_and_the_roles_run(family):
    spec = json.loads((RSI / "families" / f"{family}.json").read_text())["sandbox"]
    scene = is_scene(family)
    needed = scripts_needed(scene)
    missing = sorted(m for m in needed if (RSI / f"{m}.py").exists() and f"scripts/rsi/{m}.py" not in spec["keep"])
    assert not missing, f"{family}: sandbox.keep lacks {missing}"
    if scene:
        assert "scripts/rsi/scene" in spec["keep"], f"{family}: a scene family keeps the plug-in directory"
    else:
        assert not [k for k in spec["keep"] if k.startswith("scripts/rsi/scene")], (
            f"{family}: no plug-in in a benchmark sandbox"
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_the_closure_reaches_every_src_module_a_kept_script_imports(family):
    spec = json.loads((RSI / "families" / f"{family}.json").read_text())["sandbox"]
    reached = src_reached(spec["closure"])
    kept = [k for k in spec["keep"] if k.startswith("scripts/rsi/") and k.endswith(".py")]
    if is_scene(family):
        kept += [f"scripts/rsi/scene/{m}.py" for m in scripts_needed(True) if (SCENE / f"{m}.py").exists()]
    wanted = {x for k in kept for x in imported(ROOT / k) if x.startswith("locomani") and src_file(x)}
    assert not sorted(wanted - reached), f"{family}: closure does not reach {sorted(wanted - reached)}"


def test_there_is_a_family_with_a_keep_list():
    assert "grasp_atomic" in FAMILIES and "grasp_prep" in FAMILIES


def test_every_acceptance_pool_is_big_enough_for_the_gate():
    """2026-10-26: grasp_floor's acceptance pool had 24 starts, and the gate refuses any candidate when the pool has
    fewer than STAGE1 + 1 (stage 1 alone is STAGE1 pairs) -- iteration 0's two calls were refused before a round ran."""
    stage1 = int(re.search(r"^STAGE1 = (\d+)", (RSI / "gate.py").read_text(), re.M).group(1))
    small = {}
    for p in sorted((RSI / "families").glob("*.json")):
        accept = json.loads(p.read_text()).get("pools", {}).get("accept")
        if accept and accept.get("n", 0) < stage1 + 1:
            small[p.stem] = accept["n"]
    assert not small, f"acceptance pools under {stage1 + 1} starts: {small}"
