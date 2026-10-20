#!/usr/bin/env python3
"""Check the small navigation surface and prevent new forbidden import edges.

Existing dependency debt is explicit; stale exceptions fail so the list shrinks.
This is a static import check, not proof about arbitrary runtime/dynamic imports.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DEBT = "tests/fixtures/architecture_debt.json"
SPAWNS = "tests/fixtures/interpreter_spawns.json"
GUIDES = {
    "AGENTS.md": 80,
    "CLAUDE.md": 12,
    # 2026-09-28 reader-facing rewrite: README is the document list, each runbook only the commands, what they do
    # and what they produce; shared mechanics live in running.md, everything else in the line's design page.
    "README.md": 80,
    "STATE.md": 60,
    "docs/README.md": 60,
    # 2026-09-30 (user): the three runbooks cut back to commands, one-sentence explanations and step totals; each cap
    # is its page's length then plus about 10%. Details move to the design page, not into a raised cap.
    "docs/single_arm.md": 115,
    "docs/bimanual.md": 125,
    "docs/warehouse.md": 300,  # 2026-10-04 (user): raised from 220 (2026-09-30: + step 8 wrap-up, stop_unreal.sh)
    "docs/running.md": 180,
    "docs/setup.md": 490,  # 2026-09-29: + step 8 (Unreal); 2026-10-20: + steps 9-12 (scene RSI: /venv/main, ADT, ScaleBFM, Claude CLI)
    "docs/warehouse_design.md": 122,  # 2026-10-20: + the archive note
    # 2026-10-12: the GraspNet-object line (grasp cache or graspnet), a side line with its own runbook and design page
    "docs/object_grasp.md": 130,
    "docs/object_grasp_design.md": 95,
    # 2026-10-20 (user): the scene RSI line is the main line; its runbook (four levels, the check of recorded results)
    # and its design page. The VLA pages above are archived in place.
    "docs/rsi.md": 215,
    "docs/rsi_design.md": 150,
    "docs/agents/doc-ownership.md": 50,
    "docs/reference/change-map.md": 40,
    "docs/reference/change-map-bottle.md": 45,
    "docs/reference/change-map-crate.md": 35,
    "docs/reference/change-map-runtime.md": 45,
    "docs/reference/change-map-rendering.md": 35,
    "docs/reference/standards.md": 100,
    "docs/agents/session-workflow.md": 50,
    "docs/handoff/agent-maintenance-plan.md": 120,
}
BYTE_BUDGETS = {"docs/reference/change-map.md": 3000}

# Runbooks: documents a reader executes line by line. Every repository path they
# name must exist, because the 2026-09-17 cold-start walkthrough found five of its
# eleven findings here -- guides still naming deleted scripts (run_wall_pilot.sh)
# and paths missing a directory level. A doc that is not present is skipped: GUIDES
# owns "does this document exist", this owns "is what it tells you to run there".
# README prints the quickstart, so it has to name commands that exist. It was NOT checked
# until 2026-09-22, which is how it kept printing `scripts/collect_data.sh` for a whole
# refactor after that script was deleted.
COMMAND_DOCUMENTS = (
    "README.md",
    "docs/single_arm.md",
    "docs/bimanual.md",
    "docs/warehouse.md",
    "docs/running.md",
    "docs/object_grasp.md",
    "docs/rsi.md",  # 2026-10-20: the main line's runbook
)
# Only the source tree. `outputs/` is runtime and `<name>`/`*`/`$VAR` are placeholders.
COMMAND_PATH = re.compile(r"(?<![\w/-])((?:scripts|configs|src|tests)/[\w./-]*[\w-])")

# A runbook that says "look in the log for X" has to name an X something prints.
# 2026-09-18: both runbooks told the reader to grep train.log for `Saving checkpoint`.
# Nothing in the repository has ever printed the word "Saving"; the recipe writes
# `Copying ... to .../checkpoint-100`. A reader following the doc concludes the
# training failed. Checked per word, not per string, because the real messages are
# assembled at runtime -- `f"  finger pads {'HARDENED ...' if hard else ...}"` and
# `f"{path} exists; ..."` never contain the doc's phrase contiguously.
QUOTED = re.compile(r"`([^`\n]+)`")
NOT_A_MESSAGE = re.compile(r"^(python|bash|export|cp|sed|make|git)\s|[×]|^[\w.]+:\s")
MESSAGE_WORD = re.compile(r"[A-Za-z][\w.]{3,}")


def imports(source: str, filename: str) -> set[str]:
    """Include function-local imports and resolve package-relative imports."""
    package = Path(filename).with_suffix("").parts[1:-1]
    targets = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join((*package[: len(package) - node.level + 1], *base.split("."))).rstrip(".")
            targets.update(f"{base}.{alias.name}" for alias in node.names)
    return targets


def forbidden(source: str, target: str) -> bool:
    if source.startswith("src/locomani/backends/") and target.startswith("locomani.tasks"):
        return True
    if source.startswith("src/locomani/primitives/") and target.startswith(("locomani.tasks", "locomani.pipeline")):
        return True
    if target.startswith("scripts.experiments"):
        return True
    if source.endswith("/cli_args.py") and target.startswith("locomani.tasks.bottle.grasp_gym"):
        return True
    if source.endswith(("config.py", "config_base.py", "/cli_args.py")) and target.split(".")[0] in {
        "mujoco",
        "torch",
        "isaacsim",
    }:
        return True
    return False


def dependency_edges(root: Path) -> set[tuple[str, str]]:
    edges = set()
    for path in sorted((root / "src/locomani").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        targets = imports(path.read_text(), name)
        edges.update((name, target) for target in targets if forbidden(name, target))
    return edges


def dependency_errors(root: Path) -> list[str]:
    rows = json.loads((root / DEBT).read_text())
    allowed = {(row["source"], row["target"]) for row in rows}
    errors = []
    if len(allowed) != len(rows) or any(not row.get("reason", "").strip() for row in rows):
        errors.append(f"{DEBT}: exceptions must be unique and have a reason")
    actual = dependency_edges(root)
    for source, target in sorted(actual - allowed):
        errors.append(f"{source}: forbidden import {target}; pass data through the owning interface (see standards)")
    for source, target in sorted(allowed - actual):
        errors.append(f"{DEBT}: remove resolved exception {source} -> {target}")
    return errors


# Which interpreter a spawn starts. A spawn of an EXTERNAL tool (pgrep, ffmpeg, nvidia-smi) is not
# listed and is never checked -- the rule is about starting our own code in another python.
INTERPRETERS = {
    "sys.executable": "same-interpreter",
    "VENV_SIM_PY": "venv_sim",
    ".venv_sim/bin/python": "venv_sim",
    "VENV_DATA_PY": "venv_data_collection",
    ".venv_data_collection/bin/python": "venv_data_collection",
    "TRAIN_PY": "isaac_gr00t",
    "TORCHRUN": "isaac_gr00t",
    "Isaac-GR00T/.venv": "isaac_gr00t",
    "RENDER_PY": "render",
}
SPAWNING_CALLS = {"run", "call", "Popen", "check_call", "check_output"}


def interpreter_spawns(root: Path) -> set[tuple[str, str]]:
    """Every place src/locomani builds a command line for a python interpreter, and which one.

    A subprocess is the seam this repository keeps losing configuration across, so starting one is
    allowed only where a venv boundary makes it unavoidable, or for a reason written down in
    `SPAWNS`. Four drivers assembled a command line for a second interpreter of the SAME venv,
    which then parsed it straight back; they were deleted on 2026-09-22
    (docs/handoff/flat-functional-scripts-20260922.md, decision 5).

    The signal is a list whose FIRST element names an interpreter -- which is how every one of
    these is written, and it catches the spawn wherever it is finally handed to (`subprocess`
    directly, or `Stage.run`, which is how the pipeline stages do it). Static and literal: a
    command assembled from parts at runtime is not caught, and an external tool (pgrep, ffmpeg,
    nvidia-smi) never is, because the rule is about starting OUR code in another python.
    """
    found = set()
    for path in sorted((root / "src/locomani").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        source = path.read_text(errors="ignore")
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError:
            continue
        # Module-level constants, so a spawn cannot hide behind a local name for the interpreter
        # (`SIM = ROOT / ".venv_sim/bin/python"` and then `[SIM, ...]`).
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                constants[node.targets[0].id] = ast.get_source_segment(source, node.value) or ""
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
                continue
            head = ast.get_source_segment(source, node.elts[0]) or ""
            for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", head):
                head += " " + constants.get(word, "")
            for token, interpreter in INTERPRETERS.items():
                if token in head:
                    found.add((name, interpreter))
    return found


def spawn_errors(root: Path) -> list[str]:
    rows = json.loads((root / SPAWNS).read_text())
    allowed = {(row["source"], row["interpreter"]) for row in rows}
    errors = []
    if len(allowed) != len(rows) or any(not row.get("reason", "").strip() for row in rows):
        errors.append(f"{SPAWNS}: entries must be unique and have a reason")
    actual = interpreter_spawns(root)
    for source, interpreter in sorted(actual - allowed):
        errors.append(
            f"{source}: starts a {interpreter} python. A subprocess is only for a venv boundary; "
            f"call the function in this process, or record the reason in {SPAWNS}"
        )
    for source, interpreter in sorted(allowed - actual):
        errors.append(f"{SPAWNS}: remove the resolved entry {source} -> {interpreter}")
    return errors


def command_path_errors(root: Path, name: str, body: str) -> list[str]:
    """Every scripts/configs/src/tests path a runbook prints has to be on disk.

    Except one a reader creates by following the page: a runbook that opens with
    `cp <a template> configs/experiments/my_line.yaml` is not assuming that file,
    it is making it. Exempt each `cp` destination, and only that.
    """
    created = set(re.findall(r"^\s*cp\s+\S+\s+(\S+)", body, re.MULTILINE))
    return [
        f"{name}: names {path}, which does not exist; a runbook may only print commands that run"
        for path in sorted(set(COMMAND_PATH.findall(body)) - created)
        if not (root / path).exists()
    ]


def source_text(root: Path) -> str:
    """Everything the repository could print: string literals only, never comments.

    The first draft of this read whole files and passed, because the only place the
    word "Saving" appears in the repository is the comment above explaining that
    nothing prints it. Literals are also the honest reading of "something prints
    this" -- and an f-string's fixed parts are Constants, so runtime-assembled
    messages still resolve word by word.
    """
    parts = []
    for base in ("src", "scripts"):
        for path in sorted((root / base).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".py":
                try:
                    tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
                except SyntaxError:
                    continue
                docstrings = {
                    id(n.value)
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                }
                parts += [
                    n.value
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
                ]
            elif path.suffix == ".sh":
                parts += [line.split("#", 1)[0] for line in path.read_text(errors="ignore").splitlines()]
    return "\n".join(parts)


def message_errors(name: str, body: str, sources: str) -> list[str]:
    """Quoted multi-word output a runbook tells you to look for must be printable."""
    errors = []
    for quoted in sorted({q for q in QUOTED.findall(body) if " " in q and "/" not in q}):
        if NOT_A_MESSAGE.search(quoted):
            continue
        missing = [w for w in MESSAGE_WORD.findall(quoted) if w not in sources]
        if missing:
            errors.append(f"{name}: `{quoted}` names {missing}, which nothing in src/ or scripts/ prints")
    return errors


def navigation_errors(
    root: Path, guides: dict[str, int] | None = None, documents: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    errors = []
    cached: list[str] = []  # read the source tree at most once, and only if a runbook changed

    def sources() -> str:
        if not cached:
            cached.append(source_text(root))
        return cached[0]

    checked: dict[str, int | None] = dict(GUIDES if guides is None else guides)
    for name in documents:
        if name.endswith(".md") and (root / name).is_file():
            checked.setdefault(name, None)
    for name, limit in checked.items():
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: missing navigation document")
            continue
        body = path.read_text()
        if limit is not None and len(body.splitlines()) > limit:
            errors.append(f"{name}: over {limit} lines; move task-specific detail behind a relevant link")
        byte_limit = BYTE_BUDGETS.get(name)
        if byte_limit is not None and len(body.encode("utf-8")) > byte_limit:
            errors.append(f"{name}: over {byte_limit} bytes; keep the router focused on domains")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(link.split("#", 1)[0])
            if not (path.parent / target).exists():
                errors.append(f"{name}: broken link {link}")
        if name in COMMAND_DOCUMENTS:
            errors += command_path_errors(root, name, body)
            errors += message_errors(name, body, sources())
        if name == "docs/reference/change-map.md" or name.startswith("docs/reference/change-map-"):
            for test in re.findall(r"`(test_[\w]+\.py)`", body):
                if not (root / "tests" / test).is_file():
                    errors.append(f"{name}: missing tests/{test}; name an actual verification entry")
    return errors


def artifact_errors(tracked: list[str]) -> list[str]:
    return [
        f"{path}: runtime artifacts and their symlinks must remain untracked"
        for path in tracked
        if path.split("/", 1)[0] in {"outputs", "third_party", "datasets"}
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation-only", action="store_true", help="check documentation without scanning code")
    parser.add_argument("--changed-docs", action="store_true", help="also check links in modified Markdown files")
    args = parser.parse_args()
    documents = set()
    if args.changed_docs:
        from check_changes import changed_paths

        documents = changed_paths(ROOT)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    errors = navigation_errors(ROOT, documents=documents) + artifact_errors(tracked)
    if not args.navigation_only:
        errors += dependency_errors(ROOT) + spawn_errors(ROOT)
    for error in errors:
        print(f"[repository] {error}")
    print(f"[repository] {'FAIL' if errors else 'PASS'}: {len(errors)} issues")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
