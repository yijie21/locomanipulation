"""TODO 54 phase 2 (review 2026-11-25, rules_prompts.md F1, F5, F7, F13): the rules a role reads, assembled for it.

There is one core (`rules_core.md`: stable ids, what every family shares) and one appendix per family. The appendix is
generated: the tasks (the family's tasks file), the editable files and refused modules (the adapter's guard.py), the
bed's fixed switches (the family file's `bed_fixed`), the pools and caps, the failure classes (the backend's table) --
plus a short hand-written notes file (`rules_notes/<benchmark>.md`, the family file's `rules` key) for the facts nothing
derives. The scene line's appendix is hand-written (`rules_scene.md`). A role gets the core rules it acts on, by id,
and its slice of the appendix, inline in its prompt: it never reads a rules file.

    python scripts/rsi/rules_build.py <family> [role]     # print the assembled rules (all roles' sizes without a role)
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE = HERE / "rules_core.md"
SCENE = HERE / "rules_scene.md"

ROLES = ("reviewer", "engineer", "composer", "manager", "reflector")

# The core rules each role acts on (ids of rules_core.md). A rule appears where the role works, once.
ROLE_CORE = {
    "reviewer": "frozen hidden win branch levels escalate envelope evidence noise pool regress layer try classes segments "
    "mechanisms check-half",
    "engineer": "frozen hidden candidate guard bed-fixed win branch levels escalate envelope noise pool regress try scope tunable",
    "composer": "frozen hidden win levels noise pool regress gate check-half scope segments mechanisms",
    "manager": "hidden void levels escalate noise pool gate check-half stop sealed mechanisms scope tunable library",
    "reflector": "win noise regress levels",
}
# Which parts of the generated appendix and the notes a role gets.
ROLE_PARTS = {
    "reviewer": ("tasks", "notes", "classes", "switches", "pools"),
    "engineer": ("tasks", "guard", "notes", "switches"),
    "composer": ("tasks", "guard", "notes_task"),
    "manager": ("tasks", "pools"),
    "reflector": (),
}
# The scene appendix's blocks (`### [scene-...]`) by role: a block goes to a role when its id contains one of the words
# (a block whose id matches nothing goes to every role that takes the scene appendix).
SCENE_ROLE_WORDS = {
    "reviewer": ("robot", "park", "held", "classes", "stage", "prep", "probe", "keypoints", "twohand", "strict"),
    "engineer": ("robot", "stage", "prep", "keypoints", "twohand", "strict", "held"),
    "composer": ("classes", "strict", "candidates", "status"),
    "manager": ("park", "status", "candidates", "redraw", "seed", "prep", "probe", "held", "library"),
    "reflector": (),
}

BENCH_NAME = {
    "simple_pick": "the SIMPLE benchmark (pick and handover tasks)",
    "simple_art": "the SIMPLE benchmark (articulated tasks)",
    "lml": "the LocoManiLab benchmark",
    "bigym2": "the BiGym 2.0 benchmark",
    "humanoidarena": "the HumanoidArena benchmark",
}
BLOCK = re.compile(r"^###\s+\[([a-z0-9-]+)\]", re.M)


def blocks(text: str) -> dict[str, str]:
    """{id: text} of the `### [id] title` blocks of a rules file, in file order; a block ends at the next `###`/`##`."""
    out: dict[str, str] = {}
    cur, buf = None, []
    for line in text.splitlines():
        m = BLOCK.match(line)
        if m or (cur and line.startswith("## ")):
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = (m.group(1) if m else None), ([line] if m else [])
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def core_ids() -> list[str]:
    return list(blocks(CORE.read_text()))


def core_slice(role: str, core: Path = CORE) -> str:
    """The core rules `role` acts on, in the core's order."""
    want = set(ROLE_CORE.get(role, "").split())
    return "\n\n".join(t for i, t in blocks(core.read_text()).items() if i in want)


def scene_slice(role: str, path: Path = SCENE) -> str:
    """The scene appendix's blocks for a role (see SCENE_ROLE_WORDS)."""
    if not path.exists() or role == "reflector":
        return ""
    words = SCENE_ROLE_WORDS.get(role, ())
    bl = blocks(path.read_text())
    keep = [
        t
        for i, t in bl.items()
        if any(w in i for w in words) or not any(w in i for ws in SCENE_ROLE_WORDS.values() for w in ws)
    ]
    return "\n\n".join(keep)


def sections(text: str) -> dict[str, str]:
    """{heading: text} of the `## heading` sections of a notes file."""
    out: dict[str, str] = {}
    cur, buf = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = line[3:].strip(), [line]
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


# ---------------------------------------------------------------------------------------------- generated parts
def tasks_part(spec: dict) -> str:
    rows = []
    f = spec.get("tasks_file")
    if f and (REPO / f).exists():
        rows = [json.loads(x) for x in (REPO / f).read_text().splitlines() if x.strip()]
    if not rows:
        return ""
    if len(rows) > 8:
        return f"## Tasks\n{len(rows)} tasks (their ids and instructions are in the context's `tasks`); the notes name the kinds."
    out = ["## Tasks"]
    for r in rows:
        lim = (
            f" (limit {r['limit_s']:g} s)"
            if r.get("limit_s")
            else (f" (budget {r['limit_steps']} steps)" if r.get("limit_steps") else "")
        )
        out.append(f'- {r["id"]}{lim}: "{r.get("goal", "")}"')
    return "\n".join(out)


def guard_module(spec: dict):
    """The adapter's guard module (stdlib only, constants at module level), or None."""
    name = (spec.get("bench") or {}).get("name") or {"simple": "simple", "lml": "locomanilab"}.get(
        spec.get("backend") or ""
    )
    if not name:
        return None
    f = REPO / "src" / "locomani" / "bench" / name / "guard.py"
    if not f.exists():
        return None
    try:
        s = importlib.util.spec_from_file_location(f"_rsi_guard_{name}", f)
        mod = importlib.util.module_from_spec(s)
        sys.modules[s.name] = mod  # dataclass-free module; a name for the loader's bookkeeping
        s.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001  a guard that does not load is reported as such, not guessed
        return None


def _short_modules(mods) -> str:
    by: dict[str, list[str]] = {}
    for m in mods:
        head, _, tail = m.rpartition(".")
        by.setdefault(head, []).append(tail)
    return "; ".join(f"{h}.{{{', '.join(t)}}}" if len(t) > 1 else f"{h}.{t[0]}" for h, t in by.items())


def guard_part(spec: dict, scene: bool = False) -> str:
    if scene:
        return (
            "## What you may edit\nThe skills package `src/locomani/tasks/scene/skills/` only; any other changed file voids the "
            "candidate (the driver checks it)."
        )
    g = guard_module(spec)
    if g is None:
        return "## What you may edit\nThe skill package the family file names; the static guard's list is in the adapter's guard.py."
    lines = ["## What you may edit (the adapter's guard.py, generated)"]
    lines.append(
        f"- editable files: {', '.join(g.EDITABLE)}; frozen byte for byte: {', '.join(g.FROZEN)}; no other file in the package; "
        f"a file may grow by at most {g.MAX_GROWTH} lines."
    )
    locked = getattr(g, "LOCKED", {})
    if locked:
        lines.append("- locked definitions: " + "; ".join(f"{f}: {', '.join(n)}" for f, n in locked.items()))
    lines.append(
        f"- imports: relative ones from the package and the public modules the base file already imports, besides "
        f"{', '.join(g.NEW_IMPORTS_OK)}."
    )
    forb = getattr(g, "FORBIDDEN_MODULES", ())
    if forb:
        lines.append("- refused modules (the simulator side of the bed): " + _short_modules(forb))
    return "\n".join(lines)


def switches_part(spec: dict) -> str:
    bf = spec.get("bed_fixed") or {}
    keys = [k for k in bf if not k.startswith("_")]
    if not keys:
        return ""
    why = re.sub(r"^rule 46[^:]*:\s*", "", bf.get("_about", "")).strip()
    return f"## Fixed switches on in every round ([bed-fixed])\n{', '.join(f'`{k}`' for k in keys)}. {why}".strip()


def pools_part(spec: dict, caps: dict) -> str:
    pools = spec.get("pools") or {}
    bits = []
    if "diag" in pools:
        bits.append(f"diagnostic pool {pools['diag'].get('n')} starts (half of them the check half, hidden)")
    if "accept" in pools:
        bits.append(f"acceptance pool {pools['accept'].get('n')} (hidden)")
    if "sealed" in pools:
        bits.append(f"sealed pool {pools['sealed'].get('n')} (hidden)")
    cap_keys = ("p_max", "alpha_task", "alpha_task_check", "check_min_net", "stop_no_progress")
    cs = ", ".join(f"{k} {caps[k]}" for k in cap_keys if caps.get(k) not in (None, False, 0))
    return "## Pools and caps\n" + "; ".join(bits) + ("." if bits else "") + (f" Caps: {cs}." if cs else "")


def classes_part(caps: dict, class_suggest: dict | None) -> str:
    if not class_suggest:
        return ""
    lines = ["## Failure classes of this family ([classes])"]
    lines += [f"- `{k}`: {v}" for k, v in class_suggest.items()]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- assembly
def appendix(role: str, spec: dict, caps: dict, rules_path: Path | None, class_suggest: dict | None = None) -> str:
    """The family appendix for a role: the generated parts and the notes file's sections."""
    scene = rules_path is not None and Path(rules_path).name == SCENE.name
    parts = ROLE_PARTS.get(role, ())
    out: list[str] = []
    if scene:
        s = scene_slice(role, Path(rules_path))
        t = tasks_part(spec) if "tasks" in parts else ""
        return "\n\n".join(
            x
            for x in (
                t,
                guard_part(spec, True) if "guard" in parts else "",
                s,
                pools_part(spec, caps) if "pools" in parts else "",
            )
            if x
        )
    gen = {
        "tasks": lambda: tasks_part(spec),
        "guard": lambda: guard_part(spec),
        "switches": lambda: switches_part(spec),
        "pools": lambda: pools_part(spec, caps),
        "classes": lambda: classes_part(caps, class_suggest),
    }
    for p in ("tasks", "guard", "switches", "pools", "classes"):
        if p in parts and (g := gen[p]()):
            out.append(g)
    if rules_path is not None and Path(rules_path).exists() and ("notes" in parts or "notes_task" in parts):
        sec = sections(Path(rules_path).read_text())
        if "notes_task" in parts:
            sel = {h: t for h, t in sec.items() if re.search(r"task|compl|win", h, re.I)}
            sec = sel or sec
        if role == "engineer":  # the class hints are for the roles that read cards and gate tallies
            sec = {h: t for h, t in sec.items() if not re.search(r"class", h, re.I)}
        out.append("\n\n".join(sec.values()))
    return "\n\n".join(out)


def rules_for(
    role: str, spec: dict, caps: dict, rules_path: Path | None, class_suggest: dict | None = None, core: Path = CORE
) -> str:
    """What a role reads of the rules, inline: its core rules by id, then the family appendix."""
    head = core_slice(role, core)
    if not spec.get("bed_fixed"):  # no fixed switches in this family: the rule would only be noise
        head = re.sub(r"### \[bed-fixed\].*?(?=\n### |\Z)", "", head, flags=re.S).strip()
    app = appendix(role, spec, caps, rules_path, class_suggest)
    return (head + ("\n\n# This family\n\n" + app if app else "")).strip()


FRAG = re.compile(r"\{>(scene|bench)_([a-z]+)\}")


def expand(template: str, prompts: Path, scene: bool) -> str:
    """Insert the family-kind fragments: `{>scene_<role>}` (prompts/scene/<role>.md) on the scene line, `{>bench_<role>}`
    (prompts/bench/<role>.md) on a benchmark; the other kind's placeholder becomes empty."""

    def one(m: re.Match) -> str:
        kind, role = m.groups()
        f = Path(prompts) / kind / f"{role}.md"
        if (kind == "scene") != scene or not f.exists():
            return ""
        return "\n" + f.read_text().strip() + "\n"

    return FRAG.sub(one, template)


def role_prompt(
    role: str, prompts: Path, spec: dict, caps: dict, rules_path: Path | None, class_suggest: dict | None = None
) -> str:
    """A role's prompt template with its fragments and its rules slice inserted (the loop's `fill` then sets the run's
    values). A template without `{RULES}` (a test's fake prompts) gets no slice."""
    scene = rules_path is not None and Path(rules_path).name == SCENE.name
    t = expand((Path(prompts) / f"{role}.md").read_text(), prompts, scene)
    if "{RULES}" in t:
        t = t.replace("{RULES}", rules_for(role, spec, caps, rules_path, class_suggest))
    return t


def sizes(spec: dict, caps: dict, rules_path: Path | None, class_suggest: dict | None = None) -> dict[str, int]:
    return {r: len(rules_for(r, spec, caps, rules_path, class_suggest).encode()) for r in ROLES}


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import backend as BK

    fam = sys.argv[1]
    spec = json.loads((HERE / "families" / f"{fam}.json").read_text())
    lay = BK.layout(spec)
    if len(sys.argv) > 2:
        print(rules_for(sys.argv[2], spec, lay["caps"], lay["rules"], lay.get("class_suggest")))
    else:
        print(json.dumps(sizes(spec, lay["caps"], lay["rules"], lay.get("class_suggest")), indent=1))
