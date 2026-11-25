"""The anti-leaderboard walls around a role call: the hidden tree's lock, the rounds' lock, the transcript audit for off-limits paths, voided iterations."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import stat
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import backend as BK  # noqa: E402
import ledger as LG  # noqa: E402
import state as S  # noqa: E402


def role_tmp() -> Path:
    return S.SCRATCH_ROOT / f"rsi_scene_{S.REPO.name}"


_HID = dict(n=0, mode=None)  # reference-counted lock on the hidden tree (L9)


_HID_LOCK = threading.Lock()


def hidden_mode_file() -> Path:
    return S.STATE / "hidden_mode.txt"


@contextlib.contextmanager
def hidden_locked():
    """TODO 36 (review 2026-10-22, L9): while any role runs, the family's hidden directory has mode 000, so a `grep -r
    outputs/` or `ls -R` cannot read pools or verdicts (the audit sees only tool inputs, never what a command's output
    shows). Parallel roles share one lock; the original mode is kept in STATE so cmd_run can restore it after a
    crash. The gate and the hidden pools are used between role calls only."""
    with _HID_LOCK:
        if _HID["n"] == 0 and hidden_lockable():
            _HID["mode"] = stat.S_IMODE(S.HIDDEN.stat().st_mode)
            hidden_mode_file().write_text(oct(_HID["mode"]))
            S.HIDDEN.chmod(0)
        _HID["n"] += 1
    try:
        yield
    finally:
        with _HID_LOCK:
            _HID["n"] -= 1
            if _HID["n"] == 0 and _HID["mode"] is not None:
                S.HIDDEN.chmod(_HID["mode"])
                hidden_mode_file().unlink(missing_ok=True)
                _HID["mode"] = None


def hidden_lockable() -> bool:
    """Only a family directory inside ..._hidden/ is ever locked: a test once ran a role call with HIDDEN left at
    Path() -- the working directory -- and the repository root went to mode 000 (2026-10-22)."""
    return S.HIDDEN.parent.name == "scene_rsi_20261016_hidden" and S.HIDDEN.is_dir() and not S.HIDDEN.is_symlink()


def unlock_hidden() -> None:
    """A crash while a role ran leaves the hidden tree at mode 000: put it back (cmd_run, at start)."""
    if not hidden_lockable():
        return
    f = hidden_mode_file()
    if f.exists():
        S.HIDDEN.chmod(int(f.read_text().strip() or "755", 8))
        f.unlink()
    elif stat.S_IMODE(S.HIDDEN.stat().st_mode) & 0o700 == 0:
        S.HIDDEN.chmod(0o755)
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
    for d in (S.LOOP / "diag").glob("incumbent_*"):
        if stat.S_IMODE(d.stat().st_mode) & 0o700 == 0:
            d.chmod(0o755)


def transcript(session: str | None) -> Path | None:
    if not session:
        return None
    p = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(S.REPO)) / f"{session}.jsonl"
    return p if p.exists() else None


def scene_links() -> list[Path]:
    """Where the sandbox's scene links point in the origin (sandbox.py LINKS / scene_exclude): the whole
    outputs/scene_discovery, or that scene's linked entries and its physics copy. Empty outside a sandbox."""
    root = S.REPO / "outputs" / "scene_discovery"
    if S.ORIGIN == S.REPO or not root.exists():
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
    pats = {"gate": re.escape(S.HIDDEN.parent.name)}
    pats["kit"] = r"rsi_kit|scene_rsi_20261016/kit/"  # kit.py: every family's pools and outcomes (2026-10-20)
    if S.ORIGIN != S.REPO:
        # the origin checkout, except what the sandbox links into it: third_party, the scene's linked entries, and the
        # scene directories themselves named bare -- a role that resolves a link reaches the origin's scene directory,
        # and three drafts were voided for naming it (user 2026-10-24: 越界检查要放行原仓库场景目录本身); an entry
        # the sandbox does not link (scene_exclude), or a glob over the directory, is still flagged
        links = [p for p in scene_links() if p.is_relative_to(S.ORIGIN)]
        linked = ["/" + re.escape(str(p.relative_to(S.ORIGIN))) + r"(?:/|\b)" for p in links]
        scene_root = S.ORIGIN / "outputs" / "scene_discovery"
        dirs = sorted({p.parent for p in links if p.parent != scene_root and p.parent.is_relative_to(scene_root)})
        bare = ["/" + re.escape(str(d.relative_to(S.ORIGIN))) + rf"/?(?={BARE_END})" for d in dirs]
        pats["origin"] = re.escape(str(S.ORIGIN)) + rf"(?!/third_party{''.join('|' + x for x in linked + bare)})"
        pats["climb"] = r"third_party/\.\."
        # Claude Code's own home (other sessions' transcripts, settings), not a `.claude/` inside the checkout: the
        # iteration-5 Engineer only looked for the sandbox's project settings and that voided its round (2026-10-23)
        pats["claude_home"] = re.escape(str(Path.home() / ".claude")) + r"(?:/|\b)|~/\.claude"
        pats["sibling"] = re.escape(str(S.REPO.parent)) + "/" + rf"(?!{re.escape(S.REPO.name)}(?:/|\b))"
        pats["scratch"] = re.escape(str(S.SCRATCH_ROOT)) + "/" + rf"(?!{re.escape(role_tmp().name)}(?:/|\b))"
    if BK.of(S.FAMILY):  # the backend's own forbidden paths (layout()["leak"]: regexes, e.g. released episodes)
        pats.update({f"backend_{i}": rx for i, rx in enumerate(BK.layout(S.FAMILY).get("leak") or [])})
    return pats


def own_link_targets() -> list[str]:
    """The real paths the sandbox's own symlinks point to: third_party, the scene entries under outputs/scene_discovery
    (three levels) and the *_physics links at its root -- places a role may legitimately walk into and `..` out of."""
    links = [S.REPO / "third_party", *S.REPO.glob("*_physics")]
    base = S.REPO / "outputs" / "scene_discovery"
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
    repo = os.path.realpath(S.REPO)
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
    targets = own_link_targets() if S.ORIGIN != S.REPO else None
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


def hide_void(n: int, *paths: Path) -> None:
    """TODO 36 (review 2026-10-22, L5): a voided draft (leak or stray) leaves the visible loop tree for
    HIDDEN/<family>/void/iter_NN/, so the Composer and later roles cannot read code written after the audit flagged
    its role. (HIDDEN is unlocked between role calls.)"""
    dst = S.HIDDEN / "void" / f"iter_{n:02d}"
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
    LG.ledger(dict(step="void_iter", iter=n, role=role, leaks=leaks))
    print(f"iteration {n} void: the {role} named off-limits paths ({', '.join(leaks)})")
