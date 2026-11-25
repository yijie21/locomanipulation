"""Role calls (`claude -p`), their prompts, schemas and the Reviewer step."""

from __future__ import annotations

import collections
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "src")]
import audit as AU  # noqa: E402
import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import cascade as CS  # noqa: E402
import ledger as LG  # noqa: E402
import rules_build as RB  # noqa: E402
import segments as SG  # noqa: E402
import state as S  # noqa: E402

HYPOTHESES_SCHEMA = {
    "type": "object",
    "required": ["hypotheses", "notes"],
    "properties": {
        "notes": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "maxItems": S.MAX_HYPOTHESES,
            "items": {
                "type": "object",
                "required": ["id", "claim", "instead", "evidence", "delta", "predicted_fixed", "predicted_broken"],
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "instead": {"type": "string"},  # TODO 33.17: what to change, in one sentence
                    "status": {"enum": list(S.STATUSES)},  # set by the driver after the probe (probe_status)
                    "evidence": {"type": "string"},
                    "delta": {"type": "object", "properties": {"set": {"type": "object"}}, "required": ["set"]},
                    "needs_code": {"type": "string"},  # the probe never runs it; its `delta` is a hint for the Engineer
                    "predicted_fixed": {"type": "array", "items": {"type": "string"}},
                    "predicted_broken": {"type": "array", "items": {"type": "string"}},
                    "tasks": {"type": "array", "items": {"type": "string"}},  # a slow bed: where the probe runs
                },
            },
        },
    },
}


def hypotheses_schema() -> dict:
    """The Reviewer's schema as the family's caps allow it (TODO 54, review verify.md 12 and design.md: a benchmark
    Reviewer was handed fields for infeasibility claims, design probes, record-library `measure`s and a `sweep` that its
    backend answers with nothing). `infeasible` needs caps claims, `design` caps design_probes, `measure` caps
    measure_claims, `sweep` caps prober (the Prober phase; a lean family sweeps only the Engineers' `tunable` keys); with
    the program as Manager (caps manager "program") each hypothesis must name its `tasks` and may name `starts`, which
    is what the assignment of Engineers reads."""
    sch = copy.deepcopy(HYPOTHESES_SCHEMA)
    items = sch["properties"]["hypotheses"]["items"]
    props = items["properties"]
    extra = S.H.schema_fields(S.cap)  # the scene line's fields (infeasible, design, measure, sweep), each by its cap
    sch["properties"].update(extra.get("top", {}))
    props.update(extra.get("hypothesis", {}))
    if S.cap("manager", "llm") == "program":
        props["starts"] = {"type": "array", "items": {"type": "string"}}  # the start labels the hypothesis targets
        items["required"] = [*items["required"], "tasks"]
    return sch


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
        "scope": {  # [scope]: what the draft says it changes; the driver measures what it did change (footprint)
            "type": "object",
            "required": ["when", "starts"],
            "properties": {
                "when": {"type": "string"},
                "starts": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def pool_shape(pool: Path, fallback: int = 0) -> tuple[int, str]:
    """(starts in all, starts per task as text) of a pool directory; `fallback` in all when the pool is not there. A
    pool whose tasks hold unequal counts reads "a-b"."""
    labels = B.pool_labels(pool) if pool.exists() else []
    if not labels:
        return fallback, str(fallback)
    per = collections.Counter(LG.task_of(x) for x in labels)
    lo, hi = min(per.values()), max(per.values())
    return len(labels), str(lo) if lo == hi else f"{lo}-{hi}"


def role_template(role: str) -> str:
    """TODO 54 phase 2: a role's prompt with its fragments and its rules (the core rules it acts on by id, the family's
    appendix) inserted; `fill` sets the run's values after."""
    return RB.role_prompt(role, S.PROMPTS, S.family_spec(), S.CAPS, S.RULES, S.CLASS_SUGGEST)


def prompt_vars() -> dict:
    """What a family sets in the role prompts (TODO 31): the plan the roles read (empty when the family has none in
    the sandbox, TODO 54: three benchmark prompt sets pointed every role at a file the sandbox did not hold), a note
    on the seed's skills, the scene the prompts' first line names (`scene_note`, TODO 38.10), and the pool sizes.
    TODO 54 (review verify.md 4): `N_DIAG` is the number of diagnostic starts the roles see IN ALL, `N_DIAG_TASK`
    the number per task (the template used the per-task count as a total: "the incumbent's 15 diagnostic rounds" of
    60); `N_DIAG_FULL_TASK` and `N_DIAG_FULL` count the starts with the check half. Counted from the pool
    directories, which the check split has already halved."""
    spec = S.family_spec()
    pools = spec.get("pools") or {}
    seen, seen_task = pool_shape(S.LOOP / "pool_diag", int((pools.get("diag") or {}).get("n", 50)))
    held, _ = pool_shape(CS.check_pool()) if S.cap("check_half") else (0, "0")
    n_accept, accept_task = pool_shape(S.HIDDEN / "pool_accept", int((pools.get("accept") or {}).get("n", 60)))
    full = seen + held
    return dict(
        PLAN=S.PLAN if S.PLAN and Path(S.PLAN).exists() else "",
        SKILLS_NOTE=spec.get("skills_note") or S.SKILLS_NOTE,
        SCENE_NOTE=spec.get("scene_note") or S.SCENE_NOTE,
        WHAT=(spec.get("scene_note") or S.SCENE_NOTE)
        if S.RULES.name == "rules_scene.md"
        else "on " + RB.BENCH_NAME.get(S.RULES.stem, S.FAMILY),
        N_DIAG=seen,
        N_DIAG_TASK=seen_task,
        N_DIAG_FULL=full,
        N_DIAG_FULL_TASK=str(
            int(round(full / max(1, len({LG.task_of(x) for x in B.pool_labels(S.LOOP / "pool_diag")}))))
        ),
        N_ACCEPT=n_accept,
        N_ACCEPT_TASK=accept_task,
        CANDIDATES_NOTE=S.CANDIDATES_NOTE.replace("{K}", str(S.CANDIDATES)) if S.CANDIDATES > 1 else "",
        LIBRARY=S.LIBRARY_NOTE if (S.HERE / "experience" / "INDEX.md").exists() else "",
    )


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


class RoleFailed(RuntimeError):
    pass


class RoleLate(RoleFailed):
    """TODO 37.3: the driver stopped the role (an Engineer past --engineer-late-min after the other drafts were in)."""


def role_dir(role: str) -> Path:
    """TODO 36 (review 2026-10-22, L5): a role that runs beside others (engineer_2, reviewer_P) gets its own scratch
    directory under role_tmp(), so same-named scripts and output directories do not collide."""
    return AU.role_tmp() / role if "_" in role else AU.role_tmp()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=S.REPO, capture_output=True, text=True).stdout.strip()


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
        subprocess.run(["git", "reset", "--soft", head0], cwd=S.REPO, capture_output=True)
        subprocess.run(["git", "reset", "-q"], cwd=S.REPO, capture_output=True)
    return dict(was=head0, moved_to=head1)


def claude(
    role: str, prompt: str, schema: dict, it_dir: Path, tools: str, stub=None, stop: threading.Event | None = None,
) -> tuple[dict, dict]:  # fmt: skip
    """One role call. `stop` (TODO 37.3): an event the driver sets to give the role up; run_role then kills the
    session and raises RoleLate."""
    prompt += fill(S.ISOLATION, REPO=S.REPO, TMP=role_dir(role))
    (it_dir / f"{role}_prompt.md").write_text(prompt)
    if stub is not None:
        return stub, dict(cost_usd=0.0, session=None, seconds=0.0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    role_dir(role).mkdir(parents=True, exist_ok=True)
    # TODO 36 (review 2026-10-22): no memory file from the sandbox's ancestors (the 23 KB machine guide in /workspace)
    # or the operator's home reaches a role; tested with one session
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", TMPDIR=str(role_dir(role)),
               # TODO 37.1: try.py's round budget is per role and iteration ([try])
               RSI_TRY_LEDGER=str(role_dir(role) / f"try_{it_dir.name}.jsonl"))  # fmt: skip
    cmd = [
        "claude", "-p", "--model", S.MODEL, "--output-format", "json", "--json-schema", json.dumps(schema),
        "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config",
        "--permission-mode", "bypassPermissions", "--allowedTools", "Bash", "--disallowedTools", tools,
        *(["--effort", S.EFFORT.get(role) or S.EFFORT[role.split("_")[0]]] if (role in S.EFFORT or role.split("_")[0] in S.EFFORT) else []),
        *(["--add-dir", *AU.allowed_dirs()] if AU.allowed_dirs() else []),
        *(["--max-budget-usd", str(max(1.0, S.COST_CAP / 8))] if S.COST_CAP is not None else []),
    ]  # fmt: skip
    session = str(uuid.uuid4())
    how = ["--session-id", session]
    before = AU.transcript(session)
    seen = before.read_text(errors="replace").count(ANSWERED) if before is not None else 0
    t0 = time.time()
    head0 = git_head()
    with AU.hidden_locked():
        stdout, stderr, rc, killed = run_role([*cmd, *how], prompt, env, session, stop=stop, seen=seen)
    committed = undo_commits(head0)
    (it_dir / f"{role}_raw.json").write_text(stdout or stderr)
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
    cmd: list[str], prompt: str, env: dict, session: str, grace: float = S.ROLE_GRACE_S, poll: float = 10.0,
    stop: threading.Event | None = None, seen: int = 0,
) -> tuple[str, str, int, list[str]]:  # fmt: skip
    """Run a role's `claude -p` to its end. claude -p does not exit while a background shell it started still runs,
    and a role once left `until ! pgrep -f "<dir>"; do sleep 10; done`, which matches itself and never ends (iteration
    2, 2026-10-16: the driver stalled until a human killed it). So once the transcript shows the structured answer,
    the role's leftover processes get `grace` seconds, then the driver kills them (listed in the ledger). `stop` set
    (TODO 37.3): the session and everything under it are killed and RoleLate is raised. `seen`: the answers already in
    the transcript when a finished session takes another turn ([scope])."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=S.REPO, env=env)  # fmt: skip
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
            p = AU.transcript(session)
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
    p = AU.transcript(session)
    so = None
    for line in p.open() if p else ():
        if '"StructuredOutput"' not in line:
            continue
        for c in (json.loads(line).get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "StructuredOutput":
                so = c.get("input")
    return so


def rerun_note(role: str, leaks: list[str]) -> str:
    """role_rerun: what the second attempt is told."""
    return (f"\n\nNOTE: an earlier attempt of this {role} in this iteration was discarded because its tool calls named "
            f"off-limits paths ({', '.join(leaks)}). Do not name the hidden tree or those paths in any command, not even "
            f"to exclude them or to check that the package does not use them: the driver checks that.\n")  # fmt: skip


def discard_attempt(n: int, role: str, acct: dict, leaks: list[str], *paths: Path) -> None:
    """role_rerun: the first attempt's files leave the visible tree (hide_void), its own scratch directory is emptied
    (an unsuffixed role shares role_tmp() with the others and their try.py ledgers: that one stays), and a `rerun`
    row keeps its cost (run_cost sums the rows)."""
    AU.hide_void(n, *paths)
    d = role_dir(role)
    if d.is_dir() and d != AU.role_tmp():
        shutil.rmtree(d, ignore_errors=True)
    LG.ledger(dict(step="rerun", iter=n, role=role, leaks=leaks, cost_usd=acct.get("cost_usd", 0.0),
                session=acct.get("session"), seconds=acct.get("seconds")))  # fmt: skip
    print(f"iteration {n}: the {role} named off-limits paths ({', '.join(leaks)}); it runs once more")


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


def parameters_of(inc: dict) -> dict:
    """The incumbent's overridable parameters (skills.parameters) with its overrides applied."""
    if be := BK.of(S.FAMILY):  # a second backend reads its package's parameters from the source text
        try:
            pkg = be.load_skills(Path(inc["dir"]) / "skills", "scene_skills", S.FAMILY)
            pkg.apply_set(pkg, inc["set"])
            return json.loads(json.dumps(pkg.parameters(pkg), default=str))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[-500:]}
    return S.H.parameters_of(inc)  # a scene family: the skills loaded in its own process


def reviewer_cards(n: int, base: Path, it_dir: Path, log: bool = True) -> Path:
    """The failure cards of the incumbent's diagnostic rounds (cards.py, [cards]), in iter_NN/cards."""
    out = it_dir / "cards"
    if be := BK.of(S.FAMILY):  # the backend writes its own failure cards (cards.py needs a scene file)
        out.mkdir(parents=True, exist_ok=True)
        be.cards(list(B.outcomes(base).values()), out)
        if log:
            LG.ledger(cards_row(n, out))
        return out / "cards.md"
    return S.H.reviewer_cards(n, base, it_dir, log)  # a scene family: cards.py on the rounds and the scene file


NOTE_EXT = (".py", ".sh", ".md", ".txt", ".json", ".csv", ".tsv")


NOTE_MAX_FILE = 256 * 1024
NOTE_MAX_ALL = 4 * 1024 * 1024


def review_sources() -> list[tuple[str, Path, bool]]:
    """Where the Reviewer keeps working files: (tag, directory, recurse). It shares role_tmp() with every other role, so
    only its top-level files count."""
    return [("", AU.role_tmp(), False)]


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


def cards_row(n: int, out: Path) -> dict:
    imgs = list((out / "img").glob("*.png"))
    return dict(step="cards", iter=n, cards_bytes=(out / "cards.md").stat().st_size, images=len(imgs),
                image_bytes=sum(p.stat().st_size for p in imgs))  # fmt: skip


def review_common(n: int, inc: dict, base: Path, cards: Path, attribution: dict | None) -> tuple[dict, dict]:
    """What the Reviewer's prompt and the later roles' prompts are filled from: (`common` variables, context `extra`)."""
    common = dict(**prompt_vars(), LOOP=S.LOOP, SIM_PY=S.SIM_PY, DIAG_INC=base, SKILLS=S.SKILLS, FAMILY=S.FAMILY,
                  FAMILY_FILE=S.HERE / "families" / f"{S.FAMILY}.json", POOL=S.LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=AU.role_tmp(), CARDS=cards)  # fmt: skip
    extra = dict(diag_summary=summarize(base), parameters=parameters_of(inc), focus=LG.jload(S.STATE / "focus.json", {}),
                 parked=LG.jload(S.STATE / "parked.json", {}),
                 segments=SG.segment_table(B.outcomes(base).values()))  # fmt: skip  # TODO 48.17.6 ([segments])
    if attribution:
        extra["attribution"] = attribution
    prev = S.STATE / f"iter_{n - 1:02d}" / "designs.json"  # TODO 33.19: last iteration's design probes
    if prev.exists():
        extra["designs_last"] = LG.jload(prev)
    return common, extra


def run_review(n: int, it_dir: Path, inc: dict, base: Path, common: dict, extra: dict, stub: bool, note: str = ""):
    """The Reviewer of iteration n on the incumbent `inc` and its rounds `base`: (hypotheses, accounting, sessions, leaks).
    Writes hypotheses.json and review_notes; the ledger row is the caller's. `note`: appended to the prompt (a rerun)."""
    ctx = it_dir / "reviewer_context.json"
    LG.jdump(ctx, LG.context(inc, extra))
    prompt = fill(role_template("reviewer"), CONTEXT=ctx, **common) + note
    stub_h = dict(notes="stub", hypotheses=[dict(id="H1", claim="stub", instead="stub", evidence="stub",
                  delta={"set": {"PLAN_HELD": "pick_simple"} if "PLAN_HELD" in extra["parameters"] else {}},
                  predicted_fixed=[], predicted_broken=[])])  # fmt: skip
    t0 = time.time()
    with AU.rounds_locked(base):
        hyps, acct = claude("reviewer", prompt, hypotheses_schema(), it_dir, "Edit,Write,NotebookEdit,Agent",
                            stub_h if stub else None)  # fmt: skip
    LG.jdump(it_dir / "hypotheses.json", hyps)
    sessions = acct.pop("sessions", None) or [acct["session"]]
    leaks = sorted({x for sess in sessions for x in AU.transcript_leaks(sess)})
    acct["review_notes"] = copy_review_notes(it_dir, review_sources(), t0) if not leaks else None
    common["REVIEW_NOTES"] = review_notes_dir(it_dir)
    return hyps, acct, sessions, leaks


def review_row(n: int, hyps: dict, acct: dict, sessions: list, leaks: list, **more) -> None:
    n_claims = len(hyps["infeasible"]) if isinstance(hyps.get("infeasible"), list) else 0
    LG.ledger(dict(step="review", iter=n, n_hypotheses=len(hyps["hypotheses"]), n_claims=n_claims, void=bool(leaks),
                leaks=leaks, **acct, **({"sessions": sessions} if len(sessions) > 1 else {}), **more))  # fmt: skip
