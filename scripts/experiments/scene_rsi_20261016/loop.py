"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md §2, D6-D8, D11): the scene RSI inner loop on one task
family -- the warehouse loop (scalebfm_warehouse_rsi_20261014/loop.py) with skills as candidates. One iteration:
Reviewer (claude -p) -> Prober (the hypotheses' parameter overrides on the diagnostic pool) -> Manager (claude -p) ->
Engineer (claude -p: code in the skills package and/or overrides) -> Gate (gate.py) -> ledger. State lives in files under
<sandbox>/outputs/scene_rsi_20261016/<family>/loop, so the driver stops and resumes at any iteration.

Runs inside a sandbox checkout (sandbox.py). The roles see the sandbox, rules.md and the plan; the acceptance and sealed
pools and the gate live in the hidden tree (`..._hidden/`), and every role transcript is grepped for that name and for
paths outside the sandbox: a hit voids the iteration (the warehouse loop's audit, own tool-results excepted).

The Engineer may change only the sandbox's skills package (src/locomani/tasks/scene/skills/); anything else it touches
voids the call and is restored. On a rejection the package goes back to the incumbent's snapshot.

Stops (D8 and the warehouse budgets): the incumbent's acceptance successes >= STOP_RATE of the pool, gate calls >= 12,
wall clock >= MAX_WALL_H, model cost >= COST_CAP, 3 rejections in a row with no new hypothesis, a Manager `stop`.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/experiments/scene_rsi_20261016/loop.py run pick_single > /dev/null 2>&1 &
    $PY scripts/experiments/scene_rsi_20261016/loop.py status pick_single
    $PY scripts/experiments/scene_rsi_20261016/loop.py run pick_single --stub --max-iters 1   # plumbing, no model calls
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]
import batch as B  # noqa: E402
import family as F  # noqa: E402
from gate import candidate_hash, overrides  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"
PLAN = REPO / "docs" / "handoff" / "scene-rsi-plan-20261016.md"
RULES = HERE / "rules.md"
MODEL = os.environ.get("RSI_MODEL", "claude-opus-5-5")
COST_CAP = float(os.environ.get("RSI_COST_CAP_USD", "150"))
MAX_WALL_H = float(os.environ.get("RSI_MAX_WALL_H", "6"))
MAX_GATE_CALLS = 12
MAX_REJECT_STREAK = 3
MAX_HYPOTHESES = 4
STOP_RATE = 0.70  # D8: the inner loop stops when the incumbent wins this share of the acceptance pool
PROCS = 12
GPUS = "0,1"
ORIGIN = (REPO / "third_party").resolve().parent  # a sandbox links third_party to the origin checkout
SCRATCH_ROOT = Path("/workspace/tmp")
ROLE_GRACE_S = 120  # after a role answers, its leftover processes get this long before the driver kills them
ISOLATION = (
    "\n\nThis checkout ({REPO}) and your scratch directory ({TMP}) are the whole world for this run. Do not read "
    "anything outside them: no other checkout of this project, nothing under ~/.claude, no path through third_party/.. "
    "A role call that does is void.\n"
)
FAMILY = ""
LOOP = HIDDEN = STATE = GATE_DIR = Path()

HYPOTHESES_SCHEMA = {
    "type": "object",
    "required": ["hypotheses", "notes"],
    "properties": {
        "notes": {"type": "string"},
        "infeasible": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "maxItems": MAX_HYPOTHESES,
            "items": {
                "type": "object",
                "required": ["id", "claim", "evidence", "delta", "predicted_fixed", "predicted_broken"],
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "delta": {"type": "object", "properties": {"set": {"type": "object"}}, "required": ["set"]},
                    "needs_code": {"type": "string"},
                    "predicted_fixed": {"type": "array", "items": {"type": "string"}},
                    "predicted_broken": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}
MANAGER_SCHEMA = {
    "type": "object",
    "required": ["decision", "reason", "summary_zh"],
    "properties": {
        "decision": {"enum": ["engineer", "review", "stop"]},
        "hypothesis": {"type": "string"},
        "focus": {"type": "string"},
        "reason": {"type": "string"},
        "escalation": {"type": "string"},
        "infeasible": {"type": "string"},
        "summary_zh": {"type": "string"},
    },
}
ENGINEER_SCHEMA = {
    "type": "object",
    "required": ["mechanism", "files_changed", "set", "tests"],
    "properties": {
        "mechanism": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "set": {"type": "object"},
        "tests": {"type": "string"},
    },
}


def configure(family: str) -> None:
    global FAMILY, LOOP, HIDDEN, STATE, GATE_DIR
    FAMILY = family
    LOOP = REPO / "outputs" / "scene_rsi_20261016" / family
    HIDDEN = REPO / "outputs" / "scene_rsi_20261016_hidden" / family  # acceptance / sealed pools and the gate
    STATE, GATE_DIR = LOOP / "loop", HIDDEN / "gate"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def ledger(entry: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": now(), **entry}, ensure_ascii=False, default=str) + "\n")


def entries() -> list[dict]:
    p = STATE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def accept_pool() -> Path:
    return HIDDEN / "pool_accept"


def stop_reason(meta: dict) -> str | None:
    es = entries()
    accepted = [e for e in es if e.get("step") == "gate" and e.get("accepted") and not e.get("void")]
    n_accept = len(B.pool_labels(accept_pool()))
    if accepted and accepted[-1]["result"]["candidate_ok"] >= STOP_RATE * n_accept:
        return f"the incumbent wins {accepted[-1]['result']['candidate_ok']}/{n_accept} >= {STOP_RATE:.0%}: sealed set"
    gates = [e for e in es if e.get("step") == "gate" and not e.get("void")]
    if len(gates) >= MAX_GATE_CALLS:
        return f"gate calls {len(gates)} >= {MAX_GATE_CALLS}"
    if (time.time() - meta["started_s"]) / 3600 >= MAX_WALL_H:
        return f"wall clock >= {MAX_WALL_H} h"
    cost = sum(e.get("cost_usd", 0.0) for e in es)
    if cost >= COST_CAP:
        return f"model cost {cost:.0f} >= {COST_CAP:.0f} USD"
    streak = 0
    for e in reversed(es):
        if e.get("step") == "gate":
            if e.get("accepted"):
                break
            streak += 1
    reviews = [e for e in es if e.get("step") == "review"]
    if streak >= MAX_REJECT_STREAK and reviews and reviews[-1].get("n_hypotheses", 0) == 0:
        return f"{streak} rejections in a row and the last review had no new hypothesis"
    m = [e for e in es if e.get("step") == "manager"]
    if m and m[-1].get("decision") == "stop":
        return "manager: stop"
    return None


# -- model calls (the warehouse loop's) ----------------------------------------------------------------------------


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


class RoleFailed(RuntimeError):
    pass


def role_tmp() -> Path:
    return SCRATCH_ROOT / f"rsi_scene_{REPO.name}"


def claude(role: str, prompt: str, schema: dict, it_dir: Path, tools: str, stub=None) -> tuple[dict, dict]:
    prompt += fill(ISOLATION, REPO=REPO, TMP=role_tmp())
    (it_dir / f"{role}_prompt.md").write_text(prompt)
    if stub is not None:
        return stub, dict(cost_usd=0.0, session=None, seconds=0.0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    role_tmp().mkdir(parents=True, exist_ok=True)
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", TMPDIR=str(role_tmp()))
    cmd = [
        "claude", "-p", "--model", MODEL, "--output-format", "json", "--json-schema", json.dumps(schema),
        "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config",
        "--permission-mode", "bypassPermissions", "--disallowedTools", tools,
        "--max-budget-usd", str(max(1.0, COST_CAP / 8)),
    ]  # fmt: skip
    session = str(uuid.uuid4())
    t0 = time.time()
    stdout, stderr, rc, killed = run_role([*cmd, "--session-id", session], prompt, env, session)
    (it_dir / f"{role}_raw.json").write_text(stdout or stderr)
    try:
        out = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RoleFailed(f"{role}: no JSON from claude -p (rc {rc}): {(stderr or stdout)[-300:]}") from e
    if out.get("is_error"):
        raise RoleFailed(f"{role}: {out.get('subtype')} {str(out.get('result'))[-300:]}")
    acct = dict(cost_usd=float(out.get("total_cost_usd") or 0.0), session=out.get("session_id"),
                seconds=round(time.time() - t0), usage=out.get("usage"), killed_leftovers=killed)  # fmt: skip
    so = out.get("structured_output") or structured_from_transcript(acct["session"])
    if so is None:
        raise RoleFailed(f"{role}: no structured output; result: {str(out.get('result'))[-300:]}")
    return so, acct


def run_role(
    cmd: list[str], prompt: str, env: dict, session: str, grace: float = ROLE_GRACE_S, poll: float = 10.0
) -> tuple[str, str, int, list[str]]:
    """Run a role's `claude -p` to its end. claude -p does not exit while a background shell it started still runs,
    and a role once left `until ! pgrep -f "<dir>"; do sleep 10; done`, which matches itself and never ends (iteration
    2, 2026-10-16: the driver stalled until a human killed it). So once the transcript shows the structured answer,
    the role's leftover processes get `grace` seconds, then the driver kills them (listed in the ledger)."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=REPO, env=env)  # fmt: skip
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
        if answered is None:
            p = transcript(session)
            if p is not None and "Structured output provided successfully" in p.read_text(errors="replace"):
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


def transcript(session: str | None) -> Path | None:
    if not session:
        return None
    p = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(REPO)) / f"{session}.jsonl"
    return p if p.exists() else None


def structured_from_transcript(session: str | None) -> dict | None:
    p = transcript(session)
    so = None
    for line in p.open() if p else ():
        if '"StructuredOutput"' not in line:
            continue
        for c in (json.loads(line).get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "StructuredOutput":
                so = c.get("input")
    return so


def leak_patterns() -> dict[str, str]:
    pats = {"gate": re.escape(HIDDEN.parent.name)}
    if ORIGIN != REPO:
        pats["origin"] = re.escape(str(ORIGIN)) + r"(?!/third_party)"
        pats["climb"] = r"third_party/\.\."
        pats["claude_home"] = r"\.claude/|~/\.claude"
        pats["sibling"] = re.escape(str(REPO.parent)) + "/" + rf"(?!{re.escape(REPO.name)}(?:/|\b))"
        pats["scratch"] = re.escape(str(SCRATCH_ROOT)) + "/" + rf"(?!{re.escape(role_tmp().name)}(?:/|\b))"
    return pats


def transcript_leaks(session: str | None) -> list[str]:
    p = transcript(session)
    if p is None:
        return []
    own = re.compile(r"[^\"\s]*\.claude/projects/[^\"\s/]+/" + re.escape(str(session)) + r"/tool-results/[^\"\s]*")
    pats, found = leak_patterns(), set()
    with p.open() as fh:
        for line in fh:
            if '"tool_use"' in line:
                line = own.sub("", line)
                found |= {k for k, pat in pats.items() if re.search(pat, line)}
    return sorted(found)


# -- steps ------------------------------------------------------------------------------------------------------------


def incumbent() -> dict:
    """The gate's incumbent copied into the loop's tree (roles must not open the gate's files)."""
    inc = jload(GATE_DIR / "incumbent.json")
    src = Path(inc["dir"]) if inc else None
    if src is None:  # before the first gate call: the sandbox's own skills
        src = STATE / "seed_candidate"
        if not src.exists():
            shutil.copytree(SKILLS, src / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (src / "candidate.json").write_text('{"set": {}}')
    h = candidate_hash(src)
    dst = LOOP / "incumbents" / h
    if not dst.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dict(dir=str(dst), hash=h, set=overrides(dst), since_gate_call=(inc or {}).get("since"))


def diag_incumbent(inc: dict, it_dir: Path) -> Path:
    d = LOOP / "diag" / f"incumbent_{inc['hash']}"
    side = LOOP / "diag" / f"candidate_{inc['hash']}"
    if not d.exists() and side.exists():
        side.rename(d)
    B.run_jobs([B.Job(Path(inc["dir"]) / "skills", d, inc["set"], None, True)], FAMILY, LOOP / "pool_diag", PROCS,
               it_dir / "prober.log", GPUS)  # fmt: skip
    return d


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


def probe(hyps: list[dict], inc: dict, base: Path, it_dir: Path) -> list[dict]:
    jobs = [B.Job(Path(inc["dir"]) / "skills", it_dir / "probe" / h["id"], {**inc["set"], **h["delta"].get("set", {})})
            for h in hyps]  # fmt: skip
    failed = None
    try:
        B.run_jobs(jobs, FAMILY, LOOP / "pool_diag", PROCS, it_dir / "prober.log", GPUS)
    except Exception as e:
        failed = f"{type(e).__name__}: {e}"
    before = {k: r["ok"] for k, r in B.outcomes(base).items()}
    res = []
    for h in hyps:
        after = {k: r["ok"] for k, r in B.outcomes(it_dir / "probe" / h["id"]).items()}
        if failed or not after:
            res.append(dict(id=h["id"], error=failed or "no rounds ran (an unknown parameter? see prober.log)"))
            continue
        fixed = sorted(k for k in after if after[k] and not before.get(k))
        broken = sorted(k for k in after if before.get(k) and not after[k])
        pf, pb = set(h["predicted_fixed"]), set(h["predicted_broken"])
        res.append(dict(id=h["id"], n=len(after), ok_before=sum(before.values()), ok_after=sum(after.values()),
                        fixed=fixed, broken=broken, predicted_fixed_hit=len(pf & set(fixed)), predicted_fixed_n=len(pf),
                        predicted_broken_hit=len(pb & set(broken)), predicted_broken_n=len(pb),
                        summary=summarize(it_dir / "probe" / h["id"])))  # fmt: skip
    return res


def parameters_of(inc: dict) -> dict:
    """The incumbent's overridable parameters (skills.parameters) with its overrides applied."""
    r = subprocess.run([str(SIM_PY), "-c", (
        "import json,sys; sys.path[:0]=[sys.argv[1], 'src']; from scene_round import load_skills; "
        "p=load_skills(__import__('pathlib').Path(sys.argv[2])); p.apply_set(p, json.loads(sys.argv[3])); "
        "print(json.dumps(p.parameters(p), default=str))"), str(HERE), str(Path(inc["dir"]) / "skills"),
        json.dumps(inc["set"])], cwd=REPO, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))  # fmt: skip
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}


def context(inc: dict, extra: dict) -> dict:
    es = entries()
    return dict(
        family=FAMILY,
        incumbent=inc,
        budget=dict(
            gate_calls=len([e for e in es if e.get("step") == "gate" and not e.get("void")]),
            gate_calls_max=MAX_GATE_CALLS,
            cost_usd=round(sum(e.get("cost_usd", 0.0) for e in es), 2),
            cost_cap_usd=COST_CAP,
            stop_rate=STOP_RATE,
        ),  # fmt: skip
        history=[{k: v for k, v in e.items() if k not in ("usage",)} for e in es],
        rejected_hashes=sorted(
            {e["hash"] for e in es if e.get("step") == "gate" and not e.get("accepted") and e.get("hash")}
        ),
        **extra,
    )


def put_skills(src: Path) -> None:
    """The sandbox's skills package := `src` (the incumbent's snapshot)."""
    shutil.rmtree(SKILLS)
    shutil.copytree(src, SKILLS, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def changed_files() -> list[str]:
    r = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=REPO, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        f = line[3:].split(" -> ")[-1].strip('"')
        if not f.startswith(("outputs/", "__pycache__")) and "__pycache__" not in f:
            out.append(f)
    return sorted(out)


def iterate(n: int, meta: dict, stub: bool) -> str | None:
    it_dir = STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    inc = incumbent()
    put_skills(Path(inc["dir"]) / "skills")
    base_changes = changed_files()
    base = diag_incumbent(inc, it_dir)
    common = dict(RULES=RULES, PLAN=PLAN, LOOP=LOOP, SIM_PY=SIM_PY, DIAG_INC=base, SKILLS=SKILLS, FAMILY=FAMILY,
                  FAMILY_FILE=HERE / "families" / f"{FAMILY}.json", POOL=LOOP / "pool_diag", INC_DIR=inc["dir"])  # fmt: skip
    extra = dict(diag_summary=summarize(base), parameters=parameters_of(inc), focus=jload(STATE / "focus.json", {}))
    ctx = it_dir / "reviewer_context.json"
    jdump(ctx, context(inc, extra))
    prompt = fill((HERE / "prompts" / "reviewer.md").read_text(), CONTEXT=ctx, **common)
    stub_h = dict(notes="stub", hypotheses=[dict(id="H1", claim="stub", evidence="stub",
                  delta={"set": {"PLAN_HELD": "pick_simple"}}, predicted_fixed=[], predicted_broken=[])])  # fmt: skip
    hyps, acct = claude(
        "reviewer", prompt, HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent", stub_h if stub else None
    )
    jdump(it_dir / "hypotheses.json", hyps)
    leaks = transcript_leaks(acct["session"])
    ledger(dict(step="review", iter=n, n_hypotheses=len(hyps["hypotheses"]), void=bool(leaks), leaks=leaks, **acct))
    if leaks:
        return f"reviewer of iteration {n} named off-limits paths ({', '.join(leaks)})"
    return after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub) -> str | None:
    """Prober -> Manager -> Engineer -> Gate for a reviewed iteration (also `run --reprobe N`)."""
    pr = probe(hyps["hypotheses"][:MAX_HYPOTHESES], inc, base, it_dir)
    jdump(it_dir / "probe.json", pr)
    ledger(dict(step="probe", iter=n, results=[{k: v for k, v in p.items() if k != "summary"} for p in pr]))
    ctx = it_dir / "manager_context.json"
    jdump(ctx, context(inc, dict(extra, hypotheses=hyps, probe=pr)))
    prompt = fill((HERE / "prompts" / "manager.md").read_text(), CONTEXT=ctx, **common)
    stub_m = dict(decision="engineer", hypothesis="H1", reason="stub", summary_zh="桩测试")
    dec, acct = claude(
        "manager", prompt, MANAGER_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent,Bash", stub_m if stub else None
    )
    jdump(it_dir / "manager.json", dec)
    leaks = transcript_leaks(acct["session"])
    ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), reason=dec["reason"],
                summary_zh=dec["summary_zh"], escalation=dec.get("escalation"), infeasible=dec.get("infeasible"),
                leaks=leaks, **acct))  # fmt: skip
    if leaks:
        return f"manager of iteration {n} named off-limits paths ({', '.join(leaks)})"
    if dec["decision"] == "stop":
        return "manager: stop"
    if dec["decision"] == "review":
        jdump(STATE / "focus.json", dict(iter=n, focus=dec.get("focus", ""), reason=dec["reason"]))
        return None
    chosen = next((h for h in hyps["hypotheses"] if h["id"] == dec.get("hypothesis")), None)
    ctx = it_dir / "engineer_context.json"
    jdump(ctx, context(inc, dict(hypothesis=chosen, probe=[p for p in pr if p.get("id") == dec.get("hypothesis")],
                                 manager=dec)))  # fmt: skip
    prompt = fill((HERE / "prompts" / "engineer.md").read_text(), CONTEXT=ctx, ITER=it_dir, **common)
    stub_e = dict(mechanism="stub", files_changed=[], set={**inc["set"], **(chosen or {}).get("delta", {}).get("set", {})},
                  tests="none")  # fmt: skip
    eng, acct = claude("engineer", prompt, ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_e if stub else None)
    return after_engineer(n, it_dir, inc, eng, acct, base_changes, stub)


def after_engineer(
    n: int, it_dir: Path, inc: dict, eng: dict, acct: dict, base_changes: list, stub: bool
) -> str | None:
    rel = SKILLS.relative_to(REPO).as_posix() + "/"
    now_changed = changed_files()
    changed = [f for f in now_changed if f not in base_changes or f.startswith(rel)]
    stray = [f for f in now_changed if f not in base_changes and not f.startswith(rel)]
    leaks = transcript_leaks(acct["session"])
    ledger(dict(step="engineer", iter=n, changed=changed, stray=stray, mechanism=eng["mechanism"], set=eng["set"],
                void=bool(leaks), leaks=leaks, **acct))  # fmt: skip
    cand = it_dir / "candidate"
    if cand.exists():
        shutil.rmtree(cand)
    shutil.copytree(SKILLS, cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (cand / "candidate.json").write_text(json.dumps({"set": eng.get("set") or {}}, indent=1))
    patch = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(Path(inc["dir"]) / "skills"),
                            str(cand / "skills")], capture_output=True, text=True).stdout  # fmt: skip
    (it_dir / "code.patch").write_text(patch)
    for f in stray:  # back as they were
        if (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=REPO, capture_output=True)
        elif (REPO / f).exists():
            (REPO / f).unlink()
    put_skills(Path(inc["dir"]) / "skills")
    if leaks:
        return f"engineer of iteration {n} named off-limits paths ({', '.join(leaks)})"
    if stray:
        ledger(
            dict(
                step="gate",
                iter=n,
                accepted=False,
                void=True,
                reason=f"engineer changed files outside the skills package: {stray}",
            )
        )
        return None
    gate_args = [str(SIM_PY), str(HERE / "gate.py"), FAMILY, "--gate", str(GATE_DIR), "--pool", str(accept_pool()),
                 "--candidate", str(cand), "--gpus", GPUS]  # fmt: skip
    if stub:
        gate_args += ["--dry-run"]
    share = max(1, PROCS * 5 // 12)
    gate_args += ["--procs", str(PROCS - share if not stub else PROCS)]
    stop, side = threading.Event(), None
    if not stub:  # the candidate's diagnostic rounds beside the gate (the next incumbent's, if accepted)
        pre = LOOP / "diag" / f"candidate_{candidate_hash(cand)}"
        side = threading.Thread(target=B.run_jobs, args=([B.Job(cand / "skills", pre, overrides(cand), None, True)],
                                FAMILY, LOOP / "pool_diag", share, it_dir / "prober.log", GPUS), kwargs=dict(stop=stop),
                                daemon=True)  # fmt: skip
        side.start()
    r = subprocess.run(gate_args, cwd=REPO, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))
    (it_dir / "gate.log").write_text(r.stdout + r.stderr)
    verdict = json.loads((GATE_DIR / "ledger.jsonl").read_text().splitlines()[-1])
    jdump(it_dir / "gate.json", verdict)
    if side is not None:
        if not verdict["accepted"]:
            stop.set()
        side.join()
    ledger(dict(step="gate", iter=n, accepted=verdict["accepted"], reason=verdict["reason"], hash=verdict["hash"],
                result=verdict.get("result"), stage1=verdict.get("stage1"), void=stub))  # fmt: skip
    return None


def reprobe(n: int, stub: bool) -> str | None:
    """Re-run iteration n's probes on its Reviewer's unchanged hypotheses and continue from the Manager: for probes
    lost to the harness or never run because a driver bug stopped the loop (the hypotheses never saw a probe result,
    so they stand). The earlier Manager call is voided
    in the ledger; the human fix that made this necessary goes in interventions.jsonl."""
    it_dir = STATE / f"iter_{n:02d}"
    hyps = jload(it_dir / "hypotheses.json")
    if hyps is None:
        raise SystemExit(f"{it_dir} has no hypotheses")
    for f in ("probe", "manager.json", "probe.json"):
        p = it_dir / f
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.rename(p.with_name(p.name + ".void"))
    (STATE / "focus.json").unlink(missing_ok=True)
    ledger(dict(step="void", iter=n, reason="re-probing the same hypotheses after a driver fix (interventions.jsonl)"))
    inc = incumbent()
    put_skills(Path(inc["dir"]) / "skills")
    base_changes = changed_files()
    base = diag_incumbent(inc, it_dir)
    common = dict(RULES=RULES, PLAN=PLAN, LOOP=LOOP, SIM_PY=SIM_PY, DIAG_INC=base, SKILLS=SKILLS, FAMILY=FAMILY,
                  FAMILY_FILE=HERE / "families" / f"{FAMILY}.json", POOL=LOOP / "pool_diag", INC_DIR=inc["dir"])  # fmt: skip
    extra = dict(diag_summary=summarize(base), parameters=parameters_of(inc), focus={})
    return after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def cmd_run(a) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    meta = jload(STATE / "meta.json") or dict(started=now(), started_s=time.time(), cost_cap_usd=COST_CAP, model=MODEL,
                                              family=FAMILY, max_wall_h=MAX_WALL_H)  # fmt: skip
    meta.update(cost_cap_usd=COST_CAP, model=MODEL, max_wall_h=MAX_WALL_H)  # this run's limits (roles read meta.json)
    jdump(STATE / "meta.json", meta)
    if not (LOOP / "pool_diag").exists() or not accept_pool().exists():
        raise SystemExit("draw the pools first (family.py draw ... diag / accept)")
    n = len(list(STATE.glob("iter_*")))
    iters = a.max_iters
    if a.reprobe is not None:  # the re-probed iteration counts as one of --max-iters
        why = reprobe(a.reprobe, a.stub)
        if why:
            ledger(dict(step="stop", reason=why))
            print("stop:", why)
            return 0
        iters -= 1
    for _ in range(iters):
        why = stop_reason(meta)
        if why is None:
            try:
                why = iterate(n, meta, a.stub)
            except RoleFailed as e:
                why = f"role call failed: {e}"
            n += 1
        if why:
            ledger(dict(step="stop", reason=why))
            print("stop:", why)
            return 0
    return 0


def cmd_status(_a) -> int:
    es = entries()
    meta = jload(STATE / "meta.json", {})
    print(f"family {FAMILY}; started {meta.get('started')}; entries {len(es)}; "
          f"cost {sum(e.get('cost_usd', 0) for e in es):.2f} USD")  # fmt: skip
    for e in es[-12:]:
        print(json.dumps({k: v for k, v in e.items() if k not in ("usage", "results")}, ensure_ascii=False)[:300])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("family")
    r.add_argument("--max-iters", type=int, default=100)
    r.add_argument("--stub", action="store_true", help="canned role outputs, gate --dry-run: plumbing only")
    r.add_argument(
        "--reprobe", type=int, default=None, help="re-probe iteration N's hypotheses first (probes lost or not run)"
    )
    s = sub.add_parser("status")
    s.add_argument("family")
    a = ap.parse_args(argv)
    configure(a.family)
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
