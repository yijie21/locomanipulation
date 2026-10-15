"""TODO 25.9.1 loop driver (docs/handoff/scalebfm-warehouse-rsi-20261014.md §5, §6, §9): one iteration =
Reviewer (claude -p) -> Prober (bfm_round.py forks on the diagnostic pool) -> Manager (claude -p) -> Engineer
(claude -p) -> Gate (accept_gate.py) -> ledger. The state lives in files under outputs/scalebfm_warehouse_rsi_20261014/loop/,
so the driver can stop and resume at any iteration.

Roles run headless: Opus 5.5, hooks off, auto memory off, no MCP servers, so the project knowledge they get is
rules.md, the plan and the repository's CLAUDE.md / AGENTS.md contract (D3). The acceptance and sealed pools and the
gate live outside the loop's tree (`..._hidden/`); every role transcript is grepped for that path afterwards and a
mention voids the iteration's gate call.

Hard stops (§9), checked before each step: gate calls >= 12, wall clock >= 36 h since the loop started, model cost
>= COST_CAP USD, 3 consecutive rejections with no new hypothesis, a Manager `stop`, a SONIC stack running.

Phase 4 (plan §17) runs the loop cold, once per arm, each in its own sandbox checkout built by phase4_sandbox.py:
`--arm A` is this loop without phase 1's results; `--arm B` is the flat loop -- the Reviewer reads the trajectories and
proposes changes directly, with no Prober, no envelope and no handoff records. In an arm the transcript audit also
voids a role call that reads the origin checkout or ~/.claude (where the answers are).

Run with the sim venv (the spec loader imports the round's modules):
    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py run > /dev/null 2>&1 &
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py status
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py run --stub --max-iters 1 --state /tmp/x   # plumbing, no model calls
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py run --arm B --gpu 0   # in an arm's sandbox
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOOP = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014"
HIDDEN = (
    REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden"
)  # acceptance / sealed pools and the gate: no role reads it
STATE = LOOP / "loop"
PLAN = REPO / "docs" / "handoff" / "scalebfm-warehouse-rsi-20261014.md"
RULES = HERE / "rules.md"
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
sys.path.insert(0, str(HERE))
import extensions  # noqa: E402
from bfm_round import load_spec, spec_hash  # noqa: E402

MODEL = os.environ.get("RSI_MODEL", "claude-opus-5-5")
COST_CAP = float(os.environ.get("RSI_COST_CAP_USD", "400"))
MAX_GATE_CALLS = 12
MAX_WALL_H = 36.0
MAX_REJECT_STREAK = 3
MAX_HYPOTHESES = 4
SONIC_MARGIN = 5  # plan §9: stop when the accepted incumbent wins >= SONIC + 5 on the acceptance pool
PROCS = 4  # ScaleBFM processes per loop (run --procs); rounds are shard-invariant (phase 0), so only wall time changes
SMOKE = "sonic_s7000_04"
# files the Engineer may change (§7, D2): the extension hooks and the crate task code (default-off fields only)
ENGINEER_PATHS = ("scripts/experiments/scalebfm_warehouse_rsi_20261014/extensions.py", "src/locomani/tasks/crate/")
GATE_DIR = HIDDEN / "gate"
# phase 4 (plan §17), set by `run --arm/--gpu`: None is the phase-2 loop
ARM: str | None = None
GPU = "1"
# a sandbox links third_party to the origin checkout; anything else under the origin is off limits in an arm
ORIGIN = (REPO / "third_party").resolve().parent
ISOLATION = (
    "\n\nThis checkout ({REPO}) and your scratch directory ({TMP}) are the whole world for this run. Do not read "
    "anything outside them: no other checkout of this project, nothing under ~/.claude, no path through third_party/.. "
    "A role call that does is void.\n"
)

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
                "required": ["id", "claim", "evidence", "phase", "delta", "predicted_fixed", "predicted_broken"],
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "phase": {"enum": ["walk", "ready", "reach"]},
                    "delta": {"type": "object"},
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
    "required": ["candidate", "mechanism", "files_changed", "tests"],
    "properties": {
        "candidate": {"type": "string"},
        "mechanism": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests": {"type": "string"},
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def ledger(entry: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps({"t": now(), **entry}, ensure_ascii=False) + "\n")


def entries() -> list[dict]:
    p = STATE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def sonic_up() -> bool:
    r = subprocess.run(["pgrep", "-f", "g1_deploy_onnx_ref|run_agent_server"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def sonic_accept_ok() -> int:
    """SONIC's successes on the acceptance pool (the driver may read the hidden tree; roles may not)."""
    return sum(bool(jload(p)["ok"]) for p in (HIDDEN / "pool_accept" / "states").glob("*.json"))


def stop_reason(meta: dict) -> str | None:
    es = entries()
    accepted = [e for e in es if e.get("step") == "gate" and e.get("accepted") and not e.get("void")]
    if accepted:
        best, sonic = accepted[-1]["result"]["candidate_ok"], sonic_accept_ok()
        if best >= sonic + SONIC_MARGIN:
            return f"the incumbent's {best} acceptance successes reach SONIC's {sonic} + {SONIC_MARGIN}: phase 3"
    gates = [e for e in es if e.get("step") == "gate" and not e.get("void")]
    if len(gates) >= MAX_GATE_CALLS:
        return f"gate calls {len(gates)} >= {MAX_GATE_CALLS}"
    if (time.time() - meta["started_s"]) / 3600 >= MAX_WALL_H:
        return f"wall clock >= {MAX_WALL_H} h"
    cost = sum(e.get("cost_usd", 0.0) for e in es)
    if cost >= COST_CAP:
        return f"model cost {cost:.0f} >= {COST_CAP:.0f} USD"
    streak = 0  # rejections since the last acceptance
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


# -- model calls ------------------------------------------------------------------------------------------------


def fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


class RoleFailed(RuntimeError):
    pass


def role_tmp() -> Path:
    return Path("/workspace/tmp") / f"rsi_phase4_{ARM}" if ARM else Path("/workspace/tmp")


def claude(role: str, prompt: str, schema: dict, it_dir: Path, tools: str, stub=None) -> tuple[dict, dict]:
    """One headless role call; returns (structured output, accounting)."""
    if ARM:
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
    t0 = time.time()
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=REPO, env=env, timeout=4 * 3600)
    (it_dir / f"{role}_raw.json").write_text(r.stdout or r.stderr)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RoleFailed(f"{role}: no JSON from claude -p (rc {r.returncode}): {(r.stderr or r.stdout)[-300:]}") from e
    if out.get("is_error"):
        raise RoleFailed(f"{role}: {out.get('subtype')} {str(out.get('result'))[-300:]}")
    acct = dict(
        cost_usd=float(out.get("total_cost_usd") or 0.0), session=out.get("session_id"), seconds=round(time.time() - t0)
    )
    acct["usage"] = out.get("usage")
    so = out.get("structured_output")
    if so is None:
        text = out.get("result", "")
        so = json.loads(text[text.find("{") : text.rfind("}") + 1])
    return so, acct


def leak_patterns() -> dict[str, str]:
    """What a role's tool inputs must not name: the gate's tree always; in an arm also the origin checkout (other than
    its third_party, which the sandbox links to), a climb out through third_party/.., and ~/.claude."""
    pats = {"gate": re.escape(HIDDEN.name)}
    if ARM and ORIGIN != REPO:
        pats["origin"] = re.escape(str(ORIGIN)) + r"(?!/third_party)"
        pats["climb"] = r"third_party/\.\."
        pats["claude_home"] = r"\.claude/|~/\.claude"
    return pats


def leaks_in(lines, pats: dict[str, str]) -> list[str]:
    """Names of the patterns that occur in a tool_use line of a transcript."""
    found = set()
    for line in lines:
        if '"tool_use"' not in line:
            continue
        found |= {k for k, p in pats.items() if re.search(p, line)}
    return sorted(found)


def transcript_leaks(session: str | None) -> list[str]:
    """The off-limits paths the role's session named in a tool input (empty: none, or no transcript)."""
    if not session:
        return []
    proj = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(REPO))
    p = proj / f"{session}.jsonl"
    if not p.exists():
        return []
    with p.open() as fh:
        return leaks_in(fh, leak_patterns())


# -- steps --------------------------------------------------------------------------------------------------------


def run_batch(jobs: list[tuple], log: Path, gpu: str = "1", script: Path | None = None) -> None:
    """Rounds on the diagnostic pool for several (spec, out, fork, only) jobs. Each job is split into PROCS shards and all
    shards share one pool of PROCS processes, so the next job's shards start as soon as a slot frees instead of after
    the previous job's slowest shard."""
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=gpu, TMPDIR="/workspace/tmp")
    env["NUMPY_MADVISE_HUGEPAGE"] = "0"
    pool = LOOP / "pool_diag" / "states"
    queue = []
    for spec, out, fork, only in jobs:
        for i in range(PROCS):
            cmd = [str(SIM_PY), str(script or HERE / "bfm_round.py"), str(pool), "--spec", str(spec), "--out", str(out),
                   "--shard", f"{i}/{PROCS}"]  # fmt: skip
            if fork:
                cmd += ["--fork", fork]
            if only:
                cmd += ["--only", ",".join(only)]
            queue.append(cmd)
    running: list[subprocess.Popen] = []
    with log.open("a") as fh:
        while queue or running:
            while queue and len(running) < PROCS:
                running.append(subprocess.Popen(queue.pop(0), cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT))
            time.sleep(0.5)
            running = [p for p in running if p.poll() is None]


def run_rounds(
    spec: Path, out: Path, log: Path, fork: str | None = None, only: list[str] | None = None, gpu: str = "1"
) -> None:
    run_batch([(spec, out, fork, only)], log, gpu)


def outcomes(d: Path) -> dict:
    return {p.stem.replace("bfm", "sonic", 1): bool(jload(p)["ok"]) for p in sorted(d.glob("bfm_*.json"))}


def incumbent() -> dict:
    """The gate's current incumbent, its spec copied into the loop's tree (roles must not open the gate's files)."""
    inc = jload(GATE_DIR / "incumbent.json") or dict(spec=str(HERE / "incumbent.yaml"))
    spec = load_spec(inc["spec"])
    h = spec_hash(spec)
    path = LOOP / "incumbents" / f"{h}.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(Path(inc["spec"]).read_text())
    return dict(path=str(path), spec=spec, hash=h, since_gate_call=inc.get("since"))


def diag_incumbent(inc: dict, it_dir: Path) -> Path:
    """The incumbent's rounds on the diagnostic pool (once per incumbent) and their handoff records (none in arm B)."""
    d = LOOP / "diag" / f"incumbent_{inc['hash']}"
    if len(list(d.glob("bfm_*.json"))) < len(list((LOOP / "pool_diag" / "states").glob("*.json"))):
        run_rounds(Path(inc["path"]), d, it_dir / "prober.log", gpu=GPU)
    rec = d / "records.jsonl"
    if not rec.exists() and ARM != "B":
        subprocess.run([str(SIM_PY), str(HERE / "handoff.py"), str(LOOP / "pool_diag" / "states"), str(d), "--out", str(rec)],
                       cwd=REPO, check=True, capture_output=True)  # fmt: skip
    return d


def probe(hyps: list[dict], inc: dict, base: Path, it_dir: Path, gpu: str = "1") -> list[dict]:
    """Each hypothesis's intervention, closed loop from its fork phase, on all diagnostic starts (one shared pool)."""
    before = outcomes(base)
    jobs = []
    for h in hyps:
        delta = it_dir / f"delta_{h['id']}.yaml"
        delta.write_text(json.dumps(h["delta"]))
        jobs.append((Path(inc["path"]), it_dir / "probe" / h["id"], f"{h['phase']}:{delta}", None))
    try:
        run_batch(jobs, it_dir / "prober.log", gpu)
        failed = None
    except Exception as e:
        failed = f"{type(e).__name__}: {e}"
    res = []
    for h in hyps:
        if failed:
            res.append(dict(id=h["id"], error=failed))
            continue
        # a delta the runner refuses (unknown key, frozen field) exits without rounds: recorded, not fatal
        after = outcomes(it_dir / "probe" / h["id"])
        if not after:
            res.append(dict(id=h["id"], error="no rounds ran (see prober.log)"))
            continue
        fixed = sorted(k for k in after if after[k] and not before.get(k))
        broken = sorted(k for k in after if before.get(k) and not after[k])
        pf, pb = set(h["predicted_fixed"]), set(h["predicted_broken"])
        res.append(dict(
            id=h["id"], n=len(after), ok_before=sum(before.values()), ok_after=sum(after.values()),
            fixed=fixed, broken=broken,
            predicted_fixed_hit=len(pf & set(fixed)), predicted_fixed_n=len(pf), observed_fixed_n=len(fixed),
            predicted_broken_hit=len(pb & set(broken)), predicted_broken_n=len(pb),
        ))  # fmt: skip
    return res


def context(inc: dict, extra: dict) -> dict:
    es = entries()
    ctx = dict(
        incumbent=inc,
        budget=dict(
            gate_calls=len([e for e in es if e.get("step") == "gate" and not e.get("void")]),
            gate_calls_max=MAX_GATE_CALLS,
            cost_usd=round(sum(e.get("cost_usd", 0.0) for e in es), 2),
            cost_cap_usd=COST_CAP,
        ),
        history=[{k: v for k, v in e.items() if k not in ("usage",)} for e in es],
        rejected_hashes=sorted(
            {e["hash"] for e in es if e.get("step") == "gate" and not e.get("accepted") and e.get("hash")}
        ),
        **extra,
    )
    if ARM is None:  # phase 1's interventions on the diagnostic pool (run by the main session before the loop)
        ctx["phase1_probes"] = jload(LOOP / "phase1" / "probe.json", [])
    return ctx


SNAPSHOT_SKIP = ("outputs/", "node_modules/", "papers/", ".serena/", "assets/")


def git_snapshot() -> dict:
    """Content hashes of every modified or untracked file outside the runtime and bulk paths: the Engineer's edits are
    the difference between two snapshots, whatever else is in flight in the tree."""
    r = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=REPO, capture_output=True, text=True)
    snap = {}
    for line in r.stdout.splitlines():
        f = line[3:].split(" -> ")[-1].strip('"')
        if f.startswith(SNAPSHOT_SKIP):
            continue
        p = REPO / f
        snap[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return snap


def iterate(n: int, meta: dict, stub: bool) -> str | None:
    it_dir = STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    inc = incumbent()
    if sonic_up():
        return "a SONIC stack is running (ScaleBFM must not share the machine)"
    base = diag_incumbent(inc, it_dir)
    importlib.reload(extensions)
    ext = sorted(extensions.HOOKS)
    common = dict(
        RULES=RULES,
        PLAN=PLAN,
        LOOP=LOOP,
        SIM_PY=SIM_PY,
        DIAG_INC=base,
        RECORDS=base / "records.jsonl",
        EXTENSIONS=ext or "none yet",
        GPU=GPU,
    )
    focus = jload(it_dir.parent / "focus.json", {})
    # Reviewer
    ctx = it_dir / "reviewer_context.json"
    jdump(ctx, context(inc, dict(focus=focus)))
    prompt = fill(prompt_file("reviewer").read_text(), CONTEXT=ctx, **common)
    stub_h = dict(notes="stub", hypotheses=[dict(id="H1", claim="stub: tracking gain 1.0", evidence="stub", phase="reach",
                  delta={"set": {"visual.tracking_gain": 1.0}}, predicted_fixed=[], predicted_broken=["sonic_s7000_04"])])  # fmt: skip
    hyps, acct = claude(
        "reviewer", prompt, HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent", stub_h if stub else None
    )
    jdump(it_dir / "hypotheses.json", hyps)
    leaks = transcript_leaks(acct["session"])
    review_void = bool(leaks)
    ledger(dict(step="review", iter=n, n_hypotheses=len(hyps["hypotheses"]), void=review_void, leaks=leaks, **acct))
    if ARM and leaks:
        return contaminated("reviewer", n, leaks)
    # Prober (the flat arm has none: the Reviewer's changes go to the gate untested)
    pr = []
    if ARM != "B":
        pr = probe(hyps["hypotheses"][:MAX_HYPOTHESES], inc, base, it_dir, gpu=GPU)
        jdump(it_dir / "probe.json", pr)
        ledger(dict(step="probe", iter=n, results=pr))
    # Manager
    if sonic_up():
        return "a SONIC stack is running"
    ctx = it_dir / "manager_context.json"
    jdump(ctx, context(inc, dict(hypotheses=hyps, probe=pr)))
    prompt = fill(prompt_file("manager").read_text(), CONTEXT=ctx, **common)
    stub_m = dict(decision="engineer", hypothesis="H1", reason="stub", summary_zh="桩测试")
    dec, acct = claude(
        "manager", prompt, MANAGER_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent,Bash", stub_m if stub else None
    )
    jdump(it_dir / "manager.json", dec)
    leaks = transcript_leaks(acct["session"])
    ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), reason=dec["reason"],
                summary_zh=dec["summary_zh"], escalation=dec.get("escalation"), infeasible=dec.get("infeasible"),
                leaks=leaks, **acct))  # fmt: skip
    if ARM and leaks:
        return contaminated("manager", n, leaks)
    if dec["decision"] == "stop":
        return "manager: stop"
    if dec["decision"] == "review":
        jdump(it_dir.parent / "focus.json", dict(iter=n, focus=dec.get("focus", ""), reason=dec["reason"]))
        return None
    # Engineer
    cand = it_dir / "candidate.yaml"
    ctx = it_dir / "engineer_context.json"
    chosen = next((h for h in hyps["hypotheses"] if h["id"] == dec.get("hypothesis")), None)
    jdump(
        ctx,
        context(
            inc, dict(hypothesis=chosen, probe=[p for p in pr if p.get("id") == dec.get("hypothesis")], manager=dec)
        ),
    )
    prompt = fill(prompt_file("engineer").read_text(), CONTEXT=ctx, CANDIDATE=cand, ITER=it_dir, SMOKE=SMOKE, **common)
    before = git_snapshot()
    saved = {f: (REPO / f).read_bytes() for f in before if (REPO / f).is_file()}
    stub_e = dict(candidate=str(cand), mechanism="stub", files_changed=[], tests="none")
    if stub:
        cand.write_text("mode: 2\nstance: near\nset:\n  visual.tracking_gain: 1.0\n")
    eng, acct = claude("engineer", prompt, ENGINEER_SCHEMA, it_dir, "NotebookEdit,Agent", stub_e if stub else None)
    after = git_snapshot()
    changed = sorted(
        f for f in set(before) | set(after) if before.get(f) != after.get(f) and not f.startswith("outputs/")
    )
    stray = [f for f in changed if not f.startswith(ENGINEER_PATHS)]
    leaks = transcript_leaks(acct["session"])
    void = review_void or bool(leaks)
    ledger(dict(step="engineer", iter=n, changed=changed, stray=stray, mechanism=eng["mechanism"], void=void,
                leaks=leaks, **acct))  # fmt: skip
    if ARM and leaks:
        restore(changed, saved)
        return contaminated("engineer", n, leaks)
    if stray or not cand.exists():
        restore(changed, saved)
        ledger(
            dict(
                step="gate", iter=n, accepted=False, void=True, reason=f"engineer changed {stray} or wrote no candidate"
            )
        )
        return None
    if changed:
        patch = subprocess.run(["git", "diff", "--", *changed], cwd=REPO, capture_output=True, text=True).stdout
        (it_dir / "code.patch").write_text(patch)
    # Gate
    if sonic_up():
        return "a SONIC stack is running"
    gate_dir = STATE / "gate_stub" if stub else GATE_DIR
    gate_args = [
        str(SIM_PY),
        str(HERE / "accept_gate.py"),
        "--candidate",
        str(cand),
        "--procs",
        str(PROCS),
        "--gate-dir",
        str(gate_dir),
    ]
    if stub:
        gate_args += ["--dry-run", "--pool", str(LOOP / "pool_diag" / "states")]
    r = subprocess.run(gate_args, cwd=REPO, capture_output=True, text=True, env=dict(os.environ, GATE_GPU=GPU))
    (it_dir / "gate.log").write_text(r.stdout + r.stderr)
    verdict = json.loads((gate_dir / "ledger.jsonl").read_text().splitlines()[-1])
    jdump(it_dir / "gate.json", verdict)
    ledger(dict(step="gate", iter=n, accepted=verdict["accepted"], reason=verdict["reason"], hash=verdict["hash"],
                result=verdict.get("result"), stage1=verdict.get("stage1"), void=void or stub))  # fmt: skip
    if not verdict["accepted"] and changed:
        restore(changed, saved)
    return None


def prompt_file(role: str) -> Path:
    """The role's prompt: phase 2's, or the arm's own where phase 4 needs one (prompts/phase4/<role>_<arm>.md)."""
    own = HERE / "prompts" / "phase4" / f"{role}_{ARM}.md"
    return own if ARM and own.exists() else HERE / "prompts" / f"{role}.md"


def contaminated(role: str, n: int, leaks: list[str]) -> str:
    """In an arm a role that read an off-limits path ends the arm: its later steps would carry what it read."""
    return f"{role} of iteration {n} named off-limits paths ({', '.join(leaks)}): the arm is contaminated from here"


def restore(changed: list[str], saved: dict) -> None:
    """Put the Engineer's files back as they were before the call (other sessions' edits elsewhere are untouched).
    `saved` holds the files that were already modified or untracked; a tracked file that was clean goes back to its
    committed content, and only a file the Engineer created is removed (phase 4 found the clean tracked case deleted)."""
    for f in changed:
        p = REPO / f
        if f in saved:
            p.write_bytes(saved[f])
        elif (
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", f], cwd=REPO, capture_output=True).returncode
            == 0
        ):
            subprocess.run(["git", "checkout", "--", f], cwd=REPO, check=True, capture_output=True)
        elif p.exists():
            p.unlink()


def cmd_run(a) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    meta = jload(STATE / "meta.json") or dict(
        started=now(), started_s=time.time(), cost_cap_usd=COST_CAP, model=MODEL, arm=ARM, gpu=GPU
    )
    if meta.get("arm") != ARM:
        raise SystemExit(f"{STATE} belongs to arm {meta.get('arm')}, not {ARM}")
    if ARM and ORIGIN == REPO:
        raise SystemExit("an arm runs in its sandbox (phase4_sandbox.py), not in the origin checkout")
    jdump(STATE / "meta.json", meta)
    n = len(list(STATE.glob("iter_*")))
    done = 0
    while done < a.max_iters:
        why = stop_reason(meta)
        if why:
            ledger(dict(step="stop", reason=why))
            print("stop:", why)
            return 0
        try:
            why = iterate(n, meta, a.stub)
        except RoleFailed as e:
            why = f"role call failed: {e}"
        if why:
            ledger(dict(step="stop", reason=why))
            print("stop:", why)
            return 0
        n += 1
        done += 1
    return 0


def cmd_status(_a) -> int:
    es = entries()
    meta = jload(STATE / "meta.json", {})
    print(f"arm {meta.get('arm')} on GPU {meta.get('gpu')}; started {meta.get('started')}; entries {len(es)}; "
          f"cost {sum(e.get('cost_usd', 0) for e in es):.2f} USD")  # fmt: skip
    for e in es[-12:]:
        print(json.dumps({k: v for k, v in e.items() if k not in ("usage", "results")}, ensure_ascii=False)[:300])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-iters", type=int, default=100)
    r.add_argument("--stub", action="store_true", help="canned role outputs, gate --dry-run: plumbing only")
    r.add_argument("--state", type=Path, default=None, help="loop state directory (tests)")
    r.add_argument("--arm", choices=("A", "B"), default=None, help="phase 4 (plan §17): A cold, B cold and flat")
    r.add_argument("--gpu", default="1", help="ScaleBFM's GPU for the diagnostic rounds, the probes and the gate")
    r.add_argument(
        "--procs", type=int, default=4, help="ScaleBFM processes (wall time only; rounds are shard-invariant)"
    )
    sub.add_parser("status")
    a = ap.parse_args(argv)
    global STATE, ARM, GPU, PROCS
    if getattr(a, "state", None):
        STATE = a.state
    if a.cmd == "run":
        ARM, GPU, PROCS = a.arm, a.gpu, a.procs
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
