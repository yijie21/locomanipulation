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

Run with the sim venv (the spec loader imports the round's modules):
    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py run > /dev/null 2>&1 &
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py status
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/loop.py run --stub --max-iters 1 --state /tmp/x   # plumbing, no model calls
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
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
PROCS = 4
SMOKE = "sonic_s7000_04"
# files the Engineer may change (§7, D2): the extension hooks and the crate task code (default-off fields only)
ENGINEER_PATHS = ("scripts/experiments/scalebfm_warehouse_rsi_20261014/extensions.py", "src/locomani/tasks/crate/")
GATE_DIR = HIDDEN / "gate"

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


def claude(role: str, prompt: str, schema: dict, it_dir: Path, tools: str, stub=None) -> tuple[dict, dict]:
    """One headless role call; returns (structured output, accounting)."""
    (it_dir / f"{role}_prompt.md").write_text(prompt)
    if stub is not None:
        return stub, dict(cost_usd=0.0, session=None, seconds=0.0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", TMPDIR="/workspace/tmp")
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


def transcript_touches_gate(session: str | None) -> bool:
    """Did the role's session read anything under gate/ (any tool input naming the path)?"""
    if not session:
        return False
    proj = Path.home() / ".claude" / "projects" / str(REPO).replace("/", "-")
    p = proj / f"{session}.jsonl"
    if not p.exists():
        return False
    return any('"tool_use"' in line and HIDDEN.name in line for line in p.open())


# -- steps --------------------------------------------------------------------------------------------------------


def run_rounds(
    spec: Path, out: Path, log: Path, fork: str | None = None, only: list[str] | None = None, gpu: str = "1"
) -> None:
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=gpu, TMPDIR="/workspace/tmp")
    env["NUMPY_MADVISE_HUGEPAGE"] = "0"
    pool = LOOP / "pool_diag" / "states"
    ps = []
    with log.open("a") as fh:
        for i in range(PROCS):
            cmd = [
                str(SIM_PY),
                str(HERE / "bfm_round.py"),
                str(pool),
                "--spec",
                str(spec),
                "--out",
                str(out),
                "--shard",
                f"{i}/{PROCS}",
            ]
            if fork:
                cmd += ["--fork", fork]
            if only:
                cmd += ["--only", ",".join(only)]
            ps.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT))
        for p in ps:
            p.wait()


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
    """The incumbent's rounds on the diagnostic pool (once per incumbent) and their handoff records."""
    d = LOOP / "diag" / f"incumbent_{inc['hash']}"
    if len(list(d.glob("bfm_*.json"))) < len(list((LOOP / "pool_diag" / "states").glob("*.json"))):
        run_rounds(Path(inc["path"]), d, it_dir / "prober.log")
    rec = d / "records.jsonl"
    if not rec.exists():
        subprocess.run([str(SIM_PY), str(HERE / "handoff.py"), str(LOOP / "pool_diag" / "states"), str(d), "--out", str(rec)],
                       cwd=REPO, check=True, capture_output=True)  # fmt: skip
    return d


def probe(hyps: list[dict], inc: dict, base: Path, it_dir: Path, gpu: str = "1") -> list[dict]:
    """Each hypothesis's intervention, closed loop from its fork phase, on all diagnostic starts."""
    before = outcomes(base)
    res = []
    for h in hyps:
        delta = it_dir / f"delta_{h['id']}.yaml"
        delta.write_text(json.dumps(h["delta"]))
        out = it_dir / "probe" / h["id"]
        try:
            run_rounds(Path(inc["path"]), out, it_dir / "prober.log", fork=f"{h['phase']}:{delta}", gpu=gpu)
            after = outcomes(out)
        except Exception as e:  # a delta the runner refuses (unknown key, frozen field): recorded, not fatal
            res.append(dict(id=h["id"], error=f"{type(e).__name__}: {e}"))
            continue
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
    return dict(
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
        # phase 1's interventions on the diagnostic pool (run by the main session before the loop; plan §13-14)
        phase1_probes=jload(LOOP / "phase1" / "probe.json", []),
        **extra,
    )


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
    )
    focus = jload(it_dir.parent / "focus.json", {})
    # Reviewer
    ctx = it_dir / "reviewer_context.json"
    jdump(ctx, context(inc, dict(focus=focus)))
    prompt = fill((HERE / "prompts" / "reviewer.md").read_text(), CONTEXT=ctx, **common)
    stub_h = dict(notes="stub", hypotheses=[dict(id="H1", claim="stub: tracking gain 1.0", evidence="stub", phase="reach",
                  delta={"set": {"visual.tracking_gain": 1.0}}, predicted_fixed=[], predicted_broken=["sonic_s7000_04"])])  # fmt: skip
    hyps, acct = claude(
        "reviewer", prompt, HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent", stub_h if stub else None
    )
    jdump(it_dir / "hypotheses.json", hyps)
    review_void = transcript_touches_gate(acct["session"])
    ledger(dict(step="review", iter=n, n_hypotheses=len(hyps["hypotheses"]), void=review_void, **acct))
    # Prober
    pr = probe(hyps["hypotheses"][:MAX_HYPOTHESES], inc, base, it_dir)
    jdump(it_dir / "probe.json", pr)
    ledger(dict(step="probe", iter=n, results=pr))
    # Manager
    if sonic_up():
        return "a SONIC stack is running"
    ctx = it_dir / "manager_context.json"
    jdump(ctx, context(inc, dict(hypotheses=hyps, probe=pr)))
    prompt = fill((HERE / "prompts" / "manager.md").read_text(), CONTEXT=ctx, **common)
    stub_m = dict(decision="engineer", hypothesis="H1", reason="stub", summary_zh="桩测试")
    dec, acct = claude(
        "manager", prompt, MANAGER_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent,Bash", stub_m if stub else None
    )
    jdump(it_dir / "manager.json", dec)
    ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), reason=dec["reason"],
                summary_zh=dec["summary_zh"], escalation=dec.get("escalation"), infeasible=dec.get("infeasible"), **acct))  # fmt: skip
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
    prompt = fill(
        (HERE / "prompts" / "engineer.md").read_text(), CONTEXT=ctx, CANDIDATE=cand, ITER=it_dir, SMOKE=SMOKE, **common
    )
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
    void = review_void or transcript_touches_gate(acct["session"])
    ledger(dict(step="engineer", iter=n, changed=changed, stray=stray, mechanism=eng["mechanism"], void=void, **acct))
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
    r = subprocess.run(gate_args, cwd=REPO, capture_output=True, text=True)
    (it_dir / "gate.log").write_text(r.stdout + r.stderr)
    verdict = json.loads((gate_dir / "ledger.jsonl").read_text().splitlines()[-1])
    jdump(it_dir / "gate.json", verdict)
    ledger(dict(step="gate", iter=n, accepted=verdict["accepted"], reason=verdict["reason"], hash=verdict["hash"],
                result=verdict.get("result"), stage1=verdict.get("stage1"), void=void or stub))  # fmt: skip
    if not verdict["accepted"] and changed:
        restore(changed, saved)
    return None


def restore(changed: list[str], saved: dict) -> None:
    """Put the Engineer's files back as they were before the call (other sessions' edits elsewhere are untouched)."""
    for f in changed:
        p = REPO / f
        if f in saved:
            p.write_bytes(saved[f])
        elif p.exists():
            p.unlink()


def cmd_run(a) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    meta = jload(STATE / "meta.json") or dict(started=now(), started_s=time.time(), cost_cap_usd=COST_CAP, model=MODEL)
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
    print(f"started {meta.get('started')}; entries {len(es)}; cost {sum(e.get('cost_usd', 0) for e in es):.2f} USD")
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
    sub.add_parser("status")
    a = ap.parse_args(argv)
    global STATE
    if getattr(a, "state", None):
        STATE = a.state
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
