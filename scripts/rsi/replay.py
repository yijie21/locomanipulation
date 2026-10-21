"""TODO 32.5: replay one iteration of a finished scene loop -- Reviewer, Prober, Manager, Engineer and gate -- in a fresh
sandbox under today's code, so a change to the loop's procedure is judged on the same starting state, several replays
side by side, instead of whole loops (user 2026-10-21: changes to the loop are checked by replays first, one whole
loop at the end).

    python3 scripts/rsi/replay.py setup <finished sandbox> <family> <iter> <new sandbox> [--ref <commit>]
    cd <new sandbox> && RSI_MAX_WALL_H=100 $PY scripts/rsi/loop.py run <family> --max-iters 1   # plus the arm's knobs
    python3 scripts/rsi/replay.py report <sandbox> [<sandbox> ...] [--out report.json]

`setup` builds the sandbox from HEAD (sandbox.py: the family's own stripping), then puts in the finished run's pools
(the same starts) and its state as iteration <iter> began: the ledger and the earlier iterations, the incumbents and
their diagnostic records, the gate's ledger, calls and incumbent, the parks, the Manager's focus, the gap files and
the interventions -- from the hidden `state_before/iter_NN` snapshot the loop writes since TODO 32.5, or rebuilt from
the ledger for runs before it. The running incumbent's diagnostic rounds and its rounds on the acceptance pool are
dropped, so the replay re-runs them under today's simulator (TODO 32.4 changed the physics); everything else is
history. Paths of the finished sandbox are rewritten to the new one. meta.json starts the wall clock now.

`report` gives, per replay: the iteration's role minutes, turns and cost, the Manager's decision, the gate's verdict
(accepted, n, candidate and incumbent successes, b, c, p) and the Engineer's mechanism.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]

import sandbox as SB  # noqa: E402

VISIBLE, HIDDEN = "outputs/scene_rsi_20261016", "outputs/scene_rsi_20261016_hidden"
TEXT = (".json", ".jsonl", ".md", ".patch", ".log", ".txt", ".py")


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write_jl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def iter_of_candidate(path: str) -> int | None:
    """The iteration a gate call came from: its candidate lives in loop/iter_NN/candidate."""
    for part in Path(path).parts:
        if part.startswith("iter_") and part[5:].isdigit():
            return int(part[5:])
    return None


def rebuild_parks(loop: Path, n: int) -> dict:
    """parked.json as iteration n began, for runs without snapshots: loop.park over every earlier iteration's verified
    claims, then the ledger's unparks."""
    parked: dict = {}
    events = [e for e in jl(loop / "ledger.jsonl") if e.get("iter") is not None and e["iter"] < n]
    for k in range(n):
        d = loop / f"iter_{k:02d}"
        hyps = json.loads((d / "hypotheses.json").read_text()) if (d / "hypotheses.json").exists() else {}
        res = json.loads((d / "claims.json").read_text()) if (d / "claims.json").exists() else []
        claims = [c for c in hyps.get("infeasible") or [] if isinstance(c, dict)]
        for r, c in zip(res, claims):
            if r.get("parked"):
                parked[r["id"]] = dict(iter=k, task=c["task"], kind=c["kind"], scope=r["scope"],
                                       starts=sorted(c["starts"]), statement=c["bound"]["statement"],
                                       actions_assumed=c["actions_assumed"], exits=c.get("exits") or [],
                                       no_exit_reason=c.get("no_exit_reason"), refuted_by=c["refuted_by"],
                                       user_confirmed=False, since="rebuilt by replay.py")  # fmt: skip
        for e in events:
            if e["iter"] == k and e.get("step") == "unpark":
                parked.pop(e["claim"], None)
    return parked


def setup(src: Path, family: str, n: int, dst: Path, ref: str = "HEAD") -> None:
    SB.build(dst, family, ref)
    s_loop, d_loop = src / VISIBLE / family, dst / VISIBLE / family
    s_hid, d_hid = src / HIDDEN / family, dst / HIDDEN / family
    state, d_state = s_loop / "loop", d_loop / "loop"
    if not (state / f"iter_{n:02d}").exists() and n != len(list(state.glob("iter_*"))):
        raise SystemExit(f"{src} has no iteration {n}")
    # the same starts
    for a, b in [(s_loop / "pool_diag", d_loop / "pool_diag")] + [
        (p, d_hid / p.name) for p in sorted(s_hid.glob("pool_*")) if p.is_dir()
    ]:
        if b.exists():
            shutil.rmtree(b)
        shutil.copytree(a, b)
    snap = s_hid / "state_before" / f"iter_{n:02d}"
    lengths = json.loads((snap / "lengths.json").read_text()) if snap.exists() else None
    # the loop's ledger and the earlier iterations
    d_state.mkdir(parents=True, exist_ok=True)
    rows = jl(state / "ledger.jsonl")
    rows = rows[: lengths["ledger"]] if lengths else [r for r in rows if r.get("iter") is not None and r["iter"] < n]
    write_jl(d_state / "ledger.jsonl", [r for r in rows if r.get("step") != "stop"])
    for k in range(n):
        shutil.copytree(state / f"iter_{k:02d}", d_state / f"iter_{k:02d}")
    if (state / "seed_candidate").exists():
        shutil.copytree(state / "seed_candidate", d_state / "seed_candidate")
    # the gate: calls made before iteration n
    s_gate, d_gate = s_hid / "gate", d_hid / "gate"
    calls = jl(s_gate / "ledger.jsonl")
    calls = (
        calls[: lengths["gate_ledger"]]
        if lengths
        else [c for c in calls if (iter_of_candidate(c["candidate"]) or 0) < n]
    )
    d_gate.mkdir(parents=True, exist_ok=True)
    write_jl(d_gate / "ledger.jsonl", calls)
    for c in calls:
        name = f"{c['call']:02d}_{c['hash']}"
        if (s_gate / name).exists():
            shutil.copytree(s_gate / name, d_gate / name)
    for name in ("_seed_src", "incumbent_seed", "incumbent_seed_rounds"):
        if (s_gate / name).exists():
            shutil.copytree(s_gate / name, d_gate / name)
    acc = [c for c in calls if c.get("accepted") and not c.get("dry_run")]
    if snap.exists() and (snap / "gate_incumbent.json").exists():
        inc = json.loads((snap / "gate_incumbent.json").read_text())
    elif acc:
        last = d_gate / f"{acc[-1]['call']:02d}_{acc[-1]['hash']}"
        inc = dict(dir=str(last / "cand"), rounds=str(last / "candidate"), since=acc[-1]["call"])
    elif (s_gate / "incumbent_seed").exists():
        inc = dict(dir=str(d_gate / "incumbent_seed"), rounds=str(d_gate / "incumbent_seed_rounds"), since=0)
    else:
        inc = None
    if inc is not None:
        inc = json.loads(json.dumps(inc).replace(str(src), str(dst)))
        (d_gate / "incumbent.json").write_text(json.dumps(inc))
        for f in Path(inc["rounds"]).glob("*"):  # re-run under today's simulator
            f.unlink()
    # the incumbents and the diagnostic records of the calls before n; the running incumbent's rounds re-run
    all_hashes = {c["hash"] for c in jl(s_gate / "ledger.jsonl")}
    hashes = {c["hash"] for c in calls}
    seed_like = sorted(d.name for d in (s_loop / "incumbents").glob("*") if d.name not in all_hashes)  # the seed
    for d in sorted((s_loop / "incumbents").glob("*")):
        if d.name in hashes or d.name in seed_like:
            shutil.copytree(d, d_loop / "incumbents" / d.name)
    for d in sorted((s_loop / "diag").glob("*")):
        h = d.name.split("_", 1)[1]
        if h in hashes or h in seed_like:
            shutil.copytree(d, d_loop / "diag" / d.name)
    running = acc[-1]["hash"] if acc else (seed_like[0] if seed_like else None)
    for name in (f"incumbent_{running}", f"candidate_{running}"):
        if (d_loop / "diag" / name).exists():
            shutil.rmtree(d_loop / "diag" / name)
    # parks, focus, gaps, interventions, meta
    if snap.exists():
        for f in ("parked.json", "focus.json", "gap_candidates.json", "gaps.jsonl", "interventions.jsonl"):
            if (snap / f).exists():
                shutil.copy2(snap / f, d_state / f)
    else:
        (d_state / "parked.json").write_text(json.dumps(rebuild_parks(state, n), indent=1, ensure_ascii=False))
        mgr = state / f"iter_{n - 1:02d}" / "manager.json"
        dec = json.loads(mgr.read_text()) if n and mgr.exists() else {}
        if dec.get("decision") == "review":
            (d_state / "focus.json").write_text(
                json.dumps(dict(iter=n - 1, focus=dec.get("focus", ""), reason=dec["reason"]))
            )
        write_jl(d_state / "gaps.jsonl", [r for r in jl(state / "gaps.jsonl") if r.get("iter", n) < n])
        write_jl(
            d_state / "interventions.jsonl", [r for r in jl(state / "interventions.jsonl") if r.get("iteration", n) < n]
        )
    meta = json.loads((state / "meta.json").read_text())
    meta.update(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), started_s=time.time(),
                replay=dict(source=str(src), iteration=n))  # fmt: skip
    (d_state / "meta.json").write_text(json.dumps(meta, indent=1))
    # paths of the finished sandbox and of its roles' scratch (loop.role_tmp) -> this one's: a role that opens a path
    # from the history must stay inside its own sandbox, or the transcript audit voids it
    subs = [(str(src), str(dst)), (f"rsi_scene_{src.name}", f"rsi_scene_{dst.name}")]
    for root in (d_loop, d_hid):
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in TEXT:
                t = f.read_text(errors="replace")
                u = t
                for a, b in subs:
                    u = u.replace(a, b)
                if u != t:
                    f.write_text(u)
    print(f"{dst}: {family} as iteration {n} began (incumbent {running}, {len(calls)} gate calls, "
          f"{'snapshot' if snap.exists() else 'rebuilt from the ledger'})")  # fmt: skip


def report(sandboxes: list[Path], family: str | None) -> list[dict]:
    out = []
    for sb in sandboxes:
        fams = [family] if family else [p.name for p in (sb / VISIBLE).iterdir() if (p / "loop" / "meta.json").exists()]
        for fam in fams:
            loop = sb / VISIBLE / fam / "loop"
            meta = json.loads((loop / "meta.json").read_text())
            n = meta.get("replay", {}).get("iteration")
            rows = [r for r in jl(loop / "ledger.jsonl") if r.get("iter") == n]
            row = dict(sandbox=str(sb), family=fam, iteration=n)
            for r in rows:
                if r.get("step") in ("review", "manager", "engineer"):
                    row[r["step"]] = dict(minutes=round((r.get("seconds") or 0) / 60, 1), usd=round(r.get("cost_usd") or 0, 2),
                                          void=r.get("void", False))  # fmt: skip
                if r.get("step") == "manager":
                    row["decision"] = r.get("decision")
                    row["hypothesis"] = r.get("hypothesis")
                if r.get("step") == "engineer":
                    row["mechanism"] = r.get("mechanism", "")[:300]
                if r.get("step") == "gate":
                    g = r.get("result") or r.get("stage1") or {}
                    row["gate"] = dict(accepted=r.get("accepted"), reason=r.get("reason"), hash=r.get("hash"),
                                       **{k: g.get(k) for k in ("n", "candidate_ok", "incumbent_ok", "b", "c", "p_one_sided")})  # fmt: skip
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("src", type=Path)
    s.add_argument("family")
    s.add_argument("iter", type=int)
    s.add_argument("dst", type=Path)
    s.add_argument("--ref", default="HEAD", help="the commit the sandbox is built from")
    r = sub.add_parser("report")
    r.add_argument("sandboxes", type=Path, nargs="+")
    r.add_argument("--family")
    r.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        setup(a.src.resolve(), a.family, a.iter, a.dst.resolve(), a.ref)
    else:
        rows = report([p.resolve() for p in a.sandboxes], a.family)
        text = json.dumps(rows, indent=1, ensure_ascii=False)
        if a.out:
            a.out.write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
