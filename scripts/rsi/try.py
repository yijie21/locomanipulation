"""The roles' own closed-loop checks, through the shared round queue (TODO 37.1, user 2026-10-23: "B 到 E 按顺序做").

Run C's roles ran about 1,780 rounds of their own in three iterations against about 600 by the driver: the Reviewer
re-ran the incumbent on 40 fresh starts and swept parameters, the Engineers checked their drafts -- all through
scripts they wrote around scene_round.py, one process at a time, polled with `until ... sleep` loops (10-minute waits
in the transcripts), while the 12 queue workers stood idle. This command is the one way a role runs rounds: the
starts it names, in parallel through batch.py's workers and machine-wide slots, a table back (the outcome, the failure
class, the first failing step, and against `--ref` the starts that flip). Bounded: at most MAX_PER_CALL starts per
call and BUDGET rounds per role per iteration (the ledger file RSI_TRY_LEDGER, which the driver sets per role and
iteration). The Prober's probes and the gate stay the evidence (rules.md 3, 7); this is for looking.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/try.py <family> --skills <package dir> [--set '<JSON>'] \\
        (--labels a,b,c | --task <task id> | --all) [--ref <incumbent diag dir>] [--out <dir>] [--steps] [--npz]

The pool is the family's diagnostic pool unless `--pool` names another start directory inside the sandbox or the
scratch directory (never one under the hidden tree). A second call with the same --out runs only the rounds still
missing (batch.py resumes) and charges only those.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402

MAX_PER_CALL = 40
BUDGET = int(os.environ.get("RSI_TRY_BUDGET", "240"))  # rounds per role per iteration
HIDDEN_NAME = "scene_rsi_20261016_hidden"
SCRATCH_ROOT = Path("/workspace/tmp")


def ledger_path() -> Path:
    env = os.environ.get("RSI_TRY_LEDGER")
    if env:
        return Path(env)
    return Path(os.environ.get("TMPDIR") or SCRATCH_ROOT) / "try_ledger.jsonl"


def used(ledger: Path) -> int:
    if not ledger.exists():
        return 0
    return sum(int(json.loads(x).get("n", 0)) for x in ledger.read_text().splitlines() if x.strip())


def charge(ledger: Path, n: int, labels: list[str], out: Path, overrides: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as fh:
        fh.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n=n, labels=labels,
                                 out=str(out), set=overrides)) + "\n")  # fmt: skip


def pool_allowed(pool: Path, repo: Path = REPO, scratch: Path = SCRATCH_ROOT) -> str | None:
    """Why `pool` may not be used: under the hidden tree, or outside the sandbox and the scratch directory."""
    p = pool.resolve()
    if HIDDEN_NAME in p.parts or HIDDEN_NAME in pool.parts:
        return f"{pool} is under the hidden tree (rules.md 3)"
    if not (p.is_relative_to(repo.resolve()) or p.is_relative_to(scratch.resolve())):
        return f"{pool} is outside the sandbox and the scratch directory"
    return None


def pick_labels(pool: Path, labels: str, task: str | None, all_: bool) -> list[str]:
    """The starts to run: `--labels a,b`, every start of `--task`, or `--all`; at most MAX_PER_CALL."""
    every = B.pool_labels(pool)
    if labels:
        out = [x for x in labels.split(",") if x]
        unknown = [x for x in out if x not in every]
        if unknown:
            raise SystemExit(f"not in {pool}: {', '.join(unknown)}")
    elif task:
        out = [x for x in every if json.loads((pool / f"{x}.json").read_text()).get("task") == task]
        if not out:
            raise SystemExit(f"no start of task {task!r} in {pool}")
    elif all_:
        out = every
    else:
        raise SystemExit("give --labels, --task or --all")
    if len(out) > MAX_PER_CALL:
        raise SystemExit(f"{len(out)} starts; at most {MAX_PER_CALL} per call -- split it (--labels or --task)")
    return out


def table(out: Path, ref: Path | None, steps: bool, fam: dict | None = None) -> str:
    """The rounds' outcomes, per task and per round; the flips against `ref` where both ran a start. `steps`: every
    step's numbers per round (digest.round_line), with the trace's lift / knock / slip when the round has an .npz and
    `fam` (family.load) is given."""
    import digest as D

    rows = B.outcomes(out)
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows.values():
        per[r["task"]][0] += bool(r["ok"])
        per[r["task"]][1] += 1
    lines = [f"{sum(bool(r['ok']) for r in rows.values())}/{len(rows)} succeed: "
             + ", ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(per.items()))]  # fmt: skip
    if ref is not None:
        before = {k: bool(r["ok"]) for k, r in B.outcomes(ref).items()}
        both = sorted(k for k in rows if k in before)
        fixed = [k for k in both if rows[k]["ok"] and not before[k]]
        broken = [k for k in both if before[k] and not rows[k]["ok"]]
        lines.append(f"against {ref.name} on {len(both)} shared starts: fixed {len(fixed)} {fixed}, "
                     f"broken {len(broken)} {broken}")  # fmt: skip
    for k in sorted(rows):
        r = rows[k]
        if steps:
            tr = {}
            if fam is not None and (out / f"{k}.npz").exists() and r["task"] in fam["rows"]:
                import family as F

                tr = D.trace(out / f"{k}.npz", r, list(F.task(fam, r["task"]).objects), fam["meta"])
            lines.append(D.round_line(r, tr))
            continue
        cls = f" [{r['failure_class']}]" if not r["ok"] and r.get("failure_class") else ""
        step, why = D.first_fail(r) if not r["ok"] else ("", "")
        tail = (
            f" | first fail {step}: {why[:90]}" if step else (f" | {(r.get('why') or '')[:90]}" if not r["ok"] else "")
        )
        flags = "".join(f for f, on in (("F", r.get("fell")), ("D", r.get("dropped"))) if on)
        lines.append(
            f"{k} {'OK  ' if r['ok'] else 'fail'}{(' ' + flags) if flags else ''}{cls} t {r.get('t_end', 0):.0f}s{tail}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--skills", type=Path, required=True, help="the skills package to run")
    ap.add_argument("--set", default="", help="parameter overrides: inline JSON or a JSON file")
    ap.add_argument("--labels", default="", help="comma-separated start labels")
    ap.add_argument("--task", help="every start of this task in the pool")
    ap.add_argument("--all", action="store_true", help=f"the whole pool (at most {MAX_PER_CALL} starts)")
    ap.add_argument("--pool", type=Path, help="a start directory (default: the family's diagnostic pool)")
    ap.add_argument("--ref", type=Path, help="rounds to compare with, start by start (the incumbent's diagnostic dir)")
    ap.add_argument("--out", type=Path, help="where the rounds go (default: <scratch>/try/<time>)")
    ap.add_argument("--npz", action="store_true", help="also save each round's motor trace")
    ap.add_argument("--steps", action="store_true", help="print every step's numbers per round (digest.round_line)")
    ap.add_argument("--procs", type=int, default=6, help="queue workers for this call (<= 12; the slots are shared)")
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    from scene_round import parse_set

    pool = a.pool or REPO / "outputs" / "scene_rsi_20261016" / a.family / "pool_diag"
    why = pool_allowed(pool, REPO, SCRATCH_ROOT)
    if why:
        raise SystemExit(why)
    if not a.skills.is_dir():
        raise SystemExit(f"{a.skills} is not a directory")
    labels = pick_labels(pool, a.labels, a.task, a.all)
    overrides = parse_set(a.set)
    out = a.out or Path(os.environ.get("TMPDIR") or SCRATCH_ROOT) / "try" / time.strftime("%H%M%S")
    motor = B.motor_signature()
    missing = [x for x in labels if not B.done(out / f"{x}.json", motor)]
    ledger = ledger_path()
    spent = used(ledger)
    if missing and spent + len(missing) > BUDGET:  # reading rounds already run costs nothing
        raise SystemExit(f"budget: {spent} of {BUDGET} rounds used this iteration (rules.md 18), {len(missing)} more "
                         f"asked; {BUDGET - spent} left -- pick fewer starts, or read the rounds you have")  # fmt: skip
    if missing:
        charge(ledger, len(missing), missing, out, overrides)
        job = B.Job(a.skills.resolve(), out, overrides, tuple(labels), a.npz)
        left = B.run_jobs([job], a.family, pool.resolve(), max(1, min(12, a.procs)), out / "_batch.log", a.gpus)
        if left.get(0):
            print(f"{len(left[0])} round(s) did not finish (see {out / '_batch.log'}): {', '.join(left[0])}")
    fam = None
    if a.steps:
        import family as F

        fam = F.load(a.family)
    print(table(out, a.ref, a.steps, fam))
    print(f"rounds in {out}; {len(missing)} run now, {spent + len(missing)} of {BUDGET} this iteration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
