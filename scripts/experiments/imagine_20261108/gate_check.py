"""Does the imagined Motor say anything about starts it has never seen (TODO 45.7.2)? The frozen arrival-error rule on the
composed candidates of the three recorded batches, on the 40 hidden gate starts, beside their real gate rounds. The error
pool was built from rounds on the diagnostic starts, so none of these starts is in it. Offline: no role reads the result.

    gate_check.py run <out> [--k 8] [--procs 6]
    gate_check.py report <out>
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arrival_report as AP  # noqa: E402
import arrival_run as AR  # noqa: E402

CHUNKS = 2
VETO = (0.25, 0.5)


def jobs() -> list[dict]:
    out = []
    for sb in sorted(glob.glob("/workspace/code/rsi-scene/[rs][shx]-single-gf?-r?")):
        sb = Path(sb)
        fam, hid = (sb / "outputs" / d / AR.FAMILY for d in ("scene_rsi_20261016", "scene_rsi_20261016_hidden"))
        h = json.loads((sorted((fam / "loop").glob("iter_*"))[-1] / "gate.json").read_text())["hash"]
        d = sorted((hid / "gate").glob(f"*_{h}"))[-1]
        out.append(job(f"{sb.name}-{h}", d / "cand", d / "candidate", hid, sb.name[:2], False))
        if not any(j["incumbent"] for j in out):
            out.append(job("incumbent", hid / "gate" / "incumbent_seed", hid / "gate" / "incumbent_seed_rounds", hid, "inc", True))  # fmt: skip
    return out


def job(jid: str, pkg: Path, real: Path, hid: Path, batch: str, incumbent: bool) -> dict:
    labels = sorted(p.stem for p in real.glob("floor_*.json"))
    sets = json.loads((pkg / "candidate.json").read_text()).get("set") or {}
    return dict(id=jid, pkg=str(pkg / "skills"), set=sets, pool=str((hid / "pool_accept").resolve()), labels=labels,
                real=str(real), group=batch, incumbent=incumbent)  # fmt: skip


def run(out: Path, k: int, procs: int) -> None:
    js = jobs()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(dict(jobs=js, k=k, noise=True, fit=True), indent=1))
    tasks = [(j, j["labels"][c::CHUNKS], str(out), k, True, True) for j in js for c in range(CHUNKS)]
    print(len(js), "packages,", sum(len(j["labels"]) for j in js), "rounds, k =", k, flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=1) as p:
        for i, jid in enumerate(p.imap_unordered(AR.work, tasks, chunksize=1)):
            print(i + 1, jid, round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)


def auc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def line(name: str, rows: list[dict], key: str) -> str:
    s, y = np.array([r[key] for r in rows]), np.array([r["y"] for r in rows])
    rng = np.random.default_rng(0)
    # resample whole packages, since the rounds of one package are not independent
    ids = sorted({r["id"] for r in rows})
    by = {i: np.array([j for j, r in enumerate(rows) if r["id"] == i]) for i in ids}
    b = [auc(s[x], y[x]) for x in (np.concatenate([by[ids[i]] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(2000))]  # fmt: skip
    return f"{name:18s} n {len(rows):4d} real win {y.mean():.3f} {key:10s} AUC {auc(s, y):.3f} (95% {np.nanpercentile(b, 2.5):.3f}..{np.nanpercentile(b, 97.5):.3f})"  # fmt: skip


def report(out: Path) -> None:
    rows, _ = AP.rows_of(out)
    drafts = [r for r in rows if not r["inc"]]
    for key in ("share_body", "share"):
        print(line("all composed", drafts, key))
        for g in ("sh", "rx", "rs"):
            print(line(f"  batch {g}", [r for r in drafts if r["group"] == g], key))
        print(line("incumbent", [r for r in rows if r["inc"]], key))
    print("\nveto on share_body (composed candidates):")
    for v in VETO:
        cut = [r for r in drafts if r["share_body"] < v]
        kept = [r for r in drafts if r["share_body"] >= v]
        wins = sum(r["y"] for r in cut)
        print(f"  < {v}: vetoed {len(cut)} rounds, {wins} real wins among them ({wins / max(1, len(cut)):.2f}); kept {len(kept)}, win {np.mean([r['y'] for r in kept]):.3f}")  # fmt: skip
    print("\nper package: imagined mean share_body, real win on the gate starts")
    pkgs, a, b = sorted({r["id"] for r in drafts}), [], []
    for p in pkgs:
        x = [r for r in drafts if r["id"] == p]
        a.append(float(np.mean([r["share_body"] for r in x])))
        b.append(float(np.mean([r["y"] for r in x])))
        print(f"  {p:40s} {a[-1]:.3f} {b[-1]:.3f}")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    print(f"Spearman over {len(pkgs)} packages: {np.corrcoef(ra, rb)[0, 1]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.out, a.k, a.procs)
    else:
        report(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
