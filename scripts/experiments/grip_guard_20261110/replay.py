"""TODO 41.2: the check before the jaw closes (locomani.tasks.scene.grip_guard), replayed on recorded rounds.

The simulation is deterministic and the check does nothing when the hand is on target, so only the rounds whose jaw
closed off target can change. `run` replays those with RSI_GRIP_GUARD=1; `report` pairs each with its recorded round.

Sets:
- gf: grasp_floor, the 16 composed candidates of the 11-09 runs and their incumbent, on the 40 gate starts;
- ga: grasp_atomic, the incumbent 5cba2fa49d31 on its 120 starts (walkmap_20261027/rounds).

    $VENV_SIM_PY scripts/experiments/grip_guard_20261110/replay.py run --procs 6
    python3 scripts/experiments/grip_guard_20261110/replay.py report
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path("/workspace/code/locomanipulation")
OUT = REPO / "outputs/scene_rsi_20261016/grip_guard_20261110"
SB = Path("/workspace/code/rsi-scene")
AB = REPO / "outputs/scene_rsi_20261016/body/comp_ab_20261026"
WM = REPO / "outputs/scene_rsi_20261016/walkmap_20261027/rounds"
GA_POOLS = {
    "diag": AB / "pool_diag_d",
    "accept": SB / "comp-ab-20261026/outputs/scene_rsi_20261016_hidden/grasp_atomic/pool_accept",
    "sealed": AB / "pool_sealed",
}
SELECT_CM = 3.0  # replay a round whose last reach before the first grip ended this far off or more


def off_target(r: dict) -> bool:
    steps = r.get("steps", [])
    gi = next((i for i, s in enumerate(steps) if s["step"] == "grip"), None)
    if gi is None:
        return False
    last = next((s for s in reversed(steps[:gi]) if "jaw_err_cm" in s), None)
    return last is not None and last["jaw_err_cm"] >= SELECT_CM


def recorded(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "label" in r:
            out[r["label"]] = r
    return out


def jobs() -> list[dict]:
    """One row per (package, pool): family, pool, skills, set, recorded rounds, where the replay goes."""
    rows = []
    for sb in sorted(glob.glob(str(SB / "r[xs]-single-gf?-r?")) + glob.glob(str(SB / "sh-single-gf?-r?"))):
        g = Path(sb) / "outputs/scene_rsi_20261016_hidden/grasp_floor"
        cands = sorted(p for p in (g / "gate").glob("[0-9][0-9]_*") if (p / "candidate").is_dir())
        if not cands:
            continue
        c = cands[-1]
        rows.append(dict(name=f"gf/{Path(sb).name}", family="grasp_floor", pool=g / "pool_accept", skills=c / "cand/skills",
                         set=json.loads((c / "cand/candidate.json").read_text()).get("set", {}), rec=c / "candidate"))  # fmt: skip
        if not any(r["name"] == "gf/incumbent" for r in rows):
            rows.append(dict(name="gf/incumbent", family="grasp_floor", pool=g / "pool_accept", skills=g / "gate/incumbent_seed/skills",
                             set=json.loads((g / "gate/incumbent_seed/candidate.json").read_text()).get("set", {}),
                             rec=g / "gate/incumbent_seed_rounds"))  # fmt: skip
    for name, pool in GA_POOLS.items():
        rows.append(dict(name=f"ga/{name}", family="grasp_atomic", pool=pool, skills=AB / "skills",
                         set={"pick.VIA_BACK_M": 0.2}, rec=WM / name))  # fmt: skip
    for r in rows:
        have = {p.stem for p in Path(r["pool"]).glob("*.json")}
        r["old"] = {k: v for k, v in recorded(Path(r["rec"])).items() if k in have}
        r["labels"] = sorted(k for k, v in r["old"].items() if off_target(v))
        r["out"] = OUT / "rounds" / r["name"]
    return rows


def run(procs: int, gpus: str) -> None:
    sys.path[:0] = [str(REPO / "scripts/rsi"), str(REPO / "src")]
    os.chdir(REPO)
    import batch as B

    for k in ("RSI_REACH_ROUTE", "RSI_ROUTE_POSE", "RSI_ROUTE_SCOPE", "RSI_REACH_COMPENSATION"):
        os.environ.pop(k, None)
    os.environ["RSI_GRIP_GUARD"] = "1"
    rows = [r for r in jobs() if r["labels"]]
    for fam in ("grasp_floor", "grasp_atomic"):
        by_pool = collections.defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by_pool[str(r["pool"]) if fam == "grasp_atomic" else "one"].append(r)
        for group in by_pool.values():  # grasp_floor: every sandbox drew the same gate pool
            js = [B.Job(Path(r["skills"]), r["out"], r["set"], tuple(r["labels"])) for r in group]
            missing = B.run_jobs(js, fam, Path(group[0]["pool"]), procs, OUT / "run.log", gpus)
            print(fam, [r["name"] for r in group], "rounds", sum(len(r["labels"]) for r in group), "missing",
                  sum(len(v) for v in missing.values()), "signature", B.motor_signature(), flush=True)  # fmt: skip


def report() -> None:
    lines, total = [], collections.Counter()
    for fam in ("gf", "ga"):
        t = collections.Counter()
        acts, rescued, harmed, after = collections.Counter(), [], [], []
        for r in jobs():
            if not r["name"].startswith(fam):
                continue
            new = recorded(r["out"]) if r["out"].is_dir() else {}
            t["packages"] += 1
            t["rounds"] += len(r["old"])
            t["wins"] += sum(v["ok"] for v in r["old"].values())
            t["selected"] += len(r["labels"])
            for k in r["labels"]:
                if k not in new:
                    t["not_run"] += 1
                    continue
                o, n = r["old"][k], new[k]
                g = n.get("grip_guard") or []
                first = g[0] if g else dict(act="none", why="no grip reached the check")
                key = first["act"] if first["act"] == "retry" else first["why"]
                acts[key] += 1
                t["replayed"] += 1
                if first["act"] != "retry":
                    t["same_when_idle"] += o["ok"] == n["ok"] and abs(o["t_end"] - n["t_end"]) < 0.05
                    t["idle"] += 1
                    continue
                t["retried"] += 1
                t["retried_old_wins"] += o["ok"]
                t["retried_new_wins"] += n["ok"]
                t["fell"] += bool(n.get("fell")) and not o.get("fell")
                after.append(
                    (first["err_cm"], first.get("after_cm"), o.get("failure_class"), n["ok"], n.get("failure_class"))
                )
                (rescued if n["ok"] and not o["ok"] else harmed if o["ok"] and not n["ok"] else []).append(
                    f"{r['name']}/{k}"
                )
        total.update(t)
        lines += [f"== {fam}: {dict(t)}", f"   what the check did: {dict(acts)}",
                  f"   rescued {len(rescued)}: {rescued}", f"   harmed {len(harmed)}: {harmed}"]  # fmt: skip
        by = collections.defaultdict(lambda: [0, 0, 0])
        for e0, e1, c0, ok, c1 in after:
            b = by[c0 or "won"]
            b[0] += 1
            b[1] += ok
            b[2] += e1 is not None and e1 < 3.0
        lines.append("   retried rounds by recorded class (n, now won, hand within 3 cm after the retry): "
                     + ", ".join(f"{c} {v[0]}/{v[1]}/{v[2]}" for c, v in sorted(by.items())))  # fmt: skip
        lines.append("   new class of retried rounds still lost: "
                     + str(dict(collections.Counter(c1 for *_, ok, c1 in after if not ok))))  # fmt: skip
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "plan"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.procs, a.gpus)
    elif a.cmd == "report":
        report()
    else:
        for r in jobs():
            print(
                r["name"],
                "recorded",
                len(r["old"]),
                "wins",
                sum(v["ok"] for v in r["old"].values()),
                "selected",
                len(r["labels"]),
            )
