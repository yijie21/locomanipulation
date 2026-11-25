"""TODO 33.15.1 (plan section 3.6): the record library -- every round the roles may see, one row each, with the
version it ran under (skills hash, parameter set, motor signature, machine), its outcome and failure class, and every
measured number flattened into a column: the start (`start.spawn_dist_m`, `start.spawn_bearing_deg`, `start.obj_z`,
`start.obj_yaw_deg`), the events (`events.reach_gap_cm`, `events.fit.margin_m`, ...) and each step's first occurrence
(`pregrasp.jaw_err_cm`, `pregrasp.pred_p_miss`, `walk.stop_err_cm`, ...).

A Reviewer's hypothesis may carry a `measure` {feature, op, value, expect, tasks?, scope?}: "rounds where FEATURE OP
VALUE fail more" (expect "fails") or "succeed more" (expect "succeeds"). When the probe could not decide it (its test
needs code), the driver splits the library's rows by the condition and writes the verdict into the hypothesis's
`status` ([scene-status]): verified -- both groups have at least MIN_GROUP rounds, Fisher's exact p <= 0.05, and the
difference points the way the hypothesis says; refuted -- both groups have at least MIN_REFUTE rounds and the success
rates differ by less than NO_EFFECT or the other way at p <= 0.05; otherwise it stays a hypothesis (too few rounds). By
default (`scope` "incumbent") only the incumbent's own version counts, because rounds of different skills differ for
other reasons; `scope` "all" takes every version with the same motor signature on this machine. A correlation is not
evidence a change works ([evidence]): it decides what to build first (33.17), the gate decides what stays.

Only rounds the roles may see go in: in a sandbox its outputs outside the hidden tree and the roles' scratch directory;
the origin's library file (`build`, run by a person over finished sandboxes) leaves out every start drawn from a
family's acceptance or sealed seed range.

    python3 scripts/rsi/records.py features [--task T ...]           columns, how many rows carry each, their range
    python3 scripts/rsi/records.py query FEATURE OP VALUE [--expect fails|succeeds] [--task T ...] [--scope all]
    python3 scripts/rsi/records.py build --out outputs/scene_rsi_20261016/records/library.jsonl SANDBOX [...]
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from locomani import paths  # noqa: E402

LIB = REPO / "outputs" / "scene_rsi_20261016" / "records" / "library.jsonl"
SCRATCH_ROOT = Path(paths.TMP)  # loop.SCRATCH_ROOT: <workspace>/tmp, the roles' scratch directories live under it
HIDDEN = "scene_rsi_20261016_hidden"
OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}  # fmt: skip
MIN_GROUP = 3  # rounds on each side of the split for a verdict of verified
MIN_REFUTE = 10  # rounds on each side for refuted
NO_EFFECT = 0.10  # success rates closer than this: no relation
HIDDEN_SEED_SPAN = 10000  # a family's acceptance / sealed seed and the next this many seeds are left out of `build`
ROUND_KEYS = ("label", "task", "ok", "steps")
META_KEYS = ("step", "ok", "why", "t0", "t1", "check", "attempt")


def machine() -> str:
    return socket.gethostname()


def _flat(prefix: str, d: dict, out: dict, depth: int = 0) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            if math.isfinite(float(v)):
                out[key] = float(v)
        elif isinstance(v, str) and len(v) <= 40:
            out[key] = v
        elif isinstance(v, dict) and depth < 2:
            _flat(f"{key}.", v, out, depth + 1)


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def start_features(start: dict, task: str) -> dict:
    """Where the robot spawned relative to the object it is after (the placement whose key is in the task id)."""
    s = start.get("start") or {}
    pl = s.get("placements") or {}
    if not pl or "robot_xy" not in s:
        return {}
    key = next((k for k in pl if k in task), next(iter(pl)))
    (px, py, pz), q = pl[key]
    rx, ry = s["robot_xy"]
    bearing = math.atan2(py - ry, px - rx) - float(s.get("robot_yaw", 0.0))
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    return {"start.spawn_dist_m": round(math.hypot(px - rx, py - ry), 3),
            "start.spawn_bearing_deg": round(math.degrees(bearing), 1), "start.obj_z": round(float(pz), 3),
            "start.obj_yaw_deg": round(math.degrees(_yaw(q)), 1), "start.seed": float(s.get("seed", -1))}  # fmt: skip


def row(rec: dict, start: dict | None, where: str, host: str, family: str) -> dict:
    out = {"label": rec["label"], "task": rec["task"], "family": family, "ok": int(bool(rec["ok"])),
           "failure_class": rec.get("failure_class"), "skills": rec.get("skills"),
           "set": json.dumps(rec.get("set") or {}, sort_keys=True), "motor": rec.get("motor"),
           "controller": rec.get("controller"), "machine": host, "source": where}  # fmt: skip
    if start:
        out.update(start_features(start, rec["task"]))
    if isinstance(rec.get("events"), dict):
        _flat("events.", rec["events"], out)
    seen = set()
    for s in rec.get("steps") or []:
        if isinstance(s, dict) and s.get("step") and s["step"] not in seen:
            seen.add(s["step"])
            out[f"{s['step']}.ok"] = int(bool(s.get("ok")))
            _flat(f"{s['step']}.", {k: v for k, v in s.items() if k not in META_KEYS}, out)
    return out


def scan(dirs: list[tuple[Path, str]], host: str) -> list[dict]:
    """Round records and start files under each (directory, family), outside any hidden tree."""
    recs, starts = [], {}
    for d, family in dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.json"):
            if HIDDEN in str(f) or f.name.startswith("_"):
                continue
            try:
                j = json.loads(f.read_text())
            except (ValueError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(j, dict):
                continue
            if all(k in j for k in ROUND_KEYS):
                recs.append((j, str(f.parent), family))
            elif "start" in j and "task" in j:
                starts.setdefault(f.stem, j)
    return dedupe([row(j, starts.get(j["label"]), where, host, fam) for j, where, fam in recs])


def dedupe(rows: list[dict]) -> list[dict]:
    """A start under one version on one machine counts once (replays and re-runs repeat the same round)."""
    out, seen = [], set()
    for r in rows:
        k = (r["label"], r.get("skills"), r.get("set"), r.get("motor"), r.get("machine"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def in_sandbox(root: Path = REPO) -> bool:
    return (root / "third_party").is_symlink()


def load(root: Path = REPO, scratch_root: Path = SCRATCH_ROOT) -> list[dict]:
    """The library a sandbox's driver and roles query: the origin's library file copied in at build, plus this
    sandbox's own visible rounds and its roles' scratch rounds. Outside a sandbox: the library file only (the origin's
    outputs hold copies of hidden rounds for the reports)."""
    lib = root / LIB.relative_to(REPO)
    rows = [json.loads(x) for x in lib.read_text().splitlines() if x.strip()] if lib.exists() else []
    if in_sandbox(root):
        base = root / "outputs" / "scene_rsi_20261016"
        dirs = [(d, d.name) for d in sorted(base.iterdir()) if d.is_dir() and d.name not in ("body", "records")]
        fam = dirs[0][1] if len(dirs) == 1 else ""
        rows += scan(dirs + [(scratch_root / f"rsi_scene_{root.name}", fam)], machine())
    return dedupe(rows)


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, k, n = a + b, c + d, a + c, a + b + c + d

    def pr(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    lo, hi = max(0, k - n2), min(k, n1)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def version_of(rounds_dir: Path) -> tuple | None:
    """(skills, set, motor) of the rounds in a directory (the incumbent's diagnostic rounds)."""
    for f in sorted(Path(rounds_dir).glob("*.json")):
        if f.name.startswith("_"):
            continue
        j = json.loads(f.read_text())
        if isinstance(j, dict) and all(k in j for k in ROUND_KEYS):
            return j.get("skills"), json.dumps(j.get("set") or {}, sort_keys=True), j.get("motor")
    return None


def judge(rows: list[dict], measure: dict, version: tuple | None = None, host: str | None = None) -> dict:
    """The measure's condition against success over the library (module docstring)."""
    feature, op, value = measure.get("feature"), measure.get("op"), measure.get("value")
    expect, scope = measure.get("expect", "fails"), measure.get("scope", "incumbent")
    if op not in OPS or not feature:
        return dict(verdict="hypothesis", why=f"unusable measure (op must be one of {sorted(OPS)})")
    tasks = set(measure.get("tasks") or [])
    sel = [r for r in rows if feature in r and (host is None or r.get("machine") == host)
           and (not tasks or r["task"] in tasks)]  # fmt: skip
    if version is not None:
        sel = [r for r in sel if (r.get("skills"), r.get("set"), r.get("motor")) == tuple(version)] if scope == "incumbent" \
            else [r for r in sel if r.get("motor") == version[2]]  # fmt: skip
    try:
        hit = [bool(OPS[op](r[feature], value)) for r in sel]
    except TypeError:
        return dict(verdict="hypothesis", why=f"{feature} and {value!r} do not compare")
    a = sum(r["ok"] for r, h in zip(sel, hit, strict=True) if h)
    n1 = sum(hit)
    c = sum(r["ok"] for r, h in zip(sel, hit, strict=True) if not h)
    n2 = len(sel) - n1
    out = dict(feature=feature, op=op, value=value, expect=expect, scope=scope, n_in=n1, ok_in=a, n_out=n2, ok_out=c)
    if not n1 or not n2:
        return dict(out, verdict="hypothesis", why="one side of the split is empty")
    r1, r2 = a / n1, c / n2
    p = fisher(a, n1 - a, c, n2 - c)
    right_way = r1 < r2 if expect == "fails" else r1 > r2
    out.update(rate_in=round(r1, 3), rate_out=round(r2, 3), p=round(p, 4))
    if n1 >= MIN_GROUP and n2 >= MIN_GROUP and p <= 0.05 and right_way:
        return dict(out, verdict="verified")
    if n1 >= MIN_REFUTE and n2 >= MIN_REFUTE and (abs(r1 - r2) < NO_EFFECT or (p <= 0.05 and not right_way)):
        return dict(out, verdict="refuted")
    return dict(out, verdict="hypothesis", why="too few rounds or no clear difference")


def hidden_ranges(families_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Each family's acceptance and sealed seed ranges, from the origin's family files."""
    out = {}
    for f in families_dir.glob("*.json"):
        pools = json.loads(f.read_text()).get("pools") or {}
        out[f.stem] = [(int(pools[k]["seed"]), int(pools[k]["seed"]) + HIDDEN_SEED_SPAN) for k in ("accept", "sealed")
                       if isinstance(pools.get(k), dict) and pools[k].get("seed") is not None]  # fmt: skip
    return out


def seed_of(r: dict) -> int:
    """A row's start seed: from its start file, else the label's `_s<seed>` suffix; -1 when neither has one."""
    if r.get("start.seed") is not None:
        return int(r["start.seed"])
    tail = r["label"].rsplit("_s", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def build(sandboxes: list[Path], out: Path, scratch_root: Path = SCRATCH_ROOT) -> dict:
    """The origin's library file over finished sandboxes (a person runs it; sandbox.py copies the file in)."""
    ranges = hidden_ranges(REPO / "scripts" / "rsi" / "families")
    rows, host = [], machine()
    for sb in sandboxes:
        base = sb / "outputs" / "scene_rsi_20261016"
        dirs = [(d, d.name) for d in sorted(base.iterdir()) if d.is_dir() and d.name not in ("body", "records")]
        fam = dirs[0][1] if len(dirs) == 1 else ""
        rows += scan(dirs + [(scratch_root / f"rsi_scene_{sb.name}", fam)], host)
    kept = [r for r in dedupe(rows) if not any(a <= seed_of(r) < b for a, b in ranges.get(r["family"], []))]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return dict(rows=len(kept), left_out_hidden=len(dedupe(rows)) - len(kept), sandboxes=len(sandboxes))


def features(rows: list[dict], tasks: list[str] | None = None) -> list[tuple[str, int, str]]:
    sel = [r for r in rows if not tasks or r["task"] in tasks]
    cnt = Counter(k for r in sel for k in r if "." in k)
    out = []
    for k, n in cnt.most_common():
        vals = [r[k] for r in sel if k in r]
        nums = sorted(v for v in vals if isinstance(v, float))
        rng = (
            f"{nums[0]:g} .. {nums[len(nums) // 2]:g} .. {nums[-1]:g}"
            if nums
            else ", ".join(sorted(set(map(str, vals)))[:6])
        )
        out.append((k, n, rng))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--task", action="append")
    f.add_argument("--min", type=int, default=10, help="only columns carried by at least this many rows")
    q = sub.add_parser("query")
    q.add_argument("feature")
    q.add_argument("op", choices=sorted(OPS))
    q.add_argument("value")
    q.add_argument("--expect", choices=("fails", "succeeds"), default="fails")
    q.add_argument("--task", action="append")
    q.add_argument("--scope", choices=("incumbent", "all"), default="all")
    q.add_argument("--version-of", type=Path, help="a rounds directory whose version the incumbent scope keeps")
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=LIB)
    b.add_argument("sandboxes", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build([p.resolve() for p in a.sandboxes], a.out)))
        return 0
    rows = load()
    if a.cmd == "features":
        print(f"{len(rows)} rows")
        for k, n, rng in features(rows, a.task):
            if n >= a.min:
                print(f"{k:48s} {n:6d}  {rng}")
        return 0
    try:
        value = float(a.value)
    except ValueError:
        value = a.value
    m = dict(feature=a.feature, op=a.op, value=value, expect=a.expect, tasks=a.task, scope=a.scope)
    print(json.dumps(judge(rows, m, version_of(a.version_of) if a.version_of else None, machine())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
