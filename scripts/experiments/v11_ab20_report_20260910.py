"""Behaviour comparison of two SONIC checkpoints over one fixed case bank.

Success rate over 20 paired rounds is a weak instrument -- McNemar on this many
discordant pairs resolves almost nothing -- so the continuous per-round
quantities carry the answer: where the robot ends up standing, how far it drifts
while settling, how it exits the approach, and how far the upper body is from
its reference at takeover. Those are read from the gym log, which prints one
line per stage, and joined to the manifest's park distances by round order.
"""

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("AB20_OUT", ROOT / "outputs/experiments/v11_bottle_ab20_20260910"))
ARMS = ["release", "sonic_v1_1"]

RE_ROUND = re.compile(r"^\[gym\] round (\d+)/")
RE_DRIFT = re.compile(r"stand drift ([\d.]+) m")
RE_TURN = re.compile(r"spawn turn ([+-]?\d+) deg -> yaw_err ([+-]?\d+) deg, moved ([\d.]+) m")
RE_PREWALK = re.compile(r"pre-walk\s*: rel \[\s*([-\d.]+)\s+([-\d.]+)\] yaw_err ([+-]?\d+) deg")
RE_APPROACH = re.compile(
    r"approach: exit=(.+?) braked at x (\S+) edge (\S+) -> stopped x ([\d.]+) edge ([\d.]+) "
    r"\(carry (\S+), plan_r ([\d.]+)\)"
)
RE_POST = re.compile(r"post-approach: rel \[\s*([-\d.]+)\s+([-\d.]+)\] yaw_err ([+-]?\d+) deg")
RE_UPPER = re.compile(r"upper takeover: reference-vs-measured ([\d.]+) deg on (\S+) \(mean ([\d.]+)\)")
RE_EPISODE = re.compile(r"^EPISODE (OK|FAILED (.*?)) in (\d+)s")


def parse_gym(path):
    """One dict per round, in order. Missing stages simply stay absent."""
    rounds, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = RE_ROUND.match(line.strip())
        if m:
            if cur is not None:
                rounds.append(cur)
            cur = {"round": int(m.group(1))}
            continue
        if cur is None:
            continue
        for rx, keys in (
            (RE_DRIFT, ("stand_drift",)),
            (RE_TURN, ("turn_deg", "turn_yaw_err", "turn_moved")),
            (RE_PREWALK, ("prewalk_x", "prewalk_y", "prewalk_yaw_err")),
            (RE_POST, ("post_x", "post_y", "post_yaw_err")),
            (RE_UPPER, ("upper_max_deg", "upper_joint", "upper_mean_deg")),
        ):
            m = rx.search(line)
            if m:
                for k, v in zip(keys, m.groups()):
                    try:
                        cur[k] = float(v)
                    except ValueError:
                        cur[k] = v
        m = RE_APPROACH.search(line)
        if m:
            cur["approach_exit"] = m.group(1).strip()
            cur["stopped_x"] = float(m.group(4))
            cur["stopped_edge"] = float(m.group(5))
            cur["plan_r"] = float(m.group(7))
        m = RE_EPISODE.match(line.strip())
        if m:
            cur["ok"] = m.group(1) == "OK"
            cur["why"] = "" if cur["ok"] else m.group(2).strip()
            cur["seconds"] = float(m.group(3))
    if cur is not None:
        rounds.append(cur)
    return rounds


def stat(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if not vals:
        return None
    return dict(
        n=len(vals),
        mean=round(statistics.mean(vals), 4),
        sd=round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
    )


def welch(a, b):
    a = [v for v in a if isinstance(v, (int, float)) and math.isfinite(v)]
    b = [v for v in b if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return None
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return None if se == 0 else round((statistics.mean(b) - statistics.mean(a)) / se, 2)


def mcnemar(pairs):
    """Exact two-sided binomial on the discordant pairs."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, round(min(1.0, 2 * tail), 4)


def main():
    data = {}
    for arm in ARMS:
        g = OUT / arm / "gym.log"
        m = OUT / arm / "manifest.jsonl"
        rounds = parse_gym(g) if g.exists() else []
        rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()] if m.exists() else []
        for i, r in enumerate(rounds):
            if i < len(rows):
                r["case_id"] = rows[i]["case_id"]
                r["park"] = rows[i].get("park")
                r["park_first"] = rows[i].get("park_first")
                r["produced"] = rows[i].get("produced")
        data[arm] = rounds
        print(f"{arm:11} {len(rounds)} rounds parsed, {sum(1 for r in rounds if r.get('ok'))} ok")
    if not all(data[a] for a in ARMS):
        print("one arm has no rounds yet")
        return 1

    n = min(len(data[a]) for a in ARMS)
    print(f"\n=== paired outcome over {n} shared cases")
    print(f"{'case':10} {ARMS[0]:<38} {ARMS[1]:<38}")
    for i in range(n):
        cells = []
        for a in ARMS:
            r = data[a][i]
            cells.append(("OK" if r.get("ok") else "FAIL " + str(r.get("why", "?"))[:32]).ljust(38))
        print(f"{data[ARMS[0]][i].get('case_id', i):10} " + " ".join(cells))
    pairs = [(data[ARMS[0]][i].get("ok"), data[ARMS[1]][i].get("ok")) for i in range(n)]
    b, c, p = mcnemar(pairs)
    print(f"\nonly {ARMS[0]} {b}, only {ARMS[1]} {c}, exact McNemar p={p}")

    print("\n=== behaviour (mean / sd, and Welch t of v1_1 minus release)")
    fields = [
        ("stand_drift", "settle drift after teleport, m"),
        ("turn_moved", "metres moved during the spawn turn"),
        ("turn_yaw_err", "yaw error left after the turn, deg"),
        ("post_x", "arrival: forward distance to object, m"),
        ("post_y", "arrival: lateral offset, m"),
        ("post_yaw_err", "arrival: yaw error, deg"),
        ("park", "park distance, m"),
        ("stopped_x", "stop x, m"),
        ("upper_max_deg", "upper takeover worst joint error, deg"),
        ("upper_mean_deg", "upper takeover mean joint error, deg"),
        ("seconds", "round wall time, s"),
    ]
    report = {"cases": n, "mcnemar": dict(only_release=b, only_v1_1=c, p=p), "fields": {}}
    for key, desc in fields:
        cols = {a: [r.get(key) for r in data[a][:n]] for a in ARMS}
        s = {a: stat(cols[a]) for a in ARMS}
        if not all(s.values()):
            continue
        t = welch(cols[ARMS[0]], cols[ARMS[1]])
        print(
            f"{desc:42} {s[ARMS[0]]['mean']:8.3f} +-{s[ARMS[0]]['sd']:.3f}   "
            f"{s[ARMS[1]]['mean']:8.3f} +-{s[ARMS[1]]['sd']:.3f}   t={t}"
        )
        report["fields"][key] = dict(desc=desc, release=s[ARMS[0]], sonic_v1_1=s[ARMS[1]], welch_t=t)

    print("\n=== approach exit reason")
    for a in ARMS:
        kinds = {}
        for r in data[a][:n]:
            kinds[r.get("approach_exit", "(no approach)")] = kinds.get(r.get("approach_exit", "(no approach)"), 0) + 1
        print(f"{a:11} {kinds}")
        report.setdefault("approach_exit", {})[a] = kinds

    print("\n=== failure kinds")
    for a in ARMS:
        kinds = {}
        for r in data[a][:n]:
            if not r.get("ok"):
                k = str(r.get("why", "?")).split("(")[0].strip()
                kinds[k] = kinds.get(k, 0) + 1
        print(f"{a:11} ok {sum(1 for r in data[a][:n] if r.get('ok'))}/{n}  {kinds}")
        report.setdefault("failures", {})[a] = kinds

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
