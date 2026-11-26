"""selfmodel_hatask_20261126: the smoke's report (revisions 1, 2 section 6; the coordinator's layout 11-26), read-only on
the episode logs:  python3 ht_smoke_stats.py <run dir>[,<run dir>...] [arm=A]

Per episode: the furthest step reached -- grasp (the hands closed), grasp-and-lift (the object >= 5 cm over its reset
height for >= 50 consecutive steps: the log's lifted_step), the search for the shelf / basket, carrying (the target
found: the approach while holding), place / over the basket, open -- and how it ended: success (HA's reward >= 1),
timeout (the 3x budget), error (the code raised: revision 2 section 6), plan failure (a command's IK failed), HA's fall,
held (the sequence ended first: a target not seen, or done). Then: episodes stuck at the shelf / basket search (the
target never found: sequence "shelf not seen" / "basket not seen"; those that ended inside the search for another
reason listed apart), and episodes that met grasp-and-lift before the search (lifted_step <= the search's first frame).
Revision 1's rule on normal ends only (revision 2 section 6): A's success <= 1/20 -> the main metric is grasp-and-lift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEARCH = {"ppbox": "search_shelf", "doubledesk": "search_basket"}
PLACE = {"ppbox": "place", "doubledesk": "over_basket"}
NOT_SEEN = {"ppbox": "shelf not seen", "doubledesk": "basket not seen"}


def how(lg: dict) -> str:
    e, s = str(lg["end"]), str(lg.get("sequence"))
    if e == "success":
        return "success"
    if e == "budget":
        return "timeout"
    if e.startswith("error"):
        return "error"
    if e == "fell":
        return "HA fall"
    if s.startswith("abandoned: ik") or "ik " in s:
        return "plan failure"
    return f"held ({s})"


def main() -> int:
    runs = sys.argv[1].split(",")
    arm = sys.argv[2] if len(sys.argv) > 2 else "A"
    rows = []
    for r in runs:
        for p in sorted(Path(r).glob(f"{arm}/log/*.json")):
            lg = json.loads(p.read_text())
            t = lg["task"]
            ph = {}
            for f, name in lg["phases"]:
                ph.setdefault(name, f)  # each phase's first frame
            steps = [("grasp", "close" in ph), ("grasp-and-lift", lg.get("lifted_step") is not None), ("search", SEARCH[t] in ph),
                     ("carry", "carry" in ph), ("place", PLACE[t] in ph), ("open", "release" in ph)]  # fmt: skip
            furthest = "before grasp"
            for k, v in steps:
                if v:
                    furthest = k
            s0 = ph.get(SEARCH[t])
            rows.append(dict(task=t, seed=lg["seed"], repeat=lg["repeat"], furthest=furthest, how=how(lg), end=lg["end"],
                             sequence=lg.get("sequence"), phase_at_end=lg.get("phase_at_end"), steps=lg["steps"],
                             success_step=lg.get("success_step"), lifted_step=lg.get("lifted_step"), search_f=s0,
                             lift_before_search=lg.get("lifted_step") is not None and s0 is not None and lg["lifted_step"] <= s0,
                             stuck_search=lg.get("sequence") == NOT_SEEN[t], ended_in_search=lg.get("phase_at_end") == SEARCH[t],
                             max_lift_cm=lg.get("max_lift_cm")))  # fmt: skip
    print(f"{'task':<11}{'seed':>5}  {'furthest step':<15}{'ended':<34}{'steps':>6}{'success':>8}{'lifted':>8}{'search_f':>9}  max_lift_cm")
    for x in rows:
        print(f"{x['task']:<11}{x['seed']:>5}  {x['furthest']:<15}{x['how']:<34}{x['steps']:>6}{str(x['success_step']):>8}"
              f"{str(x['lifted_step']):>8}{str(x['search_f']):>9}  {x['max_lift_cm']}")  # fmt: skip
    normal = [x for x in rows if x["how"] != "error"]
    err = [x for x in rows if x["how"] == "error"]
    succ = sum(x["how"] == "success" for x in normal)
    print(f"\nepisodes {len(rows)}; ended in error {len(err)} (left out of the rule, revision 2 section 6): "
          + "; ".join(f"{x['task']} s{x['seed']} at {x['phase_at_end']}: {x['end']}" for x in err))  # fmt: skip
    for t in sorted({x["task"] for x in rows}):
        xs = [x for x in normal if x["task"] == t]
        print(f"{t}: normal ends {len(xs)}; success {sum(x['how'] == 'success' for x in xs)}; grasp-and-lift {sum(x['lifted_step'] is not None for x in xs)}"
              f" (before the search {sum(x['lift_before_search'] for x in xs)}); stuck at the search (target never found) {sum(x['stuck_search'] for x in xs)};"
              f" ended inside the search otherwise {sum(x['ended_in_search'] and not x['stuck_search'] for x in xs)}; ends: "
              + ", ".join(f"{k} {sum(x['how'] == k for x in xs)}" for k in sorted({x['how'] for x in xs})))  # fmt: skip
    if err:
        print(f"\nA's success on normal ends: {succ}/{len(normal)}; {len(err)} ended in error: no decision until the code is fixed and they are rerun")
    else:
        print(f"\nA's success: {succ}/{len(normal)} -> " + ("<= 1/20: revision 1, the main metric is grasp-and-lift"
              if len(normal) and succ * 20 <= len(normal) else "> 1/20: the main metric stays success"))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
