"""faildiag revision 4: the moved episodes whose object had no robot contact from 1 s before to 0.2 s after the onset
(fd_q1_moved.py --hits over q1hit/) -> fd_run.py jobs for the FD_HITLOG=2 rerun (the object's supports)."""
import json
import sys
from collections import defaultdict

T = {"dumbbell": "IThorLivingRoomApproachTableAndLiftDumbbell", "wcan": "IThorLivingRoomApproachTableAndLiftWateringCan"}
P = json.load(open(sys.argv[1]))
g = defaultdict(list)
for x in P:
    if x.get("moved") and x.get("same_rows") and not x.get("hit_bodies_1s"):
        g[(x["backend"], x["method"], x["task"])].append(x["ep"])
lines = []
for (be, m, tn), eps in sorted(g.items()):
    eps = sorted(eps)
    for i in range(0, len(eps), 3):
        lines.append(f"run --group rerun --backend {be} --task {T[tn]} --method {m} --ep-list " + ",".join(map(str, eps[i : i + 3])))
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print(sum(len(v) for v in g.values()), "episodes,", len(lines), "jobs", {k: len(v) for k, v in g.items()})
