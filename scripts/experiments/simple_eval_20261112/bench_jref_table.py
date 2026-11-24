"""selfmodel_jref_20261124 (TODO 38.10.2): the five ways of reaching side by side, one controller's bench_via.sh output.

    cd <repo> && SIMPLE_ROOT=... $PY -I scripts/experiments/simple_eval_20261112/bench_jref_table.py <out dir> <scene dir>

Arms (rows/<arm>_<task>.jsonl, films_<arm>/): A a straight reach; M the fixed template (8 cm back, up to the pregrasp
height, over); B the planned path's via points through our whole-body IK; D B with every point 6 cm back and 2 cm up; J the
planned right-arm joint trajectory itself as the reference (simple_bench.py --joint).

1. The main measure: SIMPLE's successes within 45 s and within the official limit, per task and arm, 30 episodes each.
2. On the episodes where every path arm (M, B, D, J) ran its path (it ends at the arm's own pregrasp within 1 cm), per arm:
   the successes; the right arm (shoulder to hand) in penetration with a table or another object during the pregrasp reach,
   by its deepest penetration there (any, over 1 mm, over 2 mm), and other robot links' share apart; how far the real arm
   went from the plan -- the 24 arm spheres (arm_fk.ArmFK, cuRobo's spheres on the scene's robot) of every film frame of
   the pregrasp phase, each against the same sphere's planned path (the plan's Q set into the reach's start state, prep2.py,
   4 points between each pair of plan rows), the largest such distance over the reach, per episode; the pregrasp and entry
   errors (median, cm); the success time (median of the successes, s).
3. Pairs on those episodes, X - Y with 95 % intervals (episodes resampled 2000 times): the 45 s successes, the penetration
   shares (any, over 1 mm), the paired largest distance from the plan.
4. J's own: the reference's arm against the plan's start (deg), the path length and duration.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experiments/selfmodel_via_20261120"))
from arm_fk import FINGERS, ArmFK  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

D, SCENE = Path(sys.argv[1]), Path(sys.argv[2])
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
LIMIT = {"XMovePickTeleop": 16.0, "XMoveBendPickTeleop": 16.0, "BendPickMP": 8.0, "TabletopGraspMP": 16.0}
RIGHT = ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
ARMS = [a for a in "AMBDJ" if (D / "rows" / f"{a}_{TASKS[0]}.jsonl").exists()]
PATH_ARMS = [a for a in "MBDJ" if a in ARMS]
FK = ArmFK(SCENE)
SIZE = {0: 7, 1: 4, 2: 1, 3: 1}


def rows(arm: str) -> dict:
    out = {}
    for t in TASKS:
        f = D / "rows" / f"{arm}_{t}.jsonl"
        for line in open(f) if f.exists() else []:
            r = json.loads(line)
            out[(t, r["ep"])] = r
    return out


def stage(r: dict, name: str) -> dict:
    return next((s for s in r["stages"] if s["stage"] == name), {})


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and "goal_cm" in a and a["goal_cm"] <= 1.0 and bool(stage(r, "pregrasp"))


def deepest(
    r: dict, right: bool = True
) -> float:  # cm, the pregrasp reach's deepest penetration of the right arm (or not)
    return max([h[3] for h in r.get("hits", []) if h[0] == "pregrasp" and h[1].startswith(RIGHT) == right], default=0.0)


def boot(n: int, f, k: int = 2000) -> tuple:
    rng = np.random.default_rng(1)
    v = np.array([f(rng.integers(n, size=n)) for _ in range(k)])
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def name_of(key) -> str:
    t, e = key
    return f"{t}_L{e // 10}_e{e % 10}"


def frame_spheres(z, k: int) -> np.ndarray:
    """The right arm's 24 sphere centres in film frame k (world)."""
    q = z["qpos"][k]
    names = [str(x) for x in z["jnt"]]
    assert int(z["jnt_type"][0]) == 0, "the film's first joint is the robot's free root"
    adr = {n: int(a) for n, a in zip(names, z["jnt_adr"])}
    joints = np.array([q[adr[n]] for n in MUJOCO_JOINT_NAMES])
    fingers = np.array([q[adr[f"right_hand_{f}_joint"]] for f in FINGERS])
    a0 = int(z["jnt_adr"][0])
    return FK.spheres("right", q[a0 : a0 + 3], q[a0 + 3 : a0 + 7], joints, "mj", fingers)[0]


_PLAN: dict = {}


def plan_spheres(key, src: str) -> np.ndarray | None:
    """(N, 24, 3): the planned path of every arm sphere -- the plan's right-arm rows set into the reach's start state."""
    if key in _PLAN:
        return _PLAN[key]
    n = name_of(key)
    f = Path(src) / f"plan_{n}.npz"
    pr = D / "step2" / f"prep_{n}.npz"
    if not f.exists() or not pr.exists():
        _PLAN[key] = None
        return None
    pl, p = np.load(f), np.load(pr)
    start = {str(a): float(b) for a, b in zip(p["names"], p["q"])}
    joints0 = np.array([start[x] for x in MUJOCO_JOINT_NAMES])
    fingers = np.array([start[f"right_hand_{x}_joint"] for x in FINGERS])
    idx = [MUJOCO_JOINT_NAMES.index(str(x)) for x in pl["names"]]
    Q = pl["Q"]
    u = np.linspace(0, len(Q) - 1, 5 * (len(Q) - 1) + 1)
    Qd = np.array([np.interp(u, np.arange(len(Q)), Q[:, i]) for i in range(Q.shape[1])]).T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(p["pelvis_R"], float).reshape(9))
    out = []
    for q in Qd:
        j = joints0.copy()
        j[idx] = q
        out.append(FK.spheres("right", p["pelvis_p"], quat, j, "mj", fingers)[0])
    _PLAN[key] = np.array(out)
    return _PLAN[key]


def off_plan(arm: str, key, src: str) -> float:
    """cm: the largest distance over the pregrasp reach of any arm sphere from its own planned path."""
    P = plan_spheres(key, src)
    f = D / f"films_{arm}" / f"{name_of(key)}_perception_direct.npz"
    if P is None or not f.exists():
        return float("nan")
    z = np.load(f)
    worst = 0.0
    for k in np.flatnonzero(z["phase"] == "pregrasp"):
        X = frame_spheres(z, int(k))  # (24, 3)
        d = np.linalg.norm(P - X[None], axis=2).min(0)  # per sphere, the nearest point of its planned path
        worst = max(worst, float(d.max()))
    return 100 * worst


R = {a: rows(a) for a in ARMS}
print(f"{D.name}: arms {' '.join(ARMS)}; rows " + ", ".join(f"{a} {len(R[a])}" for a in ARMS))
print("\n1. successes, 30 episodes a task: within 45 s / within the official limit (falls)")
print(f"  {'task':22s}" + "".join(f"{a:>12s}" for a in ARMS))
for t in TASKS:
    cells = []
    for a in ARMS:
        rs = [r for (tt, _), r in R[a].items() if tt == t]
        cells.append(f"{sum(bool(r.get('success_45')) for r in rs):d} / {sum(bool(r.get('success_official')) for r in rs):d}"
                     + (f" ({sum(r.get('fell') is not None for r in rs)})" if any(r.get("fell") is not None for r in rs) else ""))  # fmt: skip
    print(f"  {t:22s}" + "".join(f"{c:>12s}" for c in cells))
print(f"  {'all 120':22s}" + "".join(f"{sum(bool(r.get('success_45')) for r in R[a].values()):d} / "
                                     f"{sum(bool(r.get('success_official')) for r in R[a].values()):d}".rjust(12) for a in ARMS))  # fmt: skip

keys = sorted(k for k in R["B"] if all(k in R[a] and routed(R[a][k]) for a in PATH_ARMS))
print(f"\n2. episodes where every path arm ({' '.join(PATH_ARMS)}) ran its path: {len(keys)} ("
      + ", ".join(f"{t} {sum(k[0] == t for k in keys)}" for t in TASKS) + ")")  # fmt: skip
src = {k: str(np.load(D / "via" / f"via_{name_of(k)}.npz")["src"]) for k in keys}  # the plan B, D and J ran
S, H, H1, H2, O, F, P = {}, {}, {}, {}, {}, {}, {}
for a in ARMS:
    rs = [R[a].get(k) for k in keys]
    S[a] = np.array([bool(r and r.get("success_45")) for r in rs])
    dp = np.array([deepest(r) if r else np.nan for r in rs])
    H[a], H1[a], H2[a] = dp > 0, dp > 0.1, dp > 0.2
    O[a] = np.array([deepest(r, right=False) > 0 if r else False for r in rs])
    F[a] = np.array([off_plan(a, k, src[k]) for k in keys])
    P[a] = rs
print(
    "  arm  45 s  official  right arm: any  >1 mm  >2 mm   other links   off plan cm: median  p75   max   <=1.5 cm"
    "   pregrasp cm  entry cm  success time s"
)
for a in ARMS:
    rs = P[a]
    pre = [stage(r, "pregrasp").get("err_cm") for r in rs if r and stage(r, "pregrasp")]
    ent = [stage(r, "entry").get("err_cm") for r in rs if r and stage(r, "entry")]
    ts = [r["t_success"] for r in rs if r and r.get("t_success") is not None]
    f = F[a][~np.isnan(F[a])]
    print(f"   {a}  {int(S[a].sum()):4d}  {sum(bool(r and r.get('success_official')) for r in rs):8d}  {H[a].mean():14.3f}  {H1[a].mean():5.3f}  "
          f"{H2[a].mean():5.3f}  {O[a].mean():12.3f}  {np.median(f) if len(f) else np.nan:19.2f}  "
          f"{np.percentile(f, 75) if len(f) else np.nan:4.2f}  {f.max() if len(f) else np.nan:5.2f}  {(f <= 1.5).mean() if len(f) else np.nan:9.2f}"
          f"  {np.median(pre) if pre else np.nan:12.2f}  {np.median(ent) if ent else np.nan:8.2f}  {np.median(ts) if ts else np.nan:14.2f}")  # fmt: skip
n = len(keys)
print("\n3. pairs on these episodes, X - Y (95 % interval)")
for x, y in (("J", "D"), ("J", "B"), ("J", "M"), ("D", "B"), ("M", "B")):
    if x not in ARMS or y not in ARMS or not n:
        continue
    lo, hi = boot(n, lambda q: S[x][q].sum() - S[y][q].sum())
    l2, h2 = boot(n, lambda q: H[x][q].mean() - H[y][q].mean())
    l3, h3 = boot(n, lambda q: H1[x][q].mean() - H1[y][q].mean())
    ok = ~np.isnan(F[x]) & ~np.isnan(F[y])
    dd = F[x][ok] - F[y][ok]
    l4, h4 = boot(len(dd), lambda q: np.median(dd[q])) if len(dd) else (np.nan, np.nan)
    print(f"  {x} - {y}: 45 s successes {int(S[x].sum())} and {int(S[y].sum())}, {int(S[x].sum()) - int(S[y].sum()):+d} ({lo:+.0f} to {hi:+.0f}), "
          f"only {x} {int((S[x] & ~S[y]).sum())}, only {y} {int((~S[x] & S[y]).sum())}; right arm any penetration "
          f"{H[x].mean() - H[y].mean():+.3f} ({l2:+.3f} to {h2:+.3f}), over 1 mm {H1[x].mean() - H1[y].mean():+.3f} ({l3:+.3f} to {h3:+.3f}); "
          f"off plan paired median {np.median(dd) if len(dd) else np.nan:+.2f} cm ({l4:+.2f} to {h4:+.2f})")  # fmt: skip
if "J" in ARMS and n:
    js = [stage(r, "pregrasp") for r in P["J"] if r]
    g = [s.get("joint_gap_deg") for s in js if s.get("joint_gap_deg") is not None]
    print(f"\n4. J: the reference's arm against the plan's start, largest joint: median {np.median(g):.2f} deg, max {max(g):.2f}; path "
          f"median {np.median([s['joint_path_cm'] for s in js if s.get('joint_path_cm')]):.1f} cm, duration median "
          f"{np.median([s['joint_dur_s'] for s in js if s.get('joint_dur_s')]):.2f} s")  # fmt: skip
json.dump({a: dict(keys=[list(k) for k in keys], success_45=S[a].tolist(), deepest_right_cm=[deepest(r) if r else None for r in P[a]],
                   off_plan_cm=F[a].tolist()) for a in ARMS}, open(D / "jref_table.json", "w"))  # fmt: skip
