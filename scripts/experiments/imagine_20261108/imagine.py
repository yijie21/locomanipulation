"""Run a skills package on the imagined Motor (kin_motor.py) and set the rounds beside the real ones (TODO 45).

    PYTHONPATH=src $VENV_SIM_PY scripts/experiments/imagine_20261108/imagine.py run --family grasp_floor \
        --candidate <dir with skills/ and candidate.json> --pool <pool_diag> --out <dir> [--labels a,b] [--procs 4]
    python3 scripts/experiments/imagine_20261108/imagine.py compare --real <diag/candidate_x> --imagined <dir>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def setup_paths() -> None:
    for p in (HERE, ROOT / "scripts" / "rsi", ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def make_world(fam: dict):
    from kin_motor import KinMotor

    from locomani.backends.scalebfm import motor as MM
    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.tasks.scene import world as W

    MM.FF_SHORT = (0.0, 0.0, 0.0)  # the walk's stop shortfall is the controller's; the imagined pelvis has none
    W.Motor = KinMotor
    policy = R.Policy("cpu", ScaleBfmBackendConfig())  # for its joint and keypoint tables; never called
    return W.SceneWorld(fam["scene_dir"], policy, controller="scalebfm", sleep=False)


def work(args: tuple) -> None:
    family, skills_dir, overrides, pool, out, labels = args
    setup_paths()
    import family as F
    import scene_round as SR

    fam = F.load(family)
    skills = SR.load_skills(Path(skills_dir))
    skills.apply_set(skills, overrides)
    h = SR.skills_hash(Path(skills_dir))
    world = make_world(fam)
    out = Path(out)
    for label in labels:
        t0 = time.time()
        SR.run_round(fam, world, skills, h, Path(pool), out, label, overrides, False, "scalebfm")
        for g in (
            world.motor.grips
        ):  # the arms as they were when the robot stopped walking: contact_gen starts a replay there
            calls = world.motor.calls
            last = max(
                [i for i in range(g["n_calls"]) if calls[i]["name"] in ("walk_to", "turn", "sidestep")], default=-1
            )
            t_stop = calls[last]["t1"] if last >= 0 else 0.0
            row = [r for r in world.motor.trace if r["t"] <= t_stop + 1e-6][-1]
            g["stance_joints"] = [round(float(v), 5) for v in row["joints"]]
        log = dict(wall_s=round(time.time() - t0, 2), events=world.motor.kin_log, grips=world.motor.grips,
                   pushed={world.sim.m.body(b).name: round(v, 3) for b, v in world.motor._pushed.items()})  # fmt: skip
        (out / f"{label}.kin.json").write_text(json.dumps(log))


def run(a) -> int:
    cand = Path(a.candidate)
    overrides = json.loads((cand / "candidate.json").read_text()).get("set") or {}
    pool, out = Path(a.pool), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = (
        a.labels.split(",") if a.labels else sorted(p.stem for p in pool.glob("*.json") if not p.stem.startswith("_"))
    )
    jobs = [
        (a.family, str(cand / "skills"), overrides, str(pool), str(out), labels[i :: a.procs]) for i in range(a.procs)
    ]
    jobs = [j for j in jobs if j[-1]]
    if len(jobs) == 1:
        work(jobs[0])
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as p:
            p.map(work, jobs, chunksize=1)
    return 0


def batch_job(job: dict) -> str:
    out = Path(job["out"])
    labels = [l for l in job["labels"] if not (out / f"{l}.json").exists()]
    if labels:
        work((job["family"], job["pkg"], job["set"], job["pool"], job["out"], labels))
    return job["id"]


def batch(a) -> int:
    """Every candidate whose real diagnostic rounds a sandbox kept, imagined once on the same starts."""
    import glob
    import hashlib

    setup_paths()
    import scene_round as SR

    since = time.mktime(time.strptime(a.since, "%Y-%m-%d")) if a.since else 0.0
    jobs, seen = [], {}
    for sb in sorted(glob.glob(a.sandboxes)):
        fam_dir = Path(sb) / "outputs" / "scene_rsi_20261016" / a.family
        pkgs = {}
        for c in (
            list(fam_dir.glob("incumbents/*"))
            + list(fam_dir.glob("loop/seed_candidate"))
            + list(fam_dir.glob("loop/iter_*/candidate*"))
        ):
            if (c / "skills" / "__init__.py").exists() and (c / "candidate.json").exists():
                s = json.loads((c / "candidate.json").read_text()).get("set") or {}
                pkgs[(SR.skills_hash(c / "skills"), json.dumps(s, sort_keys=True))] = c
        for real in sorted(fam_dir.glob("diag/*")):
            files = sorted(p for p in real.glob("*.json") if not p.name.startswith("_"))
            if not files or files[0].stat().st_mtime < since:
                continue  # copied in from a run before the current walking map
            rec = json.loads(files[0].read_text())
            key = (rec["skills"], json.dumps(rec.get("set") or {}, sort_keys=True))
            if key in seen:
                continue
            if key not in pkgs:
                print("no package for", real, flush=True)
                continue
            seen[key] = True
            jid = hashlib.sha256("|".join(key).encode()).hexdigest()[:12]
            out = Path(a.out) / jid
            out.mkdir(parents=True, exist_ok=True)
            jobs.append(dict(id=jid, family=a.family, pkg=str(pkgs[key] / "skills"), set=json.loads(key[1]), pool=str(fam_dir / "pool_diag"),
                             out=str(out), labels=[p.stem for p in files], real=str(real), incumbent=real.name.startswith("incumbent_")))  # fmt: skip
    Path(a.out, "manifest.json").write_text(json.dumps(jobs, indent=1))
    print(len(jobs), "candidates", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as p:
        for i, jid in enumerate(p.imap_unordered(batch_job, jobs, chunksize=1)):
            print(i + 1, jid, round(time.time() - t0), flush=True)
    return 0


def report(a) -> int:
    jobs = json.loads(Path(a.out, "manifest.json").read_text())
    rows, per = [], []
    base = next(j for j in jobs if j["incumbent"])
    load = lambda d, l: json.loads(Path(d, f"{l}.json").read_text())  # noqa: E731
    base_real = {l: bool(load(base["real"], l)["ok"]) for l in base["labels"]}
    base_imag = {l: bool(load(base["out"], l)["ok"]) for l in base["labels"]}
    for j in jobs:
        rr = ii = 0
        for l in j["labels"]:
            if not Path(j["out"], f"{l}.json").exists():
                continue
            r, i = load(j["real"], l), load(j["out"], l)
            rows.append(dict(id=j["id"], label=l, inc=j["incumbent"], real=bool(r["ok"]), imag=bool(i["ok"]), real_fc=r.get("failure_class"),
                             imag_fc=i.get("failure_class"), base_real=base_real[l], base_imag=base_imag[l]))  # fmt: skip
            rr += bool(r["ok"])
            ii += bool(i["ok"])
        per.append((j["id"], j["incumbent"], rr, ii))
    n = len(rows)
    tab = {
        (a_, b_): sum(1 for x in rows if x["real"] == a_ and x["imag"] == b_)
        for a_ in (True, False)
        for b_ in (True, False)
    }
    print(
        f"rounds {n}, candidates {len(jobs)}; real ok {sum(x['real'] for x in rows)}, imagined ok {sum(x['imag'] for x in rows)}"
    )
    print(
        f"real ok & imagined ok {tab[True, True]}, real ok & imagined fail {tab[True, False]}, real fail & imagined ok {tab[False, True]}, both fail {tab[False, False]}"
    )
    print(
        f"agree {(tab[True, True] + tab[False, False]) / max(n, 1):.3f}; always-say-ok {sum(x['real'] for x in rows) / max(n, 1):.3f}"
    )
    cand = [x for x in rows if not x["inc"]]
    for name, want in (("fixed", lambda b, v: not b and v), ("broken", lambda b, v: b and not v)):
        rf = [want(x["base_real"], x["real"]) for x in cand]
        imf = [want(x["base_imag"], x["imag"]) for x in cand]
        both = sum(a_ and b_ for a_, b_ in zip(rf, imf))
        print(f"{name}: real {sum(rf)}, imagined {sum(imf)}, both {both}")
    import numpy as np

    arr = np.array([(r, i) for _, inc, r, i in per if not inc], float)
    if len(arr) > 2 and arr[:, 0].std() and arr[:, 1].std():
        rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        print(
            f"per candidate wins: spearman {np.corrcoef(rk(arr[:, 0]), rk(arr[:, 1]))[0, 1]:.2f}, pearson {np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]:.2f}"
        )
    print("candidates (real, imagined):", sorted((int(r), int(i)) for _, inc, r, i in per if not inc))
    print("incumbent (real, imagined):", [(int(r), int(i)) for _, inc, r, i in per if inc])
    from collections import Counter

    print(
        "real fail class -> imagined ok:",
        Counter(x["real_fc"] for x in rows if not x["real"] and x["imag"]).most_common(8),
    )
    print(
        "real fail class -> imagined fail:",
        Counter(x["real_fc"] for x in rows if not x["real"] and not x["imag"]).most_common(8),
    )
    print("imagined fail class:", Counter(x["imag_fc"] for x in rows if not x["imag"]).most_common(8))
    Path(a.out, "rows.json").write_text(json.dumps(rows))
    return 0


def fails_at(rec: dict) -> str:
    """The first step that failed (the name a Reviewer would read), or '' for a clean round."""
    for s in rec.get("steps") or []:
        if not s.get("ok", True):
            return str(s.get("step"))
    return ""


def compare(a) -> int:
    real, imag = Path(a.real), Path(a.imagined)
    agree = total = 0
    for f in sorted(imag.glob("*.json")):
        if f.name.endswith(".kin.json") or f.name.startswith("_"):
            continue
        r, i = json.loads((real / f.name).read_text()), json.loads(f.read_text())
        total += 1
        agree += bool(r["ok"]) == bool(i["ok"])
        wall = (
            json.loads((imag / f"{f.stem}.kin.json").read_text())["wall_s"]
            if (imag / f"{f.stem}.kin.json").exists()
            else 0
        )
        print(f"{f.stem:32s} real {'ok ' if r['ok'] else 'no '}{str(r.get('failure_class')):12s} {fails_at(r):12s} t={r.get('t_end', 0):6.1f}"
              f" | imag {'ok ' if i['ok'] else 'no '}{str(i.get('failure_class')):12s} {fails_at(i):12s} t={i.get('t_end', 0):6.1f} {wall:5.1f}s"
              f" {str(i.get('error') or '')[:60]}")  # fmt: skip
    print(f"agree on ok: {agree}/{total}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--pool", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--labels", default="")
    r.add_argument("--procs", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("--real", required=True)
    c.add_argument("--imagined", required=True)
    b = sub.add_parser("batch")
    b.add_argument("--family", required=True)
    b.add_argument("--sandboxes", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--procs", type=int, default=6)
    b.add_argument("--since", default="", help="skip real rounds older than this date (YYYY-MM-DD)")
    e = sub.add_parser("report")
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    return dict(run=run, compare=compare, batch=batch, report=report)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
