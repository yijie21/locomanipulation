"""TODO 53, the loop's gain: one `python -m locomani.bench.simple run` process per setting, as the RSI loop runs them.

    python loop_level.py <scenario> <on|off> --slot K --out DIR --store STORE

  sweep  (a) a probe sweep: 5 values of one post-approach parameter (art_speed x 1.0 0.6 0.8 1.25 1.5) over 15 starts of one
         task, each value a separate `run` call (its own process, its own Isaac server).
  draft  (b) the incumbent's diagnostic round (stores), then a draft that changes only the arc (art_sweep -10 deg in the task's
         defaults, a copy of the package), each a separate `run` call, on the same 15 starts.
`on` adds --set memo=true memo_dir=STORE memo_sig=loop_level; `off` runs the same calls without. Writes timings.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import common as X

PY = "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
PRACTICE = "/workspace/code/locomanipulation-esc/outputs/escalation_20261116/practice2"
ROOT = Path(__file__).resolve().parents[3]
FACTORS = [1.0, 0.6, 0.8, 1.25, 1.5]


def call(a, tag: str, setting: dict, skills: str, ids_file: Path, log) -> dict:
    out = Path(a.out) / f"{tag}.jsonl"
    run_dir = Path(a.out) / f"run_{tag}"
    shutil.rmtree(run_dir, ignore_errors=True)
    out.unlink(missing_ok=True)
    argv = [PY, "-m", "locomani.bench.simple", "run", a.task, "--ids", str(ids_file), "--setting", json.dumps(setting),
            "--out", str(out), "--skills", skills, "--bench-set", "target=perception", "--bench-set", "sensor=stereo"]  # fmt: skip
    sets = ["kinds=Q", f"practice_dir={PRACTICE}", f"run_dir={run_dir}", "lazy_render=true", "lazy_warm=48", f"slots={a.slot}",
            f"slots_dir={Path(a.out) / 'slots'}"]  # fmt: skip
    if a.cond == "on":
        sets += ["memo=true", f"memo_dir={a.store}", "memo_sig=loop_level"]
    for kv in sets:
        argv += ["--set", kv]
    env = dict(os.environ, PYTHONPATH="src", NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2", MUJOCO_GL="egl")
    t0 = time.time()
    p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = p.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(20)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        rc = "timeout"
    wall = time.time() - t0
    rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    res = dict(tag=tag, rc=rc, wall_s=round(wall, 1), n=len(rows), setting=setting, skills=skills,
               success=[bool(r.get("success")) for r in rows], ids=[r.get("episode_id") for r in rows],
               t_success=[r.get("t_success") for r in rows], loss=[r.get("loss") for r in rows],
               counted=[r.get("counted_steps") for r in rows], row_wall=[r.get("wall_s") for r in rows],
               memo=[r.get("memo") for r in rows])  # fmt: skip
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("memo", "ids", "row_wall", "t_success", "loss", "counted")}
        ),
        flush=True,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["sweep", "draft", "repeat"])
    ap.add_argument("cond", choices=["on", "off"])
    ap.add_argument("--task", default="CloseDoorTeleop")
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ids = X.pool_ids(a.task)[: a.n]
    ids_file = Path(a.out) / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")
    base_speed = 0.45  # CloseDoorTeleop's art_speed
    results = []
    with open(Path(a.out) / "driver.log", "a") as log:
        if (
            a.scenario == "repeat"
        ):  # one value again: memo on (everything restored) or off (live): is a repeat the same round?
            f = 1.25 if a.cond == "on" else 1.0
            results.append(
                call(a, f"repeat_{f}", dict(X.SETTING, art_speed=round(base_speed * f, 6)), X.PKG, ids_file, log)
            )
        elif a.scenario == "sweep":
            for f in FACTORS:
                results.append(
                    call(a, f"sweep_{f}", dict(X.SETTING, art_speed=round(base_speed * f, 6)), X.PKG, ids_file, log)
                )
        else:
            results.append(call(a, "incumbent", dict(X.SETTING), X.PKG, ids_file, log))
            draft = Path(a.out) / "draft_pkg"
            shutil.rmtree(draft, ignore_errors=True)
            shutil.copytree(X.PKG, draft, ignore=shutil.ignore_patterns("__pycache__"))
            sk = (draft / "skill.py").read_text()
            i = sk.index("CloseDoorTeleop=dict(")
            j = sk.index("art_sweep=80.0", i)
            (draft / "skill.py").write_text(sk[:j] + "art_sweep=70.0" + sk[j + len("art_sweep=80.0") :])  # the arc only
            results.append(call(a, "draft", dict(X.SETTING), str(draft), ids_file, log))
    Path(a.out, "timings.json").write_text(json.dumps(results, indent=1, default=str))
    print("done", sum(r["wall_s"] for r in results))


if __name__ == "__main__":
    main()
