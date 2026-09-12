"""Run pinned, paired GPU regression batches through the existing launchers.

This experiment keeps gate tolerances unchanged and records fresh baselines.
All GPU batches run sequentially to avoid cross-task timing contention.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/workspace/code/locomanipulation-baseline-4cc52ba")
CAND = Path("/workspace/code/locomanipulation-maintenance-20260912")
TREES = {"baseline": (BASE, "4cc52ba"), "candidate": (CAND, "3eca6bb")}
PYTHON = ROOT / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
SEED = 20260912


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_idle(gpu):
    used = int(
        subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )
    if used > 64:
        raise RuntimeError(f"GPU {gpu} has {used} MiB allocated; refusing to contend with another run")


def environment(tree, instance, gpu, scene):
    # Preserve only host bootstrap settings; exclude inherited experiment knobs.
    keep = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        LOCOMANI_ROOT=str(tree),
        LOCOMANI_WORKSPACE="/workspace",
        PYTHONPATH=str(tree / "src"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        EXPORTER_THREADS="1",
        INSTANCE=str(instance),
        CUDA_VISIBLE_DEVICES=str(gpu),
        MUJOCO_EGL_DEVICE_ID=str(gpu),
        LOCOMANI_SIM_DIR=str(scene),
        SONIC_POLICY="policy/sonic_v1_1",
        SCENE_RANDOM="0",
        SIM_VIS="none",
        NO_ENCODER="0",
        HAND="dex1",
        WRIST_CAMS="1",
        LOCKSTEP="0",
    )
    return env


def run(cmd, tree, env, directory, name, timeout=1800, allowed=(0,)):
    cmd = [str(value) for value in cmd]
    print(f"{directory.name}/{name}: start", flush=True)
    dump(
        directory / f"{name}.command.json",
        {
            "argv": cmd,
            "cwd": str(tree),
            "settings": {
                key: value
                for key, value in env.items()
                if key not in {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL"}
            },
        },
    )
    start = time.time()
    with (directory / f"{name}.log").open("x") as log:
        result = subprocess.run(cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    dump(directory / f"{name}.result.json", {"rc": result.returncode, "start": start, "seconds": time.time() - start})
    print(f"{directory.name}/{name}: exit {result.returncode}", flush=True)
    if result.returncode not in allowed:
        raise RuntimeError(f"{directory}/{name}.log: exit {result.returncode}")
    return result.returncode


def stop_owned(tree, instance, directory):
    lock = Path(f"/workspace/tmp/locomani_locks/instance_{instance}.lock")
    if lock.exists() and str(directory) in lock.read_text():
        subprocess.run(
            ["bash", str(tree / "scripts/stack/stop_instance.sh"), str(instance)],
            check=True,
            timeout=30,
            capture_output=True,
        )
    session = Path("/workspace/tmp/p03" + (f"_i{instance}" if instance else ""))
    for name in ("sim.log", "deploy.log", "exporter.log"):
        source = session / name
        if source.exists():
            shutil.copy2(source, directory / name)


def pinned_trees():
    trees = {}
    for label, (tree, revision) in TREES.items():
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
        expected = subprocess.check_output(["git", "rev-parse", revision], cwd=tree, text=True).strip()
        if actual != expected or subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "configs", "scripts"], cwd=tree
        ):
            raise RuntimeError(f"{tree}: does not contain clean pinned code {revision}")
        trees[label] = {"path": str(tree), "revision": actual}
    return trees


def validate_replay_identity(reference):
    protocol = json.loads((reference / "protocol.json").read_text())
    if pinned_trees() != protocol["trees"]:
        raise ValueError("reference code revisions/worktrees differ from this experiment")
    deploy = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    files = [(deploy / name, expected) for name, expected in protocol["models"].items()]
    files += [
        (ROOT / "outputs/scene" / scene / name, expected)
        for scene, entries in protocol["scenes"].items()
        for name, expected in entries.items()
    ]
    for path, expected in files:
        if digest(path) != expected:
            raise ValueError(f"reference model/scene fingerprint changed: {path}")


def prepare(out, crate_gpu=1):
    provenance = {
        "seed": SEED,
        "trees": pinned_trees(),
        "scenes": {},
        "models": {},
        "schedule": "coupled A0 B0 A1 B1 A2 B2; teleop A B; crate A B",
        "gpu": f"bottle GPU 0 / instance 0; crate GPU {crate_gpu} / instance 1; sequential",
        "gate_tolerances": "unchanged scripts/gate.py",
        "limits": "50 rounds per revision; asynchronous controller is not deterministic",
    }
    for scene in ("sim", "crate_bimanual"):
        folder = ROOT / "outputs/scene" / scene
        provenance["scenes"][scene] = {
            p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()
        }
    deploy = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
    model_files = list((deploy / "policy/sonic_v1_1").glob("*"))
    model_files += [deploy / "target/release/g1_deploy_onnx_ref", deploy / "planner/target_vel/V2/planner_sonic.onnx"]
    provenance["models"] = {p.relative_to(deploy).as_posix(): digest(p) for p in model_files if p.is_file()}
    dump(out / "protocol.json", provenance)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"],
        stdout=(out / "gpu-before.csv").open("w"),
        check=True,
    )
    for strategy, rounds in (("coupled", 30), ("teleop", 10)):
        profile = "bottle_visible_teleop.json" if strategy == "teleop" else "bottle_visible.json"
        for label, (tree, _) in TREES.items():
            target = out / f"{strategy}_{label}_cases.json"
            env = environment(tree, 0, 0, ROOT / "outputs/scene/sim")
            run(
                [
                    PYTHON,
                    "-u",
                    tree / "src/locomani/tasks/bottle/grasp_gym.py",
                    "--task-config",
                    tree / "configs/tasks" / profile,
                    "--rounds",
                    rounds,
                    "--seed",
                    SEED,
                    "--sample-only",
                    "--write-cases",
                    target,
                ],
                tree,
                env,
                out,
                f"sample_{strategy}_{label}",
                timeout=120,
            )
        baseline = json.loads((out / f"{strategy}_baseline_cases.json").read_text())
        candidate = json.loads((out / f"{strategy}_candidate_cases.json").read_text())
        if baseline != candidate or len(baseline["cases"]) != rounds:
            raise RuntimeError(f"{strategy}: case generation differs or is incomplete")
        yaw_rng = random.Random(SEED)
        for case in baseline["cases"]:
            case["object_yaw"] = yaw_rng.uniform(-3.141592653589793, 3.141592653589793)
        for batch in range(rounds // 10):
            dump(
                out / f"{strategy}_cases_{batch}.json",
                {**baseline, "cases": baseline["cases"][batch * 10 : (batch + 1) * 10]},
            )


def bottle(out, strategy, label, batch):
    require_idle(0)
    tree, _ = TREES[label]
    directory = out / f"{strategy}_{label}_{batch}"
    directory.mkdir()
    (directory / "videos").mkdir()
    (directory / "trace").mkdir()
    env = environment(tree, 0, 0, ROOT / "outputs/scene/sim")
    env.update(SIM_VIDEO_DIR=str(directory / "videos"), SIM_STATE_TRACE_DIR=str(directory / "trace"))
    profile = "bottle_visible_teleop.json" if strategy == "teleop" else "bottle_visible.json"
    try:
        run(["bash", tree / "scripts/stack/start_stack.sh", "pick up the water bottle"], tree, env, directory, "stack")
        run(
            [
                PYTHON,
                "-u",
                tree / "src/locomani/tasks/bottle/grasp_gym.py",
                "--task-config",
                tree / "configs/tasks" / profile,
                "--cases",
                out / f"{strategy}_cases_{batch}.json",
                "--rounds",
                10,
                "--seed",
                SEED,
                "--stand-retries",
                0,
                "--record",
                "--video-include-init",
                "--manifest",
                directory / "manifest.jsonl",
            ],
            tree,
            env,
            directory,
            "gym",
        )
    finally:
        stop_owned(tree, 0, directory)
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != 10:
        raise RuntimeError(f"{directory}: expected 10 rows, got {len(rows)}")
    print(f"{directory.name}: {sum(bool(row.get('ok')) for row in rows)}/10 ok", flush=True)


def crate(out, label, gpu=1):
    require_idle(gpu)
    tree, _ = TREES[label]
    directory = out / f"crate_{label}"
    directory.mkdir()
    env = environment(tree, 1, gpu, ROOT / "outputs/scene/crate_bimanual")
    try:
        run(
            [
                PYTHON,
                "-u",
                tree / "src/locomani/pipeline/run_crate_pilot.py",
                "--expert",
                "wall",
                "--out",
                directory,
                "--scene",
                ROOT / "outputs/scene/crate_bimanual",
                "--rounds",
                10,
                "--seed",
                SEED,
                "--instance",
                1,
                "--gpu",
                gpu,
            ],
            tree,
            env,
            directory,
            "pilot",
        )
    finally:
        stop_owned(tree, 1, directory)
    rows = [json.loads(line) for path in directory.glob("manifest_*.jsonl") for line in path.read_text().splitlines()]
    if len(rows) != 10:
        raise RuntimeError(f"{directory}: expected 10 rows, got {len(rows)}")


def manifest_rows(out, strategy, label):
    return [
        json.loads(line)
        for path in sorted(out.glob(f"{strategy}_{label}_*/manifest.jsonl"))
        for line in path.read_text().splitlines()
    ]


def check_pairing(baseline, candidate):
    """Check delivered initial conditions, including yaw absent from old case banks."""
    keyed = []
    for rows in (baseline, candidate):
        by_id = {row["case_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("duplicate manifest case IDs")
        keyed.append(by_id)
    if not keyed[0] or keyed[0].keys() != keyed[1].keys():
        raise ValueError("baseline/candidate case IDs differ or are empty")
    for case_id, before in keyed[0].items():
        after = keyed[1][case_id]
        for field in ("init", "appearance", "target", "table_top", "obj_height", "obj_diameter"):
            if before[field] != after[field]:
                raise ValueError(f"{case_id}: delivered {field} differs")
        yaw = before["execution"]["metrics"]["object_yaw"]
        other_yaw = after["execution"]["metrics"]["object_yaw"]
        if yaw != other_yaw:
            raise ValueError(f"{case_id}: delivered object_yaw differs: {yaw} != {other_yaw}")
    return len(keyed[0])


def prepare_replay(out, reference):
    protocol = json.loads((reference / "protocol.json").read_text())
    protocol.update(
        reference_run=str(reference),
        schedule="candidate coupled batches 0,1,2; candidate teleop batch 0",
        pairing="Replay every baseline case with its recorded commanded object_yaw; verify manifests",
    )
    dump(out / "protocol.json", protocol)
    for strategy, rounds in (("coupled", 30), ("teleop", 10)):
        rows = manifest_rows(reference, strategy, "baseline")
        by_id = {row["case_id"]: row for row in rows}
        bank = json.loads((reference / f"{strategy}_baseline_cases.json").read_text())
        if len(rows) != rounds or len(by_id) != rounds or {c["case_id"] for c in bank["cases"]} != by_id.keys():
            raise ValueError(f"{strategy}: reference baseline is incomplete or duplicated")
        for case in bank["cases"]:
            case["object_yaw"] = by_id[case["case_id"]]["execution"]["metrics"]["object_yaw"]
        for batch in range(rounds // 10):
            dump(
                out / f"{strategy}_cases_{batch}.json", {**bank, "cases": bank["cases"][batch * 10 : (batch + 1) * 10]}
            )


def compare(out, strategy, baseline_out=None):
    arm = "crate" if strategy == "crate" else "bottle"
    patterns = {
        label: str(
            (baseline_out if label == "baseline" and baseline_out else out)
            / (f"crate_{label}/manifest_*.jsonl" if arm == "crate" else f"{strategy}_{label}_*/manifest.jsonl")
        )
        for label in TREES
    }
    if arm == "bottle":
        count = check_pairing(
            manifest_rows(baseline_out or out, strategy, "baseline"), manifest_rows(out, strategy, "candidate")
        )
        dump(out / f"{strategy}_pairing.json", {"paired_cases": count, "delivered_inputs_equal": True})
    base = out / f"{strategy}_fresh_baseline.json"
    env = environment(CAND, 0, 0, ROOT / "outputs/scene/sim")
    run(
        [PYTHON, ROOT / "scripts/gate.py", "record", arm, patterns["baseline"], base],
        ROOT,
        env,
        out,
        f"{strategy}_record",
    )
    run(
        [
            PYTHON,
            ROOT / "scripts/gate.py",
            "compare",
            arm,
            patterns["candidate"],
            base,
            "--json",
            out / f"{strategy}_comparison.json",
        ],
        ROOT,
        env,
        out,
        f"{strategy}_compare",
        allowed=(0, 1),
    )
    if arm == "crate":
        write_crate_acceptance(out)


def crate_coverage(rows, *, require_reach_time=True):
    """The current crate gate requires wall geometry and successful phase timings."""
    successful = [row for row in rows if row.get("ok")]
    counts = {}
    for field in ("stand_off_meas", "depth", "yaw_err_deg", "reach_s", "descend_s"):
        population = successful if field in ("reach_s", "descend_s") else rows
        counts[field] = sum(type(row.get(field)) in (int, float) and math.isfinite(row[field]) for row in population)
    not_applicable = [] if require_reach_time else ["reach_s"]
    missing = [name for name, count in counts.items() if count == 0 and name not in not_applicable]
    if not successful:
        missing.append("successful crate attempts")
    return {
        "rounds": len(rows),
        "successful": len(successful),
        "metric_samples": counts,
        "not_applicable": not_applicable,
        "missing": missing,
        "complete": not missing,
    }


def write_crate_acceptance(out):
    coverage = {}
    for label in TREES:
        folder = out / f"crate_{label}"
        rows = [
            json.loads(line)
            for path in sorted((out / f"crate_{label}").glob("manifest_*.jsonl"))
            for line in path.read_text().splitlines()
        ]
        command = (
            json.loads((folder / "pilot.command.json").read_text()) if (folder / "pilot.command.json").exists() else {}
        )
        argv = command.get("argv", [])
        expert = argv[argv.index("--expert") + 1] if "--expert" in argv else None
        task = json.loads((folder / "task_config.json").read_text()) if (folder / "task_config.json").exists() else {}
        # wall_phases takes a direct hover pose when wall_walk is false. Only
        # the walk-in path executes and times the continuous reach trajectory.
        require_reach = not (expert == "wall" and task.get("wall_walk") is False)
        coverage[label] = crate_coverage(rows, require_reach_time=require_reach)
        coverage[label]["expert"] = expert
    raw = json.loads((out / "crate_comparison.json").read_text())["verdict"]
    complete = all(value["complete"] for value in coverage.values())
    verdict = "INCOMPLETE" if raw == "PASS" and not complete else raw
    result = {"verdict": verdict, "raw_gate_verdict": raw, "coverage": coverage, "accepted": raw == "PASS" and complete}
    dump(out / "crate_acceptance.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--replay-candidate-from",
        type=Path,
        help="replay all baseline cases from an existing run using its observed object yaws",
    )
    ap.add_argument("--with-crate", action="store_true", help="also run both crate revisions after candidate replay")
    ap.add_argument("--crate-gpu", type=int, default=1)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "protocol.json").exists():
            raise SystemExit("this experiment directory already has a protocol; use a fresh directory")
        dump(out / "runner.json", {"pid": os.getpid(), "started": time.time()})

        def interrupt(*_):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, interrupt)
        shutil.copy2(__file__, out / "runner_source.py")
        try:
            if args.replay_candidate_from:
                reference = args.replay_candidate_from.resolve()
                validate_replay_identity(reference)
                prepare_replay(out, reference)
                print("waiting for the reference schedule to release its run lock", flush=True)
                with (reference / "run.lock").open("a") as reference_lock:
                    fcntl.flock(reference_lock, fcntl.LOCK_SH)
                # prepare_replay already requires every baseline bottle row. An
                # unrelated later stage may have stopped (e.g. a busy crate GPU).
                protocol = json.loads((out / "protocol.json").read_text())
                protocol["reference_schedule_complete"] = (reference / "done.json").exists()
                protocol["crate_gpu"] = args.crate_gpu if args.with_crate else None
                protocol["gpu"] = "bottle GPU 0; crate GPU " + str(args.crate_gpu) + "; sequential"
                if args.with_crate:
                    protocol["schedule"] += "; crate baseline and candidate"
                dump(out / "protocol.json", protocol)
                for strategy, batches in (("coupled", 3), ("teleop", 1)):
                    for batch in range(batches):
                        bottle(out, strategy, "candidate", batch)
                    compare(out, strategy, baseline_out=reference)
                if args.with_crate:
                    for label in TREES:
                        crate(out, label, args.crate_gpu)
                    compare(out, "crate")
                dump(out / "done.json", {"time": time.time()})
                return
            prepare(out, args.crate_gpu)
            for batch in range(3):
                for label in TREES:
                    bottle(out, "coupled", label, batch)
            compare(out, "coupled")
            for label in TREES:
                bottle(out, "teleop", label, 0)
            compare(out, "teleop")
            for label in TREES:
                crate(out, label, args.crate_gpu)
            compare(out, "crate")
        except BaseException as error:
            dump(out / "failed.json", {"error": repr(error), "time": time.time()})
            raise
        dump(out / "done.json", {"time": time.time()})


if __name__ == "__main__":
    main()
