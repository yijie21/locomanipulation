"""Frozen ScaleBFM 200-success collection -> LeRobot conversion -> statistics.

Prepare without simulation; run waits for the GPU/resource policy recorded in plan.json.
The original cancelled Goal is not used. See docs/ledger/scalebfm.md.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/experiments/scalebfm200_noretract_sonic_20260910"
PYTHON = ROOT / "third_party/Isaac-GR00T/.venv/bin/python"


def write(path, data):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(out):
    if (out / "plan.json").exists():
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Partial preparation: inspect this directory before retrying")
    out.mkdir(parents=True, exist_ok=True)
    code = out / "code"
    for package in ("locomani", "gr00t"):
        shutil.copytree(
            ROOT / "src" / package, code / "src" / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    shutil.copytree(ROOT / "configs", code / "configs")
    shutil.copy2(__file__, code / "runner.py")
    scene = out / "scene"
    scene.mkdir()
    for name in ("scene_locomani.xml", "scene_meta.json"):
        shutil.copy2(ROOT / "outputs/scene/sim" / name, scene / name)
    tree = ET.parse(scene / "scene_locomani.xml")
    if tree.findall(".//include"):
        raise ValueError("Scene includes require recursive freezing")
    assets = {scene / "scene_locomani.xml", scene / "scene_meta.json"}
    for element in tree.iter():
        if element.get("file"):
            path = Path(element.get("file"))
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"Scene asset is not a resolved file: {path}")
            assets.add(path)
    assets.update((ROOT / "assets/scalebfm").glob("*"))
    backend = json.loads((ROOT / "configs/backends/scalebfm.json").read_text())
    backend["scene_dir"] = str(scene)
    assets.add(Path(backend["checkpoint"]))
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    write(out / "backend.json", backend)
    manifest = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(assets) if p.is_file()}
    write(out / "runtime_assets.json", {"files": manifest, "new_run": True})
    plan = dict(
        out=str(out),
        target_episodes=200,
        max_attempts=800,
        scene_profile="yplus_center_v1",
        arm="B",
        recovery="none",
        seeds_start=4001,
        success_protocol="sonic_lift6cm_hold2s_v1",
        carry_back_m=0.0,
        max_s=24,
        gpu=0,
        rendering="egl",
        action_hz=25,
        state_dim=91,
        action_dim=29,
        selection="task success AND existing quality audit; cycle high_near/low_near/low_deep",
        scope="collection, conversion and statistics only; no training",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={str(p): digest(p) for p in sorted(code.rglob("*")) if p.is_file()},
        backend_sha256=digest(out / "backend.json"),
        runtime_manifest_sha256=digest(out / "runtime_assets.json"),
    )
    write(out / "plan.json", plan)
    write(out / "progress.json", dict(status="prepared", pid=None, updated=time.time()))


def busy_collectors():
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith(b"/run_learn_pilot.py") or arg.endswith(b"/grasp_gym.py") for arg in args):
            found.append(int(proc.name))
    return found


def run(out):
    plan = json.loads((out / "plan.json").read_text())
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "RESULTS.json").exists():
        return

    def status(stage, **extra):
        write(out / "progress.json", dict(status=stage, pid=os.getpid(), updated=time.time(), **extra))
        print(stage, extra, flush=True)

    try:
        gpu = str(plan["gpu"])
        shared = plan.get("allow_shared_gpu", False)
        while True:
            jobs = busy_collectors() if gpu == "0" else []
            memory, total, utilization = map(
                int,
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .split(","),
            )
            headroom = subprocess.check_output(["bash", str(ROOT / "scripts/cpu_headroom.sh"), "3"], text=True).strip()
            gpu_ready = total - memory >= 8192 and utilization < 20 if shared else memory < 500
            if not jobs and gpu_ready and "OK for a stack" in headroom:
                break
            status("waiting_for_resources", conflicting_pids=jobs, gpu=int(gpu), gpu_memory_mib=memory, cpu=headroom)
            time.sleep(30)
        for path, expected in plan["code_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Frozen source changed: {path}")
        if digest(out / "backend.json") != plan["backend_sha256"]:
            raise ValueError("Frozen backend configuration changed")
        if digest(out / "runtime_assets.json") != plan["runtime_manifest_sha256"]:
            raise ValueError("Frozen runtime manifest changed")
        if plan.get("rendering") != "egl":
            raise ValueError("Formal collection requires EGL; software rendering belongs in a separate diagnostic run")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gpu,
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID=str(plan.get("egl_device_id", gpu)),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            LP_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            LOCOMANI_ROOT=str(ROOT),
            PYTHONPATH=str(out / "code/src"),
        )

        def stage(name, module, args):
            command = [str(PYTHON), "-u", "-m", module, *map(str, args)]
            with (out / f"{name}.log").open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status(name, child_pid=child.pid, command=command)
                rc = child.wait()
            if rc:
                raise RuntimeError(f"{name} exited {rc}; inspect {name}.log")

        prefix = "locomani.tasks.bottle_visual.pipeline."
        gate_path = out / "egl_preflight_pass.json"
        if not gate_path.exists():
            stage(
                "egl_preflight",
                prefix + "collect",
                [
                    "--arm",
                    "B",
                    "--pilot",
                    "--out",
                    out / "egl_preflight",
                    "--scene-profile",
                    plan["scene_profile"],
                    "--backend",
                    out / "backend.json",
                    "--runtime-assets",
                    out / "runtime_assets.json",
                ],
            )
            cases = json.loads((out / "egl_preflight/pilot_inputs.json").read_text())
            attempts = json.loads((out / "egl_preflight/pilot_attempts.json").read_text())
            import statistics

            median_wall = statistics.median(row["wall_s"] for row in attempts)
            if len(cases) != 12 or median_wall > 60:
                raise ValueError(f"EGL preflight failed: accepted={len(cases)}/12 median wall={median_wall:.1f}s")
            # Decode every pilot video; audit already checks visibility and action/physics quality.
            for case in cases:
                for camera in ("ego_view", "right_wrist"):
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-xerror",
                            "-threads",
                            "1",
                            "-i",
                            str(Path(case) / f"{camera}.mp4"),
                            "-f",
                            "null",
                            "-",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
            write(
                gate_path,
                dict(
                    accepted=len(cases),
                    median_wall_s=median_wall,
                    rendering="egl",
                    egl_device_id=env["MUJOCO_EGL_DEVICE_ID"],
                    passed_at=time.time(),
                    video_decode="24/24 passed",
                ),
            )
        stage(
            "collect",
            prefix + "collect",
            [
                "--arm",
                "B",
                "--train-only",
                "--out",
                out / "collect",
                "--scene-profile",
                plan["scene_profile"],
                "--backend",
                out / "backend.json",
                "--runtime-assets",
                out / "runtime_assets.json",
            ],
        )
        inputs_path = out / "collect/train_inputs.json"
        inputs = json.loads(inputs_path.read_text())
        if len(inputs) != 200 or len(set(inputs)) != 200:
            raise ValueError(f"Collection ended with {len(inputs)} accepted episodes; conversion not started")
        for source in inputs:
            result = json.loads((Path(source) / "result.json").read_text())
            if not result["ok"] or result.get("success_protocol") != plan["success_protocol"]:
                raise ValueError(f"Wrong success protocol: {source}")
        stage("convert", prefix + "export", ["--inputs", inputs_path, "--out", out / "dataset"])
        stage(
            "stats",
            "gr00t.data.stats",
            [
                "--dataset-path",
                out / "dataset",
                "--embodiment-tag",
                "UNITREE_G1_SONIC",
                "--modality-config-path",
                out / "code/configs/gr00t_bfm_visual_2cam.py",
            ],
        )
        info = json.loads((out / "dataset/meta/info.json").read_text())
        if info["total_episodes"] != 200 or info["fps"] != 25:
            raise ValueError("Unexpected exported dataset count/rate")
        write(
            out / "RESULTS.json",
            dict(
                status="complete",
                dataset=str(out / "dataset"),
                episodes=200,
                frames=info["total_frames"],
                videos=info["total_videos"],
                success_protocol=plan["success_protocol"],
                training_started=False,
                rendering=plan["rendering"],
                rendering_history=plan.get("rendering_history", []),
                collection=json.loads((out / "collect/summary.json").read_text()),
            ),
        )
        status("complete", episodes=200, dataset=str(out / "dataset"))
    except Exception:
        write(out / "failure.json", dict(time=time.time(), traceback=traceback.format_exc()))
        status("failed", detail="See failure.json and stage logs; no automatic retry")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    destination = options.out.resolve()
    prepare(destination)
    if not options.prepare_only:
        run(destination)
