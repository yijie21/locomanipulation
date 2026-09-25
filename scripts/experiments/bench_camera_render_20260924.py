"""What one camera frame costs: three cameras serially in one process vs one process per camera (TODO 4.7.1).

Renders exactly what the sim's update_render_caches renders -- `mujoco.Renderer` at the CAMERA_CONFIGS size,
`update_scene` on the MJCF camera, `render()` -- over states sampled evenly from a recorded state trace, so
the poses and what is in view are those of a real episode. The live sim renders one set every IMAGE_DT =
1/30 s, so a set slower than 33.3 ms is a sim that cannot keep real time.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" \
        scripts/experiments/bench_camera_render_20260924.py outputs/diagnostics/render_speed_20260924
"""

from __future__ import annotations

import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402

RUNS = REPO / "outputs" / "runs"
SCENES = {
    "warehouse": (
        RUNS / "docrepro_20260924_warehouse_offline" / "warehouse_textured" / "scene_locomani.xml",
        RUNS / "docrepro_20260924_warehouse_offline" / "trace" / "visual_crate_s7000_00_state_trace.npz",
    ),
    # the live-render run the reproduction agent made (RTF 0.35): same carton mesh, pre-09-24 scene build
    "warehouse_live": (
        RUNS / "docrepro_20260924_warehouse" / "scene_reverse_camera" / "scene_locomani.xml",
        RUNS / "docrepro_20260924_warehouse" / "trace" / "visual_crate_s7000_00_state_trace.npz",
    ),
    "crate": (
        RUNS / "docrepro_20260924_crate" / "crate_table30_depth04" / "scene_locomani.xml",
        RUNS / "docrepro_20260924_crate" / "trace" / "visual_crate_s7000_00_state_trace.npz",
    ),
    "bottle": (
        RUNS / "docrepro_20260924_bottle" / "scene" / "scene_locomani.xml",
        RUNS / "docrepro_20260924_bottle" / "collect" / "state_trace" / "gym_1790257182_state_trace.npz",
    ),
}
STATES, WARMUP = 300, 20
IMAGE_DT_MS = 1000 / 30


def load(scene: Path, trace: Path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    qpos = np.load(trace)["qpos"]
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"{trace.name}: qpos width {qpos.shape[1]} != model nq {model.nq} of {scene}")
    pick = np.linspace(0, len(qpos) - 1, STATES).astype(int)
    return model, mujoco.MjData(model), qpos[pick]


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def render_loop(model, data, states, cams: list[str]) -> dict:
    """Per state: mj_forward, then update_scene + render per camera, as the sim does. Times in ms."""
    import mujoco

    renderers = {k: mujoco.Renderer(model, CAMERA_CONFIGS[k]["height"], CAMERA_CONFIGS[k]["width"]) for k in cams}
    for q in states[:WARMUP]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for k, r in renderers.items():
            r.update_scene(data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
            r.render()
    fwd, scene, draw, frame = [], {k: [] for k in cams}, {k: [] for k in cams}, []
    c0, w0 = cpu_s(), time.perf_counter()
    for q in states:
        t0 = time.perf_counter()
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        fwd.append(time.perf_counter() - t0)
        for k, r in renderers.items():
            t1 = time.perf_counter()
            r.update_scene(data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
            t2 = time.perf_counter()
            r.render()
            t3 = time.perf_counter()
            scene[k].append(t2 - t1)
            draw[k].append(t3 - t2)
        frame.append(time.perf_counter() - t0)
    wall, cpu = time.perf_counter() - w0, cpu_s() - c0
    for r in renderers.values():
        r.close()
    ms = lambda xs: np.asarray(xs) * 1000  # noqa: E731
    return {
        "cams": cams,
        "frame_ms": ms(frame),
        "forward_ms": ms(fwd),
        "update_scene_ms": {k: ms(v) for k, v in scene.items()},
        "render_ms": {k: ms(v) for k, v in draw.items()},
        "wall_s": wall,
        "cpu_cores": cpu / wall,
    }


def _worker(name: str, cam: str, barrier, out) -> None:
    model, data, states = load(*SCENES[name])
    barrier.wait()  # every process has built its model and context; start the timed loops together
    res = render_loop(model, data, states, [cam])
    out.put((cam, res))


def stats(xs) -> dict:
    xs = np.asarray(xs)
    return {
        "p50": round(float(np.percentile(xs, 50)), 2),
        "p90": round(float(np.percentile(xs, 90)), 2),
        "max": round(float(xs.max()), 2),
    }


def run_scene(name: str) -> dict:
    cams = list(CAMERA_CONFIGS)
    model, data, states = load(*SCENES[name])
    ser = render_loop(model, data, states, cams)
    del model, data

    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(cams)), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(name, c, barrier, out)) for c in cams]
    for p in procs:
        p.start()
    par = dict(out.get() for _ in cams)
    for p in procs:
        p.join()

    par_wall = max(r["wall_s"] for r in par.values())
    return {
        "states": STATES,
        "serial": {
            "set_ms": stats(ser["frame_ms"]),
            "forward_ms": stats(ser["forward_ms"]),
            "per_camera": {
                k: {"update_scene_ms": stats(ser["update_scene_ms"][k]), "render_ms": stats(ser["render_ms"][k])}
                for k in cams
            },
            "sets_per_s": round(STATES / ser["wall_s"], 1),
            "cpu_cores": round(ser["cpu_cores"], 2),
        },
        "parallel": {
            "per_camera": {
                k: {
                    "frame_ms": stats(r["frame_ms"]),
                    "render_ms": stats(r["render_ms"][k]),
                    "cpu_cores": round(r["cpu_cores"], 2),
                }
                for k, r in par.items()
            },
            # every camera must deliver its frame before the set is complete: the slowest process sets the pace
            "sets_per_s": round(STATES / par_wall, 1),
            "cpu_cores": round(sum(r["cpu_cores"] for r in par.values()), 2),
        },
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "diagnostics" / "render_speed_20260924"
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or list(SCENES)
    results = {"image_dt_ms": round(IMAGE_DT_MS, 1), "cameras": CAMERA_CONFIGS, "scenes": {}}
    for name in names:
        print(f"[bench] {name} ...", flush=True)
        r = results["scenes"][name] = run_scene(name)
        s, p = r["serial"], r["parallel"]
        print(
            f"  serial   set p50 {s['set_ms']['p50']} / p90 {s['set_ms']['p90']} ms, {s['sets_per_s']} sets/s, "
            f"{s['cpu_cores']} cores",
            flush=True,
        )
        for k, c in s["per_camera"].items():
            print(
                f"    {k:12s} update_scene p50 {c['update_scene_ms']['p50']} ms  render p50 {c['render_ms']['p50']} / "
                f"p90 {c['render_ms']['p90']} ms",
                flush=True,
            )
        print(f"  parallel {p['sets_per_s']} sets/s, {p['cpu_cores']} cores", flush=True)
        for k, c in p["per_camera"].items():
            print(f"    {k:12s} frame p50 {c['frame_ms']['p50']} / p90 {c['frame_ms']['p90']} ms", flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"[bench] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
