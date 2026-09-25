"""Smoke test of the Isaac mirror against a live eval stack (TODO 4.7.9.10); no checkpoint needed.

Asks the mirror for images the way ObsBuilder does (MirrorClient, one request per --interval), and
checks what a closed loop depends on: every request answered, seq moving forward, the state the images
show only milliseconds old. For five of the requests it renders MuJoCo at the exact qpos the mirror
reported (by seq, from its own unconflated subscription to the privileged stream) and writes Isaac |
MuJoCo side by side, so a wrong scene or a wrong camera shows.

    bash scripts/start_stack.sh eval <experiment with eval.image_source: isaac> --name <run> --gpu 1
    bash scripts/run_isaac_mirror.sh outputs/runs/<run> --gpu 1
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "$VENV_SIM_PY" \
        scripts/experiments/isaac_mirror_smoke_20260924.py outputs/runs/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402
from locomani.sim.isaac.mirror import MirrorClient  # noqa: E402
from locomani.stack import Stack  # noqa: E402


def qpos_by_seq(port: int, keep: dict, stop: threading.Event) -> None:
    """Every privileged message's qpos by seq (no CONFLATE: the mirror may pick any of them)."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://localhost:{port}")
    while not stop.is_set():
        if sock.poll(100):
            msg = json.loads(sock.recv_string())
            if "qpos" in msg:
                keep[int(msg["seq"])] = np.asarray(msg["qpos"])
                if len(keep) > 3000:
                    for s in sorted(keep)[:1000]:
                        keep.pop(s, None)


def pct(xs, q):
    return round(float(np.percentile(np.asarray(xs, float), q)), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--interval", type=float, default=0.4)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    cfg = __import__("yaml").safe_load((run / "config.yaml").read_text())
    stack = Stack.attach(int(cfg["collect"]["instance"]))
    ports = stack.spec.ports()
    scene_xml = stack.spec.scene_path() / "scene_locomani.xml"
    out = run / "eval" / "mirror_smoke"
    out.mkdir(parents=True, exist_ok=True)

    keep, stop = {}, threading.Event()
    th = threading.Thread(target=qpos_by_seq, args=(ports["TF_PRIVILEGED_PORT"], keep, stop), daemon=True)
    th.start()
    client = MirrorClient(ports["TF_ISAAC_MIRROR_PORT"], timeout_ms=5000)
    end = time.time() + 900
    while not client.ping():
        if time.time() > end:
            raise SystemExit(f"mirror never answered: {client.last_error}")
        time.sleep(2)

    headers, errors, shots = [], [], []
    pick = set(np.linspace(0, args.n - 1, 5).astype(int).tolist())
    for i in range(args.n):
        t0 = time.time()
        got = client.render()
        wall = time.time() - t0
        if got is None:
            errors.append(client.last_error)
        else:
            images, header = got
            headers.append(dict(header, request_s=round(wall, 4)))
            if i in pick:
                shots.append((header["seq"], images))
        time.sleep(max(0.0, args.interval - (time.time() - t0)))
    stop.set()
    client.close()

    seqs = [h["seq"] for h in headers]
    report = dict(
        requests=args.n,
        answered=len(headers),
        errors=errors[:5],
        seq_strictly_increasing=all(b > a for a, b in zip(seqs, seqs[1:])),
        seq_step_p50=pct(np.diff(seqs), 50) if len(seqs) > 1 else None,
        render_ms=dict(p50=pct([h["render_ms"] for h in headers], 50), p90=pct([h["render_ms"] for h in headers], 90)),
        state_age_ms=dict(
            p50=pct([1e3 * h["state_age_s"] for h in headers], 50),
            p90=pct([1e3 * h["state_age_s"] for h in headers], 90),
        ),
        request_ms=dict(
            p50=pct([1e3 * h["request_s"] for h in headers], 50), p90=pct([1e3 * h["request_s"] for h in headers], 90)
        ),
        stack_gpu=str(stack.spec.gpu),
    )

    import cv2
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    rs = {k: mujoco.Renderer(model, c["height"], c["width"]) for k, c in CAMERA_CONFIGS.items()}
    rows, missing = [], 0
    for seq, images in shots:
        if seq not in keep:
            missing += 1
            continue
        data.qpos[:] = keep[seq]
        mujoco.mj_forward(model, data)
        cells = []
        for k, c in CAMERA_CONFIGS.items():
            rs[k].update_scene(data, camera=c["mjcf_name"])
            pair = np.concatenate([images[k], rs[k].render()], axis=1)
            cv2.putText(
                pair, f"seq {seq} {k}: isaac | mujoco", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cells.append(pair)
        rows.append(np.concatenate(cells, axis=1))
    report["side_by_side_rows"], report["side_by_side_missing_qpos"] = len(rows), missing
    if rows:
        grid = np.concatenate(rows, axis=0)
        grid = cv2.resize(grid, (grid.shape[1] // 3, grid.shape[0] // 3), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "mirror_vs_mujoco.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "headers.jsonl").write_text("".join(json.dumps(h) + "\n" for h in headers))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
