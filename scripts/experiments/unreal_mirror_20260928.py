"""The Isaac mirror's protocol served by Unreal (URLab): a VLA closed loop that sees Unreal pixels, no other change.

Drop-in for scripts/run_isaac_mirror.sh with `images.source: isaac`: it answers {"op": "render"} / {"op": "ping"}
on the stack's TF_ISAAC_MIRROR_PORT with the same header and frames (src/locomani/sim/isaac/mirror.py), rendering
the newest privileged state (qpos by joint name) in a running URLab editor through the patched "fresh" capture
(patches/urlab_fresh_capture.md). Every rendered frame is also written to --out as JPEG for the side-by-side.

Run it from the urlab_bridge venv after the stack, with the robot + carton imported into the warehouse level
(unreal_smoke_20260928.py setup, unreal_robot_materials_20260928.py, unreal_no_nanite_20260928.py; it refuses to
start while any mesh is Nanite):

    "$BRIDGE_PY" scripts/experiments/unreal_mirror_20260928.py --sim-scene <stack scene_locomani.xml> \
        --privileged-port 5601 --mirror-port 5578 --out <run>/eval/unreal
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unreal_smoke_20260928 import require_render_defaults  # noqa: E402

from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402  (plain numpy)

STREAMS = tuple(CAMERA_CONFIGS)
WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}


def joint_table(model) -> dict[str, tuple[int, int]]:
    return {model.joint(j).name: (int(model.jnt_qposadr[j]), WIDTH[int(model.jnt_type[j])]) for j in range(model.njnt)}


def qpos_map(src: dict, dst_model) -> list[tuple[int, int, int]]:
    """(sim adr, URLab adr, width) per URLab joint; URLab prefixes names with the actor, so match by suffix."""
    out = []
    for j in range(dst_model.njnt):
        name = dst_model.joint(j).name
        key = next((k for k in src if name == k or name.endswith("_" + k) or name.endswith("/" + k)), None)
        if key is None:
            raise SystemExit(f"URLab joint {name!r} is not in the sim scene")
        s, w = src[key]
        out.append((s, int(dst_model.jnt_qposadr[j]), w))
    return out


class UnrealView:
    def __init__(self, host: str, sim_scene: Path):
        from urlab_client import URLabClient

        self.client = URLabClient(host, local_model=False, recv_timeout_ms=60000)
        try:
            self.client.connect()
        except ValueError:
            pass
        if not self.client.manager_present:
            res = self.client.sim.start(raise_on_failure=False, timeout_s=300)
            print(f"[unreal-mirror] PIE {res.state} {res.compile_error[:200]!r}", flush=True)
        self.client.local_model = True
        self.client.connect()
        self.client.runtime.set_mode("puppet")
        names = self.client.camera_names()
        self.cams = {s: next(n for n in names if n.endswith(CAMERA_CONFIGS[s]["mjcf_name"])) for s in STREAMS}
        self.all_cams = names
        sim = mujoco.MjModel.from_xml_path(str(sim_scene))
        self.nq = sim.nq
        self.map = qpos_map(joint_table(sim), self.client.model)
        print(f"[unreal-mirror] cameras {self.cams}; sim nq {sim.nq}, URLab nq {self.client.model.nq}", flush=True)

    def _push(self, include=None) -> dict:
        d = self.client.data
        req = {
            "mode": "puppet",
            "n_steps": 0,
            "observations": "minimal",
            "time": float(d.time),
            "qpos": np.asarray(d.qpos, dtype=np.float64).tolist(),
            "qvel": [0.0] * self.client.model.nv,
            "per_articulation": {},
        }
        if include:
            req["include_cameras"] = include
        return self.client._rpc("step", req, expected_op="step_ok")

    def render(self, qpos: np.ndarray, cams=None) -> dict:
        d = self.client.data
        for s, t, w in self.map:
            d.qpos[t : t + w] = qpos[s : s + w]
        mujoco.mj_forward(self.client.model, d)
        self._push()
        want = cams or list(self.cams.values())
        reply = self._push({n: "fresh" for n in want})
        out = {}
        for stream, name in self.cams.items():
            blk = reply.get("cameras", {}).get(name)
            if blk is None:
                if cams is None:
                    raise RuntimeError(f"no frame for {name}")
                continue
            px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)
            out[stream] = np.ascontiguousarray(px[..., [2, 1, 0]])
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-scene", type=Path, required=True)
    ap.add_argument("--privileged-port", type=int, required=True)
    ap.add_argument("--mirror-port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    args = ap.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    require_render_defaults()
    view = UnrealView(args.host, args.sim_scene)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://localhost:{args.privileged_port}")
    latest = None
    deadline = time.time() + 60
    while latest is None and time.time() < deadline:
        if sub.poll(200):
            latest = json.loads(sub.recv_string())
    if latest is None or len(latest.get("qpos", [])) != view.nq:
        raise SystemExit(f"no usable privileged state on :{args.privileged_port} (nq {view.nq})")
    # one capture of every camera: each switches to on-demand, so unrequested ones stop rendering per frame
    view.render(np.asarray(latest["qpos"]), cams=view.all_cams)
    view.render(np.asarray(latest["qpos"]))

    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(f"tcp://*:{args.mirror_port}")
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    print(f"isaac mirror listening on :{args.mirror_port} (unreal; state from :{args.privileged_port})", flush=True)
    log = (args.out / "unreal_mirror.jsonl").open("a")
    while not stop:
        if not rep.poll(200):
            continue
        try:
            op = json.loads(rep.recv()).get("op")
        except ValueError:
            op = None
        if op == "ping":
            rep.send_multipart([json.dumps({"ok": True, "streams": list(STREAMS)}).encode()])
            continue
        if op != "render":
            rep.send_multipart([json.dumps({"ok": False, "error": f"unknown op {op!r}"}).encode()])
            continue
        while sub.poll(0):
            latest = json.loads(sub.recv_string())
        t0 = time.perf_counter()
        try:
            images = view.render(np.asarray(latest["qpos"], dtype=np.float64))
        except Exception as exc:  # noqa: BLE001  (answer the client rather than leave its REQ hanging)
            rep.send_multipart([json.dumps({"ok": False, "error": str(exc)}).encode()])
            continue
        header = dict(
            ok=True,
            seq=int(latest["seq"]),
            t=float(latest["t"]),
            state_wall=float(latest["wall"]),
            state_age_s=round(time.time() - float(latest["wall"]), 4),
            render_ms=round((time.perf_counter() - t0) * 1e3, 2),
            streams=list(STREAMS),
            shapes={k: list(images[k].shape) for k in STREAMS},
            renderer="unreal",
        )
        rep.send_multipart([json.dumps(header).encode(), *[images[k].tobytes() for k in STREAMS]])
        for k in STREAMS:
            Image.fromarray(images[k]).save(args.out / "frames" / f"{header['seq']:07d}_{k}.jpg", quality=92)
        log.write(json.dumps(header) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
