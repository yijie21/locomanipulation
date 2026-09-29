"""Smoke test URLab (MuJoCo inside Unreal) as the renderer that would replace the Isaac mirror.

Drives a running URLab editor over its bridge (ZMQ, step port 5559) from the urlab_bridge venv:

    setup      new level, MjManager, import the exported MJCF, spawn it, a directional light, save
    replay     puppet mode: push state-trace rows as qpos (mapped by joint name), read the three dataset cameras,
               write Unreal | MuJoCo side by side per row
    alternate  the pose-binding check: alternate two far-apart rows and classify every image against settled
               references of both; reports, per wait policy, how often the image shows the pose just pushed

At efbc5e2 the plugin pauses the camera ZMQ publishers in every step mode but live, and its "sync" camera read
returns whatever readback the tick has in flight: about half the images of a puppet push show the previous pose until
~200 ms after it. The loco-fresh-capture branch of the plugin (patches/urlab_fresh_capture.patch) adds
include_cameras "fresh": snapshot -> transforms -> capture -> synchronous readback inside the request. A capture is
two RPCs: push the pose, wait --delay-ms (0 with "fresh"), then a zero-step with include_cameras --camera-mode.

    . /workspace/ue_57/ue_env.sh
    MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_smoke_20260928.py setup --xml <abs robot.xml>
    MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_smoke_20260928.py replay --xml ... --trace <npz> \
        --trace-joints <scene_joints.json> --out outputs/unreal_smoke_20260928/robot
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

CAMERAS = ("head_camera", "left_wrist", "right_wrist")
ACTOR_ID = "g1"


def uecmd(code: str) -> str:
    """Run Python in the editor through ue_cmd_server.py (TCP 127.0.0.1:18801); returns its stdout."""
    s = socket.create_connection(("127.0.0.1", 18801), timeout=60)
    s.sendall(code.encode() + b"\n__END__\n")
    out = b""
    while chunk := s.recv(1 << 16):
        out += chunk
    return out.decode()


NANITE_CHECK = """
sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
world = unreal.EditorLevelLibrary.get_editor_world()
nanite = sorted({c.static_mesh.get_name() for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
                 for c in a.get_components_by_class(unreal.StaticMeshComponent)
                 if c.static_mesh and sub.get_nanite_settings(c.static_mesh).enabled})
cvar = unreal.SystemLibrary.get_console_variable_int_value
print(cvar('r.Shadow.Nanite'), cvar('ShowFlag.Rendering'), len(nanite), ' '.join(nanite[:8]))
"""


def require_render_defaults() -> None:
    """Refuse to render unless the editor has the render defaults: no Nanite mesh in the level, r.Shadow.Nanite 0,
    the editor viewport off (ShowFlag.Rendering 0).

    Nanite shadows cost 13 ms per capture however little moves (TODO 12.7.4), and the dataset / eval images are
    defined without them; the viewport shifts the captures' indirect light by ~2/255 (TODO 12.7.5). A fresh URLab
    import builds Nanite meshes, so this catches a skipped unreal_no_nanite_20260928.py. Needs the editor's command
    server (UE_EXEC_CMDS=... ue_cmd_server.py).
    """
    try:
        shadow, viewport, count, names = (uecmd(NANITE_CHECK).strip().split(" ", 3) + [""])[:4]
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[render-defaults] cannot ask the editor (ue_cmd_server on :18801): {exc}") from exc
    problems = []
    if int(count):
        problems.append(f"{count} Nanite meshes ({names}): run scripts/experiments/unreal_no_nanite_20260928.py")
    if int(shadow):
        problems.append(f"r.Shadow.Nanite {shadow}: the host project's DefaultEngine.ini sets 0")
    if int(viewport):
        problems.append(f"ShowFlag.Rendering {viewport}: the host project's DefaultEngine.ini sets 0")
    if problems:
        raise SystemExit("[render-defaults] " + "; ".join(problems))


def connect(host: str, *, local_model: bool):
    from urlab_client import URLabClient

    client = URLabClient(host, local_model=local_model, recv_timeout_ms=60000)
    try:
        client.connect()
    except ValueError as exc:  # first discover before PIE carries no model
        print(f"[connect] {exc}")
    return client


def cmd_setup(args) -> None:
    client = connect(args.host, local_model=False)
    if client.manager_present:
        client.sim.stop()
        time.sleep(1.0)
    if args.existing_level:
        client.scene.load_level(args.existing_level)
    else:
        client.scene.create_level(args.level, force_overwrite=True)
    client.scene.ensure_manager()
    bp = client.scene.import_xml(str(Path(args.xml).resolve()), force_reimport=True)
    print(f"[setup] imported {bp}")
    client.scene.spawn_actor(bp, actor_id=ACTOR_ID)
    if not args.existing_level:
        client.scene.spawn_light("directional", actor_id="sun", rotation_euler=(0.0, -50.0, 30.0))
    client.scene.save_level()
    # URLab's import (MjLevelOps ImportXmlSync) leaves the Blueprint in memory only; its meshes and material instances
    # are saved by the later steps, the level by save_level. Unsaved, the next editor session loads a level whose robot
    # actor has no class, and every session had to re-import (TODO 12.11).
    asset = bp.class_path.split(".")[0]  # /Game/MuJoCoImports/dynamic.dynamic_C -> /Game/MuJoCoImports/dynamic
    print(uecmd(f"print(unreal.EditorAssetLibrary.save_asset({asset!r}, only_if_is_dirty=False))").strip())
    print(f"[setup] level {client.scene.current_level()} and {asset} saved")
    client.close()


def start_puppet(args):
    client = connect(args.host, local_model=False)
    if not client.manager_present:
        res = client.sim.start(raise_on_failure=False, timeout_s=args.pie_timeout)
        print(f"[pie] state={res.state} compile_error={res.compile_error[:300]!r}")
    client.local_model = True
    client.connect()
    if client.model is None:
        raise SystemExit("[pie] handshake carried no model (MuJoCo version mismatch?)")
    client.runtime.set_mode("puppet")
    names = client.camera_names()
    cams = {c: next((n for n in names if n.split("/")[-1].endswith(c)), None) for c in CAMERAS}
    print(f"[pie] ue nq={client.model.nq} cameras={cams}")
    missing = [c for c, n in cams.items() if n is None]
    if missing:
        raise SystemExit(f"[pie] cameras {missing} not found among {names}")
    return client, cams


def qpos_map(src_joints: dict, dst_model) -> list[tuple[int, int, int]]:
    """(src adr, dst adr, width) per joint, matched by name with URLab's actor prefix stripped."""
    width = {0: 7, 1: 4, 2: 1, 3: 1}
    src = {j["name"]: (j["qposadr"], width[j["type"]]) for j in src_joints["joints"]}
    out = []
    for j in range(dst_model.njnt):
        name = dst_model.joint(j).name
        key = next((k for k in src if name == k or name.endswith("_" + k) or name.endswith("/" + k)), None)
        if key is None:
            raise SystemExit(f"[map] UE joint {name!r} has no trace column")
        s_adr, w = src[key]
        out.append((s_adr, int(dst_model.jnt_qposadr[j]), w))
    return out


def apply_row(client, mapping, row) -> None:
    for s, d, w in mapping:
        client.data.qpos[d : d + w] = row[s : s + w]
    client.data.qvel[:] = 0
    mujoco.mj_forward(client.model, client.data)


def push(client, include: dict | None = None) -> dict:
    """One puppet step of the current client.data (no integration); with `include`, the reply carries cameras."""
    request = {
        "mode": "puppet",
        "n_steps": 0,
        "observations": "minimal",
        "time": float(client.data.time),
        "qpos": np.asarray(client.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(client.data.qvel, dtype=np.float64).tolist(),
        "per_articulation": {},
    }
    if include:
        request["include_cameras"] = include
    return client._rpc("step", request, expected_op="step_ok")


CAMERA_MODE = "fresh"


def capture(client, cams, row_qpos, mapping, delay_ms: float) -> dict:
    """Push `row_qpos`, wait `delay_ms`, read the three cameras (RGB uint8) in CAMERA_MODE."""
    apply_row(client, mapping, row_qpos)
    push(client)
    if delay_ms > 0:
        time.sleep(delay_ms / 1e3)
    reply = push(client, {n: CAMERA_MODE for n in cams.values()})
    out = {}
    for c, n in cams.items():
        blk = reply.get("cameras", {}).get(n)
        if blk is None:
            raise SystemExit(f"[capture] no frame for {n}; reply keys {sorted(reply)}")
        img = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)
        out[c] = img[..., [2, 1, 0]].copy()  # BGRA -> RGB
    return out


def mujoco_views(xml: str, rows_qpos: list[np.ndarray], mapping) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    out = []
    with mujoco.Renderer(model, 480, 640) as r:
        for row in rows_qpos:
            for s, d, w in mapping:
                data.qpos[d : d + w] = row[s : s + w]
            mujoco.mj_forward(model, data)
            views = {}
            for c in CAMERAS:
                r.update_scene(data, camera=c)
                views[c] = r.render().copy()
            out.append(views)
    return out


def isaac_frames(video: str, indices: list[int]) -> list[np.ndarray]:
    """Frames `indices` of an Isaac video (render_isaac_videos.sh writes every 4th trace row), RGB."""
    import cv2

    cap = cv2.VideoCapture(video)
    out = []
    for k in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = cap.read()
        out.append(bgr[..., ::-1].copy() if ok else np.zeros((480, 640, 3), np.uint8))
    cap.release()
    return out


def cmd_replay(args) -> None:
    trace = np.load(args.trace)
    src_joints = json.loads(Path(args.trace_joints).read_text())
    client, cams = start_puppet(args)
    mapping = qpos_map(src_joints, client.model)
    local = mujoco.MjModel.from_xml_path(args.xml)
    local_map = qpos_map(src_joints, local)
    out = Path(args.out) / "replay"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(range(0, len(trace["qpos"]), args.stride))[: args.rows]
    ref = mujoco_views(args.xml, [trace["qpos"][i] for i in rows], local_map)
    isaac = isaac_frames(args.isaac_ego, [i // args.isaac_stride for i in rows]) if args.isaac_ego else None
    timing = []
    for n, i in enumerate(rows):
        t0 = time.perf_counter()
        frames = capture(client, cams, trace["qpos"][i], mapping, args.delay_ms)
        t1 = time.perf_counter()
        timing.append({"row": int(i), "seq": int(trace["seq"][i]), "capture_ms": (t1 - t0) * 1e3})
        top = np.concatenate([frames[c] for c in CAMERAS], axis=1)
        bottom = np.concatenate([ref[n][c] for c in CAMERAS], axis=1)
        grid = [top, bottom]
        if isaac is not None:
            row = [isaac[n] if c == "head_camera" else np.zeros_like(top[:, :640]) for c in CAMERAS]
            grid.append(np.concatenate(row, axis=1))
        Image.fromarray(np.concatenate(grid, axis=0)).save(out / f"row_{i:05d}.png")
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    ms = np.array([t["capture_ms"] for t in timing])
    print(
        f"[replay] {len(rows)} rows -> {out}; capture (delay {args.delay_ms:g} ms) p50 {np.median(ms):.1f} ms p90 {np.percentile(ms, 90):.1f} ms"
    )
    client.close()


def cmd_alternate(args) -> None:
    trace = np.load(args.trace)
    src_joints = json.loads(Path(args.trace_joints).read_text())
    client, cams = start_puppet(args)
    mapping = qpos_map(src_joints, client.model)
    rows = {"A": args.rows_ab[0], "B": args.rows_ab[1]}
    refs = {lab: capture(client, cams, trace["qpos"][rows[lab]], mapping, 1000.0) for lab in ("A", "B")}
    again = {lab: capture(client, cams, trace["qpos"][rows[lab]], mapping, 1000.0) for lab in ("A", "B")}
    sep = {c: float(np.abs(refs["A"][c].astype(np.int16) - refs["B"][c]).mean()) for c in CAMERAS}
    noise = {
        c: max(float(np.abs(refs[lab][c].astype(np.int16) - again[lab][c]).mean()) for lab in "AB") for c in CAMERAS
    }
    print(f"[alternate] A|B separation per camera {sep}; settled re-capture noise {noise}")
    usable = [c for c in CAMERAS if sep[c] > 5 * max(noise[c], 0.2)]
    print(f"[alternate] cameras that can tell A from B: {usable}")
    report = {"separation": sep, "noise": noise, "usable": usable}
    for delay in args.delays_ms:
        wrong = {c: 0 for c in usable}
        ms = []
        for n in range(args.cycles * 2):
            lab = "AB"[n % 2]
            t0 = time.perf_counter()
            frames = capture(client, cams, trace["qpos"][rows[lab]], mapping, delay)
            ms.append((time.perf_counter() - t0) * 1e3)
            for c in usable:
                img = frames[c].astype(np.int16)
                d_own = np.abs(img - refs[lab][c]).mean()
                d_other = np.abs(img - refs["B" if lab == "A" else "A"][c]).mean()
                wrong[c] += bool(d_own >= d_other)
        report[f"delay_{delay:g}ms"] = {
            "per_camera_previous_pose": wrong,
            "captures": args.cycles * 2,
            "capture_ms_p50": float(np.median(ms)),
        }
        print(
            f"[alternate] delay {delay:g} ms: previous pose {wrong} of {args.cycles * 2}; capture p50 {np.median(ms):.0f} ms"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alternate.json").write_text(json.dumps({"rows": rows, "report": report}, indent=1))
    client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--pie-timeout", type=float, default=300.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--xml", required=True)
    s.add_argument("--level", default="URLabSmoke")
    for name in ("replay", "alternate"):
        p = sub.add_parser(name)
        p.add_argument("--xml", required=True, help="the exported MJCF (MuJoCo reference renders)")
        p.add_argument("--trace", required=True)
        p.add_argument("--trace-joints", required=True, help="<variant>_joints.json of the model that wrote the trace")
        p.add_argument("--out", required=True)
    sub.choices["replay"].add_argument("--rows", type=int, default=12)
    sub.choices["replay"].add_argument("--stride", type=int, default=240)
    sub.choices["replay"].add_argument("--delay-ms", type=float, default=0.0)
    sub.choices["alternate"].add_argument("--rows-ab", type=int, nargs=2, default=[0, 1500])
    sub.choices["alternate"].add_argument("--cycles", type=int, default=10)
    sub.choices["alternate"].add_argument("--delays-ms", type=float, nargs="+", default=[0, 20, 50, 100, 200])
    for name in ("replay", "alternate"):
        sub.choices[name].add_argument("--camera-mode", choices=["fresh", "sync"], default="fresh")
        sub.choices[name].add_argument(
            "--cameras", nargs="+", help="MJCF camera names (default: the three dataset ones)"
        )
    sub.choices["setup"].add_argument("--existing-level", help="load this level instead of creating --level")
    sub.choices["replay"].add_argument("--isaac-ego", help="Isaac ego video of the same trace (third row)")
    sub.choices["replay"].add_argument("--isaac-stride", type=int, default=4)
    args = ap.parse_args()
    global CAMERA_MODE, CAMERAS
    CAMERA_MODE = getattr(args, "camera_mode", CAMERA_MODE)
    CAMERAS = tuple(getattr(args, "cameras", None) or CAMERAS)
    {"setup": cmd_setup, "replay": cmd_replay, "alternate": cmd_alternate}[args.cmd](args)


if __name__ == "__main__":
    main()
