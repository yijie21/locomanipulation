"""Vision RSI D2 step 2d (docs/handoff/vision-rsi-design-20261023.md section 17.4): each pool round's memory of its target,
from the round's own mapping tour. The tour's walk is the same every round (tour_record.py); what differs is where the
target stands, so the frames that see its support surface are re-rendered with the round's placements (tour_sensors'
Replay: MuJoCo, doors open, the robot's own pixels removed), SAM 3 is prompted with the target's category phrase
(design D6), each instance is back-projected with the stereo error model and placed with the estimated poses
(map_poses.py), the size check drops instances too large for the category, and the rest merge into nodes (8 cm).

Which node is the target: the instruction names the object's category and the surface it stands on ("the small vase
on the bottom shelf of the wide display unit"; the scene description's surface phrase), so the node is the one standing
on that surface (inside its footprint, padded 5 cm, at most 40 cm above its top) seen in the most views. None there:
null, and the round's prior is the surface's centre (perceive.RenderConfig.memory).

The frames: every head-scan frame at a viewpoint the tour assigned the support to, and every walking frame whose view
holds the support's centre within 3.5 m (by the estimated camera pose).

    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/rsi/map_memory.py \\
        outputs/vision_rsi_20261023/tour outputs/vision_rsi_20261023/pool_sealed outputs/vision_rsi_20261023/tour/memory.json \\
        [--poses kiss]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from map_build import BASELINE_M, MAX_DEPTH_M, STEREO_PX, kmat, se3  # noqa: E402
from tour_sensors import H, Replay, W, head_quat  # noqa: E402

from locomani.tasks.scene.services import SAM3_ADDRESS, Client, prompt_of, unpack_mask  # noqa: E402
from locomani.tasks.scene.tour import in_box  # noqa: E402

MIN_SCORE, SIZE_FACTOR, MERGE_M = 0.4, 1.6, 0.08


def frames_seeing(surface: dict, tour: dict, frames: list[dict], cams: np.ndarray, K: np.ndarray) -> list[int]:
    """Scan frames at the support's viewpoints, and walking frames whose (estimated) view holds its centre within
    3.5 m."""
    vps = {k for k, v in enumerate(tour["viewpoints"]) if surface["key"] in v["surfaces"]}
    c = np.r_[surface["xy"], float(surface["top"])]
    out = []
    for f in frames:
        if f["kind"] == "scan":
            if f["viewpoint"] in vps:
                out.append(f["i"])
            continue
        T = cams[f["i"]]
        p = T[:3, :3].T @ (c - T[:3, 3])  # camera frame: x right, y up, z backward
        if not 0.3 < -p[2] < 3.5:
            continue
        u, v = K[0, 0] * p[0] / -p[2] + K[0, 2], -K[1, 1] * p[1] / -p[2] + K[1, 2]
        if 20 <= u < W - 20 and 20 <= v < H - 20:
            out.append(f["i"])
    return out


def nodes_of(dets: list[dict]) -> list[dict]:
    """Greedy merge: an instance joins the first node whose centre is within MERGE_M."""
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        for n in out:
            if np.linalg.norm(np.subtract(n["centre"], d["centre"])) < MERGE_M:
                n["members"].append(d)
                n["centre"] = np.median([m["centre"] for m in n["members"]], axis=0).tolist()
                break
        else:
            out.append(dict(centre=list(d["centre"]), members=[d]))
    return [dict(centre=np.round(n["centre"], 4).tolist(), views=len(n["members"]),
                 score=round(max(m["score"] for m in n["members"]), 3)) for n in out]  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tour", type=Path, help="the tour directory (record/, sensors/, poses/)")
    ap.add_argument("pool", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--poses", default="kiss", help="a pose file in <tour>/poses (without .npy), or 'truth'")
    a = ap.parse_args(argv)
    tour = json.loads((a.tour / "record" / "tour.json").read_text())
    traj = dict(np.load(a.tour / "record" / "trajectory.npz"))
    scene_dir = REPO / tour["scene"]
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    lz = dict(np.load(a.tour / "sensors" / "lidar.npz"))
    cz = dict(np.load(a.tour / "sensors" / "camera.npz"))
    frames = json.loads((a.tour / "sensors" / "frames.json").read_text())
    K = kmat(cz["K"])
    lt = np.array([se3(R, p) for R, p in zip(lz["R"], lz["pos"], strict=True)])
    le = lt if a.poses == "truth" else np.load(a.tour / "poses" / f"{a.poses}.npy")
    near = [int(np.argmin(np.abs(lz["t"] - t))) for t in cz["t"]]
    cams = np.array([le[n] @ np.linalg.inv(lt[n]) @ se3(cz["R"][i], cz["pos"][i]) for i, n in enumerate(near)])
    size = {}
    for k, o in meta["objects"].items():
        if o.get("free"):
            ph = prompt_of(o, k)
            size[ph] = max(size.get(ph, 0.0), 2 * float(np.linalg.norm(o["box"]["half_extent"])))
    sam = Client(SAM3_ADDRESS)
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    diag_path = a.out.with_name(a.out.stem + "_diag.json")
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    rng = np.random.default_rng(0)
    for start_file in sorted(a.pool.glob("pick_*.json")):
        st = json.loads(start_file.read_text())
        seed = str(st["start"]["seed"])
        key = st["task"].removeprefix("pick_")
        if seed in out:
            continue
        t0 = time.time()
        phrase, sk = prompt_of(meta["objects"][key], key), meta["objects"][key]["surface"]
        surf = dict(meta["surfaces"][sk], key=sk)
        pick = frames_seeing(surf, tour, frames, cams, K)
        rp = Replay(scene_dir, traj, st["start"]["placements"], meta)
        m, d = rp.m, rp.d
        cid = m.camera("head_camera").id
        mount = m.cam_quat[cid].copy()
        robot = m.body_rootid[m.geom_bodyid] == m.body_rootid[m.body("pelvis").id]
        r = mujoco.Renderer(m, H, W, max_geom=int(m.ngeom) + 1000)
        dets = []
        for i in pick:
            f = frames[i]
            m.cam_quat[cid] = head_quat(mount, f["pan"], f["tilt"])
            rp.pose(rp.row(f["t"]))
            r.update_scene(d, camera=cid)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(d, camera=cid)
            z = r.render().copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cid)
            seg = r.render()
            r.disable_segmentation_rendering()
            mine = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
            mine[mine] = robot[seg[..., 0][mine]]
            z = z + rng.normal(size=z.shape) * z**2 * STEREO_PX / (K[1, 1] * BASELINE_M)
            res = sam.call(dict(image=rgb, prompts=[phrase]))
            if "error" in res:
                raise SystemExit(res["error"])
            for score, packed in res["results"][phrase]:
                if score < MIN_SCORE:
                    continue
                mask = unpack_mask(packed) & (z > 0.1) & (z < MAX_DEPTH_M) & ~mine
                if mask.sum() < 15:
                    continue
                v, u = np.nonzero(mask)
                dd = z[v, u]
                p = np.stack([(u - K[0, 2]) / K[0, 0] * dd, -(v - K[1, 2]) / K[1, 1] * dd, -dd], -1)
                w = p @ cams[i][:3, :3].T + cams[i][:3, 3]
                w = w[np.linalg.norm(w - np.median(w, axis=0), axis=1) < 0.5]
                if len(w) < 15:
                    continue
                ext = float(np.linalg.norm(np.percentile(w, 95, axis=0) - np.percentile(w, 5, axis=0)))
                if ext > SIZE_FACTOR * size[phrase]:
                    continue
                dets.append(dict(frame=i, score=float(score), centre=np.median(w, axis=0).tolist()))
        m.cam_quat[cid] = mount
        r.close()
        nodes = nodes_of(dets)
        top = float(surf["top"])
        on = [n for n in nodes if in_box(n["centre"][:2], surf, 0.05) and top - 0.02 <= n["centre"][2] <= top + 0.4]
        best = max(on, key=lambda n: (n["views"], n["score"])) if on else None
        truth = np.asarray(st["start"]["placements"][key][0], float)
        out[seed] = {key: best["centre"] if best else None}
        diag[seed] = dict(task=st["task"], phrase=phrase, frames=len(pick), detections=len(dets), nodes=len(nodes),
                          on_support=len(on), views=best["views"] if best else 0,
                          err_cm=round(100 * float(np.linalg.norm(np.subtract(best["centre"], truth))), 1) if best else None,
                          nearest_on_support_cm=round(100 * min(float(np.linalg.norm(np.subtract(n["centre"], truth))) for n in on), 1) if on else None,
                          seconds=round(time.time() - t0, 1))  # fmt: skip
        print(seed, json.dumps(diag[seed]), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        diag_path.write_text(json.dumps(diag, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
