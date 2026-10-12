"""TODO 21.3.4: draw each round's planned grasp into its video as the magenta ghost gripper, offline -- no replanning.

The runs had SIM_VIS=none (the default since 2026-09-10): grasp_planned's set_ghost did put the scene's ghost hand at
the chosen grasp, but in geom group 4, which the recorder draws only when SIM_VIS asks for it. No state was recorded per
frame, so a frame cannot be re-rendered; but the recorder films with MuJoCo's pelvis tracking camera
(sim_video.recording_camera), whose lookat is the robot's centre of mass, unsmoothed -- so each frame's camera is one
unknown point plus a fixed offset (2.5 m, azimuth 135, elevation -20). This recovers that point per frame by registering
the frame against a robot-free render of the static scene (floor, table, the target where record.json saw it):
Gauss-Newton on the camera's translation (plus gain and bias) over a pyramid, Tukey-weighted so the robot, its shadow
and its reflection drop out. It renders the ghost alone from that camera at the pose grasp_planned gave set_ghost
(record.json grasp: wrist = tool - R @ OFFSET, attitude R) and lays it over the frame at ALPHA -- over the robot as
well, so the real hand stays readable under it -- never over the ego / wrist insets. The ghost shows from the frame
after the stand-alignment teleport, so it is already on while the planner plans (set_ghost came 2-17 s later): the
teleport is the largest frame-to-frame change in the first CUT_WITHIN frames (frame 4 or 5 in all 130 rounds, >= 10.2
grey levels against <= 7.1 anywhere else). Registration's own failure does not find it -- registered backwards, five
rounds slid across the 0.4 m jump to a wrong camera near the last one. Frames are registered from the last one back to
that frame, each starting from the next one's camera. A round without a chosen grasp is left as it is.

Writes <root>/ghost_rounds/obj_XXX/round_NN.mp4 and round_NN.json (onset frame, per-frame lookat and inlier share);
with --tag _fixed the runs gnb_obj_XXX_fixed and <root>/fixed/ghost_rounds.
--reuse takes the camera from an existing round_NN.json instead of registering again (about 2 min a round): for a change
to the drawing only.

    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/graspnet_bottle_20261008/ghost_overlay.py outputs/graspnet_bottle_20261008 [--jobs 8] [ids...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# numpy asks for transparent huge pages on every array >= 4 MB, and each frame here makes several; with the host's memory
# fragmented a fault on them stalls in direct compaction. 2026-10-09, another project's Isaac running: 7 frames in 45 s,
# 38 s of it in the kernel; with this, 43 frames, 5 s.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from locomani.pipeline.sim_video import FPS, HEIGHT, INSETS, WIDTH  # noqa: E402
from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402

DISTANCE, AZIMUTH, ELEVATION = 2.5, 135.0, -20.0  # sim_video.recording_camera's tracking view
COM_BELOW_PELVIS = 0.068  # the standing robot's subtree com under the pelvis origin: the first guess, refined per frame
ROUNDS = ("round_00_ik", "round_01_curobo")
ALPHA = 0.6
FAR_M = 8.0  # registered pixels: nearer than this (the far floor aliases), not sky
LEVELS = (3, 2, 1)  # pyramid levels registered, coarse to fine; level 1 is 640x480
ITERS = 8
CUT_WITHIN = 40  # frames: grasp_planned aligns the stand first thing
# The previews halve the frame and the gripper is ~70 px at full size, so a 3 cm miss would be 7 px: a magnified crop
# around the ghost's grip centre goes into the corner the recorder's insets leave free (top right), under the label
# bar make_videos.py draws (114 of 720 rows there, 152 here).
ZOOM, ZOOM_W, ZOOM_H = 2.5, 480, 360
ZOOM_X, ZOOM_Y = WIDTH - ZOOM_W - 8, 164


def view_rows() -> np.ndarray:
    """World -> OpenCV camera rotation (rows: right, down, forward) of the tracking view."""
    a, e = np.radians(AZIMUTH), np.radians(ELEVATION)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return np.stack([np.cross(fwd, up), -up, fwd])


def free_camera(lookat) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = DISTANCE, AZIMUTH, ELEVATION
    return cam


class Scene:
    """The round's scene with the robot moved out of shot and the target where record.json saw it."""

    def __init__(self, xml: Path, target_body: str, obj_pos, obj_quat):
        m = mujoco.MjModel.from_xml_path(str(xml))
        m.vis.global_.offwidth, m.vis.global_.offheight = WIDTH, HEIGHT
        d = mujoco.MjData(m)
        base = m.jnt_qposadr[m.joint("floating_base_joint").id]
        d.qpos[base : base + 3] = (60.0, 60.0, -50.0)  # under the floor
        a = m.jnt_qposadr[m.body(target_body).jntadr[0]]
        d.qpos[a : a + 3], d.qpos[a + 3 : a + 7] = obj_pos, obj_quat
        ghost = m.body("ghost_hand")
        self.mocap = int(m.body_mocapid[ghost.id])
        geoms = np.flatnonzero(m.geom_bodyid == ghost.id)
        m.geom_rgba[geoms, 3] = 1.0  # opaque here; the blend is ours
        self.target_geoms = set(np.flatnonzero(m.geom_bodyid == m.body(target_body).id).tolist())
        self.bead = m.geom_pos[m.geom("ghost_hand_bead").id].copy()  # the grip centre, ghost body frame
        mujoco.mj_forward(m, d)
        self.m, self.d = m, d
        self.f = 0.5 * HEIGHT / np.tan(np.radians(0.5 * m.vis.global_.fovy))
        self.static = mujoco.Renderer(m, HEIGHT, WIDTH)
        self.ghost = mujoco.Renderer(m, HEIGHT, WIDTH)
        for flag in ("mjRND_SKYBOX", "mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_FOG"):
            self.ghost.scene.flags[getattr(mujoco.mjtRndFlag, flag)] = 0
        self.ghost_opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.ghost_opt)
        self.ghost_opt.geomgroup[:] = 0
        self.ghost_opt.geomgroup[4] = 1
        self.ghost_opt.sitegroup[:] = 0

    def render_static(self, lookat):
        """BGR, z-depth (m), target mask of the static scene seen from `lookat`."""
        r, cam = self.static, free_camera(lookat)
        r.update_scene(self.d, camera=cam)
        bgr = r.render()[:, :, ::-1].copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=cam)
        seg = r.render()[:, :, 0].copy()
        r.disable_segmentation_rendering()
        return bgr, depth, np.isin(seg, list(self.target_geoms))

    def place_ghost(self, pos, quat) -> None:
        self.d.mocap_pos[self.mocap], self.d.mocap_quat[self.mocap] = pos, quat
        mujoco.mj_forward(self.m, self.d)

    def render_ghost(self, lookat):
        """(row slice, column slice, BGR, coverage 0..1) of the ghost alone, cropped to where it is; None out of view."""
        r, cam = self.ghost, free_camera(lookat)
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=cam, scene_option=self.ghost_opt)
        rows, cols = np.nonzero(r.render()[:, :, 0] >= 0)
        r.disable_segmentation_rendering()
        if not len(rows):
            return None
        ys = slice(max(rows.min() - 3, 0), rows.max() + 4)
        xs = slice(max(cols.min() - 3, 0), cols.max() + 4)
        r.update_scene(self.d, camera=cam, scene_option=self.ghost_opt)
        bgr = r.render()[ys, xs, ::-1]
        cover = np.zeros(bgr.shape[:2], np.float32)
        cover[rows - ys.start, cols - xs.start] = 1.0
        return ys, xs, bgr, cv2.GaussianBlur(cover, (0, 0), 0.7)


def gray(bgr) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def sample(images, u, v):
    """Bilinear samples of each image at (u, v); remap takes the points as a block (its maps stay under 32767 rows)."""
    n = len(u)
    pad = -n % 1024
    mu = np.r_[u, np.zeros(pad)].astype(np.float32).reshape(-1, 1024)
    mv = np.r_[v, np.zeros(pad)].astype(np.float32).reshape(-1, 1024)
    return [cv2.remap(im, mu, mv, cv2.INTER_LINEAR).ravel()[:n].astype(np.float64) for im in images]


class Registrar:
    """Camera translation of a frame against the static render at `lookat0`: per pyramid level, the reference's valid
    pixels back-projected to 3-D, then Gauss-Newton on (translation, gain, bias) of the frame's warped intensities."""

    def __init__(self, ref_bgr, depth, valid, f):
        self.R = view_rows()
        g = gray(ref_bgr).astype(np.float32) / 255.0
        edge = cv2.dilate((np.abs(cv2.Laplacian(depth, cv2.CV_32F)) > 0.05).astype(np.uint8), np.ones((5, 5)))
        valid = valid & (depth < FAR_M) & (edge == 0)
        self.levels = []
        for lv in LEVELS:
            s = 0.5**lv
            w, h = round(WIDTH * s), round(HEIGHT * s)
            t = cv2.resize(cv2.GaussianBlur(g, (0, 0), 0.5 / s), (w, h), interpolation=cv2.INTER_AREA)
            z = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            ok = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            ok[:2], ok[-2:], ok[:, :2], ok[:, -2:] = False, False, False, False
            v, u = np.nonzero(ok)
            fl, cx, cy = f * s, w / 2 - 0.5, h / 2 - 0.5
            zz = z[v, u].astype(np.float64)
            P = np.stack([(u - cx) * zz / fl, (v - cy) * zz / fl, zz], 1)
            self.levels.append(dict(s=s, w=w, h=h, f=fl, cx=cx, cy=cy, P=P, T=t[v, u].astype(np.float64)))

    def solve(self, frame_gray, delta0, levels=None):
        x = np.r_[np.asarray(delta0, dtype=float), 1.0, 0.0]  # translation (m), gain, bias
        R = self.R
        for L in self.levels if levels is None else [self.levels[i] for i in levels]:
            img = frame_gray.astype(np.float32) / 255.0
            img = cv2.resize(
                cv2.GaussianBlur(img, (0, 0), 0.5 / L["s"]), (L["w"], L["h"]), interpolation=cv2.INTER_AREA
            )
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            P, T, fl, cx, cy = L["P"], L["T"], L["f"], L["cx"], L["cy"]
            for _ in range(ITERS):
                q = P - (R @ x[:3])[None]
                z = q[:, 2]
                u = cx + fl * q[:, 0] / z
                v = cy + fl * q[:, 1] / z
                inside = (u > 1) & (u < L["w"] - 2) & (v > 1) & (v < L["h"] - 2)
                Iw, Gx, Gy = sample((img, gx, gy), u, v)
                r = x[3] * Iw + x[4] - T
                du = fl * (-R[0][None] * z[:, None] + q[:, :1] * R[2][None]) / z[:, None] ** 2
                dv = fl * (-R[1][None] * z[:, None] + q[:, 1:2] * R[2][None]) / z[:, None] ** 2
                J = np.column_stack([x[3] * (Gx[:, None] * du + Gy[:, None] * dv), Iw, np.ones_like(Iw)])
                rr = r[inside]
                c = 4.685 * max(1.4826 * np.median(np.abs(rr - np.median(rr))), 0.01)
                wgt = np.where(inside & (np.abs(r) < c), (1 - (r / c) ** 2) ** 2, 0.0)
                Jw = J * wgt[:, None]
                step = np.linalg.solve(Jw.T @ J + 1e-6 * np.eye(5), -Jw.T @ r)
                x += step
                if np.linalg.norm(step[:3]) < 2e-4:
                    break
        inlier = float(np.mean(np.abs(r[inside]) < 0.04)) if inside.any() else 0.0
        return x[:3], inlier


def project(lookat, point, f: float) -> np.ndarray:
    """Full-frame pixel (column, row) of a world point seen from the tracking view at `lookat`, focal length `f` px."""
    R = view_rows()
    q = R @ (np.asarray(point) - (np.asarray(lookat) - DISTANCE * R[2]))
    return np.array([WIDTH / 2 - 0.5 + f * q[0] / q[2], HEIGHT / 2 - 0.5 + f * q[1] / q[2]])


def paste_zoom(im: np.ndarray, centre) -> None:
    """The ZOOM-times crop of `im` around `centre`, laid into the free corner like sim_video's insets, in place."""
    cw, ch = round(ZOOM_W / ZOOM), round(ZOOM_H / ZOOM)
    x = int(np.clip(round(centre[0] - cw / 2), 0, WIDTH - cw))
    y = int(np.clip(round(centre[1] - ch / 2), 0, HEIGHT - ch))
    crop = cv2.resize(im[y : y + ch, x : x + cw], (ZOOM_W, ZOOM_H), interpolation=cv2.INTER_CUBIC)
    im[ZOOM_Y : ZOOM_Y + ZOOM_H, ZOOM_X : ZOOM_X + ZOOM_W] = crop
    cv2.rectangle(im, (ZOOM_X - 1, ZOOM_Y - 1), (ZOOM_X + ZOOM_W, ZOOM_Y + ZOOM_H), (255, 255, 255), 1)
    for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            im, f"grasp x{ZOOM:g}", (ZOOM_X + 6, ZOOM_Y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA
        )


def frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, im = cap.read()
        if not ok:
            break
        yield im
    cap.release()


def teleport(grays, keep) -> tuple[int, float]:
    """(first frame after the alignment teleport, its mean grey-level change) -- the largest jump early in the round."""
    k = keep[::4, ::4] > 0
    small = [g[::4, ::4].astype(np.float32) for g in grays[: CUT_WITHIN + 1]]
    jumps = [float(np.mean(np.abs(b - a)[k])) for a, b in zip(small, small[1:])]
    i = int(np.argmax(jumps))
    return i + 1, jumps[i]


def insets_mask() -> np.ndarray:
    keep = np.ones((HEIGHT, WIDTH), np.float32)
    for _cam, _label, w, h, x, y in INSETS:
        keep[max(y - 2, 0) : y + h + 2, max(x - 2, 0) : x + w + 2] = 0.0
    return keep


def write_mp4(path: Path, images) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{FPS}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for im in images:
        proc.stdin.write(np.ascontiguousarray(im).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed on {path}")


def register(sc: Scene, rec: dict, video: Path, keep) -> dict:
    """Each frame's tracking-camera lookat from the teleport on, with the registration's own measures."""
    lookat0 = np.asarray(rec["pelvis"], dtype=float) - [0.0, 0.0, COM_BELOW_PELVIS]
    ref, depth, target_px = sc.render_static(lookat0)
    reg = Registrar(ref, depth, (keep > 0) & ~target_px, sc.f)
    grays = [gray(im) for im in frames(video)]  # registered backwards; the colour frames are read again to compose
    n = len(grays)
    onset, jump = teleport(grays, keep)
    delta = np.full((n, 3), np.nan)
    inlier = np.zeros(n)
    d = np.zeros(3)
    scale = None
    for i in range(n - 1, onset - 1, -1):
        est, share = reg.solve(grays[i], d)
        if scale is None:
            scale = share
        inlier[i] = share
        if share > 0.6 * scale and (i == n - 1 or np.linalg.norm(est - d) < 0.05):
            delta[i], d = est, est
            scale = 0.9 * scale + 0.1 * share
    # an isolated unregistered frame takes its neighbours' camera
    idx = np.arange(n)
    have = ~np.isnan(delta[:, 0])
    for j in range(3):
        delta[onset:, j] = np.interp(idx[onset:], idx[have], delta[have, j])
    lookat = lookat0 + delta
    return dict(
        frames=n,
        onset=onset,
        onset_s=onset / FPS,
        teleport_jump=jump,
        lookat0=lookat0.tolist(),
        com_minus_pelvis=(lookat[onset] - np.asarray(rec["pelvis"])).tolist(),
        lookat_span_m=float(np.linalg.norm(np.ptp(lookat[onset:], axis=0))),
        inlier_median=float(np.median(inlier[onset:][inlier[onset:] > 0])),
        interpolated=int(np.sum(~have[onset:])),
        lookat=np.round(lookat[onset:], 4).tolist(),
    )


def overlay_round(root: Path, oid: int, k: int, check: bool = False, reuse: bool = False, tag: str = "") -> dict:
    run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{tag}"
    rec = json.loads((run / "grasp_planned" / ROUNDS[k] / "record.json").read_text())
    video = run / "videos" / f"round_{k:02d}.mp4"
    out = (root / tag.strip("_") if tag else root) / "ghost_rounds" / f"obj_{oid:03d}"
    out.mkdir(parents=True, exist_ok=True)
    grasp = rec.get("grasp")
    report = dict(object=oid, round=k, planner=rec.get("planner"), video=str(video.relative_to(REPO)), ghost=False)
    if not grasp or not video.exists():
        report["why"] = "no chosen grasp" if not grasp else "no video"
        return report
    scene_dir = root / "scenes" / f"obj_{oid:03d}"
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    target = meta["objects"][meta["target_object"]]["body"]
    sc = Scene(scene_dir / "scene_locomani.xml", target, rec["object_pos"], rec["object_quat"])
    R = np.asarray(grasp["R"], dtype=float)
    wrist = np.asarray(grasp["tool"], dtype=float) - R @ OFFSET
    sc.place_ghost(wrist, Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]])
    grip = wrist + R @ sc.bead

    keep = insets_mask()
    prior = out / f"round_{k:02d}.json"
    camera = json.loads(prior.read_text()) if reuse and prior.exists() else {}
    if camera.get("ghost"):  # an earlier pass's camera: only the drawing changed
        camera = {key: v for key, v in camera.items() if key not in report and key != "wrist"}
    else:
        camera = register(sc, rec, video, keep)
    n, onset = camera["frames"], camera["onset"]
    lookat = np.full((n, 3), np.nan)
    lookat[onset:] = camera["lookat"]
    alpha = ALPHA * keep
    checks = sorted({onset, (onset + n) // 2, n - 1}) if check else []

    def composed():
        for i, im in enumerate(frames(video)):
            if i >= onset:
                im_out = im.copy()
                ghost = sc.render_ghost(lookat[i])
                if ghost is not None:
                    ys, xs, g, cover = ghost
                    a = (alpha[ys, xs] * cover)[:, :, None]
                    im_out[ys, xs] = (im[ys, xs] * (1 - a) + g * a).astype(np.uint8)
                paste_zoom(im_out, project(lookat[i], grip, sc.f))
            else:
                im_out = im
            if i in checks:
                stat, _, _ = sc.render_static(lookat[i])
                cv2.imwrite(str(out / f"round_{k:02d}_check_{i:03d}.png"), np.hstack([im, stat, im_out]))
            yield im_out

    write_mp4(out / f"round_{k:02d}.mp4", composed())
    report.update(ghost=True, wrist=wrist.tolist(), **camera)
    (out / f"round_{k:02d}.json").write_text(json.dumps(report))
    return report


def job(args) -> str:
    root, oid, k, check, reuse, tag = args
    cv2.setNumThreads(1)
    try:
        r = overlay_round(root, oid, k, check, reuse, tag)
    except Exception as exc:  # noqa: BLE001 -- one bad round must not stop the batch
        return f"object {oid:03d} round {k}: ERROR {exc!r}"
    if not r["ghost"]:
        return f"object {oid:03d} round {k}: no ghost ({r['why']})"
    return (
        f"object {oid:03d} round {k}: onset {r['onset']} ({r['onset_s']:.1f} s), inlier {r['inlier_median']:.2f}, "
        f"camera span {100 * r['lookat_span_m']:.1f} cm, interpolated {r['interpolated']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="also write registration check images")
    ap.add_argument("--reuse", action="store_true", help="take each round's camera from its round_NN.json, if any")
    ap.add_argument("--tag", default="", help="run-name suffix (run_batch.sh's tag); outputs go to <root>/<tag>/")
    args = ap.parse_args()
    root = args.root.resolve()
    ids = args.ids or [o["id"] for o in json.loads((root / "objects.json").read_text())["objects"]]
    work = [
        (root, oid, k, args.check, args.reuse, args.tag)
        for oid in ids
        for k in range(len(ROUNDS))
        if (
            REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{args.tag}" / "grasp_planned" / ROUNDS[k] / "record.json"
        ).exists()
    ]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(job, work):
                print(line, flush=True)
    else:
        for w in work:
            print(job(w), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
