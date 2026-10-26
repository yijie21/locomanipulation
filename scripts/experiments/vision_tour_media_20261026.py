"""Vision RSI D2 (docs/handoff/vision-rsi-design-20261023.md section 17.8): videos and a figure of the mapping walk for
the progress report.

  video   (sim venv) one walk sped up (--speed, default 8x): on the left the robot from straight above, following it, north up, the head
          camera's view in the corner; on the right the walking map the robot builds as it goes, from its own head
          camera depth and LiDAR placed with its estimated poses (map_build.py's rules: obstacle 3+ points at
          0.08-1.7 m, floor 3+ below 0.04 m, the rest not seen), its path, the head camera's field of view, the
          viewpoints (filled once scanned); for the exploration walk also the frontier (seen floor next to unseen)
  figure  (vision_map venv) the finished map of the planned tour: its point cloud from above coloured by height, and
          its walking map against the scene file's, the surfaces it found and where it put the targets

    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/vision_tour_media_20261026.py video outputs/vision_rsi_20261023/tour <out>/tour_map.mp4
    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/vision_tour_media_20261026.py video outputs/vision_rsi_20261023/tour_explore <out>/explore_map.mp4 --explore
    PYTHONPATH=src /workspace/envs/vision_map/bin/python scripts/experiments/vision_tour_media_20261026.py figure \\
        outputs/vision_rsi_20261023/tour <out>/tour_map.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "rsi"), str(REPO / "src")]

from map_common import MAX_DEPTH_M, kmat, se3, stereo_noise  # noqa: E402

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
SPEED, FPS, MAX_BYTES = 8.0, 12, 490_000
CROP = ((-4.35, 2.95), (-8.8, 3.95))  # the living area (x, y), metres
OBST_Z, FLOOR_Z, MIN_HITS, STRIDE = (0.08, 1.7), 0.04, 3, 4
RES = 0.05
SHELLS_ABOVE = {
    "adt_shell_10"
}  # the building shell around the bathroom and hall: its top faces cover the overhead shot


def grid_shape():
    nx = int(np.ceil((CROP[0][1] - CROP[0][0]) / RES))
    ny = int(np.ceil((CROP[1][1] - CROP[1][0]) / RES))
    return nx, ny


def cells(xy: np.ndarray) -> np.ndarray:
    """Flat indices of the crop's cells under world xy points (-1 outside)."""
    nx, ny = grid_shape()
    c = np.floor((xy - np.array([CROP[0][0], CROP[1][0]])) / RES).astype(int)
    ok = (c[:, 0] >= 0) & (c[:, 0] < nx) & (c[:, 1] >= 0) & (c[:, 1] < ny)
    return np.where(ok, c[:, 0] * ny + c[:, 1], -1)


def contributions(walk: Path, poses: np.ndarray):
    """Per sensor frame, in time order: (t, obstacle cells, floor cells) from the head camera and every second LiDAR
    frame, placed with the estimated LiDAR poses (map_build.py's chain)."""
    s = walk / "sensors"
    lz, cz = dict(np.load(s / "lidar.npz")), dict(np.load(s / "camera.npz"))
    lt = np.array([se3(R, p) for R, p in zip(lz["R"], lz["pos"], strict=True)])
    K = kmat(cz["K"])
    v, u = np.mgrid[0:480:STRIDE, 0:640:STRIDE]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u, float)], -1)
    rng = np.random.default_rng(0)
    out, cams = [], []
    for i, t in enumerate(cz["t"]):
        z = cv2.imread(str(s / "depth" / f"{i:05d}.png"), cv2.IMREAD_UNCHANGED)[::STRIDE, ::STRIDE] / 1000.0
        z = stereo_noise(z, K[1, 1], rng)
        ok = (z > 0.1) & (z < MAX_DEPTH_M)
        own = s / "self" / f"{i:05d}.png"
        if own.exists():
            ok &= cv2.imread(str(own), cv2.IMREAD_UNCHANGED)[::STRIDE, ::STRIDE] == 0
        n = int(np.argmin(np.abs(lz["t"] - t)))
        cam = poses[n] @ np.linalg.inv(lt[n]) @ se3(cz["R"][i], cz["pos"][i])
        w = (rays[ok] * z[ok][:, None]) @ cam[:3, :3].T + cam[:3, 3]
        o, f = (w[:, 2] > OBST_Z[0]) & (w[:, 2] < OBST_Z[1]), w[:, 2] < FLOOR_Z
        out.append((float(t), cells(w[o, :2]), cells(w[f, :2])))
        cams.append(cam)
    off = lz["offsets"]
    for k in range(0, len(off) - 1, 2):
        p = lz["points"][off[k] : off[k + 1]].astype(np.float64) @ poses[k][:3, :3].T + poses[k][:3, 3]
        o = (p[:, 2] > OBST_Z[0]) & (p[:, 2] < OBST_Z[1])
        out.append((float(lz["t"][k]), cells(p[o, :2]), np.zeros(0, int)))
    out.sort(key=lambda r: r[0])
    return out, cz, np.array(cams), lz["t"], poses


class Text:
    def __init__(self):
        self.big, self.small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 15)

    def put(self, dr: ImageDraw.ImageDraw, xy, text: str, big=False, fill=(255, 255, 255), box=True) -> None:
        font = self.big if big else self.small
        if box:
            l, t, r, b = dr.textbbox(xy, text, font=font)
            dr.rectangle((l - 4, t - 3, r + 4, b + 3), fill=(0, 0, 0, 160))
        dr.text(xy, text, font=font, fill=fill)


def map_panel(obst, floor, path, here, yaw, fov, vps, scanned, frontier, w: int, h: int) -> Image.Image:
    """The walking map so far, north up, scaled into w x h."""
    nx, ny = grid_shape()
    o, f = obst.reshape(nx, ny) >= MIN_HITS, floor.reshape(nx, ny) >= MIN_HITS
    img = np.full((nx, ny, 3), 55, np.uint8)
    img[f & ~o] = (232, 232, 226)
    img[o] = (52, 86, 150)
    if frontier is not None:
        img[frontier] = (230, 70, 200)
    img = np.flipud(img.transpose(1, 0, 2))  # rows from the top = north
    scale = min(w / nx, h / ny)
    big = cv2.resize(img, (int(nx * scale), int(ny * scale)), interpolation=cv2.INTER_NEAREST)
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    ox, oy = (w - big.shape[1]) // 2, (h - big.shape[0]) // 2
    canvas.paste(Image.fromarray(big), (ox, oy))
    dr = ImageDraw.Draw(canvas, "RGBA")

    def px(p):
        return (ox + (p[0] - CROP[0][0]) / RES * scale, oy + (CROP[1][1] - p[1]) / RES * scale)

    if len(path) > 1:
        dr.line([px(p) for p in path], fill=(255, 160, 40, 230), width=2)
    for k, v in enumerate(vps):
        q = px(v)
        r = 6
        dr.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), outline=(250, 220, 60), width=2,
                   fill=(250, 220, 60, 220) if k in scanned else None)  # fmt: skip
    q = px(here)
    a0, a1 = yaw - fov / 2, yaw + fov / 2
    reach = 3.0 / RES * scale
    dr.polygon([q, (q[0] + reach * np.cos(a0), q[1] - reach * np.sin(a0)),
                (q[0] + reach * np.cos(a1), q[1] - reach * np.sin(a1))], fill=(255, 80, 60, 60))  # fmt: skip
    dr.ellipse((q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6), fill=(235, 50, 40))
    dr.line([q, (q[0] + 14 * np.cos(yaw), q[1] - 14 * np.sin(yaw))], fill=(235, 50, 40), width=3)
    return canvas


def frontier_cells(obst, floor) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    nx, ny = grid_shape()
    o, f = obst.reshape(nx, ny) >= MIN_HITS, floor.reshape(nx, ny) >= MIN_HITS
    free = f & ~o
    unknown = ~free & ~o
    return free & binary_dilation(unknown, structure=np.ones((3, 3), bool))


def encode(frames: list[np.ndarray], path: Path) -> tuple[int, int]:
    h, w = frames[0].shape[:2]
    for crf in range(26, 50, 2):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]  # fmt: skip
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {path}")
        if path.stat().st_size <= MAX_BYTES:
            return crf, path.stat().st_size
    raise SystemExit(f"{path.name}: still {path.stat().st_size} bytes at CRF 48")


def hide_ceilings(m, d) -> int:
    """For the overhead shot only: scene geoms lying wholly above 2 m (ceiling panels) and SHELLS_ABOVE go to group 5,
    out of the renderer's default groups. The sensors keep them."""
    import mujoco

    mujoco.mj_forward(m, d)
    n = 0
    for g in range(m.ngeom):
        if m.geom_group[g] > 2:
            continue
        c, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        low = float((d.geom_xpos[g] + R @ c)[2] - np.abs(R[2]) @ half)
        if low > 2.0 or m.geom(g).name in SHELLS_ABOVE:
            m.geom_group[g] = 5
            n += 1
    return n


def video(a) -> int:
    import mujoco
    from tour_sensors import Replay

    walk = a.walk
    tour = json.loads((walk / "record" / "tour.json").read_text())
    traj = dict(np.load(walk / "record" / "trajectory.npz"))
    meta = json.loads((REPO / tour["scene"] / "scene_meta.json").read_text())
    poses = np.load(walk / "poses" / "kiss_planar.npy")
    contrib, cz, cams, lidar_t, _ = contributions(walk, poses)
    rp = Replay(REPO / tour["scene"], traj, None, meta)
    m, d = rp.m, rp.d
    hide_ceilings(m, d)
    LW, RW, H = 600, 360, 540
    r = mujoco.Renderer(m, H, LW, max_geom=int(m.ngeom) + 1000)
    text = Text()
    nx, ny = grid_shape()
    obst, floor = np.zeros(nx * ny, np.int32), np.zeros(nx * ny, np.int32)
    vps = [np.asarray(v["xy"], float) for v in tour["viewpoints"]]
    holds = tour["holds"]
    fov = 2 * np.arctan(320.0 / kmat(cz["K"])[0, 0])
    t_end = float(traj["t"][-1])
    n = int(t_end * FPS / SPEED) + 1
    frames, ci, path = [], 0, []
    for k in range(n + FPS * 2):  # two still seconds at the end
        t = min(k * SPEED / FPS, t_end)
        while ci < len(contrib) and contrib[ci][0] <= t:
            _, o, f = contrib[ci]
            np.add.at(obst, o[o >= 0], 1)
            np.add.at(floor, f[f >= 0], 1)
            ci += 1
        li = int(np.clip(np.searchsorted(lidar_t, t), 0, len(lidar_t) - 1))
        here = poses[li][:3, 3]
        path.append(here[:2].copy())
        cj = int(np.clip(np.searchsorted(cz["t"], t, side="right") - 1, 0, len(cz["t"]) - 1))
        fwd = cams[cj][:3, :3] @ np.array([0.0, 0.0, -1.0])  # the head camera's view direction
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        scanned = {h["viewpoint"] for h in holds if h["t1"] <= t}
        holding = next((h for h in holds if h["t0"] <= t < h["t1"]), None)
        rp.pose(rp.row(t))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [*d.xpos[m.body("pelvis").id][:2], 0.4]
        cam.azimuth, cam.elevation, cam.distance = 90.0, -89.9, 4.5
        r.update_scene(d, camera=cam)
        left = Image.fromarray(r.render().copy())
        inset = Image.open(walk / "sensors" / "rgb" / f"{cj:05d}.jpg").convert("RGB").resize((208, 156))
        left.paste(inset, (10, H - 156 - 10))
        # the exploration's stops are chosen as it goes: draw only the ones reached
        shown = [j for j in range(len(vps)) if not a.explore or j in scanned or (holding or {}).get("viewpoint") == j]
        right = map_panel(obst, floor, path, here[:2], yaw, fov, [vps[j] for j in shown],
                          {i for i, j in enumerate(shown) if j in scanned},
                          frontier_cells(obst, floor) if a.explore else None, RW, H - 60)  # fmt: skip
        canvas = Image.new("RGB", (LW + RW, H + 34), (24, 24, 24))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (LW, 0))
        dr = ImageDraw.Draw(canvas, "RGBA")
        title = (f"机器人自己探索房间（{SPEED:g} 倍速）" if a.explore
                 else f"巡房：机器人按规划的路线走一圈（{SPEED:g} 倍速）")  # fmt: skip
        text.put(dr, (10, 8), title, big=True)
        text.put(dr, (10, 36), f"第 {t:.0f} 秒")
        if holding is not None:
            what = f"在第 {holding['viewpoint'] + 1} 个停留点转头看一圈" if not a.explore else "停下转头看一圈"
        else:
            what = "去下一个看不到的地方" if a.explore else "走向下一个观察点"
        text.put(dr, (10, 60), what)
        text.put(dr, (14, H - 156 - 34), "头部相机看到的")
        text.put(dr, (LW + 10, 8), "它自己建的地图", big=True)
        ly = H - 52
        swatches = [((232, 232, 226), "看到的地面"), ((52, 86, 150), "障碍"), ((55, 55, 55), "还没看到")]
        if a.explore:
            swatches.append(((230, 70, 200), "交界"))
        x = LW + 10
        for col, name in swatches:
            dr.rectangle((x, ly + 4, x + 12, ly + 16), fill=col, outline=(200, 200, 200))
            text.put(dr, (x + 16, ly), name, box=False)
            x += 26 + int(dr.textlength(name, font=text.small))
        y2 = ly + 22
        dr.ellipse((LW + 10, y2 + 3, LW + 22, y2 + 15), fill=(235, 50, 40))
        text.put(dr, (LW + 26, y2), "机器人和相机朝向", box=False)
        dr.ellipse((LW + 168, y2 + 3, LW + 180, y2 + 15), outline=(250, 220, 60), width=2)
        text.put(dr, (LW + 184, y2), "停留点" if a.explore else "观察点", box=False)
        cap = ("左：从正上方跟拍机器人；右：它用自己的深度相机和激光雷达、按自己估计的位置拼出的地图，桔线是走过的路"
               if not a.explore else "不给路线：每次走向“看到的地面和没看到的地方的交界”（粉色）最多处，到了转头看一圈，直到无处可去")  # fmt: skip
        text.put(dr, (10, H + 8), cap, box=False, fill=(225, 225, 225))
        frames.append(np.asarray(canvas))
    r.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    crf, size = encode(frames, a.out)
    Image.fromarray(frames[-1]).save(a.out.with_suffix(".jpg"), quality=85)
    print(
        json.dumps(dict(out=str(a.out), frames=len(frames), seconds=round(len(frames) / FPS, 1), crf=crf, bytes=size))
    )
    return 0


def figure(a) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    from matplotlib import font_manager
    from matplotlib.patches import Polygon as Poly

    from locomani.primitives.navigator import build as build_navigation

    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    walk = a.walk
    tour = json.loads((walk / "record" / "tour.json").read_text())
    meta = json.loads((REPO / tour["scene"] / "scene_meta.json").read_text())
    mp = walk / "map_kiss_planar"
    pts = np.asarray(o3d.io.read_point_cloud(str(mp / "points.ply")).points)
    pts = pts[(pts[:, 2] < 2.0) & (pts[:, 2] > -0.05)]
    g = dict(np.load(mp / "grid.npz"))
    surfaces = json.loads((mp / "surfaces.json").read_text())
    match = json.loads((mp / "surface_match.json").read_text())
    objs = json.loads((walk / "map_kiss" / "objects.json").read_text())
    occ, _ = build_navigation(meta, body_radius=0.35)
    fig, ax = plt.subplots(1, 2, figsize=(13, 11.5), dpi=100)
    order = np.argsort(pts[:, 2])
    ax[0].scatter(pts[order, 0], pts[order, 1], c=pts[order, 2], s=0.15, cmap="viridis", vmin=0, vmax=1.9,
                  linewidths=0, rasterized=True)  # fmt: skip
    ax[0].set_title("① 点云（俯视，颜色是高度：深紫=地面，黄=约 1.9 m）", fontsize=15)
    # right: the walking map against the scene file's
    ext = (g["origin"][0], g["origin"][0] + g["obstacle"].shape[0] * g["res"],
           g["origin"][1], g["origin"][1] + g["obstacle"].shape[1] * g["res"])  # fmt: skip
    img = np.full(g["obstacle"].shape + (3,), 0.25)
    img[g["free"]] = (0.93, 0.93, 0.9)
    img[g["obstacle"]] = (0.2, 0.34, 0.6)
    ax[1].imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
    truth = np.where(occ.occupied.T, 1.0, np.nan)
    ax[1].contour(np.arange(occ.nx) * occ.res + occ.origin[0] + occ.res / 2,
                  np.arange(occ.ny) * occ.res + occ.origin[1] + occ.res / 2, np.nan_to_num(truth), levels=[0.5],
                  colors=["#ff5a5a"], linewidths=0.6)  # fmt: skip
    targets = set(tour["config"].get("must") or [])
    for k, i in match.items():
        s = surfaces[i]
        c, he, yaw = np.asarray(s["xy"]), np.asarray(s["half_extent"]), float(s["yaw"])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = c + (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * he) @ R.T
        ax[1].add_patch(Poly(corners, closed=True, fill=False, ec="#22d3ee" if k in targets else "#7dd3fc",
                             lw=2.2 if k in targets else 0.9))  # fmt: skip
    truth_pos = {k: meta["objects"][k]["init_pos"] for k in objs["targets"]}
    for k, row in objs["targets"].items():
        tp = truth_pos[k]
        ax[1].plot(tp[0], tp[1], "x", color="#ffffff", ms=7, mew=1.6)
        mine = [n for n in objs["node_list"] if n["phrase"] == row["phrase"]]
        if mine:
            best = min(mine, key=lambda n: np.linalg.norm(np.subtract(n["centre"][:2], tp[:2])))
            ax[1].plot(best["centre"][0], best["centre"][1], "o", color="#fbbf24", ms=5)
    vp = np.array([v["xy"] for v in tour["viewpoints"]])
    ax[1].plot(vp[:, 0], vp[:, 1], "o", mfc="none", mec="#facc15", ms=11, mew=1.8)
    ax[1].set_title("② 可走地图、台面、目标位置", fontsize=15)
    for x in ax:
        x.set_xlim(*CROP[0])
        x.set_ylim(*CROP[1])
        x.set_aspect("equal")
        x.set_facecolor("#111")
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    png = a.out.with_suffix(".png")
    fig.savefig(png, dpi=100, facecolor="white")
    if a.out.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(a.out), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        png.unlink()
    print(json.dumps(dict(out=str(a.out), bytes=a.out.stat().st_size)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("video", "figure"))
    ap.add_argument("walk", type=Path, help="a walk directory (record/, sensors/, poses/, map_*)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--explore", action="store_true", help="the exploration walk (frontier shown, its own wording)")
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args(argv)
    globals().update(SPEED=a.speed, FPS=a.fps)
    return video(a) if a.what == "video" else figure(a)


if __name__ == "__main__":
    raise SystemExit(main())
