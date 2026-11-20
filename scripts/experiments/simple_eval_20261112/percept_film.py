"""How the robot gets its grasp from one look, as a video per episode (2026-11-19, the SIMPLE report).

    cd third_party/SIMPLE && MUJOCO_GL=egl .venv/bin/python -I <this file> <look dumps> <cloud dir> <out dir> NAME[=TITLE] ...
        [--check]          print the chosen grasp's rank and stop
        [--unnamed sam]    the captions of an instruction that names no object follow blind.py look --unnamed
        [--truth NAME,..]  step 2 also outlines the true target, for a look that found the wrong thing (the viewer's aid)
        [--robot XML]      SIMPLE's G1 + Dex3 model, by default relative to SIMPLE's root (the hand's outline)

<look dumps>/<NAME>.npz is blind.py look's LOOK_DUMP of the episode (the head stereo pair as SIMPLE's Isaac scene draws
it, the matched and the filled depth, SAM 3's mask, every pixel's point in the robot's frame), <cloud dir> holds the
target's cloud and GraspGen-X's grasps of it (grasp_<NAME>.npz). Six steps, each drawn from the data the robot used:

  1 the look          the left image, then the right one for a moment
  2 the target        SAM 3's mask for the object the instruction names
  3 the distances     the depth by stereo matching (holes), then as LingBot-Depth fills it
  4 the points        every pixel put back in space, seen from the head camera and then from outside
  5 the grasps        GraspGen-X's palm poses as hand outlines, the ones the right hand can take (pick.graspgen_grasp's
                      rule), the best-scored of those
  6 the grasp         that hand and its approach, back in the camera's image

-> <out>/percept_<NAME>.mp4 (1280 x 720, 25 fps) and percept_<NAME>.jpg (a frame of step 5).
"""

import argparse
import os
import subprocess

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("dumps")
ap.add_argument("clouds")
ap.add_argument("out")
ap.add_argument("names", nargs="+")
ap.add_argument("--check", action="store_true")
ap.add_argument(
    "--truth",
    default="",
    help="names (comma) whose step 2 also outlines the true target, for a look that found the wrong thing",
)
ap.add_argument(
    "--unnamed",
    default="depth",
    choices=("depth", "sam"),
    help="how blind.py look found the target of an instruction that names no object (its --unnamed): the captions follow it",
)
ap.add_argument("--robot", default="data/robots/g1_sonic/g1_29dof_with_hand.xml")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

W, H, FPS = 1280, 720, 25
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
BG = np.array([14, 16, 20], np.uint8)
YEL, WHITE, INK = (242, 176, 30), (255, 255, 255), (14, 16, 20)
PALM = np.array([0.0415, 0.0, 0.0])  # pick.PALM: the Dex3 palm link in the wrist-yaw frame
AHEAD, ROLL, CLEAR, PRE = 0.7, (-30.0, 110.0), 0.06, 0.08  # pick.py's rule for a grasp the right hand can take


# ---- the hand as an outline: the palm's hull and the three fingers' chains, in the palm link's frame
def hand_outline(f_open):
    m = mujoco.MjModel.from_xml_path(a.robot)
    d = mujoco.MjData(m)
    for n, q in zip(("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"), f_open):
        d.qpos[m.joint(f"right_hand_{n}_joint").qposadr[0]] = q
    mujoco.mj_forward(m, d)
    b = m.body("right_wrist_yaw_link").id
    Rw, pw = d.xmat[b].reshape(3, 3), d.xpos[b]

    def palm(p):
        return (np.asarray(p) - pw) @ Rw - PALM

    def verts(g):
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T

    def tip(body):
        bid = m.body(body).id
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_dataid[g] >= 0)
        w = verts(g)
        return w[np.argmax(np.linalg.norm(w - d.xpos[bid], axis=1))]

    lines = []
    for f, links in (("thumb", (0, 1, 2)), ("index", (0, 1)), ("middle", (0, 1))):
        pts = [d.xpos[m.body(f"right_hand_{f}_{k}_link").id] for k in links] + [tip(f"right_hand_{f}_{links[-1]}_link")]
        lines.append(np.array([palm(p) for p in pts]))
    g = next(
        g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.mesh(m.geom_dataid[g]).name == "right_hand_palm_link"
    )
    v = palm(verts(g))
    hull = cv2.convexHull((v[:, [0, 2]] * 1000).astype(np.float32))[:, 0] / 1000
    y = float(np.median(v[:, 1]))
    ring = np.column_stack([hull[:, 0], np.full(len(hull), y), hull[:, 1]])
    lines.insert(0, np.vstack([ring, ring[:1]]))
    return lines


# ---- drawing
def font(size):
    return ImageFont.truetype(FONT, size)


def paste(img, rgba, x, y):
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    al = src[..., 3:4].astype(np.float32) / 255
    img[y0:y1, x0:x1] = (src[..., :3] * al + img[y0:y1, x0:x1] * (1 - al)).astype(np.uint8)


def chip(text, size, fg=WHITE, bg=(0, 0, 0, 170), pad=(10, 6)):
    f = font(size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("RGBA", (r - l + 2 * pad[0], size + 2 * pad[1] + 4), bg)
    ImageDraw.Draw(im).text((pad[0] - l, pad[1]), text, font=f, fill=fg + (255,))
    return np.asarray(im)


STEPS = ["看一眼", "找目标", "量距离", "变成三维点", "出抓取", "定抓取点"]


def overlay(title, step, line):
    """The frame's fixed texts: the title, the six steps with this one lit, the step's sentence at the foot."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 44], fill=(10, 12, 15, 225))
    d.text((16, 9), title, font=font(21), fill=(235, 235, 235, 255))
    x = W - 16
    for k in range(len(STEPS) - 1, -1, -1):
        t = f"{k + 1} {STEPS[k]}"
        w = int(font(17).getlength(t)) + 18
        x -= w
        on = k == step
        d.rounded_rectangle([x, 8, x + w - 6, 36], 6, fill=YEL + (255,) if on else (58, 62, 68, 255))
        d.text((x + 6, 11), t, font=font(17), fill=INK + (255,) if on else (200, 200, 200, 255))
    d.rectangle([0, H - 92, W, H], fill=(0, 0, 0, 185))
    d.text((20, H - 84), f"第 {step + 1} 步　{STEPS[step]}", font=font(27), fill=YEL + (255,))
    d.text((20, H - 44), line, font=font(21), fill=(236, 236, 236, 255))
    return np.asarray(im)


def project(P, c, Rc, f):
    pc = (P - c) @ Rc
    d = -pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return W / 2 + f * pc[..., 0] / d - 0.5, H / 2 - f * pc[..., 1] / d - 0.5, d


def splat(P, C, c, Rc, f, lo, hi):
    """The points as squares of (hi - lo + 1) pixels, the nearest one on each pixel."""
    u, v, d = project(P, c, Rc, f)
    ok = (d > 0.08) & np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 4 * W) & (np.abs(v) < 4 * H)
    ui, vi, dd, cc = (
        np.floor(u[ok] + 0.5).astype(np.int64),
        np.floor(v[ok] + 0.5).astype(np.int64),
        d[ok].astype(np.float32),
        C[ok],
    )
    z = np.full(H * W, np.inf, np.float32)
    img = np.empty((H * W, 3), np.uint8)
    img[:] = BG
    hits = []
    for dy in range(lo, hi + 1):
        for dx in range(lo, hi + 1):
            x, y = ui + dx, vi + dy
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = y[m] * W + x[m]
            np.minimum.at(z, idx, dd[m])
            hits.append((idx, m))
    for idx, m in hits:
        win = dd[m] <= z[idx] + 1e-6
        img[idx[win]] = cc[m][win]
    return img.reshape(H, W, 3)


def look_at(pos, aim):
    zb = (pos - aim) / np.linalg.norm(pos - aim)
    x = np.cross([0.0, 0.0, 1.0], zb)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(zb, x), zb])


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def sph(v):
    r = np.linalg.norm(v)
    return r, np.arctan2(v[1], v[0]), np.arcsin(v[2] / r)


def unsph(r, az, el):
    return r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def polylines(img, lines, c, Rc, f, color, thick, alpha=1.0):
    """3-D polylines (a list of n x 3) drawn over the image."""
    pts = []
    for ln in lines:
        u, v, d = project(ln, c, Rc, f)
        if np.all(d > 0.05) and np.all(np.isfinite(u)) and np.all(np.abs(u) < 3 * W) and np.all(np.abs(v) < 3 * H):
            pts.append(np.round(np.column_stack([u, v]) * 8).astype(np.int32))
    if not pts:
        return
    if alpha >= 0.999:
        cv2.polylines(img, pts, False, color, thick, cv2.LINE_AA, shift=3)
        return
    top = img.copy()
    cv2.polylines(top, pts, False, color, thick, cv2.LINE_AA, shift=3)
    cv2.addWeighted(top, alpha, img, 1 - alpha, 0, dst=img)


def turbo(depth, lo, hi):
    n = np.uint8(255 * (1 - np.clip((depth - lo) / (hi - lo), 0, 1)))  # near warm, far cool
    im = cv2.cvtColor(cv2.applyColorMap(n, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    im[depth <= 0] = BG
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)


def colorbar(lo, hi):
    im = Image.new("RGBA", (150, 250), (0, 0, 0, 150))
    bar = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(np.linspace(255, 0, 200))[:, None].repeat(18, 1), cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    im.paste(Image.fromarray(bar), (14, 36))
    d = ImageDraw.Draw(im)
    d.text((12, 8), "离相机多远", font=font(17), fill=(235, 235, 235, 255))
    d.text((40, 34), f"近 {lo:.1f} m", font=font(17), fill=(235, 235, 235, 255))
    d.text((40, 216), f"远 {hi:.1f} m", font=font(17), fill=(235, 235, 235, 255))
    return np.asarray(im)


class Writer:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27",
                                   "-preset", "slow", "-movflags", "+faststart", path], stdin=subprocess.PIPE)  # fmt: skip
        self.n = 0

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def film(name, title):
    z = np.load(f"{a.dumps}/{name}.npz", allow_pickle=False)
    g = np.load(f"{a.clouds}/grasp_{name}.npz")
    cl = np.load(f"{a.clouds}/{name}.npz")
    left, right, mask, ok, pts = z["left"], z["right"], z["mask"].astype(bool), z["ok"].astype(bool), z["pts"]
    yaw, cam_p, cam_R = float(z["yaw"]), z["cam_p"], z["cam_R"]
    f0 = 0.5 * 360 / np.tan(np.radians(float(z["fovy"])) / 2) * (W / 640)

    # the grasps in the robot's frame, best score first, and pick.graspgen_grasp's rule
    T = g["T_op"].copy()
    T[:, :3, 3] += cl["p_o"]  # T_wo: the cloud's frame is the robot's, moved to the cloud's median
    score = g["score"]
    ahead, lft = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    Rw = T[:, :3, :3]
    roll = np.degrees(np.arctan2(-Rw[:, 2, 1], Rw[:, :, 1] @ lft))
    support = np.percentile(cl["pts"][:, 2], 2)
    can = (Rw[:, :, 0] @ ahead >= AHEAD) & (roll >= ROLL[0]) & (roll <= ROLL[1]) & (T[:, 2, 3] >= support + CLEAR)
    if not can.any():
        print(name, "no grasp the right hand can take", flush=True)
        return
    best = int(np.argmax(can))
    print(name, "grasps", len(T), "the right hand can take", int(can.sum()), "chosen rank", best, "score", round(float(score[best]), 4),
          "| SAM 3", str(z["named"]), round(float(z["score"]), 3), flush=True)  # fmt: skip
    if a.check:
        return
    hand = hand_outline(g["f_open"])

    fork = [np.array([[-0.045, 0.0, 0.0], [0.05, 0.0, 0.0]])] + [ln[[0, -1]] for ln in hand[1:]]

    def hands(idx, back=0.0, glyph=None):  # the grasps idx as polylines in the robot's frame
        out = []
        for i in idx:
            p = T[i, :3, 3] - back * T[i, :3, 1]
            out += [ln @ T[i, :3, :3].T + p for ln in (glyph or hand)]
        return out

    # what the views show
    big = lambda im: cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)  # noqa: E731
    L, Rt = big(left), big(right)
    m2 = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    cnt, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dim = (L * 0.32).astype(np.uint8)
    found = np.where(m2[..., None] > 0, L, dim)
    ys, xs = np.nonzero(m2)
    dlo, dhi = 0.3, float(np.clip(np.percentile(z["lb"][ok], 97), 1.5, 4.0))
    sg, lb = turbo(z["sg"], dlo, dhi), turbo(np.where(ok, z["lb"], 0), dlo, dhi)
    P, C = pts[ok], left[ok]
    tgt = mask[ok]
    Ct = C.copy()
    Ct[tgt] = (0.55 * C[tgt] + 0.45 * np.array(YEL)).astype(np.uint8)
    tc = np.median(pts[mask & ok], axis=0)
    off = (
        np.linalg.norm(P - cam_p, axis=1) > 0.55
    ) | tgt  # from outside: without the robot's own hands under the camera
    Po, Co = P[off], Ct[off]
    dist = np.linalg.norm(Po - tc, axis=1)
    near = (dist < 0.7) & (
        Po[:, 2] > float(z["top"]) - 0.05
    )  # step 5's close views: what stands on the target's table, the rest fades out
    Cs = (0.8 * C[off] + 0.2 * Ct[off]).astype(np.uint8)
    Pn, Cl = Po[near], Cs[near]

    # the camera's way out of the head and around the target
    fwd = -cam_R[:, 2]
    aim0 = cam_p + fwd * ((tc - cam_p) @ fwd)
    r0, az0, el0 = sph(cam_p - tc)
    side = (
        -0.72 * ahead - 0.6 * lft
    )  # from behind the robot's right shoulder: the faces the robot saw, and the hand's side
    az1 = az0 + np.arctan2(np.cross(unsph(1, az0, 0), side)[2], unsph(1, az0, 0) @ side)
    R0 = Rotation.from_matrix(cam_R)

    el_out = np.radians(34)

    def cam(s, r1=1.25, el1=el_out, daz=0.0):
        """s 0: the head camera itself; 1: outside, r1 from the target."""
        e = smooth(s)
        pos = tc + unsph(r0 + e * (r1 - r0), az0 + e * (az1 + daz - az0), el0 + e * (el1 - el0))
        Rl = look_at(pos, aim0 + e * (tc - aim0))
        if s >= 0.3:
            return pos, Rl
        return pos, Slerp([0, 1], Rotation.concatenate([R0, Rotation.from_matrix(Rl)]))(smooth(s / 0.3)).as_matrix()

    out = Writer(f"{a.out}/percept_{name}.mp4")
    lines = [
        f"开局站好，头上的相机拍一张。任务指令是“{str(z['instruction'])}”",
        "相机有左右两只眼睛，各拍一张，两张图有一点错位",
        (
            f"把指令里的物体名“{str(z['named'])}”交给 SAM 3，它在图里圈出这个物体"
            if str(z["named"]) != "None"
            else (
                "这条指令没说是哪个物体：让 SAM 3 圈出图里的物体，取手够得着的范围里最大的一个"
                if a.unnamed == "sam"
                else "这条指令没说是哪个物体，规则是取桌面上立着的最大的一块东西"
            )
        ),
        "对比左右两张图的错位，算出每个像素离相机多远。对不上的地方是黑的",
        "再用 LingBot-Depth 把对不上的地方补全",
        "每个像素按远近放回空间，就是一片三维的点。黄色的是目标",
        "换个角度看：只看了一眼，所以物体背面和被挡住的地方是空的",
        f"GraspGen-X 根据目标的点，给出 {len(T)} 个手掌可以放的位置，越绿分数越高",
        f"去掉右手做不到的：手指要朝前，手掌不能压到桌面。还剩 {int(can.sum())} 个",
        "剩下的里面取分数最高的一个。白色的手是抓之前先到的位置，在后面 8 厘米",
        "抓取点定下来了。之后走过去只靠腿上的里程计，不再看第二眼",
    ]
    ov = {}

    def put(img, step, k):
        if (step, k) not in ov:
            ov[(step, k)] = overlay(title, step, lines[k])
        paste(img, ov[(step, k)], 0, 0)
        out.write(img)

    def fade(A, B, t):
        return cv2.addWeighted(B, smooth(t), A, 1 - smooth(t), 0)

    def secs(s):
        return range(int(round(s * FPS)))

    # 1 the look
    for _ in secs(2.2):
        put(L.copy(), 0, 0)
    for i in secs(2.0):
        put((Rt if (i // 6) % 2 == 0 else L).copy(), 0, 1)
    # 2 the target
    tag = chip(
        f"{str(z['named'])}　SAM 3 置信度 {float(z['score']):.2f}"
        if str(z["named"]) != "None"
        else (f"够得着的最大物体　SAM 3 置信度 {float(z['score']):.2f}" if a.unnamed == "sam" else "桌面上最大的一块"),
        21,
        fg=INK,
        bg=YEL + (255,),
    )
    wrong = name in a.truth.split(",")
    if wrong:  # for the viewer only: where the task's target really is
        t2 = cv2.resize(z["true"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        tcnt, _ = cv2.findContours(t2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tys, txs = np.nonzero(t2)
        ttag = chip("真正的目标在这里，这一局找错了", 21, fg=WHITE, bg=(190, 40, 30, 255))
    for i in secs(4.4 if wrong else 3.2):
        img = fade(L, found, i / (0.6 * FPS))
        n = int(len(cnt) and min(1.0, i / (0.9 * FPS)) * max(len(c) for c in cnt))
        for c in cnt:
            cv2.polylines(img, [c[: max(2, n)]], n >= len(c), YEL, 3, cv2.LINE_AA)
        if i > 0.9 * FPS and len(xs):
            paste(
                img,
                tag,
                int(np.clip(xs.mean() - tag.shape[1] / 2, 8, W - tag.shape[1] - 8)),
                int(max(52, ys.min() - tag.shape[0] - 10)),
            )
        if wrong and i > 1.5 * FPS and len(txs):
            cv2.drawContours(img, tcnt, -1, (230, 60, 50), 3, cv2.LINE_AA)
            paste(
                img,
                ttag,
                int(np.clip(txs.mean() - ttag.shape[1] / 2, 8, W - ttag.shape[1] - 8)),
                int(min(H - 140, tys.max() + 12)),
            )
        put(img, 1, 2)
    # 3 the distances
    bar = colorbar(dlo, dhi)
    for i in secs(2.6):
        img = fade(found, sg, i / (0.6 * FPS))
        paste(img, bar, W - 170, 60)
        put(img, 2, 3)
    for i in secs(2.4):
        img = fade(sg, lb, i / (0.6 * FPS))
        paste(img, bar, W - 170, 60)
        put(img, 2, 4)
    # 4 the points: from the head camera, then from outside
    head = splat(P, Ct, cam_p, cam_R, f0, -1, 0)
    for i in secs(1.8):
        put(fade(lb, head, i / (0.7 * FPS)), 3, 5)
    n = len(secs(3.6))
    for i in range(n):
        c, Rc = cam(i / (n - 1))
        put(splat(P if i < 2 else Po, Ct if i < 2 else Co, c, Rc, f0, -1, 1), 3, 6 if i > 0.25 * n else 5)
    for i in secs(1.2):
        c, Rc = cam(1.0, daz=np.radians(4) * i / (1.2 * FPS))
        put(splat(Po, Co, c, Rc, f0, -1, 1), 3, 6)
    # 5 the grasps: all, those the right hand can take, the best of them
    order = np.argsort(score)  # the best drawn last
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bins = [order[(score[order] >= lo) & (score[order] < hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    cols = [
        tuple(
            int(v) for v in np.array([150, 156, 165]) + (np.array([60, 230, 110]) - np.array([150, 156, 165])) * k / 4
        )
        for k in range(5)
    ]
    alph = [0.14, 0.2, 0.3, 0.45, 0.7]
    canb = [b[can[b]] for b in bins]
    daz0, RN = np.radians(4), 0.52

    def view5(t):  # 0..1 over the step: close to the target, and slowly round it
        e = smooth(t / 0.16)
        return cam(1.0, r1=1.25 - (1.25 - RN) * e, el1=np.radians(34 + 4 * e), daz=daz0 + np.radians(16) * t)

    def cloud5(t, c, Rc):
        w = smooth(t / 0.1)  # what is far from the target fades into the background
        if w >= 1:
            return splat(Pn, Cl, c, Rc, f0, -2, 2)
        col = (Co * (1 - 0.2 * w) + Cs * 0.2 * w).astype(np.float32)
        col[~near] = col[~near] * (1 - w) + BG * w
        return splat(Po, col.astype(np.uint8), c, Rc, f0, -1, 1 if w < 0.5 else 2)

    def note(img, text):
        paste(img, chip(text, 22, fg=INK, bg=YEL + (255,)), 20, 58)

    n5 = len(secs(8.4))
    poster = None
    for i in range(n5):
        t = i / (n5 - 1)
        c, Rc = view5(t)
        img = cloud5(t, c, Rc)
        if t < 0.36:  # every grasp
            al = smooth((t - 0.10) / 0.08)
            for b, col, al0 in zip(bins, cols, alph):
                if al > 0.02 and len(b):
                    polylines(img, hands(b, glyph=fork), c, Rc, f0, col, 1, al0 * al)
            if al > 0.5:
                note(img, f"GraspGen-X：{len(T)} 个候选")
            k = 7
        elif t < 0.66:  # the ones the right hand can take
            al = 1 - smooth((t - 0.36) / 0.08)
            for b, bc, col, al0 in zip(bins, canb, cols, alph):
                if al > 0.02 and len(b[~can[b]]):
                    polylines(img, hands(b[~can[b]], glyph=fork), c, Rc, f0, col, 1, al0 * al)
                if len(bc):
                    polylines(img, hands(bc, glyph=fork), c, Rc, f0, col, 1, min(1.0, al0 + 0.25))
            note(img, f"右手做得到的：{int(can.sum())} 个")
            k = 8
        else:  # the best of them, and where the hand waits before it goes in
            al = 1 - smooth((t - 0.66) / 0.08)
            if al > 0.02:
                for bc, col, al0 in zip(canb, cols, alph):
                    if len(bc):
                        polylines(img, hands(bc, glyph=fork), c, Rc, f0, col, 1, min(1.0, al0 + 0.25) * al)
            polylines(img, hands([best], PRE), c, Rc, f0, WHITE, 2, 0.95 * (1 - al))
            polylines(
                img,
                [np.array([T[best, :3, 3] - PRE * T[best, :3, 1], T[best, :3, 3]])],
                c,
                Rc,
                f0,
                WHITE,
                2,
                0.95 * (1 - al),
            )
            polylines(img, hands([best]), c, Rc, f0, YEL, 4)
            note(img, f"选中 1 个，分数 {float(score[best]):.2f}")
            k = 9
            if t > 0.9 and poster is None:
                poster = img.copy()
                paste(poster, overlay(title, 4, lines[9]), 0, 0)
        put(img, 4, k)
    # 6 the grasp, back in the camera's image
    c5 = view5(1.0)
    n6 = len(secs(2.6))
    for i in range(n6):
        e = smooth(i / (n6 - 1))
        s = 1 - e
        c, Rc = cam(s, r1=RN, el1=np.radians(38), daz=daz0 + np.radians(16))
        img = (
            splat(Pn, Cl, c, Rc, f0, -2, 2)
            if s > 0.6
            else splat(Po if i < n6 - 2 else P, Co if i < n6 - 2 else Ct, c, Rc, f0, -1, 1)
        )
        polylines(img, hands([best], PRE), c, Rc, f0, WHITE, 2, 0.95)
        polylines(img, hands([best]), c, Rc, f0, YEL, 4)
        put(img, 5, 10)
    del c5
    p0, p1 = (project(T[best, :3, 3] - b * T[best, :3, 1], cam_p, cam_R, f0) for b in (PRE, 0.0))
    for i in secs(3.2):
        img = fade(head, L, i / (0.8 * FPS))
        polylines(img, hands([best], PRE), cam_p, cam_R, f0, WHITE, 2, 0.95)
        polylines(img, hands([best]), cam_p, cam_R, f0, YEL, 4)
        if p0[2] > 0.05 and p1[2] > 0.05:
            cv2.arrowedLine(
                img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), WHITE, 2, cv2.LINE_AA, tipLength=0.3
            )
        if i > 1.1 * FPS and p1[2] > 0.05:  # the same, enlarged
            cx, cy = int(np.clip(p1[0], 130, W - 130)), int(np.clip(p1[1], 85 + 44, H - 92 - 85))
            zoom = cv2.resize(img[cy - 85 : cy + 85, cx - 130 : cx + 130], (598, 391), interpolation=cv2.INTER_CUBIC)
            x0 = 24 if cx > W / 2 else W - 598 - 24
            cv2.rectangle(img, (cx - 130, cy - 85), (cx + 130, cy + 85), YEL, 2, cv2.LINE_AA)
            img[150 : 150 + 391, x0 : x0 + 598] = zoom
            cv2.rectangle(img, (x0, 150), (x0 + 598, 150 + 391), YEL, 2, cv2.LINE_AA)
            paste(img, chip("放大看：黄色是抓的位置，白色是先到的位置", 19, fg=INK, bg=YEL + (255,)), x0, 150 - 38)
        put(img, 5, 10)
    out.close()
    if poster is not None:
        cv2.imwrite(
            f"{a.out}/percept_{name}.jpg", cv2.cvtColor(poster, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86]
        )
    print("wrote", name, out.n, "frames", round(out.n / FPS, 1), "s", flush=True)


for spec in a.names:
    name, _, title = spec.partition("=")
    film(name, title or name)
