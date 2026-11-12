"""The upstream VLM points at the target in the frame it sees; the point is carried through 3D to the wrist camera.

Naming the target ("the bottle, second from the left") is the upstream VLM's job, and it answers with a pixel in a frame where
the whole row shows. Here that pixel is given: a pixel on the target in the head camera's frame when the head sees it
(--min-px of it), else a pixel in the wrist camera's own frame. That pixel is the task's input, not a reading of the world.
  head pixel   its depth gives a point in the world the robot believes in. In the wrist frame, SAM 3's instances of the
               object's phrase and of "object" are the candidates; the one whose points are nearest that point, within --near,
               is the target. None that near: the depth cut around the point's place in the wrist frame.
  wrist pixel  the smallest candidate that holds the pixel, else the depth cut around it.
Runs in the sam3d env with scripts/rsi/sam3_server.py on --port.

    python vision_point.py <set> --objects <simready>/objects.json --wrist wristb100 --head look100 --tag wbp
-> <set>/vision/see_<tag>_L<k>_s<j>.npz (pts, p_o, R_o, mask), seg_<tag>.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from vision_phrases import phrase_of  # noqa: E402
from vision_seg import object_in_box, points  # noqa: E402


def pixel_on(mask: np.ndarray):
    """A pixel of the mask, the one nearest its centre: (row, column)."""
    rc = np.argwhere(mask)
    return rc[np.argmin(np.linalg.norm(rc - rc.mean(axis=0), axis=1))]


def depth_of(V: str, pose: str, k: int, j: int, raw) -> np.ndarray:
    lb = os.path.join(V, f"lb_{pose}_L{k}_s{j}.npz")
    return np.load(lb)["depth_lb"] if os.path.exists(lb) else raw["depth_rs"].astype(np.float32)


def main() -> None:
    from locomani.tasks.scene.services import Client, unpack_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=18832)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--min-px", type=int, default=200)
    ap.add_argument("--near", type=float, default=0.06)
    a = ap.parse_args()
    V = os.path.join(a.set, "vision")
    kinds = {o["name"]: o["type"] for o in json.load(open(a.objects))["objects"]}
    sam = Client(("127.0.0.1", a.port))
    rows = []
    for f in sorted(glob.glob(os.path.join(V, f"see_{a.wrist}_L*_s*.npz"))):
        k, j = map(int, re.search(r"_L(\d+)_s(\d+)\.npz$", f).groups())
        z = np.load(f)
        slot = json.load(open(os.path.join(a.set, f"layout_{k}", "layout.json")))["slots"][j]
        phrase = phrase_of(kinds[slot["object"]])
        true, robot = z["mask"], z["robot"]
        depth = depth_of(V, a.wrist, k, j, z)
        valid = depth > 0
        fovy = float(z["fovy"])
        P = points(np.where(valid, depth, 9.0), z["cam_p"], z["cam_R"], fovy)
        Hh, Ww = depth.shape
        foc = 0.5 * Hh / math.tan(math.radians(fovy) / 2)
        row = dict(layout=k, slot=j, station=slot["station"], surf=slot["surf"], prompt=phrase, true_px=int(true.sum()))
        fh = os.path.join(V, f"see_{a.head}_L{k}_s{j}.npz")
        zh = np.load(fh) if os.path.exists(fh) else None
        X = pix = None
        if zh is not None and zh["mask"].sum() >= a.min_px:
            r, c = pixel_on(zh["mask"])
            dh = depth_of(V, a.head, k, j, zh)
            win = dh[max(0, r - 2) : r + 3, max(0, c - 2) : c + 3]
            if (win > 0).any():
                d = float(np.median(win[win > 0]))
                fh_ = 0.5 * dh.shape[0] / math.tan(math.radians(float(zh["fovy"])) / 2)
                X = zh["cam_p"] + zh["cam_R"] @ np.array(
                    [(c - dh.shape[1] / 2 + 0.5) * d / fh_, (r - dh.shape[0] / 2 + 0.5) * d / fh_, d]
                )
                row["src"] = "head"
        if X is None and true.sum() > 0:
            pix, row["src"] = pixel_on(true), "wrist"
        mask = np.zeros_like(true)
        if X is not None or pix is not None:
            found = sam.call(dict(image=z["rgb"], prompts=[phrase, "object"]))["results"]
            cands = [m for m in (unpack_mask(m) & valid for p in (phrase, "object") for s, m in found[p] if s >= a.min_score)
                     if m.sum() >= 30 and (m & robot).sum() < 0.3 * m.sum() and m.sum() < 0.15 * m.size]  # fmt: skip
            row["n"] = len(cands)
            if X is not None:
                cx = (X - z["cam_p"]) @ z["cam_R"]
                pix = np.array([cx[1] * foc / max(cx[2], 1e-6) + Hh / 2, cx[0] * foc / max(cx[2], 1e-6) + Ww / 2])
                dist = [float(np.linalg.norm(np.median(P[m], axis=0) - X)) for m in cands]
                if dist and min(dist) < a.near:
                    mask, row["by"] = cands[int(np.argmin(dist))], "sam near the point"
                    row["off_cm"] = 100 * min(dist)
            else:
                hold = [m for m in cands if m[pix[0], pix[1]]]
                if hold:
                    mask, row["by"] = min(hold, key=lambda m: m.sum()), "sam on the pixel"
            if "by" not in row and cx_ok(pix, Hh, Ww):
                half = a.near * foc / max(float(depth[int(pix[0]), int(pix[1])]), 0.2)
                box = [pix[1] - half, pix[0] - half, pix[1] + half, pix[0] + half]
                mask, row["by"] = object_in_box(P, np.where(valid, depth, 9.0), box) & ~robot & valid, "depth cut"
        row.update(px=int(mask.sum()), iou=float((mask & true).sum() / max(1, (mask | true).sum())))
        np.savez_compressed(os.path.join(V, f"see_{a.tag}_L{k}_s{j}.npz"), pts=P[mask].astype(np.float32), p_o=z["p_o"],
                            R_o=z["R_o"], mask=mask)  # fmt: skip
        rows.append(row)
        if row["iou"] < 0.5:
            print({key: (round(val, 2) if isinstance(val, float) else val) for key, val in row.items()}, flush=True)
    json.dump(rows, open(os.path.join(V, f"seg_{a.tag}.json"), "w"))
    for src in ("head", "wrist", None):
        rs = [r for r in rows if r.get("src") == src]
        print(f"pointed in {src}: {len(rs)}, right {sum(r['iou'] >= 0.5 for r in rs)}")
    print(f"all {len(rows)}: right {sum(r['iou'] >= 0.5 for r in rows)}")


def cx_ok(pix, H: int, W: int) -> bool:
    return 0 <= pix[0] < H and 0 <= pix[1] < W


if __name__ == "__main__":
    main()
