"""Two eyes name the target on their own; where they name different things, the wide view settles it. Runs in the sam3d env.

The wrist camera (--a) and the head camera (--b) each give the target's points in the world frame the robot believes in.
  agree     the two clouds' centres are within --near: the wrist's cloud is kept.
  one       only one eye found anything: its cloud is kept.
  disagree  both candidates are boxed and numbered in the head camera's frame, which shows the whole piece of furniture, and
            Qwen3-VL is asked which box is the expression. A wrist candidate outside the head's frame cannot be boxed: the
            wrist's cloud is kept.
No truth is used; the table (agree_<tag>.json) carries each eye's overlap with the true mask for scoring afterwards.

    python vision_agree.py <set> --objects <simready>/objects.json --a wb100 --b lk100 --see-b look100 --tag wbx
-> <set>/vision/see_<tag>_L<k>_s<j>.npz (pts, p_o, R_o)
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_phrases import phrase_of  # noqa: E402
from vision_seg import Qwen, expression  # noqa: E402

H, W = 480, 640
COLOURS = ((255, 0, 0), (0, 80, 255))


def box_of(P: np.ndarray, cam_p: np.ndarray, cam_R: np.ndarray, fovy: float):
    """The box of world points in the camera's frame, or None when most of them fall outside it."""
    c = (P - cam_p) @ cam_R
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    u, v = c[:, 0] * f / np.maximum(c[:, 2], 1e-6) + W / 2, c[:, 1] * f / np.maximum(c[:, 2], 1e-6) + H / 2
    inside = (c[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.mean() < 0.5:
        return None
    (x0, x1), (y0, y1) = np.percentile(u[inside], [2, 98]), np.percentile(v[inside], [2, 98])
    return [float(x0) - 4, float(y0) - 4, float(x1) + 4, float(y1) + 4]


def which(qwen: Qwen, img: Image.Image, boxes: list, what: str):
    """The index of the box Qwen names, or None."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=COLOURS[i], width=3)
        d.text((b[0] + 3, max(0, b[1] - 12)), str(i + 1), fill=COLOURS[i])
    text = (
        f"Two objects are marked with numbered boxes: 1 is red, 2 is blue. Which box marks {what}? "
        "Answer with the number only."
    )
    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    inp = qwen.proc.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out = qwen.model.generate(**inp, max_new_tokens=8, do_sample=False)
    txt = qwen.proc.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    m = re.search(r"[12]", txt)
    return int(m.group()) - 1 if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--a", required=True, help="the wrist camera's clouds")
    ap.add_argument("--b", required=True, help="the head camera's clouds")
    ap.add_argument("--see-b", required=True, help="the head camera's own frames")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--near", type=float, default=0.06)
    ap.add_argument("--min-points", type=int, default=100)
    a = ap.parse_args()
    V = os.path.join(a.set, "vision")
    kinds = {o["name"]: o["type"] for o in json.load(open(a.objects))["objects"]}
    truth = {
        t: {(r["layout"], r["slot"]): r.get("iou", 0.0) for r in json.load(open(os.path.join(V, f"seg_{t}.json")))}
        for t in (a.a, a.b)
    }
    qwen, rows = None, []
    for f in sorted(glob.glob(os.path.join(V, f"see_{a.a}_L*_s*.npz"))):
        k, j = map(int, re.search(r"_L(\d+)_s(\d+)\.npz$", f).groups())
        za = np.load(f)
        fb = os.path.join(V, f"see_{a.b}_L{k}_s{j}.npz")
        zb = np.load(fb) if os.path.exists(fb) else None
        A = za["pts"].astype(np.float64)
        B = zb["pts"].astype(np.float64) if zb is not None else np.zeros((0, 3))
        row = dict(layout=k, slot=j, iou_a=truth[a.a].get((k, j), 0.0), iou_b=truth[a.b].get((k, j), 0.0))
        has_a, has_b = len(A) >= a.min_points, len(B) >= a.min_points
        take = "a"
        if has_a and has_b:
            row["off_cm"] = float(100 * np.linalg.norm(np.median(A, axis=0) - np.median(B, axis=0)))
            row["case"] = "agree" if row["off_cm"] < 100 * a.near else "disagree"
            if row["case"] == "disagree":
                raw = np.load(os.path.join(V, f"see_{a.see_b}_L{k}_s{j}.npz"))
                ba = box_of(A, raw["cam_p"], raw["cam_R"], float(raw["fovy"]))
                bb = box_of(B, raw["cam_p"], raw["cam_R"], float(raw["fovy"]))
                if ba is not None and bb is not None:
                    layout = json.load(open(os.path.join(a.set, f"layout_{k}", "layout.json")))
                    what = expression(layout, j, phrase_of(kinds[layout["slots"][j]["object"]]))
                    qwen = qwen or Qwen()
                    order = ("a", "b") if (k + j) % 2 == 0 else ("b", "a")  # the number a candidate gets alternates
                    i = which(qwen, Image.fromarray(raw["rgb"]), [dict(a=ba, b=bb)[o] for o in order], what)
                    row["asked"] = True
                    if i is not None:
                        take = order[i]
        elif has_a or has_b:
            row["case"], take = "one", "a" if has_a else "b"
        else:
            row["case"] = "none"
        z = za if take == "a" else zb
        row.update(take=take, iou=row["iou_" + take])
        np.savez_compressed(os.path.join(V, f"see_{a.tag}_L{k}_s{j}.npz"), pts=z["pts"], p_o=za["p_o"], R_o=za["R_o"])
        rows.append(row)
        if row["case"] != "agree":
            print({key: (round(val, 2) if isinstance(val, float) else val) for key, val in row.items()}, flush=True)
    json.dump(rows, open(os.path.join(V, f"agree_{a.tag}.json"), "w"))
    right = lambda key, rs: sum(r[key] >= 0.5 for r in rs)  # noqa: E731
    for case in ("agree", "one", "disagree", "none"):
        rs = [r for r in rows if r["case"] == case]
        print(
            f"{case}: {len(rs)}; right: wrist {right('iou_a', rs)}, head {right('iou_b', rs)}, taken {right('iou', rs)}"
        )
    print(f"all {len(rows)}: wrist {right('iou_a', rows)}, taken {right('iou', rows)}")


if __name__ == "__main__":
    main()
