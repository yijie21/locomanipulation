"""The target's mask from the head camera's colour and depth frame, with no simulator mask. Runs in the sam3d env, one GPU.

--how gd     Grounding DINO's best box for the object's phrase, the depth inside the box gives the object (what stands above the
             surface it rests on, the piece nearest the box centre). Fails on which-one: several objects share a phrase.
--how refer  the task names the object the way an upstream VLM would ("the can, second object from the left on the wooden table"):
             SAM 3 (scripts/rsi/sam3_server.py on --port) gives the instances of the phrase, Qwen3-VL-8B a box for the whole
             expression; the instance that overlaps the box most is taken, the depth cut inside the box when SAM 3 has none.
             Pixels of the robot's own body are never candidates (the robot knows where its arms are; the frame's `robot` mask).
--depth rs   points from the RealSense-like depth (vision_depth.py) instead of the clean one; lb: that depth through
             LingBot-Depth (vision_depth_fix.py); lbfill: the sensor's reading where it has one, LingBot-Depth's in the holes.

-> <set>/vision/see_<tag>_L<k>_s<j>.npz (points, p_o, R_o), seg_<tag>.json (overlap with the true mask, per stage)

    HF_HOME=/workspace/huggingface_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 /workspace/miniconda3/envs/sam3d/bin/python -s \\
        scripts/experiments/reachwm_20261030/vision_seg.py <set> --objects <simready>/objects.json
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
from PIL import Image
from scipy import ndimage
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_phrases import phrase_of  # noqa: E402

RISE, BAND = 0.008, 0.20  # m over the surface to count as the object; m of depth around the piece kept


def points(depth, cam_p, cam_R, fovy):
    H, W = depth.shape
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    v, u = np.mgrid[0:H, 0:W]
    return cam_p + np.stack([(u - W / 2 + 0.5) * depth / f, (v - H / 2 + 0.5) * depth / f, depth], -1) @ cam_R.T


def object_in_box(P, depth, box):
    """Pixels of the object inside the box: above the surface under it (the commonest height in the box's lower rows and
    around it), the connected piece nearest the box centre."""
    H, W = depth.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros_like(depth, bool)
    pad = max(4, (x1 - x0) // 3)
    ring = np.zeros_like(depth, bool)
    ring[max(0, y0 - pad) : min(H, y1 + pad), max(0, x0 - pad) : min(W, x1 + pad)] = True
    ring[y0:y1, x0:x1] = False
    zc = P[y0:y1, x0:x1, 2]
    z_obj = float(np.median(zc))
    zr = P[..., 2][ring & (depth < 3.0)]
    zr = zr[zr < z_obj + 0.005]  # the surface is under the object
    if len(zr) < 20:
        zr = zc.ravel()
    hist, edges = np.histogram(zr, bins=np.arange(zr.min() - 0.005, zr.max() + 0.01, 0.005))
    surf = float(edges[np.argmax(hist)] + 0.0025)
    m = np.zeros_like(depth, bool)
    m[y0:y1, x0:x1] = zc > surf + RISE
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    best = min(
        range(1, n + 1), key=lambda i: np.min(np.hypot(*(np.argwhere(lab == i) - [cy, cx]).T)) - 1e-3 * (lab == i).sum()
    )
    m = lab == best
    return m & (np.abs(depth - np.median(depth[m])) < BAND)


WHERE = {"tray": "on the grey tray on the floor", "low": "on the white low cabinet", "table": "on the wooden table",
         "counter": "on the white counter", "high": "on the high shelf"}  # fmt: skip
ORD = {
    3: ("leftmost", "middle", "rightmost"),
    4: ("leftmost", "second from the left", "second from the right", "rightmost"),
}


def expression(layout: dict, j: int, phrase: str) -> str:
    """How the task names slot j: its phrase and its place in the row, left to right as the robot faces the station."""
    s = layout["slots"][j]
    row = sorted(
        (t for t in layout["slots"] if t["station"] == s["station"] and t["surf"] == s["surf"]), key=lambda t: -t["lat"]
    )
    rank = next(i for i, t in enumerate(row) if t["object"] == s["object"])
    where = (
        WHERE.get(s["station"])
        or f"on the {('bottom', 'middle', 'top')[(0.30, 0.62, 0.94).index(s['surf'])]} shelf of the wooden cabinet"
    )
    return f"the {phrase} that is the {ORD[len(row)][rank]} object {where}"


class Qwen:
    def __init__(self):
        from transformers import AutoModelForImageTextToText

        name = "Qwen/Qwen3-VL-8B-Instruct"
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()

    def box(self, img: Image.Image, what: str):
        """The box of `what` in pixels, or None (Qwen answers in thousandths of the frame)."""
        text = f'Locate {what}. Output its bounding box as JSON: {{"bbox_2d": [x1, y1, x2, y2]}}.'
        msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        inp = self.proc.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=64, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", txt)
        if not m:
            return None
        x0, y0, x1, y1 = (int(v) for v in m.groups())
        W, H = img.size
        return [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]


def box_mask(shape, b):
    m = np.zeros(shape, bool)
    m[max(0, int(b[1])) : int(b[3]) + 1, max(0, int(b[0])) : int(b[2]) + 1] = True
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--pose", default="start")
    ap.add_argument("--how", default="refer", choices=["gd", "refer"])
    ap.add_argument("--depth", default="clean", choices=["clean", "rs", "lb", "lbfill"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--box", type=float, default=0.25)
    ap.add_argument("--port", type=int, default=18832)
    ap.add_argument("--min-score", type=float, default=0.3)
    a = ap.parse_args()
    tag = a.tag or a.how
    V = os.path.join(a.set, "vision")
    kinds = {o["name"]: o["type"] for o in json.load(open(a.objects))["objects"]}
    iou = lambda m, t: int((m & t).sum()) / max(1, int((m | t).sum()))  # noqa: E731
    if a.how == "gd":
        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").cuda().eval()
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
        from locomani.tasks.scene.services import Client, unpack_mask

        sam, qwen = Client(("127.0.0.1", a.port)), Qwen()
        boxes_f = os.path.join(V, f"refer_{a.pose}.json")  # Qwen's boxes do not depend on the depth: asked once
        boxes = json.load(open(boxes_f)) if os.path.exists(boxes_f) else {}
    rows = []
    for f in sorted(glob.glob(os.path.join(V, f"see_{a.pose}_L*_s*.npz"))):
        z = np.load(f)
        k, j = (int(x[1:]) for x in os.path.basename(f)[:-4].split("_")[2:])
        layout = json.load(open(os.path.join(a.set, f"layout_{k}", "layout.json")))
        slot = layout["slots"][j]
        prompt = phrase_of(kinds[slot["object"]])
        img = Image.fromarray(z["rgb"])
        true, robot = z["mask"], z["robot"]
        depth = (z["depth"] if a.depth == "clean" else z["depth_rs"]).astype(np.float32)
        if a.depth in (
            "lb",
            "lbfill",
        ):  # LingBot-Depth's frame (vision_depth_fix.py): all of it, or only where the sensor has a hole
            lb = np.load(os.path.join(V, f"lb_{a.pose}_L{k}_s{j}.npz"))["depth_lb"]
            depth = lb if a.depth == "lb" else np.where(depth > 0, depth, lb)
        valid = depth > 0
        cam_R = (
            z["cam_R_true"] if a.depth == "clean" and "cam_R_true" in z.files else z["cam_R"]
        )  # clean = no calibration error
        cam_p = z["cam_p_true"] if a.depth == "clean" and "cam_p_true" in z.files else z["cam_p"]
        P = points(np.where(valid, depth, 9.0), cam_p, cam_R, float(z["fovy"]))
        row = dict(layout=k, slot=j, station=slot["station"], surf=slot["surf"], prompt=prompt, true_px=int(true.sum()))
        mask = np.zeros_like(true)
        if a.how == "gd":
            inp = proc(images=img, text=prompt + ".", return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
            det = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=a.box, text_threshold=0.2,
                                                              target_sizes=[img.size[::-1]])[0]  # fmt: skip
            row["n"] = int(len(det["scores"]))
            if len(det["scores"]):
                b = det["boxes"][int(torch.argmax(det["scores"]))].tolist()
                mask = object_in_box(P, np.where(valid, depth, 9.0), b)
                row.update(score=float(det["scores"].max()), box=[round(v, 1) for v in b],
                           box_hit=float((true & box_mask(true.shape, b)).sum() / max(1, true.sum())))  # fmt: skip
        else:
            what = expression(layout, j, prompt)
            key = f"{k}_{j}"
            if key not in boxes:
                boxes[key] = qwen.box(img, what)
            b = boxes[key]
            found = sam.call(dict(image=z["rgb"], prompts=[prompt]))["results"][prompt]
            cands = [(s, m) for s, m in ((s, unpack_mask(m)) for s, m in found if s >= a.min_score)
                     if (m & robot).sum() < 0.3 * max(1, m.sum()) and m.sum() < 0.15 * m.size]  # fmt: skip
            row.update(what=what, n=len(cands), qwen=b is not None,
                       cand_iou=max((iou(m, true) for _, m in cands), default=0.0))  # fmt: skip
            if b is not None:
                bm = box_mask(true.shape, b)
                row.update(box=[round(v, 1) for v in b], box_hit=float((true & bm).sum() / max(1, true.sum())))
            if cands and b is not None and any((m & bm).any() for _, m in cands):
                mask, row["by"] = max(cands, key=lambda c: (c[1] & bm).sum() / max(1, (c[1] | bm).sum()))[1], "sam+qwen"
            elif cands and (b is None or len(cands) == 1):
                mask, row["by"] = max(cands, key=lambda c: c[0])[1], "sam"
            elif b is not None:
                mask, row["by"] = object_in_box(P, np.where(valid, depth, 9.0), b) & ~robot, "qwen+depth"
        mask = mask & valid
        row.update(px=int(mask.sum()), iou=iou(mask, true))
        np.savez_compressed(os.path.join(V, f"see_{tag}_L{k}_s{j}.npz"), pts=P[mask].astype(np.float32), p_o=z["p_o"],
                            R_o=z["R_o"], mask=mask)  # fmt: skip
        rows.append(row)
        print({k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in row.items() if k_ != "box"}, flush=True)
    if a.how == "refer":
        json.dump(boxes, open(boxes_f, "w"))
    json.dump(rows, open(os.path.join(V, f"seg_{tag}.json"), "w"))
