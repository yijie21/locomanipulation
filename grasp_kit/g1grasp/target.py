"""The target in the look (perception env): SAM 3's mask of the object the instruction names, its points in the base
frame, and the height of what it stands on."""

import re

import numpy as np

from .config import Config


class Sam3:
    """SAM 3 on one image and one text prompt: every instance above its confidence floor, as (score, mask)."""

    def __init__(self, threshold: float = Config.sam_threshold):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=threshold)

    def __call__(self, rgb: np.ndarray, prompt: str) -> list:
        from PIL import Image

        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(Image.fromarray(np.asarray(rgb, np.uint8)))
            o = self.proc.set_text_prompt(state=state, prompt=prompt)
            if not len(o["masks"]):
                return []
            masks = o["masks"].squeeze(1).cpu().numpy().astype(bool)
            scores = o["scores"].float().cpu().numpy()
        return [(float(s), m) for s, m in zip(scores, masks)]


def named(instruction: str):
    """The object an instruction names ("pick up the red cup from the table" -> "red cup"), or None ("the object")."""
    s = instruction.lower().strip()
    for pat in (
        r"pick up the ([a-z ]+?)(?: from|,|\.|$)",
        r"hand over (?:the )?([a-z ]+?) from",
        r"grasp the ([a-z ]+?)(?: on| from|,|\.|$)",
    ):
        m = re.search(pat, s)
        if m and m.group(1).strip() not in ("object", "thing", "item"):
            return m.group(1).strip()
    return None


def best_named(sam: Sam3, rgb, name: str, cfg: Config):
    """SAM 3's best-scored instance of `name`, as (mask, score), or (None, 0)."""
    best = (None, 0.0)
    for score, m in sam(rgb, name):
        if score > max(best[1], cfg.sam_floor):
            best = (m, score)
    return best


def largest_seen(sam: Sam3, rgb, pts, ok, cfg: Config):
    """For an instruction that names no object: the largest of SAM 3's instances of an object (cfg.generic) that is of
    an object's size and lies between cfg.own and cfg.reach ahead of the pelvis, as (mask, score), or (None, 0). The
    depth only places an instance within the reach, so a few centimetres of depth error on a plain or glossy table top
    do not decide what the target is."""
    u = pts[..., 0]
    best = (None, 0.0, 0)
    for score, m in sam(rgb, cfg.generic):
        n = int(m.sum())
        if score < cfg.sam_floor or not cfg.min_px <= n <= cfg.max_frac * m.size or (m & ok).sum() < cfg.min_seen:
            continue
        if cfg.own < float(np.median(u[m & ok])) < cfg.reach and n > best[2]:
            best = (m, score, n)
    return best[0], best[1]


def support_of(pts, ok, mask) -> float:
    """The height of what the masked thing stands on: the level of the points around its foot."""
    obj = pts[mask & ok]
    lo, c = np.percentile(obj[:, 2], 2), np.median(obj[:, :2], axis=0)
    near = ok & ~mask & (np.linalg.norm(pts[..., :2] - c, axis=2) < 0.35) & (np.abs(pts[..., 2] - lo) < 0.04)
    z = pts[near][:, 2]
    if len(z) < 50:
        return float(lo)
    h, e = np.histogram(z, bins=np.arange(lo - 0.04, lo + 0.045, 0.005))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    return float(np.median(z[np.abs(z - top) < 0.006]))


def find(sam: Sam3, rgb, pts, ok, instruction: str = "", obj: str = "", cfg: Config | None = None) -> dict:
    """The target: dict(phrase, score, mask, cloud (N x 3, base frame), table_top (m, base frame)); mask None when
    nothing was seen."""
    from scipy.ndimage import binary_erosion

    cfg = cfg or Config()
    phrase = obj or named(instruction)
    mask, score = best_named(sam, rgb, phrase, cfg) if phrase else largest_seen(sam, rgb, pts, ok, cfg)
    out = dict(phrase=phrase or cfg.generic, score=float(score), mask=mask)
    if mask is None or (mask & ok).sum() < cfg.min_seen:
        return dict(out, mask=None)
    keep = (binary_erosion(mask, iterations=cfg.erode) if cfg.erode else mask) & ok
    cloud = pts[keep]
    cloud = cloud[np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1) < cfg.radius]
    return dict(out, cloud=cloud, table_top=support_of(pts, ok, mask))
