"""A look: what one frame of the head camera gives the pipeline, as one .npz file.

rgb          H x W x 3 uint8   the colour image
depth        H x W float32     m along the optical axis, aligned to rgb, 0 where unknown      } either or both
right        H x W x 3 uint8   the right image of a rectified stereo pair whose left is rgb   } (with baseline)
baseline     float             m between the stereo pair's cameras
K            3 x 3             the colour camera's intrinsics (pixels, OpenCV's convention)
T_base_cam   4 x 4             the camera frame in the base frame (frames.py)
instruction  str               e.g. "pick up the red cup"; or
object       str               the object's name as said to the segmenter ("red cup"); wins over the instruction
"""

import numpy as np


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    look = {k: z[k] for k in z.files}
    for k in ("instruction", "object"):
        if k in look:
            look[k] = str(look[k])
    rgb = look.get("rgb")
    assert rgb is not None and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, (
        f"{path}: rgb H x W x 3 uint8"
    )
    H, W = rgb.shape[:2]
    assert "depth" in look or ("right" in look and "baseline" in look), (
        f"{path}: a depth, or a right image and a baseline"
    )
    if "depth" in look:
        assert look["depth"].shape == (H, W), f"{path}: depth {look['depth'].shape}, the image {(H, W)}"
        look["depth"] = np.nan_to_num(look["depth"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for k, shape in (("K", (3, 3)), ("T_base_cam", (4, 4))):
        assert k in look and look[k].shape == shape, f"{path}: {k} {shape}"
        look[k] = look[k].astype(np.float64)
    assert "instruction" in look or "object" in look, f"{path}: an instruction or an object"
    return look


def save(path, rgb, K, T_base_cam, depth=None, right=None, baseline=None, instruction=None, obj=None) -> None:
    out = dict(rgb=np.asarray(rgb, np.uint8), K=np.asarray(K, float), T_base_cam=np.asarray(T_base_cam, float))
    if depth is not None:
        out["depth"] = np.asarray(depth, np.float32)
    if right is not None:
        out["right"], out["baseline"] = np.asarray(right, np.uint8), float(baseline)
    if instruction:
        out["instruction"] = str(instruction)
    if obj:
        out["object"] = str(obj)
    np.savez_compressed(path, **out)
