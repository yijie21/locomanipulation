"""One image to look at (TODO 49.5b): the front camera's first frame of an episode with the truth drawn on it.

The server's rgbd reply (colour, depth, K, the camera in the pelvis frame) and the privileged object poses are projected into the
image: a cross on every object's centre with its depth and the depth's range printed. A person checks that the camera, the depth and
the truth agree (the box where the cross is). Used by the bed only (`--set overlay_dir=DIR`), never by a skill package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def project(K, T_world_cam, p) -> tuple[float, float, float] | None:
    """Pixel (u, v) and depth of a world point, or None when it is behind the camera."""
    pc = np.linalg.inv(T_world_cam) @ np.append(np.asarray(p, float), 1.0)
    if pc[2] <= 1e-3:
        return None
    return float(K[0, 0] * pc[0] / pc[2] + K[0, 2]), float(K[1, 1] * pc[1] / pc[2] + K[1, 2]), float(pc[2])


def save_first(link, priv, eid: str, out_dir: str) -> str:
    import cv2

    r = link.ask("rgbd")
    img = np.ascontiguousarray(np.asarray(r["rgb"], np.uint8)[..., ::-1])
    root = np.asarray(link.ask("state")["root"], float)
    from locomani.bench.humanoidarena.privileged import quat_to_mat

    T_wb = np.eye(4)
    T_wb[:3, :3], T_wb[:3, 3] = quat_to_mat(root[3:7]), root[:3]
    T_wc = T_wb @ np.asarray(r["T_body_cam"], float)
    d = np.asarray(r["depth"], float)
    ok = np.isfinite(d) & (d > 0)
    cv2.putText(
        img,
        f"depth {d[ok].min():.2f}-{d[ok].max():.2f} m  {eid}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    for name, o in priv.objects().items():
        pr = project(np.asarray(r["K"], float), T_wc, o[:3])
        if pr is None:
            continue
        u, v, z = pr
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            vi, ui = int(min(max(v, 0), d.shape[0] - 1)), int(min(max(u, 0), d.shape[1] - 1))
            cv2.putText(
                img,
                f"{name} z={z:.2f} depth={d[vi, ui]:.2f}",
                (int(u) + 10, int(v) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = str(path / (eid.replace("/", "_") + ".png"))
    cv2.imwrite(out, img)
    return out


def save_world(link, eid: str, name: str, out_dir: str) -> str:
    """The benchmark's third-person camera at a mark of the package, a quarter of its size: for a person to look at."""
    import cv2

    img = np.asarray(link.ask("worldcam")["rgb"], np.uint8)[..., ::-1][::2, ::2]
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = str(path / f"{eid.replace('/', '_')}_{name}.png")
    cv2.imwrite(out, np.ascontiguousarray(img))
    return out
