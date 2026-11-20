"""Depth for the look (perception env): LingBot-Depth completes and corrects a depth image from the colour image; for a
stereo pair, semi-global matching first gives it the depth to start from (how the simulated benchmark's head camera
worked)."""

import numpy as np

LINGBOT = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


class LingBot:
    def __init__(self, name: str = LINGBOT, device: str = "cuda"):
        import torch
        from mdm.model.v2 import MDMModel

        self.torch, self.device = torch, device
        self.model = MDMModel.from_pretrained(name).to(device).eval()

    def __call__(self, rgb: np.ndarray, depth_in: np.ndarray, K: np.ndarray) -> np.ndarray:
        """The completed depth (m, 0 where it has none) from the colour image and a depth to start from."""
        torch = self.torch
        H, W = depth_in.shape
        Kn = [[K[0, 0] / W, 0, (K[0, 2] + 0.5) / W], [0, K[1, 1] / H, (K[1, 2] + 0.5) / H], [0, 0, 1]]
        Kn = torch.tensor(Kn, dtype=torch.float32, device=self.device)[None]
        img = torch.tensor(rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
        d = torch.tensor(np.asarray(depth_in, np.float32), device=self.device)[None]
        with torch.no_grad():
            out = self.model.infer(img, depth_in=d, intrinsics=Kn)["depth"][0].float().cpu().numpy()
        return np.where(np.isfinite(out) & (out > 0), out, 0.0).astype(np.float32)


def stereo(left: np.ndarray, right: np.ndarray, fx: float, baseline: float, up: int = 2) -> np.ndarray:
    """Depth (m, 0 where matching fails) of the left image from a rectified pair, by semi-global matching at `up`
    times the resolution (the benchmark's settings: a 110 deg, 5 cm head pair at 640 x 360)."""
    import cv2

    g = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
         for i in (left, right)]  # fmt: skip
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64 * up, blockSize=5, P1=200, P2=800, uniquenessRatio=10,
                              speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_HH)  # fmt: skip
    d = cv2.resize(m.compute(*g).astype(np.float32) / 16.0 / up, left.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return np.where(d > 0.5, fx * baseline / np.maximum(d, 1e-3), 0.0).astype(np.float32)
