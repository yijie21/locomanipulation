"""Development viewer (SIMPLE's venv; numpy + PIL): an episode's saved looks side by side with SAM 3's masks tinted, and
a summary of its film (the phases, their step ranges, the root's path, the right palm's height).

    .venv/bin/python show_episode.py <bench out dir> <episode name, e.g. ppbox_J_s104_r0> <png>
"""

import glob
import sys

import numpy as np
from PIL import Image


def main() -> int:
    out, name, png = sys.argv[1:4]
    files = sorted(glob.glob(f"{out}/looks/{name}_*.npz"))
    ims = []
    for f in files:
        z = np.load(f)
        im = z["rgb"].copy()
        ms = z["masks"]
        for i, m in enumerate(ms[:2] if ms.ndim == 3 else []):
            col = np.array([[255, 0, 0], [0, 0, 255]][i])
            im[m] = (0.5 * im[m] + 0.5 * col).astype(np.uint8)
        ims.append(np.asarray(Image.fromarray(im).resize((320, 240))))
    if ims:
        rows = [
            np.concatenate(ims[i : i + 4] + [np.zeros_like(ims[0])] * (4 - len(ims[i : i + 4])), 1)
            for i in range(0, len(ims), 4)
        ]
        Image.fromarray(np.concatenate(rows, 0)).save(png)
    print("looks:", [f.split("/")[-1] for f in files])
    fz = np.load(f"{out}/film/{name}.npz")
    names = [str(n) for n in fz["phase_names"]]
    ph = fz["phase"]
    root = fz["root"]
    palm = fz["palm"]
    k = 0
    while k < len(ph):
        j = k
        while j < len(ph) and ph[j] == ph[k]:
            j += 1
        print(f"  {names[ph[k]]:>14} steps {k:5d}-{j - 1:5d}  root {root[k, :2].round(2)} -> {root[j - 1, :2].round(2)} "
              f"z {root[j - 1, 2]:.2f}  palm {palm[j - 1, :3].round(2)}")  # fmt: skip
        k = j
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
