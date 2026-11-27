"""look_r.py <runs> <placement> <out.png> az,el,d: check render, A's frame | R's frame (no ghost). notebook: skip figure render."""
import sys
out, v = sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "look"]
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import numpy as np
from PIL import Image
import fig_render as FR
M, st, _ = FR.load()
G = FR.masks(M)
yaw, la = FR.heading(st), FR.lookat(M, st)
az, el, d = (float(x) for x in v.split(","))
Mr = FR.prep(M, G, 900, 600, 8)
rnd = FR.mujoco.Renderer(Mr, 600, 900)
c = FR.cam(la, yaw, az, el, d)
opt = FR.option()
tiles = [FR.render(rnd, st[a]["D"], c, opt) for a in ("A", "R")]
Image.fromarray(np.concatenate(tiles, 1)).save(out)
print("saved", out)
