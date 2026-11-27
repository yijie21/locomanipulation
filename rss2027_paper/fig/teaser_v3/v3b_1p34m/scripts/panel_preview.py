"""panel_preview.py <runs> <placement> <out png> az,el,d ...: check render (not a deliverable): rows of A | R at v3b's
panel moment (after3500ms), 900 x 600 each, offsamples 8. notebook: skip figure render, not an experiment."""
import sys
runs, pi, out, views = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.argv = [sys.argv[0], runs, pi, "lib", "/tmp"]
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import fig_final as FF  # noqa: E402
M, Z, st, _ = FF.load()
G = FF.FR.masks(M)
rnd = mujoco.Renderer(FF.FR.prep(M, G, 900, 600, 8), 600, 900)
opt = FF.FR.option()
rows = []
for v in views:
    az, el, d = (float(x) for x in v.split(","))
    c, _, _ = FF.main_cam(M, st, az, el, d)
    t = [FF.FR.label(FF.FR.render(rnd, st[FF.PANEL][a]["D"], c, opt), f"{a}  az {az:+.0f} down {el:.0f} d {d:.1f}  +3.50 s") for a in ("A", "R")]
    rows.append(np.concatenate(t, 1))
Image.fromarray(np.concatenate(rows, 0)).save(out)
print("saved", out)
