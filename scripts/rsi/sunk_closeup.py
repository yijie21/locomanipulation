"""TODO 27.14: where a placed object rests against the board it was put on, from a round's trace: one close camera at
board height, the board as rendered and see-through (surface_height_audit.py says why it sinks).

    MUJOCO_GL=egl PYTHONPATH=src $PY scripts/rsi/sunk_closeup.py <round base> <t> <host body or geom prefix> <out.jpg>
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import report_media as RM  # noqa: E402
import visualize as V  # noqa: E402

base, t_query, host, out = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
r = RM.Rd(base)
sc = V.Scene()
i = int(np.searchsorted(r.z["t"], t_query))
sc.set(r, i)
obj = r.z[r.key][i, :3]
pel = r.z["robot"][i, :2]
az = math.degrees(math.atan2(obj[1] - pel[1], obj[0] - pel[0])) + 90
cam = dict(lookat=obj, distance=0.55, azimuth=az, elevation=-8)
font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 22)
a = sc.render(cam, (640, 480))
gs = [g for g in range(sc.m.ngeom) if sc.m.geom(g).name.startswith(host)]
saved = sc.m.geom_rgba[gs].copy()
sc.m.geom_rgba[gs, 3] = 0.35
b = sc.render(cam, (640, 480))
sc.m.geom_rgba[gs] = saved
sheet = Image.new("RGB", (1280, 520), (20, 20, 24))
for k, (im, txt) in enumerate(((a, "视频里看到的"), (b, "柜子半透明：物体其实在板子里面"))):
    sheet.paste(Image.fromarray(im), (640 * k, 40))
    ImageDraw.Draw(sheet).text((640 * k + 10, 6), txt, fill=(240, 240, 240), font=font)
sheet.save(out, quality=90)
print(r.label, "t", float(r.z["t"][i]), "object z", round(float(obj[2]), 4), "geoms made see-through", len(gs))
