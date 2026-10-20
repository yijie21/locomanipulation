"""TODO 27.14: close-up of one trace tick, fingers as the report videos draw them (the commanded jaw) and as physics
would hold them (closed until the finger pads touch the object, inside the joint range).

The trace's `jaw` is the gripper command: -0.06 while gripping (the grip force is setpoint depth), past the finger
joint's -0.02 limit, so `visualize.Scene.set` (and the Unreal replay, which pushes its qpos) draws the two fingers
crossed and outside the object.

    MUJOCO_GL=egl PYTHONPATH=src $PY scripts/rsi/grip_closeup.py <round base> <t> <out.jpg>
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import report_media as RM  # noqa: E402
import visualize as V  # noqa: E402

FINGER_RANGE = (-0.02, 0.0245)


def physical_jaw(sc: V.Scene, side: str, body: str, command: float) -> float:
    """The finger joint value at which the pads stop on `body`: the command when opening or clear of it, otherwise the
    first value from the command upward at which no pad penetrates the object."""
    m, d = sc.m, sc.d
    fingers = {m.body(f"{side}_dex1_finger_{k}_link").id for k in (1, 2)}
    pads = [g for g in range(m.ngeom) if m.geom_bodyid[g] in fingers and m.geom_contype[g]]
    ob = m.body(body).id
    obj = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ob and (m.geom_contype[g] or m.geom_conaffinity[g])]
    q = max(FINGER_RANGE[0], command)
    if command >= 0.0:
        return q
    for q in np.linspace(q, FINGER_RANGE[1], 120):
        for a in sc.grip[side]:
            d.qpos[a] = q
        mujoco.mj_forward(m, d)
        if min(mujoco.mj_geomDistance(m, d, p, o, 0.05, None) for p in pads for o in obj) >= -0.0005:
            break
    return float(q)


def main() -> int:
    base, t_query, out = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    r = RM.Rd(base)
    sc = V.Scene()
    i = int(np.searchsorted(r.z["t"], t_query))
    side = r.side()
    sc.set(r, i)
    jaw, _, _ = sc.hand(side)
    obj = r.z[r.key][i, :3]
    az = math.degrees(math.atan2(obj[1] - jaw[1], obj[0] - jaw[0])) + 70
    cam = dict(lookat=0.5 * (jaw + obj), distance=0.45, azimuth=az, elevation=-25)
    command = float(r.z["jaw"][i, ("left", "right").index(side)])
    drawn = sc.render(cam, (640, 480))
    q = physical_jaw(sc, side, r.key[4:], command)
    for a in sc.grip[side]:
        sc.d.qpos[a] = q
    mujoco.mj_kinematics(sc.m, sc.d)
    held = sc.render(cam, (640, 480))
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 22)
    sheet = Image.new("RGB", (1280, 520), (20, 20, 24))
    for k, (im, txt) in enumerate(
        ((drawn, f"视频里画的：手指 = 指令 {command:+.3f}"), (held, f"物理上：手指停在物体表面 {q:+.3f}"))
    ):
        sheet.paste(Image.fromarray(im), (640 * k, 40))
        ImageDraw.Draw(sheet).text((640 * k + 10, 6), txt, fill=(240, 240, 240), font=font)
    sheet.save(out, quality=90)
    print(r.label, "t", float(r.z["t"][i]), "side", side, "jaw command", command, "-> physical", round(q, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
