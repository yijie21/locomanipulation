"""selfmodel_via_20261120, fourth step, part two: the replay scene with the narrow-gap test's two real obstacles.

    python3 -I gap_scene.py <source scene dir> <new scene dir> [bar depth,thickness in m, default 0.03,0.03]

Copies scene_meta.json and scene_locomani.xml and adds two mocap bodies before </worldbody>: gap_board (the table board,
half sizes 0.30 0.60 0.02, gap_routes.py's BOARD_HALF) and gap_bar (half sizes depth/2 0.60 thickness/2), parked where
gen_reach.py's GapObstacles parks them (50 0 0.5 and 50 3 0.5). Default contact settings: they collide with every robot geom;
being mocap they make no contact with the floor or each other. Nothing else changes.
"""

import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
bar = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.03,0.03").split(",")]
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "scene_meta.json", dst / "scene_meta.json")
xml = (src / "scene_locomani.xml").read_text()
assert xml.count("</worldbody>") == 1 and "gap_board" not in xml
add = (
    '    <body name="gap_board" mocap="true" pos="50 0 0.5">\n'
    '      <geom name="gap_board" type="box" size="0.30 0.60 0.02" rgba="0.55 0.42 0.30 1"/>\n'
    "    </body>\n"
    '    <body name="gap_bar" mocap="true" pos="50 3 0.5">\n'
    f'      <geom name="gap_bar" type="box" size="{bar[0] / 2:g} 0.60 {bar[1] / 2:g}" rgba="0.25 0.25 0.25 1"/>\n'
    "    </body>\n"
)
(dst / "scene_locomani.xml").write_text(xml.replace("</worldbody>", add + "  </worldbody>"))
print(f"{dst / 'scene_locomani.xml'}: gap_board, gap_bar (bar {bar[0]} x {bar[1]} m) added")
