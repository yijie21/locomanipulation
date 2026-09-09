"""Emit a render-only MJCF in which no body carries more than one visual mesh.

The Isaac MJCF importer composes ALL of a body's visual geoms onto one prim as
several references. Every prototype names its geometry `mesh`, so the composed
children collide by name and only the strongest reference survives -- on the G1
the wrist keeps `Dex1_base_link` and silently loses `right_wrist_yaw_link`,
which renders as a gap between the forearm and the gripper.

Fixing that on the USD side does not work. Moving references onto children
displaces the robot (silhouette IoU 0.49-0.62 against the unsplit build), and a
namespace rename of the prototypes' children does not apply. So the merge is
avoided upstream instead: each extra visual geom is moved into its own massless,
jointless child body, which the importer then gives its own prim.

STATUS: written and verified, but NOT WIRED IN. Isaac 4.5's MJCF importer
aborts (SIGABRT, no Python traceback) on the jointless child bodies this
introduces, with and without an explicit inertial, so the build still imports
the original MJCF and the wrist mesh is still lost. The variant itself is
correct -- 41 bodies against the original's 35, the same 34 joints and 77
geoms, body positions identical to 0.0000 mm at qpos0 -- so this is worth
retrying on a newer importer.

This NEVER touches the simulated model. The physics MJCF is shared with
collection and evaluation; this writes a separate file used only for the visual
import. Massless and jointless means the child adds no DOF and no dynamics, so
the two models describe the same kinematics.
"""

from __future__ import annotations

import os
import re

VISUAL = 'group="1"'


def _attr(tag: str, name: str) -> str:
    m = re.search(name + r'="([^"]*)"', tag)
    return m.group(1) if m else ""


def split_multi_mesh_bodies(src: str, dst: str, log=print) -> str:
    """Write `dst` from `src`; return dst. Idempotent for a given src."""
    xml = open(src).read()
    out, moved = [], 0
    pos = 0
    # Walk bodies by their opening tags; a body's own geoms are the ones before
    # its first nested <body.
    for m in re.finditer(r"<body\b[^>]*>", xml):
        pass
    # Simpler and safe: operate per body block delimited by consecutive <body>
    starts = [m.start() for m in re.finditer(r"<body\b", xml)] + [len(xml)]
    out.append(xml[: starts[0]])
    for i in range(len(starts) - 1):
        blk = xml[starts[i] : starts[i + 1]]
        head_end = blk.find(">") + 1
        head, body = blk[:head_end], blk[head_end:]
        geoms = [g for g in re.finditer(r"<geom\b[^>]*/?>", body)]
        vis = [g for g in geoms if VISUAL in g.group(0) and 'mesh="' in g.group(0)]
        if len(vis) < 2:
            out.append(blk)
            continue
        # keep the first, re-home the rest
        parent = _attr(head, "name") or f"body{i}"
        drop = vis[1:]
        newbody = body
        extra = []
        for g in drop:
            tag = g.group(0)
            mesh = _attr(tag, "mesh")
            gpos = _attr(tag, "pos") or "0 0 0"
            gquat = _attr(tag, "quat")
            inner = re.sub(r'\spos="[^"]*"', "", tag)
            inner = re.sub(r'\squat="[^"]*"', "", inner)
            q = f' quat="{gquat}"' if gquat else ""
            extra.append(f'\n<body name="{parent}_{mesh}_vis" pos="{gpos}"{q}>{inner}</body>')
            newbody = newbody.replace(tag, "", 1)
            moved += 1
        # insert the new child bodies just before this block's own closing tail
        cut = len(newbody)
        for anchor in ("<body", "</body>"):
            k = newbody.find(anchor)
            if k != -1:
                cut = min(cut, k)
        out.append(head + newbody[:cut] + "".join(extra) + newbody[cut:])
    text = "".join(out)
    # Mesh paths in an MJCF resolve against the FILE's directory, so a variant
    # written anywhere else loses its assets. Absolutise meshdir (adding a
    # compiler tag if there is none) rather than copying a few hundred MB.
    src_dir = os.path.dirname(os.path.abspath(src))
    mc = re.search(r"<compiler\b[^>]*>", text)
    if mc:
        tag = mc.group(0)
        md = _attr(tag, "meshdir") or "."
        newtag = (
            re.sub(r'meshdir="[^"]*"', f'meshdir="{os.path.join(src_dir, md)}"', tag)
            if 'meshdir="' in tag
            else tag[:-1] + f' meshdir="{src_dir}"' + tag[-1:]
        )
        text = text.replace(tag, newtag, 1)
    else:
        text = text.replace("<mujoco", "<mujoco", 1)
        k = text.find(">") + 1
        text = text[:k] + f'\n<compiler meshdir="{src_dir}"/>' + text[k:]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    log(f"render MJCF: moved {moved} extra visual geoms into their own bodies -> {dst}")
    return dst


if __name__ == "__main__":
    import sys

    split_multi_mesh_bodies(sys.argv[1], sys.argv[2])
