"""Give every visual mesh of a multi-mesh body its own prim in the robot USD.

Why this exists. Isaac 4.5's MJCF importer writes one prototype per mesh
asset under ``/meshes/<asset>`` -- an Xform carrying the geom's pos/quat and a
child ``Mesh "mesh"`` -- and then, for a body with several visual geoms,
composes ALL of those prototypes onto the single ``/visuals/<body>`` prim as a
reference list::

    def Xform "right_wrist_yaw_link" (
        prepend references = [
            </meshes/right_wrist_yaw_link>,
            </meshes/Dex1_base_link>
        ]
    )

USD composition merges same-named children across references, and each
prototype names its geometry ``mesh``, so the composed prim ends up with ONE
``mesh`` whose every attribute comes from whichever reference is strongest.
The other prototypes' geometry is not hidden or misplaced: it is gone from the
composed stage. On this robot that is the wrist cylinder of each
``*_wrist_yaw_link`` (the dex1 housing wins), the contour shell of ``pelvis``
and three of the four meshes on ``torso_link``. The mesh files are fine; the
MJCF is fine; only the composition loses them.

The fix is to stop asking one prim to hold two prototypes: the reference list
becomes one child Xform per reference, each with a single reference. The
prototypes, their transforms and every other prim are untouched, and the
body's own ``material:binding`` stays on the parent so the renderer's colour
pass (stage.py) sees the same prim it always did.

This is a text rewrite of the exported ``.usda``, on purpose. It runs under
any python, needs no Kit and no ``pxr``, and does the edit on the LAYER rather
than through a live stage -- which is where the earlier attempts went wrong:
``ClearReferences`` on a live stage while the articulation root was still
being taken from the wrong prim displaced the whole robot, and the
``Sdf.BatchNamespaceEdit`` rename never applied. See docs/rendering.md.

Children are named ``<asset>_visual`` rather than ``<asset>`` because
stage.py clears the direct binding on every prim named after a multi-mesh
BODY, and a child named ``right_wrist_yaw_link`` would be caught by that.

Usage:
    python -m locomani.sim.isaac.split_visual_refs IN.usda OUT.usda [--check]

Exit status 0 with a summary line; the summary lists each split body and its
meshes so a build log shows what changed.
"""

from __future__ import annotations

import re
import sys

_DEF = re.compile(r'^(?P<ind>\s*)def Xform "(?P<name>[^"]+)" \($')
_REF = re.compile(r"^\s*</(?P<path>[^>]+)>,?\s*$")


def _split_block(lines: list[str]) -> tuple[list[str], list[str]] | None:
    """One `def Xform "<body>" ( ... ) { ... }` block -> rewritten block.

    Returns (new_lines, mesh_names), or None when the block has no reference
    list and should be left exactly as it is.
    """
    ind = _DEF.match(lines[0]).group("ind")
    try:
        i0 = next(i for i, l in enumerate(lines) if l.strip() == "prepend references = [")
    except StopIteration:
        return None
    i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].strip() == "]")
    refs = []
    for l in lines[i0 + 1 : i1]:
        m = _REF.match(l)
        if not m:
            raise ValueError(f"unexpected reference line: {l!r}")
        refs.append(m.group("path"))
    if len(refs) < 2:
        return None
    # metadata without the reference list
    head = lines[:i0] + lines[i1 + 1 :]
    # body: everything between the `{` line and the closing `}` line
    ib = next(i for i, l in enumerate(head) if l.rstrip() == ind + "{")
    close = head[-1]
    assert close.rstrip() == ind + "}", close
    body = head[ib + 1 : -1]
    new = head[: ib + 1] + body
    if body and body[-1].strip():
        new.append("")
    cind = ind + "    "
    for path in refs:
        asset = path.rsplit("/", 1)[-1]
        new += [
            f'{cind}def Xform "{asset}_visual" (',
            f"{cind}    prepend references = </{path}>",
            f"{cind})",
            f"{cind}{{",
            f"{cind}}}",
            "",
        ]
    if new[-1] == "":
        new.pop()
    new.append(close)
    return new, [p.rsplit("/", 1)[-1] for p in refs]


def split_visual_refs(text: str) -> tuple[str, dict[str, list[str]]]:
    lines = text.split("\n")
    # Only the /visuals scope. /collisions carries the same reference lists,
    # but it is invisible and PhysX merges its meshes itself
    # (PhysxMeshMergeCollisionAPI); leave it as the importer wrote it.
    try:
        s0 = next(i for i, l in enumerate(lines) if l == 'def Scope "visuals"')
    except StopIteration:
        raise SystemExit("no top-level /visuals scope in this layer")
    s1 = next(
        (i for i in range(s0 + 1, len(lines)) if lines[i].startswith("def ") or lines[i].startswith("over ")),
        len(lines),
    )
    out, done = lines[:s0], {}
    i = s0
    while i < s1:
        m = _DEF.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ind = m.group("ind")
        j = next(k for k in range(i + 1, s1) if lines[k].rstrip() == ind + "}")
        block = lines[i : j + 1]
        r = _split_block(block)
        if r is None:
            out += block
        else:
            out += r[0]
            done[m.group("name")] = r[1]
        i = j + 1
    out += lines[s1:]
    return "\n".join(out), done


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    if len(argv) != 2:
        print(__doc__.split("Usage:")[1].split("\n")[1].strip())
        return 2
    src, dst = argv
    text = open(src).read()
    new, done = split_visual_refs(text)
    for body, meshes in done.items():
        print(f"split {body}: {' + '.join(meshes)}")
    print(f"split {len(done)} multi-mesh bodies, {sum(len(v) for v in done.values())} meshes")
    if not check:
        open(dst, "w").write(new)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
