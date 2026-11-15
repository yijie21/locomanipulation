"""The textured looks of SIMPLE's objects for MuJoCo's camera (2026-11-12).

SIMPLE renders its objects from USD files (data/assets/<res>/ruled_models/<uid>_ruled.usd: a visual mesh with texture
coordinates and one diffuse texture). This writes each as an OBJ with texture coordinates and a PNG, which episodes.py adds
to the scene as a look with no contacts, so the head camera's colour image shows what a segmenter can read. Needs pxr.

    /venv/main/bin/python visuals.py <SIMPLE>/data/assets <out>/visual      # all objects of graspnet and objects
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade


def export(usd: Path, out: Path, name: str) -> dict:
    st = Usd.Stage.Open(str(usd))
    mesh = next(p for p in st.Traverse() if p.GetTypeName() == "Mesh" and p.GetName() == "visual")
    g = UsdGeom.Mesh(mesh)
    if g.GetPointsAttr().Get() is None:  # the basket: its look is its collision mesh, with no material
        return dict(look=False)
    M = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T  # column vectors
    pts = np.array(g.GetPointsAttr().Get()) @ M[:3, :3].T + M[:3, 3]
    counts, idx = np.array(g.GetFaceVertexCountsAttr().Get()), np.array(g.GetFaceVertexIndicesAttr().Get())
    assert (counts == 3).all(), "triangles only"
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uv, uvi = np.array(pv.Get()), pv.GetIndices()
    uvi = np.array(uvi) if uvi else np.arange(len(uv))
    assert pv.GetInterpolation() == "faceVarying" and len(uvi) == len(idx)
    tex, flat = None, (0.6, 0.6, 0.6)
    for p in st.Traverse():
        if p.GetTypeName() == "Shader":
            for i in UsdShade.Shader(p).GetInputs():
                v = i.Get()
                if i.GetBaseName() == "diffuse_texture" and v is not None:
                    tex = (usd.parent / v.path).resolve()
                if i.GetBaseName() == "diffuse_color_constant" and v is not None:
                    flat = tuple(v)
    if tex and tex.exists():
        Image.open(tex).convert("RGB").save(out / f"{name}.png")
    else:  # an object of one colour
        Image.new("RGB", (8, 8), tuple(int(255 * c) for c in flat)).save(out / f"{name}.png")
    with open(out / f"{name}.obj", "w") as f:
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pts)
        f.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        for k in range(0, len(idx), 3):
            f.write("f " + " ".join(f"{idx[k + j] + 1}/{uvi[k + j] + 1}" for j in range(3)) + "\n")
    return dict(
        textured=bool(tex and tex.exists()),
        lo=pts.min(0).round(4).tolist(),
        hi=pts.max(0).round(4).tolist(),
        up=UsdGeom.GetStageUpAxis(st),
    )


if __name__ == "__main__":
    assets, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for res in ("graspnet", "objects"):
        for usd in sorted((assets / res / "ruled_models").glob("*_ruled.usd")):
            print(res, usd.name, export(usd, out, f"{res}_{usd.name[:3]}"), flush=True)
