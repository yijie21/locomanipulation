"""Realistic objects for the grasp room (2026-11-09): NVIDIA SimReady assets (huggingface.co/datasets/nvidia/simready-assets,
CC-BY-4.0, 2677 USD assets) made into MuJoCo scenes of the 65-object line's format, so grasp_cache.py and grasp_room.py read
them unchanged. All under outputs/reachwm_20261030/simready/. Downloads are data: each phase is given paths, nothing of theirs
is executed.

The shipped physics is not used: each rigid part is one whole-object convex hull (a mug's cavity is filled) and the mass comes
from that solid (mug 1.33 kg, a can 2.83 kg). Collision is redone with CoACD, the mass is the line's rule (volume x 400 kg/m^3,
clipped to 0.03-0.40 kg). Left out: assets of more than one rigid body (pump, cap, lid, scissors), and what does not fit the
jaw or the room's slots.

Phases, each with its own interpreter (P3 = python3 -I, USD = /venv/main/bin/python -I, CD = simready/venv_coacd/bin/python -I,
SIM = third_party/GR00T-WholeBodyControl/.venv_sim/bin/python):
  pick      P3   candidates by name: per-group rules below -> picks.json
  fetch     P3   download; `fetch geom` USD files only, `fetch tex` also the textures of the kept objects
  scan      USD  size, rigid bodies, joints -> scan.json
  keep      P3   apply the size and rigidity rules, round-robin over each group's types up to its cap -> kept.json
  pick_more, keep_more  P3   the second round, appended (see MORE); then fetch geom, scan, fetch tex and the rest again
  extract   USD  one OBJ per albedo texture, 1024 px PNG, the merged surface; origin at the bounding box centre -> mj/NAME/
  coacd     CD   convex pieces, volume, mass
  scenes    SIM  scenes/obj_XXX in the 65-object format: rest pose found by settling, objects.json
  then      GRASP_CACHE_ROOT=<simready> GRASP_PAIR_FAR=1 grasp_cache.py ID over every id (grasp_cache_summary.json in each scene).
            GRASP_PAIR_FAR: the farthest opposed point, hollow objects' inner wall otherwise ends every antipodal pair

Result 2026-11-09: 173 kept, 171 scenes (2 never at rest), 128 with at least 10 validated grasps, 71 types: produce 34, beverage 27,
processed 22, packaged 11, cleaning 11, dishware 8, consumable 6, kitchenware 4, medicine 4, decor 1. Flat thin things (forks,
spoons, pencils, screwdrivers, stationery 0 of 8, tools 0 of 11) lying on a table give no grasp for the Dex1.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "outputs" / "reachwm_20261030" / "simready"
DL, MJ = ROOT / "dl", ROOT / "mj"
HF = "https://huggingface.co/datasets/nvidia/simready-assets/resolve/main/"
API = "https://huggingface.co/api/datasets/nvidia/simready-assets/tree/main"
UA = {"User-Agent": "research-client"}
MAX_FILE = 40_000_000
JAW, MAX_Z, MAX_XY = 0.09, 0.26, 0.24  # narrow side, height, longest side (the room's slots are 0.28 m apart)
TABLE_TOP, DENSITY, MASS_RANGE = 0.62, 400.0, (0.03, 0.40)
# group, name pattern, names to leave out, how many of each type to look at, how many of the group to keep
GROUPS = (
    (
        "produce",
        r"^food_produce_",
        r"squash|bocChoy|broccoli|corn|dragonFruit|grape|banana_bunch|bitterMelon|cabbage|lettuce|melon|pumpkin|pineapple|celery|eggplant",
        2,
        30,
    ),
    ("beverage", r"^food_beverage_", r"$^", 3, 24),
    ("processed", r"^food_processed_", r"chips", 3, 20),
    ("packaged", r"^food_(canned|can|condiment|packaged)_", r"box_(close|open)|bread|pasta", 1, 14),
    ("kitchenware", r"^kitchenware_", r"pan_|cuttingBoard|tray|kettle|rollingPin", 3, 20),
    ("dishware", r"^dishware_", r"Plate|tray|stacked|teapot", 2, 10),
    ("utensil", r"^utensil_", r"$^", 2, 5),
    ("tool", r"^(handTool|tool)_", r"sledge|socket|pry|Bar|bar_", 1, 15),
    ("stationery", r"^stationery_", r"tacks|box", 1, 8),
    ("cleaning", r"^cleaning_", r"cloth|towel|wipes|bucket|caddy|squeegee|iron|painted", 2, 8),
    (
        "consumable",
        r"^consumable_",
        r"towel|hose|belt|seal|bearing|filter|fuse|lightBulb|ball|pad|roll|paper|shaped|timing|serpentine",
        1,
        8,
    ),
    ("decor", r"^decor_", r"$^", 3, 3),
    ("medicine", r"^medicine_", r"$^", 4, 4),
)


# second round (the first kept 110, of which 70 had grasps: flat thin things cannot be picked off a table by the Dex1): more of
# what works. Assets of several bodies are allowed here: extract merges them, a cap or a pump stays on as one rigid body.
# group, how many of each type to look at now, how many more to keep
MORE = (
    ("produce", 5, 14),
    ("beverage", 8, 18),
    ("processed", 8, 14),
    ("packaged", 3, 6),
    ("cleaning", 3, 4),
    ("consumable", 2, 6),
    ("decor", 6, 3),
    ("medicine", 11, 7),
)
MIN_SIDE = 0.02  # thinner than this along any axis, lying on a table: no grasp


def typ(a: str) -> str:
    return re.sub(r"_[a-z]\d+$", "", a)


def http(url: str, timeout: float = 120.0):
    for k in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 5:
                raise
        time.sleep(5 * 2**k)
    raise RuntimeError(url)


def pick() -> None:
    names, url, pages = [], API + "?limit=1000", 0
    while url and pages < 30:
        r = http(url)
        names += [x["path"] for x in json.load(r) if x.get("type") == "directory"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url, pages = m.group(1) if m else None, pages + 1
    out = []
    for g, pat, drop, per_type, _cap in GROUPS:
        seen: dict[str, int] = {}
        for a in sorted(names):
            if re.search(pat, a) and not re.search(drop, a):
                t = typ(a)
                if seen.get(t, 0) < per_type:
                    seen[t] = seen.get(t, 0) + 1
                    out.append(dict(asset=a, group=g, type=t))
        print(g, len([o for o in out if o["group"] == g]), "candidates of", len(seen), "types")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "picks.json").write_text(json.dumps(out, indent=1))
    print(len(out), "candidates of", len(names), "assets")


def pick_more() -> None:
    picks = json.load(open(ROOT / "picks.json"))
    have = {o["asset"] for o in picks}
    names, url, pages = [], API + "?limit=1000", 0
    while url and pages < 30:
        r = http(url)
        names += [x["path"] for x in json.load(r) if x.get("type") == "directory"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url, pages = m.group(1) if m else None, pages + 1
    groups = {g[0]: g for g in GROUPS}
    new = []
    for g, per_type, _ in MORE:
        _, pat, drop, _, _ = groups[g]
        seen: dict[str, int] = {}
        for a in sorted(names):
            if re.search(pat, a) and not re.search(drop, a):
                t = typ(a)
                if seen.get(t, 0) < per_type:
                    seen[t] = seen.get(t, 0) + 1
                    if a not in have:
                        new.append(dict(asset=a, group=g, type=t))
    (ROOT / "picks.json").write_text(json.dumps(picks + new, indent=1))
    print(len(new), "more candidates;", len(picks) + len(new), "in all")


def keep_more() -> None:
    S, kept = json.load(open(ROOT / "scan.json")), json.load(open(ROOT / "kept.json"))
    have = {o["asset"] for o in kept}
    picks, added = json.load(open(ROOT / "picks.json")), []
    for g, _, cap in MORE:
        by: dict[str, list] = {}
        for o in picks:
            v = S.get(o["asset"], {"error": 1})
            if o["group"] != g or o["asset"] in have or "error" in v or v["up"] != "Z" or v["mpu"] != 1.0:
                continue
            if (
                min(v["size"][:2]) <= JAW
                and max(v["size"][:2]) <= MAX_XY
                and v["size"][2] <= MAX_Z
                and min(v["size"]) >= MIN_SIDE
            ):
                by.setdefault(o["type"], []).append(o)
        chosen: list = []
        while len(chosen) < cap and any(by.values()):
            for t in sorted(by):
                if by[t] and len(chosen) < cap:
                    chosen.append(by[t].pop(0))
        added += chosen
        print(f"{g}: {len(chosen)} more, of {len({c['type'] for c in chosen})} types")
    (ROOT / "kept.json").write_text(
        json.dumps(kept + added, indent=1)
    )  # appended: the first 111 keep their scene numbers
    print(len(added), "added;", len(kept) + len(added), "in all")


def fetch_one(a: str, tex: bool) -> str:
    t = [x for x in json.load(http(f"{API}/{urllib.parse.quote(a)}?recursive=true&limit=1000")) if x["type"] == "file"]
    n = big = 0
    for x in t:
        p = x["path"]
        if ".thumbs" in p and not ("256x256" in p and p.endswith(".usd.png")):
            continue
        if "/Textures/" in p and not tex:
            continue
        if p.endswith((".mdl",)) or "/source_assets/" in p or "/runnables/" in p:
            continue
        if x.get("size", 0) > MAX_FILE:
            big += 1
            continue
        dst = DL / p
        if dst.exists() and dst.stat().st_size == x.get("size", -1):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with http(HF + urllib.parse.quote(p)) as r, open(str(dst) + ".part", "wb") as f:
            f.write(r.read())
        os.replace(str(dst) + ".part", dst)
        n += 1
    return f"{a}: {n} files, {big} over the size limit"


def fetch(what: str) -> None:
    src = "picks.json" if what == "geom" else "kept.json"
    assets = [o["asset"] for o in json.load(open(ROOT / src))]
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fetch_one, a, what == "tex"): a for a in assets}
        for k, f in enumerate(cf.as_completed(futs)):
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001
                msg = f"FAILED {futs[f]}: {e}"
            if k % 25 == 0 or msg.startswith("FAILED"):
                print(k, msg, flush=True)


def usd_of(a: str) -> str | None:
    us = sorted(glob.glob(str(DL / a / "simready_usd" / "*.usd*")))
    return us[0] if us else None


def open_stage(u: str):
    from pxr import Usd

    st = Usd.Stage.Open(u, Usd.Stage.LoadAll)
    r = st.GetDefaultPrim()
    r.GetVariantSets().GetVariantSet("MuJoCo").SetVariantSelection("Enabled")
    return st, r


def scan() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    out = {}
    for o in json.load(open(ROOT / "picks.json")):
        a, u = o["asset"], usd_of(o["asset"])
        if u is None:
            out[a] = dict(error="no usd")
            continue
        try:
            st, r = open_stage(u)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            bb = cache.ComputeWorldBound(r).ComputeAlignedRange()
            mpu = UsdGeom.GetStageMetersPerUnit(st)
            size = [(bb.GetMax()[i] - bb.GetMin()[i]) * mpu for i in range(3)]
            rigid = joints = meshes = 0
            for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                t = p.GetTypeName()
                rigid += p.HasAPI(UsdPhysics.RigidBodyAPI)
                joints += "Joint" in t
                meshes += t == "Mesh"
            out[a] = dict(size=size, rigid=rigid, joints=joints, meshes=meshes, up=UsdGeom.GetStageUpAxis(st), mpu=mpu)
        except Exception as e:  # noqa: BLE001
            out[a] = dict(error=str(e)[:200])
    (ROOT / "scan.json").write_text(json.dumps(out, indent=1))
    bad = sum("error" in v for v in out.values())
    print(len(out), "scanned,", bad, "errors")


def keep() -> None:
    S, kept, why = json.load(open(ROOT / "scan.json")), [], {}
    picks = json.load(open(ROOT / "picks.json"))
    for g, *_rest, cap in GROUPS:
        by: dict[str, list] = {}
        for o in picks:
            if o["group"] != g:
                continue
            v, a = S.get(o["asset"], {"error": "unscanned"}), o["asset"]
            if "error" in v:
                why[a] = "error"
            elif v["rigid"] != 1 or v["joints"]:
                why[a] = "several bodies"
            elif v["up"] != "Z" or v["mpu"] != 1.0:
                why[a] = "frame"
            elif min(v["size"][:2]) > JAW or max(v["size"][:2]) > MAX_XY or v["size"][2] > MAX_Z:
                why[a] = "size"
            else:
                by.setdefault(o["type"], []).append(o)
        chosen: list = []
        while len(chosen) < cap and any(by.values()):
            for t in sorted(by):
                if by[t] and len(chosen) < cap:
                    chosen.append(by[t].pop(0))
        kept += chosen
        print(f"{g}: kept {len(chosen)} of cap {cap}, types {len({c['type'] for c in chosen})}")
    (ROOT / "kept.json").write_text(json.dumps(kept, indent=1))
    c: dict[str, int] = {}
    for w in why.values():
        c[w] = c.get(w, 0) + 1
    print(len(kept), "kept; left out", c)


def extract() -> None:
    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdShade

    Image.MAX_IMAGE_PIXELS = None
    done = 0
    for o in json.load(open(ROOT / "kept.json")):
        a = o["asset"]
        d = MJ / a
        if (d / "meta.json").exists():
            continue
        try:
            st, r = open_stage(usd_of(a))
            chunks = []
            for p in Usd.PrimRange(r, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                if p.GetTypeName() != "Mesh":
                    continue
                m = UsdGeom.Mesh(p)
                M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0.0))
                pts = np.array(m.GetPointsAttr().Get(), float)
                w = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
                c = np.array(m.GetFaceVertexCountsAttr().Get())
                idx = np.array(m.GetFaceVertexIndicesAttr().Get())
                off = np.r_[0, np.cumsum(c)[:-1]]
                pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
                uv = np.array(pv.ComputeFlattened(), float) if pv and pv.HasValue() else np.zeros((len(idx), 2))
                if len(uv) != len(idx):
                    uv = np.zeros((len(idx), 2))
                subs = UsdGeom.Subset.GetAllGeomSubsets(m)
                groups = [
                    (
                        np.array(s.GetIndicesAttr().Get()),
                        UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0],
                    )
                    for s in subs
                ]
                groups = groups or [(np.arange(len(c)), UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0])]
                for fs, mat in groups:
                    tri = [(off[f], off[f] + j, off[f] + j + 1) for f in fs for j in range(1, c[f] - 1)]
                    chunks.append(dict(pts=w, idx=idx, uv=uv, tri=np.array(tri, int), mat=mat))
            allp = np.concatenate([ch["pts"][ch["idx"][ch["tri"].ravel()]] for ch in chunks])
            lo, hi = allp.min(0), allp.max(0)
            centre = (lo + hi) / 2
            d.mkdir(parents=True, exist_ok=True)
            mats: dict = {}
            for ch in chunks:
                file, color = None, None
                if ch["mat"]:
                    for sh in ch["mat"].GetPrim().GetChildren():
                        if sh.GetTypeName() != "Shader":
                            continue
                        if sh.GetName() == "Albedo":
                            v = UsdShade.Shader(sh).GetInput("file").Get()
                            file = v.resolvedPath or v.path
                        if sh.GetName() == "Principled_BSDF":
                            i = UsdShade.Shader(sh).GetInput("diffuseColor")
                            if i and i.Get() is not None:
                                color = [float(x) for x in i.Get()]
                mats.setdefault(file or str(color), dict(file=file, color=color, c=[]))["c"].append(ch)
            vis, AV, AF, nv = [], [], [], 0
            for k, mm in enumerate(mats.values()):
                V = np.concatenate([(ch["pts"][ch["idx"][ch["tri"]]] - centre).reshape(-1, 3) for ch in mm["c"]])
                T = np.concatenate([ch["uv"][ch["tri"]].reshape(-1, 2) for ch in mm["c"]])
                u_, inv = np.unique(np.c_[np.round(V, 6), np.round(T, 5)], axis=0, return_inverse=True)
                F = inv.ravel().reshape(-1, 3)
                with open(d / f"vis_{k}.obj", "w") as f:
                    f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in u_[:, :3]))
                    f.write("".join(f"vt {s:.5f} {t:.5f}\n" for s, t in u_[:, 3:]))
                    f.write("".join(f"f {i + 1}/{i + 1} {j + 1}/{j + 1} {l + 1}/{l + 1}\n" for i, j, l in F))
                tex = None
                if mm["file"] and os.path.exists(mm["file"]):
                    im = Image.open(mm["file"]).convert("RGB")
                    im.thumbnail((1024, 1024))
                    tex = f"tex_{k}.png"
                    im.save(d / tex)
                vis.append(dict(obj=f"vis_{k}.obj", tex=tex, color=mm["color"], faces=len(F)))
                AV.append(u_[:, :3])
                AF.append(F + nv)
                nv += len(u_)
            V, F = np.concatenate(AV), np.concatenate(AF)
            r_, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
            F = inv.ravel()[F]
            with open(d / "all.obj", "w") as f:
                f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in r_))
                f.write("".join(f"f {i + 1} {j + 1} {l + 1}\n" for i, j, l in F))
            (d / "meta.json").write_text(
                json.dumps(dict(asset=a, group=o["group"], type=o["type"], size_m=(hi - lo).tolist(), vis=vis))
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            print("FAILED extract", a, str(e)[:200], flush=True)
    print("extracted", done, flush=True)


def coacd_all() -> None:
    import coacd
    import numpy as np
    import trimesh

    coacd.set_log_level("error")
    assets = [o["asset"] for o in json.load(open(ROOT / "kept.json"))]
    ids = assets[int(sys.argv[2]) :: int(sys.argv[3])] if len(sys.argv) > 3 else assets
    for a in ids:
        d = MJ / a
        if not (d / "meta.json").exists():
            continue
        meta = json.load(open(d / "meta.json"))
        if "mass" in meta:
            continue
        try:
            m = trimesh.load(d / "all.obj", process=False, force="mesh")
            parts = coacd.run_coacd(
                coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)),
                threshold=0.05,
                max_convex_hull=30,
                preprocess_mode="auto",
                seed=0,
            )
            vol, names = 0.0, []
            for i, (v, f) in enumerate(parts):
                h = trimesh.Trimesh(v, f, process=True).convex_hull
                vol += float(h.volume)
                h.export(d / f"col_{i}.obj")
                names.append(f"col_{i}.obj")
            mass = float(np.clip(vol * DENSITY, *MASS_RANGE))
            meta.update(pieces=names, volume=vol, mass=mass, rho=mass / vol)
            (d / "meta.json").write_text(json.dumps(meta))
            print(a, len(names), "pieces", round(mass, 3), "kg", flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAILED coacd", a, str(e)[:200], flush=True)


def scenes() -> None:
    import xml.etree.ElementTree as ET

    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    base = HERE.parents[2] / "outputs" / "scene" / "sim_lowcom"
    BODY, KEY = "water_bottle_object", "water_bottle"
    base_meta = json.loads((base / "scene_meta.json").read_text())

    def verts(p: Path) -> np.ndarray:
        return np.array([list(map(float, ln.split()[1:4])) for ln in open(p) if ln.startswith("v ")])

    def narrow(v: np.ndarray) -> float:
        a = np.linspace(0, np.pi, 72, endpoint=False)
        pr = v[:, :2] @ np.c_[np.cos(a), np.sin(a)].T
        return float((pr.max(0) - pr.min(0)).min())

    def fmt(x) -> str:
        return " ".join(f"{float(v):.8g}" for v in x)

    def write(name: str, d: Path, meta: dict, R: np.ndarray, out: Path) -> tuple[float, float]:
        v = verts(d / "all.obj") @ R.T
        lo, hi = v.min(0), v.max(0)
        centre, half = (hi + lo) / 2, (hi - lo) / 2
        q, pos = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]], -centre
        tree = ET.parse(base / "scene_locomani.xml")
        root = tree.getroot()
        body = root.find(f".//body[@name='{BODY}']")
        geoms = body.findall("geom")
        contact = {k: geoms[2].get(k) for k in ("friction", "solref", "solimp", "condim") if geoms[2].get(k)}
        for g in geoms:
            body.remove(g)
        for tag in ("inertial",):
            for e in body.findall(tag):
                body.remove(e)
        asset = root.find("asset")
        for k, vs in enumerate(meta["vis"]):
            ET.SubElement(asset, "mesh", name=f"{name}_v{k}", file=str(d / vs["obj"]))
            attrs = dict(
                type="mesh",
                mesh=f"{name}_v{k}",
                contype="0",
                conaffinity="0",
                group="1",
                density="0",
                pos=fmt(pos),
                quat=fmt(q),
            )
            if vs["tex"]:
                ET.SubElement(asset, "texture", name=f"{name}_t{k}", type="2d", file=str(d / vs["tex"]))
                ET.SubElement(
                    asset, "material", name=f"{name}_m{k}", texture=f"{name}_t{k}", specular="0.3", shininess="0.3"
                )
                attrs["material"] = f"{name}_m{k}"
            else:
                c = vs["color"] or (0.7, 0.7, 0.7)
                attrs["rgba"] = f"{c[0]} {c[1]} {c[2]} 1"
            ET.SubElement(body, "geom", attrs)
        for i, p in enumerate(meta["pieces"]):
            ET.SubElement(asset, "mesh", name=f"{name}_c{i}", file=str(d / p))
            ET.SubElement(
                body,
                "geom",
                dict(
                    type="mesh",
                    mesh=f"{name}_c{i}",
                    rgba="0.8 0.8 0.8 0",
                    density=f"{meta['rho']:.4f}",
                    pos=fmt(pos),
                    quat=fmt(q),
                    **contact,
                ),
            )
        x, y = base_meta["objects"][KEY]["init_pos"][:2]
        z = TABLE_TOP + half[2] + 0.002
        body.set("pos", f"{x} {y} {z:.6f}")
        out.mkdir(parents=True, exist_ok=True)
        tree.write(out / "scene_locomani.xml", encoding="unicode")
        return float(2 * half[2]), narrow(v)

    def settle(xml: Path) -> tuple[float, float, np.ndarray]:
        m = mujoco.MjModel.from_xml_path(str(xml))
        d = mujoco.MjData(m)
        adr = m.jnt_qposadr[m.body_jntadr[m.body(BODY).id]]
        mujoco.mj_forward(m, d)
        p0 = d.qpos[adr : adr + 3].copy()
        # the robot is free in this scene and would fall on the table: pinned for the 2 s by resetting everything else
        keep_q, keep_v = d.qpos.copy(), d.qvel.copy()
        idx = np.ones(m.nq, bool)
        idx[adr : adr + 7] = False
        vdx = np.ones(m.nv, bool)
        vadr = m.jnt_dofadr[m.body_jntadr[m.body(BODY).id]]
        vdx[vadr : vadr + 6] = False
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            d.qpos[idx], d.qvel[vdx] = keep_q[idx], keep_v[vdx]
        q = d.qpos[adr + 3 : adr + 7]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return float(np.linalg.norm(d.qpos[adr : adr + 3] - p0)), float(np.degrees(R.magnitude())), R.as_matrix()

    rows, skipped = [], []
    for oid, o in enumerate(json.load(open(ROOT / "kept.json"))):
        a, d = o["asset"], MJ / o["asset"]
        if not (d / "meta.json").exists():
            skipped.append((a, "no mesh"))
            continue
        meta = json.load(open(d / "meta.json"))
        if "mass" not in meta:
            skipped.append((a, "no pieces"))
            continue
        name, dest, R, row = f"sr{oid:03d}", ROOT / "scenes" / f"obj_{oid:03d}", np.eye(3), None
        for attempt in range(4):
            h, w = write(name, d, meta, R, dest)
            if w > JAW or h > MAX_Z:
                skipped.append((a, f"rest pose too wide or tall: w {w:.3f} h {h:.3f}"))
                break
            try:
                moved, turned, Rr = settle(dest / "scene_locomani.xml")
            except Exception as e:  # noqa: BLE001
                skipped.append((a, "compile: " + str(e)[:120]))
                break
            if moved <= 0.01 and turned <= 3.0:
                row = dict(id=oid, name=name, asset=a, group=o["group"], type=o["type"], height=h, width=w, mass=meta["mass"], pieces=len(meta["pieces"]),
                           scene=str(dest), settle_moved_m=moved, settle_deg=turned, settle_rebuilds=attempt)  # fmt: skip
                m_ = json.loads(json.dumps(base_meta))
                m_["objects"][KEY].update(height=h, scale=1.0, mesh=str(d / "all.obj"))
                (dest / "scene_meta.json").write_text(json.dumps(m_, indent=1))
                break
            yaw = np.arctan2(Rr[1, 0], Rr[0, 0])
            R = Rotation.from_euler("z", -yaw).as_matrix() @ Rr @ R
        else:
            skipped.append((a, "never at rest"))
        if row:
            rows.append(row)
            print(
                f"{name} {a[:36]} h {row['height']:.3f} w {row['width']:.3f} m {row['mass']:.3f} rebuilt {attempt}",
                flush=True,
            )
    (ROOT / "objects.json").write_text(json.dumps(dict(objects=rows, skipped=skipped), indent=1))
    print(len(rows), "scenes;", len(skipped), "skipped", flush=True)


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "pick":
        pick()
    elif ph == "fetch":
        fetch(sys.argv[2])
    elif ph == "scan":
        scan()
    elif ph == "keep":
        keep()
    elif ph == "pick_more":
        pick_more()
    elif ph == "keep_more":
        keep_more()
    elif ph == "extract":
        extract()
    elif ph == "coacd":
        coacd_all()
    elif ph == "scenes":
        scenes()
    else:
        raise SystemExit(__doc__)
