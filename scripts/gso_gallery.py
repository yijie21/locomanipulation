#!/usr/bin/env python3
"""A local HTML gallery of every Google Scanned Object in the catalog (user 2026-11-05): each object rendered alone by
MuJoCo offscreen from one fixed viewpoint (its visual mesh and scan texture on a light floor, the camera framing its
bounding box), beside what catalog.json says about it: key, name, dataset category, single / bimanual / none and
why, mass, size, split. The page filters by kind, category and split, searches names and sorts by mass or size;
a click shows the object large with every field.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $VENV_SIM_PY scripts/gso_gallery.py [--out outputs/gso_gallery]
        [--catalog assets/mujoco_scanned_objects/catalog.json] [--models assets/mujoco_scanned_objects/models]
        [--size 384] [--workers 8] [--force]

Writes `<out>/index.html` (open it in a browser; the pictures are `<out>/img/<key>.jpg`, beside it). A picture
already rendered is kept unless --force; the page is always rewritten. About 2 minutes for the 1030 objects.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/mujoco_scanned_objects"
FOVY = 40.0
AZIMUTH, ELEVATION = 135.0, -28.0  # three-quarter view from above, the same for every object
FRAME = 1.08  # the bounding sphere's margin in the frame


def camera_for(bbox_min, bbox_max, fovy: float = FOVY) -> dict:
    """A free camera (lookat, distance, azimuth, elevation) that frames the bounding box's sphere at `fovy`."""
    lo, hi = [float(v) for v in bbox_min], [float(v) for v in bbox_max]
    centre = [(a + b) / 2 for a, b in zip(lo, hi)]
    radius = 0.5 * math.dist(lo, hi)
    distance = FRAME * radius / math.sin(math.radians(fovy) / 2)
    return {"lookat": centre, "distance": distance, "azimuth": AZIMUTH, "elevation": ELEVATION}


def object_xml(model_dir: Path, rec: dict) -> str:
    """The object's visual mesh and texture alone (no hulls: nothing collides, nothing to compile slowly), on a floor
    disc at its bottom wide enough that its edge stays out of the frame, under a flat light-grey sky; `statistic` keeps
    the shadow map on the object (the wide floor otherwise spreads it and the floor speckles)."""
    d = Path(model_dir).resolve()
    ext = max(float(e) for e in rec["extent_m"])
    z0 = float(rec["bbox_min_m"][2])
    return f"""<mujoco>
  <visual><global offwidth="1024" offheight="1024" fovy="{FOVY}"/><quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.93 0.93 0.93" rgb2="0.93 0.93 0.93" width="8" height="8"/>
    <texture name="tex" type="2d" file="{d / "texture.png"}"/>
    <material name="mat" texture="tex" specular="0.3" shininess="0.4"/>
    <material name="floor" rgba="0.95 0.95 0.95 1" specular="0" reflectance="0"/>
    <mesh name="vis" file="{d / "model.obj"}"/>
  </asset>
  <statistic extent="{2 * ext}" center="0 0 {z0 + ext / 2}"/>
  <worldbody>
    <light pos="{ext} {-ext} {z0 + 3 * ext}" dir="-1 1 -3" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <geom type="cylinder" size="{40 * ext} 0.0005" pos="0 0 {z0 - 0.0005}" material="floor"/>
    <geom type="mesh" mesh="vis" material="mat"/>
  </worldbody>
</mujoco>"""


def render_one(job: tuple) -> tuple[str, str | None]:
    """(key, error or None): render one object to its jpg."""
    key, model_dir, rec, out_jpg, size = job
    try:
        import mujoco
        from PIL import Image

        model = mujoco.MjModel.from_xml_string(object_xml(Path(model_dir), rec))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        c = camera_for(rec["bbox_min_m"], rec["bbox_max_m"])
        cam.lookat[:] = c["lookat"]
        cam.distance, cam.azimuth, cam.elevation = c["distance"], c["azimuth"], c["elevation"]
        with mujoco.Renderer(model, size, size) as r:
            r.update_scene(data, camera=cam)
            Image.fromarray(r.render()).save(out_jpg, quality=88)
        return key, None
    except Exception as e:  # one broken scan must not stop the other 1029
        return key, f"{type(e).__name__}: {e}"


def entries(catalog: dict) -> list[dict]:
    """The page's rows: one per object, the catalog fields the page shows."""
    excluded = set(catalog.get("thresholds", {}).get("excluded_categories") or [])
    rows = []
    for key, o in catalog["objects"].items():
        rows.append({
            "key": key, "name": o["name"], "phrase": o["phrase"], "category": o.get("category") or "(none)",
            "kind": o["graspable"], "why": o["why"], "split": o["split"], "mass": o["mass_kg"],
            "extent": [round(float(e) * 100, 1) for e in o["extent_m"]], "grasp_h": o.get("grasp_height_m"),
            "excluded": (o.get("category") in excluded), "stands": (o.get("rest") or {}).get("stands"),
            "img": f"img/{key}.jpg",
        })  # fmt: skip
    return rows


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GSO Object Gallery</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1c1c1c;--mute:#6b6b6b;--line:#e2e2df;--single:#1f7a4d;--bimanual:#2457a6;
--none:#8a8a8a;--excl:#b3261e;--chip:#efefec}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#202020;--ink:#ececec;--mute:#a0a0a0;--line:#333;
--single:#4fbf86;--bimanual:#6f9cf0;--none:#8f8f8f;--excl:#ef6b62;--chip:#2a2a2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mute);font-size:13px;margin-bottom:10px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
input[type=search]{min-width:220px;flex:1 1 220px}.count{color:var(--mute);margin-left:auto}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.card img{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:#ededed}
.meta{padding:8px 10px}.ph{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.k{color:var(--mute);font-size:12px}.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.chip{font-size:11px;padding:1px 6px;border-radius:10px;background:var(--chip);color:var(--mute)}
.single{color:#fff;background:var(--single)}.bimanual{color:#fff;background:var(--bimanual)}
.none{color:#fff;background:var(--none)}.excl{color:#fff;background:var(--excl)}
dialog{border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink);max-width:min(900px,94vw);padding:0}
dialog::backdrop{background:rgba(0,0,0,.55)}.dl{display:flex;flex-wrap:wrap}
.dl img{width:min(480px,94vw);aspect-ratio:1}.dl .info{padding:16px;flex:1 1 280px;min-width:0}
.dl table{border-collapse:collapse;width:100%;font-size:13px}.dl td{border-top:1px solid var(--line);padding:5px 4px;vertical-align:top}
.dl td:first-child{color:var(--mute);white-space:nowrap;padding-right:12px}.dl td{overflow-wrap:anywhere}
.close{float:right;font:inherit;background:none;border:0;color:var(--mute);cursor:pointer;font-size:20px}
</style></head><body>
<header><h1>Google Scanned Objects: __N__ objects</h1>
<div class="sub">__SUB__</div>
<div class="bar">
<input type="search" id="q" placeholder="Search name, phrase, key">
<select id="kind"><option value="">All kinds</option><option>single</option><option>bimanual</option><option>none</option></select>
<select id="cat"><option value="">All categories</option></select>
<select id="split"><option value="">All splits</option><option>train</option><option>held_out</option></select>
<select id="ex"><option value="">Drawable and excluded</option><option value="in">Drawable only</option><option value="out">Excluded categories only</option></select>
<select id="sort"><option value="key">Sort: key</option><option value="mass">Sort: mass</option><option value="size">Sort: largest side</option><option value="name">Sort: name</option></select>
<span class="count" id="count"></span></div></header>
<main id="grid"></main>
<dialog id="dlg"><div class="dl"><img id="big" alt=""><div class="info"><button class="close" id="x" aria-label="Close">×</button>
<div class="ph" id="bt" style="white-space:normal;font-size:15px;margin-bottom:8px"></div><table id="tb"></table></div></div></dialog>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const cats=[...new Set(D.map(d=>d.category))].sort();
for(const c of cats){const o=document.createElement('option');o.textContent=c;$('cat').appendChild(o);}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function chips(d){return `<span class="chip ${d.kind}">${d.kind}</span>`+(d.excluded?'<span class="chip excl">excluded</span>':'')+
 `<span class="chip">${esc(d.category)}</span>`+(d.split==='held_out'?'<span class="chip">held out</span>':'');}
function draw(){
 const q=$('q').value.trim().toLowerCase(),k=$('kind').value,c=$('cat').value,s=$('split').value,e=$('ex').value,so=$('sort').value;
 let rows=D.filter(d=>(!q||(d.name+' '+d.phrase+' '+d.key).toLowerCase().includes(q))&&(!k||d.kind===k)&&(!c||d.category===c)
  &&(!s||d.split===s)&&(!e||(e==='in'?!d.excluded:d.excluded)));
 const key={key:d=>d.key,name:d=>d.name.toLowerCase(),mass:d=>-d.mass,size:d=>-Math.max(...d.extent)}[so];
 rows.sort((a,b)=>{const x=key(a),y=key(b);return x<y?-1:x>y?1:0;});
 $('count').textContent=`${rows.length} shown`;
 $('grid').innerHTML=rows.map(d=>`<div class="card" data-k="${d.key}"><img loading="lazy" src="${d.img}" alt="${esc(d.phrase)}">
 <div class="meta"><div class="ph" title="${esc(d.phrase)}">${esc(d.phrase)}</div><div class="k">${d.key} · ${d.mass} kg · ${d.extent.join(' × ')} cm</div>
 <div class="chips">${chips(d)}</div></div></div>`).join('');
}
const byKey=Object.fromEntries(D.map(d=>[d.key,d]));
$('grid').addEventListener('click',ev=>{const el=ev.target.closest('.card');if(!el)return;const d=byKey[el.dataset.k];
 $('big').src=d.img;$('bt').textContent=d.phrase;
 const f=[['key',d.key],['name',d.name],['category',d.category+(d.excluded?' (excluded from the logistics room)':'')],
  ['graspable',d.kind],['why',d.why],['mass',d.mass+' kg'],['size x × y × z',d.extent.join(' × ')+' cm'],
  ['grasp height',d.grasp_h+' m'],['rest test',d.stands===null?'not run':(d.stands?'stands':'falls')],['split',d.split]];
 $('tb').innerHTML=f.map(([a,b])=>`<tr><td>${a}</td><td>${esc(b)}</td></tr>`).join('');$('dlg').showModal();});
$('x').onclick=()=>$('dlg').close();$('dlg').addEventListener('click',ev=>{if(ev.target===$('dlg'))$('dlg').close();});
for(const id of ['q','kind','cat','split','ex','sort'])$(id).addEventListener('input',draw);
draw();
</script></body></html>
"""


def page(catalog: dict) -> str:
    rows = entries(catalog)
    c = catalog["counts"]
    sub = (
        f"single {c.get('single')} · bimanual {c.get('bimanual')} · none {c.get('none')} "
        f"(of which excluded categories {c.get('excluded', 0)}: "
        f"{', '.join(catalog['thresholds'].get('excluded_categories') or []) or 'none'}) · held out {c.get('held_out')}"
        " · rendered by MuJoCo from the scan mesh and texture; kinds and masses from catalog.json"
    )
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("__N__", str(len(rows))).replace("__SUB__", html.escape(sub)).replace("__DATA__", data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=ASSETS / "catalog.json")
    ap.add_argument("--models", type=Path, default=ASSETS / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/gso_gallery")
    ap.add_argument("--size", type=int, default=384, help="px; each picture is square")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true", help="render every picture again")
    a = ap.parse_args(argv)
    if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"):
        print("[gallery] MUJOCO_GL is not egl or osmesa: the offscreen render needs one of them", file=sys.stderr)
    catalog = json.loads(a.catalog.read_text())
    img = a.out / "img"
    img.mkdir(parents=True, exist_ok=True)
    jobs = [
        (k, str(a.models / o["name"]), o, str(img / f"{k}.jpg"), a.size)
        for k, o in catalog["objects"].items()
        if a.force or not (img / f"{k}.jpg").is_file()
    ]
    t0, failed = time.time(), {}
    if jobs:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, (key, err) in enumerate(pool.map(render_one, jobs, chunksize=4), 1):
                if err:
                    failed[key] = err
                if i % 100 == 0:
                    print(f"[gallery] {i} / {len(jobs)} rendered ({time.time() - t0:.0f} s)", flush=True)
    (a.out / "index.html").write_text(page(catalog))
    print(json.dumps({"page": str(a.out / "index.html"), "rendered": len(jobs) - len(failed), "kept": len(catalog["objects"]) - len(jobs),
                      "failed": failed, "seconds": round(time.time() - t0, 1)}, indent=1))  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
