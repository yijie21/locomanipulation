#!/usr/bin/env python3
"""A local HTML gallery of every Google Scanned Object in the catalog (user 2026-11-05): each object rendered alone by
MuJoCo offscreen from one fixed viewpoint (its visual mesh and scan texture on a light floor, the camera framing its
bounding box), beside what catalog.json says about it: key, name, dataset category, single / bimanual / none and
why, mass, size, split and two shape scores (gso_objects.shape_scores: `fill`, how box-like; `solid`, how far from an
open container). The page filters by kind, category, split and the scene pool, searches names and sorts by mass, size
or shape; the (i) button shows the object large with every field.

Choosing by hand (user 2026-11-17): every one of the 1030 objects shows two things, each first by the rules and each
settable by hand. Excluded or included: by its category at first; the card's button flips it. Its shape kind: what
the shape rules (size, rest test) give, and under the `shape` tag three buttons (or the keys 1 / 2 / 3 over the card,
0 back to the rule; x flips the exclusion) set it to single-arm pick, two-arm pick or none. The object is drawn as its
shape kind when it is included, and is none when excluded. Header buttons set every card shown at once. A choice is
kept even where the rules agree (only `back to the rule` drops it): with --serve every click is saved at once to
`assets/mujoco_scanned_objects/hand_choices.json` ({"objects": {model name: {"excluded": bool, "exclude_why": str,
"shape": single | bimanual | none, "shape_why": str}}}, either choice or both; tracked; scripts/gso_catalog.py reads
it). Opened as a plain file the page keeps the choices in the browser and a Download button writes the same file to
put there.

Picking the test scenes' objects (user 2026-11-17): `--pick` also writes `<out>/pick.html`
(scripts/gso_scene_pick.py), the 10 single-hand objects of `logistics_room_gso_single` and the 5-9 two-hand objects
of `logistics_room_gso_bimanual`, each click saved to `--picks` and placed on the platform as the builder will.
Rebuild the catalog first (scripts/gso_catalog.py) so the page offers the kinds the hand choices give.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $VENV_SIM_PY scripts/gso_gallery.py [--out outputs/gso_gallery]
        [--catalog assets/mujoco_scanned_objects/catalog.json] [--models assets/mujoco_scanned_objects/models]
        [--hand-choices assets/mujoco_scanned_objects/hand_choices.json]
        [--size 384] [--workers 8] [--force] [--serve [--port 8765] [--host 127.0.0.1]]
        [--pick [--picks assets/mujoco_scanned_objects/scene_picks.json]]

Writes `<out>/index.html` (the pictures are `<out>/img/<key>.jpg`, the shape scores `<out>/shape.json`, beside it). A
picture or score already there is kept unless --force; the page is always rewritten. About 2 minutes for the 1030
objects the first time.
"""

from __future__ import annotations

import argparse
import html
import http.server
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
REASONS = (
    "irregular shape",
    "toy or figurine",
    "open container",
    "hard to grasp",
    "bad scan",
    "checked by eye",
)  # the page's picks


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


def shape_one(job: tuple) -> tuple[str, dict | None]:
    """(key, gso_objects.shape_scores or None when the scan cannot be read)."""
    key, model_dir = job
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from locomani.sim.gso_objects import shape_scores

        return key, shape_scores(Path(model_dir))
    except Exception:  # noqa: BLE001 -- one broken scan sorts last, it does not stop the page
        return key, None


def shape_kind(o: dict, thresholds) -> tuple[str, str]:
    """(kind, why) by the shape rules alone (size, rest test; no category, no hand-kept list): what the object would be
    if it were included."""
    from locomani.sim.gso_objects import graspable

    return graspable(o["extent_m"], thresholds, o.get("rest"), None)


def entries(catalog: dict, shapes: dict | None = None) -> list[dict]:
    """The page's rows: one per object, the catalog fields the page shows. `excluded`: its category is not used;
    `kind` / `why`: the shape rules alone (`shape_kind`); `auto`: the kind by the rules before any hand choice (none
    for an unused category, else `kind`); `fits`: it fits the carton (the scenes draw only those). The page applies
    the hand choices on top."""
    sys.path.insert(0, str(ROOT / "src"))
    from dataclasses import fields

    from locomani.sim.gso_objects import Thresholds

    used = (catalog.get("thresholds") or {}).get("used_categories")
    known = {f.name for f in fields(Thresholds)}
    th = Thresholds(**{k: tuple(v) if isinstance(v, list) else v for k, v in (catalog.get("thresholds") or {}).items()
                       if k in known and k != "used_categories"}, used_categories=None)  # fmt: skip
    shapes = shapes or {}
    rows = []
    for key, o in catalog["objects"].items():
        sh = shapes.get(key) or {}
        ruled = o.get("by_hand") is None and (used is None or (o.get("category") or "(none)") in used)
        # the catalog's own verdict when the shape rules gave it (it saw the unrounded extents: the Bundt pan is
        # 9.99 cm tall and 0.1 m in the record); recomputed only for objects the category or a hand choice decided
        kind, why = (o["graspable"], o["why"]) if ruled else shape_kind(o, th)
        rows.append({
            "key": key, "name": o["name"], "phrase": o["phrase"], "category": o.get("category") or "(none)",
            "kind": kind, "why": why, "split": o["split"], "mass": o["mass_kg"],
            "extent": [round(float(e) * 100, 1) for e in o["extent_m"]], "grasp_h": o.get("grasp_height_m"),
            "excluded": used is not None and (o.get("category") or "(none)") not in used,
            "stands": (o.get("rest") or {}).get("stands"), "img": f"img/{key}.jpg", "fits": bool(o.get("fits_carton")),
            "auto": "none" if used is not None and (o.get("category") or "(none)") not in used else kind,
            "fill": sh.get("fill"), "solid": sh.get("solid"),
        })  # fmt: skip
    return rows


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GSO Object Gallery</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1c1c1c;--mute:#6b6b6b;--line:#e2e2df;--single:#1f7a4d;--bimanual:#2457a6;
--none:#8a8a8a;--excl:#b3261e;--chip:#efefec;--hand:#c77700}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#202020;--ink:#ececec;--mute:#a0a0a0;--line:#333;
--single:#4fbf86;--bimanual:#6f9cf0;--none:#8f8f8f;--excl:#ef6b62;--chip:#2a2a2a;--hand:#f0a640}}
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
.card{position:relative;user-select:none;cursor:default}
.card.k-none img{opacity:.35;filter:grayscale(1)}.card.out{border-color:var(--excl)}
.card.hand{box-shadow:0 0 0 3px var(--hand)}
.card .mark{position:absolute;top:8px;left:8px;right:44px;color:#fff;background:var(--none);font-size:12px;
font-weight:600;padding:2px 8px;border-radius:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card .mark.hand{background:var(--hand)}.card .mark.cat{background:var(--excl)}
.info-btn{position:absolute;top:6px;right:6px;width:30px;height:30px;border-radius:15px;border:1px solid var(--line);
background:var(--card);color:var(--ink);font:600 14px/1 system-ui;cursor:pointer}
.row{display:flex;align-items:center;gap:6px;margin-top:8px}.row .lab{font-size:11px;color:var(--mute);width:38px;flex:none}
.seg{display:flex;flex:1;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{flex:1;font:600 12px/1 system-ui;padding:8px 2px 10px;border:0;border-right:1px solid var(--line);background:var(--card);
color:var(--mute);cursor:pointer;position:relative}.seg button:last-child{border-right:0}
.seg button.on.single{background:var(--single);color:#fff}.seg button.on.bimanual{background:var(--bimanual);color:#fff}
.seg button.on.none{background:var(--none);color:#fff}.seg.handset{outline:2px solid var(--hand)}
.seg button.auto::after{content:"rule";position:absolute;bottom:1px;left:0;right:0;font:500 9px/1 system-ui;opacity:.8}
.exb{flex:1;font:600 12px/1 system-ui;padding:8px 4px;border-radius:6px;cursor:pointer;border:1px solid var(--line);
background:var(--card);color:var(--single)}.exb.out{background:var(--excl);border-color:var(--excl);color:#fff}
.exb.handset{outline:2px solid var(--hand)}
.bar2{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
button.act{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}
#state{font-size:13px;color:var(--mute)}#state.bad{color:var(--excl);font-weight:600}
.tally b{font-weight:600}.hint{color:var(--mute);font-size:12px}
</style></head><body>
<header><h1>Google Scanned Objects: __N__ objects</h1>
<div class="sub">__SUB__</div>
<div class="bar">
<input type="search" id="q" placeholder="Search name, phrase, key">
<select id="kind"><option value="">Shape: any</option><option value="single">Shape: single</option><option value="bimanual">Shape: bimanual</option><option value="none">Shape: none</option></select>
<select id="cat"><option value="">All categories</option></select>
<select id="split"><option value="">All splits</option><option>train</option><option>held_out</option></select>
<select id="ex"><option value="">All objects</option><option value="eout">Excluded now</option><option value="ein">Included now</option><option value="drawn">Included, graspable, fits the carton</option><option value="hand">Anything set by hand</option><option value="hex">Exclusion set by hand</option><option value="hshape">Shape set by hand</option><option value="fits">Fits the carton</option><option value="nofit">Does not fit the carton</option><option value="out">Category excluded</option></select>
<select id="sort"><option value="key">Sort: key</option><option value="fill">Sort: least box-like first</option><option value="solid">Sort: most hollow first</option><option value="mass">Sort: mass</option><option value="size">Sort: largest side</option><option value="name">Sort: name</option></select>
<span class="count" id="count"></span></div>
<div class="bar2"><span class="hint">Per card: the button excludes / includes it; under <b>shape</b> pick single, bimanual or none. Over a card: 1 / 2 / 3 shape, 0 shape back to the rule, x flip exclusion. Reason:</span>
<select id="reason">__REASONS__<option value="__other__">other…</option></select>
<input id="other" placeholder="your reason" style="display:none">
<span class="tally" id="tally"></span><span id="state"></span></div>
<div class="bar2"><span class="hint">All shown — shape:</span><button class="act" data-shape="single">Single</button>
<button class="act" data-shape="bimanual">Bimanual</button><button class="act" data-shape="none">None</button>
<button class="act" data-shape="">Rule</button>
<span class="hint">exclusion:</span><button class="act" data-ex="1">Exclude</button><button class="act" data-ex="0">Include</button>
<button class="act" data-ex="">Rule</button>
<button class="act" id="dl" style="display:none">Download hand_choices.json</button></div></header>
<main id="grid"></main>
<dialog id="dlg"><div class="dl"><img id="big" alt=""><div class="info"><button class="close" id="x" aria-label="Close">×</button>
<div class="ph" id="bt" style="white-space:normal;font-size:15px;margin-bottom:8px"></div><table id="tb"></table></div></div></dialog>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const cats=[...new Set(D.map(d=>d.category))].sort();
for(const c of cats){const o=document.createElement('option');o.textContent=c;$('cat').appendChild(o);}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let CH=__MARKS__;  // hand choices {model name: {excluded?, exclude_why?, shape?, shape_why?}}; the server's copy wins when served
let served=false,saving=null,dirty=false;
const LS='gso_hand_choices',KINDS=['single','bimanual','none'],LABEL={single:'Single',bimanual:'Bimanual',none:'None'};
const isOut=d=>CH[d.name]&&'excluded' in CH[d.name]?CH[d.name].excluded:d.excluded;
const shapeOf=d=>CH[d.name]&&CH[d.name].shape?CH[d.name].shape:d.kind;
const drawn=d=>isOut(d)?'none':shapeOf(d);
const handEx=d=>!!(CH[d.name]&&'excluded' in CH[d.name]),handShape=d=>!!(CH[d.name]&&CH[d.name].shape);
function tidy(d){const c=CH[d.name];if(c&&!('excluded' in c)&&!c.shape)delete CH[d.name];}
function setShape(d,k,why){const c=CH[d.name]||(CH[d.name]={});if(k){c.shape=k;c.shape_why=why;}else{delete c.shape;delete c.shape_why;}tidy(d);}
function setExcl(d,v,why){const c=CH[d.name]||(CH[d.name]={});if(v===null){delete c.excluded;delete c.exclude_why;}else{c.excluded=v;c.exclude_why=why;}tidy(d);}
function chips(d){return (d.excluded?'<span class="chip excl">category excluded</span>':'')+
 `<span class="chip">${esc(d.category)}</span>`+(d.split==='held_out'?'<span class="chip">held out</span>':'')+(d.fits?'':'<span class="chip">no carton fit</span>')+
 (d.fill==null?'':`<span class="chip" title="fill: convex hull / principal-axis box">fill ${d.fill}</span><span class="chip" title="solid: V-HACD hulls / convex hull">solid ${d.solid}</span>`);}
function reason(){const r=$('reason').value;return r==='__other__'?($('other').value.trim()||'other'):r;}
function status(){
 const c={single:0,bimanual:0,none:0};let out=0,pick=0;for(const d of D){c[drawn(d)]++;if(isOut(d))out++;if(d.fits&&drawn(d)!=='none')pick++;}
 $('tally').innerHTML=`excluded <b>${out}</b> · drawn as single <b>${c.single}</b> · bimanual <b>${c.bimanual}</b> · none <b>${c.none}</b> · can enter a scene <b>${pick}</b> · set by hand <b>${Object.keys(CH).length}</b>`;
 const st=$('state');st.className='';
 if(served){st.textContent=saving?'saving…':(dirty?'not saved':'saved to hand_choices.json');if(dirty&&!saving)st.className='bad';}
 else{st.textContent='file mode: choices stay in this browser; download the file and put it beside catalog.json';$('dl').style.display='';}
}
const body=()=>({objects:Object.fromEntries(Object.entries(CH).sort())});
async function save(){
 if(!served){try{localStorage.setItem(LS,JSON.stringify(body()));}catch(e){}dirty=false;status();return;}
 dirty=true;if(saving)return;
 while(dirty){dirty=false;saving=true;status();
  try{const r=await fetch('api/choices',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body())});
   if(!r.ok)throw new Error(await r.text());}
  catch(e){dirty=true;saving=null;status();$('state').textContent='save failed: '+e.message;return;}
  saving=null;}
 status();
}
function label(d){const c=CH[d.name]||{};
 if(isOut(d))return handEx(d)?['hand','excluded by hand: '+c.exclude_why]:['cat','excluded: category '+d.category];
 if(handEx(d)&&d.excluded)return ['hand','included by hand: '+c.exclude_why];
 if(handShape(d))return ['hand',`shape ${c.shape} by hand: ${c.shape_why}`];
 if(d.kind==='none')return ['','shape rules: none'];
 return null;}
function card(d){const l=label(d),k=shapeOf(d),out=isOut(d);
 return `<div class="card k-${drawn(d)}${out?' out':''}${CH[d.name]?' hand':''}" data-k="${d.key}"><img loading="lazy" src="${d.img}" alt="${esc(d.phrase)}">
 ${l?`<span class="mark ${l[0]}">${esc(l[1])}</span>`:''}<button class="info-btn" title="details" aria-label="details">i</button>
 <div class="meta"><div class="ph" title="${esc(d.phrase)}">${esc(d.phrase)}</div><div class="k">${d.key} · ${d.mass} kg · ${d.extent.join(' × ')} cm</div>
 <div class="chips">${chips(d)}</div>
 <div class="row"><span class="lab" title="${esc('shape rules: '+d.kind+' — '+d.why)}">shape</span><div class="seg${handShape(d)?' handset':''}">${KINDS.map(x=>`<button data-v="${x}" class="${x}${x===k?' on':''}${x===d.kind?' auto':''}" title="${x===d.kind?esc('the shape rules: '+d.why):''}">${LABEL[x]}</button>`).join('')}</div></div>
 <div class="row"><span class="lab">use</span><button class="exb${out?' out':''}${handEx(d)?' handset':''}" title="${d.excluded?'its category is excluded by the rules':'its category is used'}">${out?'Excluded — click to include':'Included — click to exclude'}</button></div>
 </div></div>`;}
let shown=[];
function draw(){
 const q=$('q').value.trim().toLowerCase(),k=$('kind').value,c=$('cat').value,s=$('split').value,e=$('ex').value,so=$('sort').value;
 const exOk=d=>!e||{eout:isOut(d),ein:!isOut(d),drawn:d.fits&&drawn(d)!=='none',hand:!!CH[d.name],hex:handEx(d),hshape:handShape(d),
  fits:d.fits,nofit:!d.fits,out:d.excluded}[e];
 let rows=D.filter(d=>(!q||(d.name+' '+d.phrase+' '+d.key).toLowerCase().includes(q))&&(!k||shapeOf(d)===k)&&(!c||d.category===c)
  &&(!s||d.split===s)&&exOk(d));
 const key={key:d=>d.key,name:d=>d.name.toLowerCase(),mass:d=>-d.mass,size:d=>-Math.max(...d.extent),
  fill:d=>d.fill==null?9:d.fill,solid:d=>d.solid==null?9:d.solid}[so];
 rows.sort((a,b)=>{const x=key(a),y=key(b);return x<y?-1:x>y?1:a.key<b.key?-1:1;});
 shown=rows;$('count').textContent=`${rows.length} shown`;
 $('grid').innerHTML=rows.map(card).join('');status();
}
const byKey=Object.fromEntries(D.map(d=>[d.key,d]));
function details(d){
 $('big').src=d.img;$('bt').textContent=d.phrase;const c=CH[d.name]||{};
 const f=[['key',d.key],['name',d.name],['drawn as',drawn(d)],
  ['excluded',(isOut(d)?'yes':'no')+(handEx(d)?' (by hand: '+c.exclude_why+')':' (by the category rule)')],
  ['shape kind',shapeOf(d)+(handShape(d)?' (by hand: '+c.shape_why+')':' (by the shape rules)')],
  ['shape rules say',d.kind+': '+d.why],['category',d.category+(d.excluded?' (not a used category)':'')],
  ['fits the carton',d.fits?'yes':'no'],['mass',d.mass+' kg'],['size x × y × z',d.extent.join(' × ')+' cm'],
  ['grasp height',d.grasp_h+' m'],['shape: fill / solid',d.fill==null?'not computed':d.fill+' / '+d.solid],
  ['rest test',d.stands===null?'not run':(d.stands?'stands':'falls')],['split',d.split]];
 $('tb').innerHTML=f.map(([a,b])=>`<tr><td>${a}</td><td>${esc(b)}</td></tr>`).join('');$('dlg').showModal();}
function redraw(el){const d=byKey[el.dataset.k];el.outerHTML=card(d);save();return document.querySelector(`.card[data-k="${d.key}"]`);}
$('grid').addEventListener('click',ev=>{const el=ev.target.closest('.card');if(!el)return;const d=byKey[el.dataset.k];
 if(ev.target.closest('.info-btn')){details(d);return;}
 const b=ev.target.closest('.seg button');if(b){setShape(d,b.dataset.v,reason());redraw(el);return;}
 if(ev.target.closest('.exb')){setExcl(d,!isOut(d),reason());redraw(el);}});
let hover=null;$('grid').addEventListener('mouseover',ev=>{hover=ev.target.closest('.card');});
$('grid').addEventListener('mouseleave',()=>{hover=null;});
document.addEventListener('keydown',ev=>{const t=ev.target;
 if(!hover||!hover.isConnected||$('dlg').open||(t instanceof Element&&t.closest('input,select,textarea')))return;
 const d=byKey[hover.dataset.k],key=ev.key.toLowerCase();
 if(key==='x')setExcl(d,!isOut(d),reason());
 else{const k={'1':'single','2':'bimanual','3':'none','0':''}[key];if(k===undefined)return;setShape(d,k,reason());}
 ev.preventDefault();hover=redraw(hover);});
function bulk(what,apply){if(!shown.length||!confirm(`Set ${what} for the ${shown.length} objects shown?`))return;
 const why=reason();for(const d of shown)apply(d,why);draw();save();}
for(const b of document.querySelectorAll('[data-shape]'))b.onclick=()=>{const k=b.dataset.shape;
 bulk(k?'shape '+k:'shape back to the rule',(d,why)=>setShape(d,k,why));};
for(const b of document.querySelectorAll('[data-ex]'))b.onclick=()=>{const v=b.dataset.ex;
 bulk(v===''?'exclusion back to the rule':(v==='1'?'excluded':'included'),(d,why)=>setExcl(d,v===''?null:v==='1',why));};
$('x').onclick=()=>$('dlg').close();$('dlg').addEventListener('click',ev=>{if(ev.target===$('dlg'))$('dlg').close();});
$('reason').addEventListener('input',()=>{$('other').style.display=$('reason').value==='__other__'?'':'none';});
$('dl').onclick=()=>{const b=new Blob([JSON.stringify(body(),null,1)+'\\n'],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='hand_choices.json';a.click();};
window.addEventListener('beforeunload',ev=>{if(served&&(dirty||saving)){ev.preventDefault();ev.returnValue='';}});
for(const id of ['q','kind','cat','split','ex','sort'])$(id).addEventListener('input',draw);
(async()=>{
 try{const r=await fetch('api/choices',{cache:'no-store'});if(r.ok){CH=(await r.json()).objects||{};served=true;}}catch(e){}
 if(!served){try{const v=JSON.parse(localStorage.getItem(LS)||'null');if(v)CH=v.objects||{};}catch(e){}}
 draw();
})();
</script></body></html>
"""


def page(catalog: dict, shapes: dict | None = None, choices: dict | None = None) -> str:
    """The page: every catalog object, its shape scores and the hand choices it starts from ({name: entry})."""
    rows = entries(catalog, shapes)
    c = catalog["counts"]
    sub = (
        f"single {c.get('single')} · bimanual {c.get('bimanual')} · none {c.get('none')} "
        f"(of which excluded categories {c.get('excluded', 0)}: "
        f"{', '.join(catalog.get('not_used_categories') or []) or 'none'}) · held out {c.get('held_out')}"
        f" · set by hand in this catalog: {c.get('by_hand') or 'none'}"
        " · rendered by MuJoCo from the scan mesh and texture; kinds and masses from catalog.json"
    )
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    marks_js = json.dumps(choices or {}, ensure_ascii=False).replace("</", "<\\/")
    reasons = "".join(f"<option>{html.escape(r)}</option>" for r in REASONS)
    return (PAGE.replace("__N__", str(len(rows))).replace("__SUB__", html.escape(sub)).replace("__REASONS__", reasons)
            .replace("__MARKS__", marks_js).replace("__DATA__", data))  # fmt: skip


def save_marks(path: Path, body: dict, names: set[str]) -> dict:
    """Write the hand-kept list from the page's POST body {"objects": {name: entry}}: every name a catalog object,
    every entry as gso_objects.check_hand_choice wants it; written sorted, atomically. Returns what was written."""
    sys.path.insert(0, str(ROOT / "src"))
    from locomani.sim.gso_objects import check_hand_choice

    objs = body.get("objects") if isinstance(body, dict) else None
    if not isinstance(objs, dict):
        raise ValueError('body must be {"objects": {model name: {"excluded": ..., "shape": ...}}}')
    bad = sorted(k for k in objs if k not in names)
    if bad:
        raise ValueError(f"not catalog objects: {bad[:3]}")
    doc = {
        "note": "choices by hand for the logistics room's objects (user 2026-11-17): `excluded` over the category "
        "rule, `shape` (single / bimanual / none) over the shape rules; written by scripts/gso_gallery.py --serve, "
        "read by scripts/gso_catalog.py",
        "objects": {k: check_hand_choice(k, c) for k, c in sorted(objs.items())},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return doc


def serve(out: Path, marks_path: Path, names: set[str], host: str, port: int, extra: dict | None = None) -> None:
    """Serve `out` (the page and pictures) and `api/choices` (GET the list, POST a new one) until Ctrl-C; `extra`
    adds endpoints {"api/<name>": (GET -> doc or None, POST body -> doc or None)} (the picking page's)."""
    extra = extra or {}

    def route(path: str):
        path = path.split("?")[0]
        return next((v for k, v in extra.items() if path.endswith("/" + k)), None)

    from locomani.sim.gso_objects import load_hand_choices

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(out), **kw)

        def log_message(self, fmt, *args):  # the pictures would flood the terminal
            pass

        def _json(self, code: int, doc) -> None:
            body = json.dumps(doc, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            ext = route(self.path)
            if ext and ext[0]:
                return self._json(200, ext[0]())
            if self.path.split("?")[0].endswith("/api/choices"):
                return self._json(200, {"objects": load_hand_choices(marks_path)})
            return super().do_GET()

        def do_POST(self):
            ext = route(self.path)
            if ext and ext[1]:
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    return self._json(200, ext[1](body))
                except (ValueError, json.JSONDecodeError) as e:
                    return self._json(400, {"error": str(e)})
            if not self.path.split("?")[0].endswith("/api/choices"):
                return self._json(404, {"error": "no such endpoint"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                doc = save_marks(marks_path, body, names)
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            print(f"[gallery] {len(doc['objects'])} objects set by hand -> {marks_path}", flush=True)
            return self._json(200, {"objects": doc["objects"]})

    with http.server.ThreadingHTTPServer((host, port), Handler) as httpd:
        print(
            f"[gallery] open http://{host}:{port}/index.html  (clicks save to {marks_path}; Ctrl-C stops)", flush=True
        )
        if "api/picks" in extra:
            print(f"[gallery] pick the scene objects at http://{host}:{port}/pick.html", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=ASSETS / "catalog.json")
    ap.add_argument("--models", type=Path, default=ASSETS / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/gso_gallery")
    ap.add_argument("--size", type=int, default=384, help="px; each picture is square")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--hand-choices", type=Path, default=ASSETS / "hand_choices.json",
                    help="the hand-kept kinds the page starts from and --serve writes")  # fmt: skip
    ap.add_argument("--force", action="store_true", help="render every picture and score again")
    ap.add_argument("--serve", action="store_true", help="serve the page on localhost; clicks save to --hand-choices")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pick", action="store_true", help="also write and serve pick.html, the picking page of the two "
                    "test scenes (scripts/gso_scene_pick.py); its clicks save to --picks")  # fmt: skip
    ap.add_argument("--picks", type=Path, default=ASSETS / "scene_picks.json",
                    help="the objects picked for logistics_room_gso_single / _bimanual")  # fmt: skip
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from locomani.sim.gso_objects import load_hand_choices

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
    shape_path = a.out / "shape.json"
    shapes = {} if a.force or not shape_path.is_file() else json.loads(shape_path.read_text())
    todo = [(k, str(a.models / o["name"])) for k, o in catalog["objects"].items() if k not in shapes]
    if todo:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            shapes.update(pool.map(shape_one, todo, chunksize=8))
        shape_path.write_text(json.dumps(shapes, separators=(",", ":")))
    choices = load_hand_choices(a.hand_choices)
    (a.out / "index.html").write_text(page(catalog, shapes, choices))
    print(json.dumps({"page": str(a.out / "index.html"), "rendered": len(jobs) - len(failed), "kept": len(catalog["objects"]) - len(jobs),
                      "shapes_computed": len(todo), "set_by_hand": len(choices), "failed": failed,
                      "seconds": round(time.time() - t0, 1)}, indent=1))  # fmt: skip
    extra = None
    if a.pick:
        import gso_scene_pick as SP

        from locomani.sim.gso_objects import load_picks

        (a.out / "pick.html").write_text(SP.page(catalog, load_picks(a.picks, catalog)))
        surf = SP.surface()
        extra = {"api/picks": (lambda: load_picks(a.picks, catalog), lambda b: SP.save(a.picks, b, catalog)),
                 "api/place": (None, lambda b: SP.place(b, catalog, surf))}  # fmt: skip
        print(f"[gallery] picking page {a.out / 'pick.html'}", flush=True)
    if a.serve:
        serve(a.out, a.hand_choices, {o["name"] for o in catalog["objects"].values()}, a.host, a.port, extra)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
