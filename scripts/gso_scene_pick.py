#!/usr/bin/env python3
"""The picking page for the two hand-picked test scenes (user 2026-11-17), served by `scripts/gso_gallery.py --pick
--serve` beside the gallery: `logistics_room_gso_single` gets exactly 20 single-hand objects and no carton, `logistics_room_gso_bimanual`
5-9 two-hand objects (as many as fit) and the carton. The page offers each scene's pool (gso_objects.pick_pool: its kind
in the built catalog, hand choices applied, training split only), shows the pictures the gallery rendered, and on
every click saves `assets/mujoco_scanned_objects/scene_picks.json` and asks the server to place the set on the
platform as the scene builder will (gso_objects.place_picked: every object on the 0.35 m reach band of an edge, the
carton on its hug band, long side along the edge) and draws the result from above, or names what found no spot.
`scripts/prepare_logistics_scene.py --picks single|bimanual` builds the scene from the file.

Not run on its own: `$VENV_SIM_PY scripts/gso_gallery.py --pick --serve` writes `<out>/pick.html` and serves it.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from locomani.sim import gso_objects as G  # noqa: E402


def surface() -> dict:
    """The platform top of the packaged logistics room, as prepare_logistics_scene places on it."""
    import xml.etree.ElementTree as ET

    import prepare_logistics_scene as P

    geom = (
        ET.parse(P.DEFAULT_SOURCE / "scene_locomani.xml").getroot().find("worldbody").find("geom[@name='platform_col']")
    )
    return P.platform_surface(geom)


def place(body: dict, catalog: dict, surf: dict) -> dict:
    """The page's `api/place`: {scene, seed, objects: [names]} -> where the builder would put them and the carton:
    {ok, attempt, error, surface: {xy, half, reach}, items: [{name, phrase, xy, yaw, half, carton}]}."""
    import prepare_logistics_scene as P

    which = body.get("scene") if isinstance(body, dict) else None
    if which not in G.PICK_SCENES:
        raise ValueError(f"scene must be one of {sorted(G.PICK_SCENES)}")
    doc = G.check_picks(
        {"scenes": {which: {"seed": body.get("seed", 0), "objects": body.get("objects") or []}}}, catalog
    )
    keys = G.picked_keys(catalog, doc, which)
    recs = {k: catalog["objects"][k] for k in keys}
    boxes = {k: G.object_box(r)["half_extent"] for k, r in recs.items()}
    carton = G.PICK_SCENES[which]["carton"]
    if carton:
        boxes[P.BOX_KEY] = P.BOX_HALF
    out = {"surface": {"xy": surf["xy"], "half": surf["half_extent"], "reach": P.REACH_DEPTH_M}, "items": [],
           "ok": False, "attempt": None, "error": None}  # fmt: skip
    try:
        placed, out["attempt"] = G.place_picked(boxes, surf, doc["scenes"][which]["seed"], P.REACH_DEPTH_M,
                                                constraints={P.BOX_KEY: P.CARTON_PLACEMENT} if carton else None)  # fmt: skip
    except RuntimeError as e:
        names = {k: r["name"] for k, r in recs.items()} | {P.BOX_KEY: "the carton"}
        msg = str(e)
        for k, n in names.items():
            msg = msg.replace(k, n)
        out["error"] = msg
        return out
    out["ok"] = True
    for k, p in placed.items():
        carton = k == P.BOX_KEY
        out["items"].append({"name": "carton" if carton else recs[k]["name"],
                             "phrase": P.BOX_PHRASE if carton else recs[k]["phrase"], "xy": p["xy"], "yaw": p["yaw"],
                             "half": boxes[k][:2], "carton": carton})  # fmt: skip
    return out


def save(path: Path, body: dict, catalog: dict) -> dict:
    """The page's `api/picks` POST: the picks checked (an unfinished set is kept), written sorted, atomically."""
    doc = G.check_picks(body, catalog)
    doc = {
        "note": "objects picked by hand for the two RSI test scenes (user 2026-11-17): written by scripts/gso_gallery.py "
        "--pick --serve, read by scripts/prepare_logistics_scene.py --picks single|bimanual; `seed` places them",
        **doc,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return doc


def rows(catalog: dict) -> dict[str, list[dict]]:
    """Each scene's pool as the page's cards."""
    out = {}
    for which in G.PICK_SCENES:
        out[which] = []
        for k in G.pick_pool(catalog, which):
            o = catalog["objects"][k]
            half = G.object_box(o)["half_extent"]
            out[which].append({
                "key": k, "name": o["name"], "phrase": o["phrase"], "category": o.get("category") or "(none)",
                "mass": o["mass_kg"], "extent": [round(float(e) * 100, 1) for e in o["extent_m"]],
                "area": round(4 * half[0] * half[1], 5), "fits": bool(o.get("fits_carton")), "img": f"img/{k}.jpg",
                "hand": bool(o.get("by_hand")),
            })  # fmt: skip
    return out


def page(catalog: dict, picks: dict) -> str:
    rules = {w: {"count": list(s["count"]), "scene": s["scene"], "kind": s["kind"], "carton": s["carton"]}
             for w, s in G.PICK_SCENES.items()}  # fmt: skip
    data = {"rows": rows(catalog), "rules": rules, "picks": picks.get("scenes", {})}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("__DATA__", blob).replace("__FILE__", html.escape(G.PICKS_FILE))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scene Object Picker</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1c1c1c;--mute:#6b6b6b;--line:#e2e2df;--single:#1f7a4d;--bimanual:#2457a6;
--sel:#c77700;--bad:#b3261e;--chip:#efefec;--plat:#e9e4d8;--band:#d6ead9;--box:#a8763e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161616;--card:#202020;--ink:#ececec;--mute:#a0a0a0;
--line:#333;--single:#4fbf86;--bimanual:#6f9cf0;--sel:#f0a640;--bad:#ef6b62;--chip:#2a2a2a;--plat:#3a352c;--band:#24402c;--box:#c99a5e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
h1{font-size:18px;margin:0 0 8px}.tabs{display:flex;gap:6px;margin-bottom:8px}
.tabs button{font:600 13px system-ui;padding:7px 14px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}
.tabs button.on{background:var(--ink);color:var(--bg)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
input[type=search]{flex:1 1 200px;min-width:0}input[type=number]{width:72px}
#state{font-size:13px;color:var(--mute)}#state.bad{color:var(--bad);font-weight:600}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;padding:16px}
@media (max-width:900px){.wrap{grid-template-columns:minmax(0,1fr)}}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;align-content:start}
.card{position:relative;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;user-select:none}
.card img{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:#ededed}
.card.on{box-shadow:0 0 0 3px var(--sel)}.card .n{position:absolute;top:6px;left:6px;min-width:26px;height:26px;border-radius:13px;
background:var(--sel);color:#fff;font:700 13px/26px system-ui;text-align:center;padding:0 6px}
.meta{padding:7px 9px}.ph{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.k{color:var(--mute);font-size:12px}.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.chip{font-size:11px;padding:1px 6px;border-radius:10px;background:var(--chip);color:var(--mute)}
aside{position:sticky;top:150px;align-self:start;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
aside h2{font-size:15px;margin:0 0 6px}.count{font-size:22px;font-weight:700}.count.bad{color:var(--bad)}.count.ok{color:var(--single)}
svg{width:100%;height:auto;display:block;margin:8px 0}#place{font-size:13px}#place.bad{color:var(--bad);font-weight:600}
ol{margin:8px 0 0;padding-left:22px;font-size:13px}ol li{margin:2px 0}ol button{font:inherit;border:0;background:none;color:var(--bad);cursor:pointer}
button.act{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}
.hint{color:var(--mute);font-size:12px}
</style></head><body>
<header><h1>Pick the objects of the two RSI test scenes</h1>
<div class="tabs"><button data-tab="single">Single-hand scene (no carton)</button><button data-tab="bimanual">Two-hand scene (+ carton)</button></div>
<div class="bar"><input type="search" id="q" placeholder="Search name, phrase, category">
<select id="cat"><option value="">All categories</option></select>
<select id="show"><option value="">All</option><option value="on">Picked</option><option value="fits">Fits the carton</option></select>
<select id="sort"><option value="key">Sort: key</option><option value="area">Sort: footprint, small first</option>
<option value="-area">Sort: footprint, large first</option><option value="mass">Sort: mass</option><option value="name">Sort: name</option></select>
<span class="hint">seed</span><input type="number" id="seed" min="0" step="1" value="0">
<span id="state"></span><button class="act" id="dl" style="display:none">Download __FILE__</button></div></header>
<div class="wrap"><main id="grid"></main>
<aside><h2 id="title"></h2><div><span class="count" id="count"></span> <span class="hint" id="want"></span></div>
<div id="place" class="hint"></div><svg id="map" viewBox="0 0 520 400" role="img" aria-label="platform from above"></svg>
<div class="hint">From above: the platform, the green band within reach of an edge, the carton (brown, two-hand scene) and the picked
objects as their bounding boxes. Every click places the set again as the scene builder will.</div><ol id="list"></ol></aside></div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
let tab='single',served=false,saving=false,dirty=false,seq=0;
const P={};for(const w in D.rules)P[w]={seed:(D.picks[w]||{}).seed||0,objects:((D.picks[w]||{}).objects||[]).slice()};
const LS='gso_scene_picks';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function want(w){const [lo,hi]=D.rules[w].count;return lo===hi?`${lo}`:`${lo}-${hi}`;}
function okCount(w){const [lo,hi]=D.rules[w].count,n=P[w].objects.length;return n>=lo&&n<=hi;}
function body(){return {scenes:P};}
function cats(){const s=$('cat'),v=s.value,c=[...new Set(D.rows[tab].map(r=>r.category))].sort();
 s.innerHTML='<option value="">All categories</option>'+c.map(x=>`<option>${esc(x)}</option>`).join('');s.value=c.includes(v)?v:'';}
function draw(){
 for(const b of document.querySelectorAll('.tabs button'))b.classList.toggle('on',b.dataset.tab===tab);
 const q=$('q').value.trim().toLowerCase(),cat=$('cat').value,show=$('show').value,sort=$('sort').value,on=P[tab].objects;
 let rs=D.rows[tab].filter(r=>(!q||(r.name+' '+r.phrase+' '+r.category+' '+r.key).toLowerCase().includes(q))&&(!cat||r.category===cat)
  &&(!show||(show==='on'?on.includes(r.name):r.fits)));
 const by={key:(a,b)=>a.key<b.key?-1:1,area:(a,b)=>a.area-b.area,'-area':(a,b)=>b.area-a.area,mass:(a,b)=>a.mass-b.mass,name:(a,b)=>a.name<b.name?-1:1};
 rs.sort(by[sort]);
 $('grid').innerHTML=rs.map(r=>{const i=on.indexOf(r.name);return `<div class="card${i>=0?' on':''}" data-name="${esc(r.name)}">
 ${i>=0?`<span class="n">${i+1}</span>`:''}<img loading="lazy" src="${esc(r.img)}" alt="${esc(r.phrase)}"><div class="meta">
 <div class="ph" title="${esc(r.name)}">${esc(r.phrase)}</div><div class="k">${esc(r.key)} · ${esc(r.category)}</div>
 <div class="chips"><span class="chip">${r.extent.join(' × ')} cm</span><span class="chip">${r.mass.toFixed(2)} kg</span>
 <span class="chip">${(r.area*1e4).toFixed(0)} cm²</span>${r.fits?'<span class="chip">fits carton</span>':''}${r.hand?'<span class="chip">set by hand</span>':''}</div></div></div>`;}).join('');
 const n=on.length;$('title').textContent=D.rules[tab].scene;$('count').textContent=`${n} picked`;
 $('count').className='count '+(okCount(tab)?'ok':'bad');$('want').textContent=`want ${want(tab)}`+(D.rules[tab].carton?' (as many as fit) + the carton':', no carton');
 const rmap=Object.fromEntries(D.rows[tab].map(r=>[r.name,r]));
 $('list').innerHTML=on.map(nm=>`<li>${esc((rmap[nm]||{}).phrase||nm)} <button data-rm="${esc(nm)}" aria-label="remove">✕</button></li>`).join('');
 $('seed').value=P[tab].seed;
}
function toggle(name){const on=P[tab].objects,i=on.indexOf(name),hi=D.rules[tab].count[1];
 if(i>=0)on.splice(i,1);else{if(on.length>=hi){state(`at most ${hi} objects in this scene`,true);return;}on.push(name);}
 changed();}
function state(t,bad){$('state').textContent=t;$('state').className=bad?'bad':'';}
function changed(){draw();save();layout();}
async function save(){
 if(!served){try{localStorage.setItem(LS,JSON.stringify(body()));}catch(e){}state('kept in this browser: Download writes the file');return;}
 if(saving){dirty=true;return;}saving=true;dirty=false;
 try{const r=await fetch('api/picks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body())});
  const j=await r.json();state(r.ok?'saved to __FILE__':'not saved: '+j.error,!r.ok);}catch(e){state('not saved: '+e,true);}
 saving=false;if(dirty)save();
}
async function layout(){
 const svg=$('map'),me=++seq;
 if(!served){svg.innerHTML='';$('place').textContent='placement preview needs --serve';return;}
 $('place').className='hint';$('place').textContent='placing…';
 let j;try{const r=await fetch('api/place',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({scene:tab,seed:P[tab].seed,objects:P[tab].objects})});j=await r.json();if(!r.ok)throw j.error;}
 catch(e){if(me===seq){$('place').className='bad';$('place').textContent='placement failed: '+e;}return;}
 if(me!==seq)return;
 const s=j.surface,W=520,H=400,k=Math.min((W-20)/(2*s.half[0]),(H-20)/(2*s.half[1]));
 const X=x=>W/2+(x-s.xy[0])*k,Y=y=>H/2-(y-s.xy[1])*k;
 const hx=s.half[0]-s.reach,hy=s.half[1]-s.reach;
 let g=`<rect x="${X(s.xy[0]-s.half[0])}" y="${Y(s.xy[1]+s.half[1])}" width="${2*s.half[0]*k}" height="${2*s.half[1]*k}" fill="var(--band)" stroke="var(--line)"/>`;
 if(hx>0&&hy>0)g+=`<rect x="${X(s.xy[0]-hx)}" y="${Y(s.xy[1]+hy)}" width="${2*hx*k}" height="${2*hy*k}" fill="var(--plat)"/>`;
 j.items.forEach(it=>{const deg=-it.yaw*180/Math.PI,w=2*it.half[0]*k,h=2*it.half[1]*k,cx=X(it.xy[0]),cy=Y(it.xy[1]);
  const i=P[tab].objects.indexOf(it.name);
  g+=`<g transform="translate(${cx} ${cy}) rotate(${deg})"><rect x="${-w/2}" y="${-h/2}" width="${w}" height="${h}" rx="2"
  fill="${it.carton?'var(--box)':'var(--'+D.rules[tab].kind+')'}" fill-opacity="${it.carton?0.9:0.75}" stroke="var(--ink)" stroke-width="0.8"><title>${esc(it.phrase)}</title></rect></g>`;
  if(!it.carton)g+=`<text x="${cx}" y="${cy+4}" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">${i+1}</text>`;});
 svg.innerHTML=g;
 if(j.ok){$('place').className='hint';$('place').textContent=`all placed on the reach band`+(j.attempt?` (placement try ${j.attempt+1})`:'')+'; the build still checks the robot can start each task';}
 else{$('place').className='bad';$('place').textContent=j.error+' — drop a large object or try another seed';}
}
$('grid').addEventListener('click',ev=>{const c=ev.target.closest('.card');if(c)toggle(c.dataset.name);});
$('list').addEventListener('click',ev=>{const b=ev.target.closest('[data-rm]');if(b){const on=P[tab].objects;on.splice(on.indexOf(b.dataset.rm),1);changed();}});
for(const b of document.querySelectorAll('.tabs button'))b.onclick=()=>{tab=b.dataset.tab;cats();draw();layout();};
for(const id of ['q','cat','show','sort'])$(id).addEventListener('input',draw);
$('seed').addEventListener('change',()=>{const v=Math.max(0,parseInt($('seed').value||'0',10)||0);P[tab].seed=v;changed();});
$('dl').onclick=()=>{const b=new Blob([JSON.stringify(body(),null,1)+'\\n'],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='__FILE__';a.click();};
(async()=>{
 try{const r=await fetch('api/picks',{cache:'no-store'});if(r.ok){const j=await r.json();for(const w in j.scenes)P[w]=j.scenes[w];served=true;}}catch(e){}
 if(!served){$('dl').style.display='';try{const v=JSON.parse(localStorage.getItem(LS)||'null');if(v&&v.scenes)for(const w in v.scenes)P[w]=v.scenes[w];}catch(e){}}
 cats();draw();layout();
})();
</script></body></html>
"""
