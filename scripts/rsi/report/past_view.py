"""TODO 39.7 (user 2026-11-02): a round's PAST scene-graph records (`<label>.past.jsonl`, TODO 39) as one self-contained
HTML page for people -- not a role's tool (scripts/rsi/report/ is stripped from every sandbox).

Per step: a top-down map in world xy (left) -- surfaces, furniture, the stance disc and the approach corridor at their
real size, objects as boxes, hands as dots, the robot with its heading -- with the step's start faint and its end solid,
a moved object's path as an arrow, and the relations that have a place drawn between their entities; the relations as
a list grouped by class (right) with before -> after, numbers and the expectation's mark; the events in between under
the map. The view is fixed for the whole round (the room's shell left out), with a button to zoom to the step; a
toggle keeps only what flipped or missed its expectation. `--compare` puts a second round beside it, steps aligned by
primitive kind (as `past.py compare`), the edges whose values differ marked in both. `--frames` adds a MuJoCo frame of
each step's start and end from the round's .npz (needs EGL; on SRA0889 MUJOCO_EGL_DEVICE_ID=0).

Schema 0.2 records (2026-11-06) carry each object's `affordance`: the page opens with one row per object (the scene
file's labels, the one-hand jaw check with its angles in the object's frame, the hug's spans and mass, a container's
opening, the scene grasp height and capability, each field's source), the map's tooltips name an object's grasp kind
and mass, and a step lists the objects whose one-hand check is stale (tilted since the start). 0.1 records show none.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/report/past_view.py <diag dir>/<label> [--compare <round>] [--frames] [--out <page.html>]

Records written before the 2026-11-02 recorder fix flag the room's shell and the target's own furniture as blocking the
stance; the page drops those edges and their expectation misses.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(REPO / "src")]

from locomani.tasks.scene import past as PAST  # noqa: E402

SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "past_view"
PAD_M = 0.3
FRAME_SIZE = (640, 480)

# relation -> class; classes colour the edges and group the list
REL = {"touching": "contact", "gripped": "holding", "held": "holding", "rel_pose": "holding", "supported_by": "support",
       "in": "support", "above": "spatial", "beside": "spatial", "near_edge": "spatial", "blocking": "spatial",
       "dist": "spatial", "facing": "spatial", "reach_vec": "relpose", "approach": "relpose", "at": "relpose",
       "upright": "predicate", "moved": "event", "slipped": "event"}  # fmt: skip
CLASSES = ("contact", "holding", "support", "spatial", "relpose", "predicate", "event")
TYPES = ("object", "effector", "robot", "surface", "furniture", "region")

# every word the page shows, per language (`--lang`; past_view_en.py is the English entry point)
LANG = {
    "zh": dict(
        html_lang="zh",
        gloss={
            "touching": "接触：两个刚体在这一刻有接触点（MuJoCo 接触）",
            "gripped": "两指夹住：这只手的两根手指都碰到物体",
            "held": "持握（谓词）：夹住且离开所有台面",
            "rel_pose": "物体在手里的偏置（手腕减物体位置，m）",
            "supported_by": "支撑：物体底面在台面顶 ±2 cm 且在台面足迹内、未被持",
            "in": "在容器内（谓词）",
            "above": "上方净空：目标正上方 0.5 m 内的面，gap 是到目标顶面的距离",
            "beside": "邻居：0.30 m 内的物体，gap 是两个包围盒之间的缝，side 是在目标的哪一侧",
            "near_edge": "离台沿：目标到最近一条台沿的距离",
            "blocking": "阻挡：包围盒和站位圆盘或接近走廊相交，depth 是相交深度",
            "dist": "机器人骨盆到目标的水平距离，bearing 是目标相对朝向的方位角",
            "facing": "机器人朝向和目标方位的夹角（度）",
            "reach_vec": "伸手目标点在骨盆系的前伸 / 朝手外侧 / 高度（m）",
            "approach": "手从物体的哪一面接近",
            "at": "到位：手（夹爪中心）或机器人到目标点的误差",
            "upright": "竖直（谓词）：物体竖直轴偏离不超过 15°",
            "moved": "物体位移超过 2 cm",
            "slipped": "被持物体和手腕的距离变化超过 1 cm",
        },
        cls={
            "contact": "接触",
            "holding": "持握",
            "support": "支撑",
            "spatial": "空间",
            "relpose": "相对位姿",
            "predicate": "谓词",
            "event": "事件",
        },  # fmt: skip
        type={
            "object": "物体",
            "effector": "手",
            "robot": "机器人",
            "surface": "台面",
            "furniture": "家具",
            "region": "区域",
        },
        src={"observed": "观测", "predicate": "谓词", "derived": "几何推导", "approx": "近似", "unknown": "未知"},
        unknown="未知",
        sep="，",
        colon="：",
        lp="（",
        rp="）",
        gap="　",
        semi="；",
        zoom_in="放大到这一步",
        zoom_out="整局视野",
        only="只看翻转 / 差异",
        theme="切换深浅色",
        legend_ent="实体",
        legend_rel="关系",
        leg_dash="虚线 = 未知",
        leg_bold="加粗 = 这一步翻转",
        leg_red="红 = 期待没成立",
        leg_faint="淡色 = 步骤开始时的位置",
        at_start="开始时",
        above="上方 {gap:.2f} m：{s}",
        expect="期待",
        observed="观测",
        no_rel="这一步没有关系。",
        head=("主体", "关系", "客体", "前 → 后", "数值", "来源", "期待"),
        no_ev="这一步中途没有翻转。",
        chatter="（抖动 {n} 次）",
        joined="{id} 加入范围（{why}）",
        events="中途事件",
        ok="成功",
        fail="失败",
        roles="角色",
        check="检查值",
        evidence="证据：{npz} 拍 {k0}–{k1}",
        task="任务",
        round_ok="整局 成功",
        round_fail="整局 失败",
        no_records="没有记录",
        no_pair="这一局没有对应的步骤。",
        one_side="（单边）",
        n_diff=" ·差 {n}",
        cmp_note="步骤按原语种类对齐（最长公共子序列）；两边值不同的边标黄。",
        f_start="开始",
        f_end="结束",
        f_cap="{step} {which}（拍 {k}）",
        afford="物体可供性",
        afford_note="来自场景文件和几何检查（每个物体取它第一次出现的那一步）；夹爪角度在物体坐标系，从物体 +x 轴方向接近为 0°。",
        afford_head=("物体", "可抓方式", "单手夹爪", "双手抱", "容器口", "抓取高度", "能力表", "来源"),
        grasp={"single": "单手", "bimanual": "双手", "none": "不可抓"},
        jaw_below="夹爪高度处没有截面",
        jaw_fits="可从 {runs}° 夹",
        jaw_none="任何方向都夹不下",
        narrowest="最窄 {w} cm @ {deg:+d}°（每侧 {m:+.1f} cm）",
        rim_pinch="可夹沿（壁厚 {w} cm）",
        top_pinch="可从顶部夹 {w} cm",
        grasp_z="抓取点高出原点 {z:+.2f} m",
        oh_error="单手检查失败：{e}",
        hug="跨度 x {x} / y {y} cm，高 {h} cm",
        opening="{x}×{y} cm",
        rim_h="沿高 {r:.2f} m",
        in_band="在伸手高度带内",
        out_band="不在伸手高度带内",
        stale="可供性已过时（物体倾斜了，单手检查用的是开始时的位姿）：",
        stale_item="{e} 倾斜 {d:.0f}°",
    ),
    "en": dict(
        html_lang="en",
        gloss={
            "touching": "contact: the two bodies touch at this instant (MuJoCo contact)",
            "gripped": "gripped: both fingers of this hand touch the object",
            "held": "held (predicate): gripped and off every surface",
            "rel_pose": "the object's offset in the hand (wrist minus object position, m)",
            "supported_by": "support: the object's underside within ±2 cm of the surface top, inside its footprint, not held",
            "in": "inside the container (predicate)",
            "above": "overhang: a surface within 0.5 m above the target; gap is the clearance to the target's top",
            "beside": "neighbour: an object within 0.30 m; gap is the space between the boxes, side is which side of the target",
            "near_edge": "edge distance: from the target to the nearest edge of its surface",
            "blocking": "blocking: the box intersects the stance disc or the approach corridor; depth is how far",
            "dist": "horizontal distance from the pelvis to the target; bearing is the target's angle off the heading",
            "facing": "angle between the robot's heading and the target's bearing (deg)",
            "reach_vec": "reach target in the pelvis frame: forward / outward of the hand / height (m)",
            "approach": "the face of the object the hand comes from",
            "at": "landing: the hand's (jaw centre's) or the robot's error to its target point",
            "upright": "upright (predicate): the object's vertical axis within 15°",
            "moved": "the object moved more than 2 cm",
            "slipped": "the held object's distance to the wrist changed by more than 1 cm",
        },
        cls={
            "contact": "contact",
            "holding": "holding",
            "support": "support",
            "spatial": "spatial",
            "relpose": "relative pose",
            "predicate": "predicate",
            "event": "event",
        },  # fmt: skip
        type={
            "object": "object",
            "effector": "hand",
            "robot": "robot",
            "surface": "surface",
            "furniture": "furniture",
            "region": "region",
        },  # fmt: skip
        src={
            "observed": "observed",
            "predicate": "predicate",
            "derived": "geometry",
            "approx": "approximate",
            "unknown": "unknown",
        },  # fmt: skip
        unknown="unknown",
        sep=", ",
        colon=": ",
        lp=" (",
        rp=")",
        gap="  ",
        semi="; ",
        zoom_in="Zoom to this step",
        zoom_out="Whole round",
        only="Only flips / misses",
        theme="Light / dark",
        legend_ent="Entities",
        legend_rel="Relations",
        leg_dash="dashed = unknown",
        leg_bold="bold = flipped in this step",
        leg_red="red = expectation not met",
        leg_faint="faint = where it was when the step began",
        at_start="at the step's start",
        above="{gap:.2f} m above: {s}",
        expect="expected",
        observed="observed",
        no_rel="No relations in this step.",
        head=("subject", "relation", "object", "before → after", "values", "source", "expected"),
        no_ev="Nothing flipped during this step.",
        chatter=" (flickered {n} times)",
        joined="{id} joined the scope ({why})",
        events="Events during the step",
        ok="OK",
        fail="FAILED",
        roles="Roles",
        check="Checks",
        evidence="Evidence: {npz}, ticks {k0}–{k1}",
        task="Task",
        round_ok="round succeeded",
        round_fail="round failed",
        no_records="no records",
        no_pair="No matching step in this round.",
        one_side=" (one side only)",
        n_diff=" · {n} differ",
        cmp_note="Steps aligned by primitive kind (longest common subsequence); edges whose values differ are marked yellow.",
        f_start="start",
        f_end="end",
        f_cap="{step} {which} (tick {k})",
        afford="Object affordances",
        afford_note="From the scene file and the geometry checks (each object from the first step it appears in); jaw angles in the object's frame, an approach from its +x axis is 0°.",
        afford_head=(
            "object",
            "grasp",
            "one-hand jaw",
            "two-hand hug",
            "opening",
            "grasp height",
            "capability",
            "source",
        ),
        grasp={"single": "one hand", "bimanual": "two hands", "none": "not graspable"},
        jaw_below="nothing at the jaw height",
        jaw_fits="jaw fits from {runs}°",
        jaw_none="jaw fits from no approach",
        narrowest="narrowest {w} cm at {deg:+d}° ({m:+.1f} cm a side)",
        rim_pinch="rim pinch ({w} cm wall)",
        top_pinch="top pinch {w} cm",
        grasp_z="grasp point {z:+.2f} m above the origin",
        oh_error="one-hand check failed: {e}",
        hug="spans x {x} / y {y} cm, {h} cm tall",
        opening="{x}×{y} cm",
        rim_h="rim at {r:.2f} m",
        in_band="in the reach band",
        out_band="outside the reach band",
        stale="Affordance stale (the object tilted; the one-hand check is from the start pose): ",
        stale_item="{e} tilted {d:.0f}°",
    ),
}
T = LANG["zh"]


def set_lang(lang: str) -> None:
    global T
    T = LANG[lang]


def gloss(r: str) -> str:
    return T["gloss"].get(r, r)


DRAWN = {"touching", "gripped", "held", "supported_by", "in", "beside", "blocking", "approach"}  # between two entities
FROM_PELVIS = {"dist", "facing", "reach_vec", "at"}  # arrows from the robot or the hand

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--dim:#6b7079;--line:#d9dbe0;--panel:#ffffff;--warn:#d4372c;--diff:#e2a400;
--object:#2f6fdb;--effector:#d0499a;--robot:#4b5563;--surface:#a7b0bd;--furniture:#c9b79c;--region:#2aa198;
--contact:#e8590c;--holding:#c2255c;--support:#2f9e44;--spatial:#1c7ed6;--relpose:#7048e8;--predicate:#868e96;--event:#e8590c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16181d;--fg:#e7e9ee;--dim:#9aa1ad;
--line:#2c3038;--panel:#1d2026;--surface:#56606e;--furniture:#6e6250;--robot:#aab2bf}}
:root[data-theme="dark"]{--bg:#16181d;--fg:#e7e9ee;--dim:#9aa1ad;--line:#2c3038;--panel:#1d2026;--surface:#56606e;
--furniture:#6e6250;--robot:#aab2bf}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:16px 0 6px}.dim{color:var(--dim)}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}.bar button{border:1px solid var(--line);background:var(--panel);
color:var(--fg);padding:4px 9px;border-radius:6px;cursor:pointer;font:inherit}.bar button.on{border-color:var(--fg);font-weight:600}
.bar button.fail{color:var(--warn)}.step{display:none}.step.on{display:block}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}@media (max-width:820px){.grid{grid-template-columns:1fr}}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media (max-width:820px){.pair{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
svg.map{width:100%;height:auto;background:var(--bg);border:1px solid var(--line);border-radius:6px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}td.wrap{white-space:normal;min-width:9em}
th{color:var(--dim);font-weight:500}tr.disc td{color:var(--warn)}tr.diff td{background:color-mix(in srgb,var(--diff) 18%,transparent)}
.hideplain tr.plain,.hideplain .e.plain{display:none}.mark{font-weight:700}.ok{color:var(--support)}.bad{color:var(--warn)}
.legend span{display:inline-block;margin-right:10px;white-space:nowrap}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.frames{display:flex;gap:8px;flex-wrap:wrap}.frames figure{margin:0;flex:1 1 280px}.frames img{width:100%;border-radius:6px}
.frames figcaption{font-size:12px;color:var(--dim)}ul.ev{margin:4px 0;padding-left:18px}abbr{text-decoration:none;border-bottom:1px dotted var(--dim)}
"""

JS = """
function show(sec, i){document.querySelectorAll('#'+sec+' .step').forEach((s,k)=>s.classList.toggle('on',k==i));
document.querySelectorAll('#'+sec+' .bar.steps button').forEach((b,k)=>b.classList.toggle('on',k==i));}
function zoom(btn){const svg=btn.closest('.step').querySelector('svg.map');const z=svg.dataset.zoomed=='1';
svg.setAttribute('viewBox',z?svg.dataset.round:svg.dataset.step);svg.dataset.zoomed=z?'0':'1';btn.textContent=z?btn.dataset.in:btn.dataset.out;}
function only(btn){document.body.classList.toggle('hideplain');btn.classList.toggle('on');}
window.addEventListener('DOMContentLoaded',()=>{const m=location.hash.match(/^#(\d+)$/);
const sec=document.querySelector('section');if(m&&sec)show(sec.id,+m[1]);});
function theme(btn){const r=document.documentElement;r.dataset.theme=r.dataset.theme=='dark'?'light':'dark';}
"""


# -- reading --------------------------------------------------------------------------------------------------------
def side_file(p: Path) -> Path:
    p = Path(p)
    if p.name.endswith(".past.jsonl"):
        return p
    return p.with_name((p.stem if p.suffix == ".json" else p.name) + ".past.jsonl")


def load(p: Path) -> tuple[list[dict], list[dict]]:
    """(records, note / error lines) of a round."""
    lines = PAST.read(side_file(p))
    return [r for r in lines if "record_id" in r], [x for x in lines if "record_id" not in x]


def edge_key(e: dict) -> tuple:
    return (e["s"], e["r"], e.get("o"))


def legacy_blockers(rec: dict, meta: dict) -> set[str]:
    """Entities a record from before the 2026-11-02 recorder fix marks as blocking but never block: the room's shell
    and the furniture the target stands on."""
    out = {k for k in (rec.get("entities") or {}) if PAST.is_shell(meta, k)}
    sup = (rec.get("roles") or {}).get("support")
    of = (meta.get("surfaces") or {}).get(sup, {}).get("of_object") if sup else None
    return out | ({of} if of else set())


def clean(rec: dict, meta: dict) -> dict:
    """The record without the legacy blocking false positives (edges and the expectation misses they caused)."""
    drop = legacy_blockers(rec, meta)
    if not drop:
        return rec
    r = dict(rec)
    for g in ("G_before", "G_after"):
        r[g] = dict(rec[g], edges=[e for e in rec[g]["edges"] if not (e["r"] == "blocking" and e["s"] in drop)])
    left = {e["o"] for e in r["G_after"]["edges"] if e["r"] == "blocking" and e.get("v")}
    r["discrepancies"] = [d for d in rec.get("discrepancies") or []
                          if not (d["edge"][1] == "blocking" and d["edge"][0] == "*" and d["edge"][2] not in left)]  # fmt: skip
    r["events"] = [e for e in rec.get("events") or [] if not (e["edge"][1] == "blocking" and e["edge"][0] in drop)]
    return r


# -- geometry -------------------------------------------------------------------------------------------------------
def quat_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def shape_of(eid: str, ent: dict, state: dict | None, meta: dict) -> dict | None:
    """World-xy drawing of one entity: kind (rect / disc / dot / seg / robot), centre, half sizes, yaw."""
    t = ent.get("type")
    objs, surfs = meta.get("objects") or {}, meta.get("surfaces") or {}
    if t == "object" and state and "p" in state:
        half = ent.get("half_extent_m") or (objs.get(eid, {}).get("box") or {}).get("half_extent") or [0.04, 0.04]
        return dict(kind="rect", c=state["p"][:2], half=half[:2], yaw=quat_yaw(state.get("q") or (1, 0, 0, 0)))
    if t == "effector" and state and "p" in state:
        return dict(kind="dot", c=state["p"][:2], r=0.03)
    if t == "robot" and state and "pelvis_p" in state:
        return dict(kind="robot", c=state["pelvis_p"][:2], yaw=float(state.get("yaw") or 0.0), r=0.18)
    if t == "surface" and eid in surfs:
        s = surfs[eid]
        return dict(
            kind="rect",
            c=s["xy"],
            half=(s.get("flat_half_extent") or s["half_extent"])[:2],
            yaw=float(s.get("yaw") or 0),
        )
    if t == "furniture" and eid in objs and objs[eid].get("box"):
        o = objs[eid]
        c = [o["box"]["centre"][i] + (o.get("init_pos") or (0, 0, 0))[i] for i in (0, 1)]
        return dict(
            kind="rect", c=c, half=o["box"]["half_extent"][:2], yaw=quat_yaw(o.get("init_quat") or (1, 0, 0, 0))
        )
    if t == "region":
        if ent.get("shape") == "disc":
            return dict(kind="disc", c=ent["centre"][:2], r=ent["radius_m"])
        if ent.get("shape") == "segment":
            return dict(kind="seg", a=ent["a"][:2], b=ent["b"][:2], w=ent["half_width_m"])
        if ent.get("shape") == "point":
            return dict(kind="dot", c=ent["p"][:2], r=0.02)
        if ent.get("shape") == "rect":
            return dict(kind="rect", c=ent["centre"][:2], half=ent["half_extent_m"][:2], yaw=float(ent.get("yaw") or 0))
    return None


def corners(sh: dict) -> list[tuple[float, float]]:
    c, s = math.cos(sh["yaw"]), math.sin(sh["yaw"])
    hx, hy = float(sh["half"][0]), float(sh["half"][1])
    return [
        (sh["c"][0] + c * x - s * y, sh["c"][1] + s * x + c * y)
        for x, y in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    ]


def extent(sh: dict) -> list[tuple[float, float]]:
    k = sh["kind"]
    if k == "rect":
        return corners(sh)
    if k == "seg":
        return [tuple(sh["a"]), tuple(sh["b"])]
    r = sh.get("r", 0.05)
    return [(sh["c"][0] - r, sh["c"][1] - r), (sh["c"][0] + r, sh["c"][1] + r)]


def view_box(shapes: list[dict]) -> str:
    pts = [p for sh in shapes for p in extent(sh)]
    if not pts:
        return "-1 -1 2 2"
    xs, ys = [p[0] for p in pts], [-p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - PAD_M, max(xs) + PAD_M, min(ys) - PAD_M, max(ys) + PAD_M
    w, h = max(x1 - x0, 0.6), max(y1 - y0, 0.6)
    return f"{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}"


def centre(sh: dict) -> tuple[float, float]:
    if sh["kind"] == "seg":
        return ((sh["a"][0] + sh["b"][0]) / 2, (sh["a"][1] + sh["b"][1]) / 2)
    return (float(sh["c"][0]), float(sh["c"][1]))


def round_shapes(recs: list[dict], meta: dict) -> list[dict]:
    """Every entity of every step at its start and end: the round's fixed view (the room's shell left out)."""
    out = []
    for r in recs:
        for eid, ent in (r.get("entities") or {}).items():
            if PAST.is_shell(meta, eid):
                continue
            for g in ("G_before", "G_after"):
                sh = shape_of(eid, ent, (r[g].get("states") or {}).get(eid), meta)
                if sh:
                    out.append(sh)
    return out


# -- SVG ------------------------------------------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def svg_shape(sh: dict, cls: str, title: str, faint: bool) -> str:
    op = ' opacity="0.35"' if faint else ""
    t = f"<title>{esc(title)}</title>"
    k = sh["kind"]
    if k == "rect":
        pts = " ".join(f"{x:.3f},{-y:.3f}" for x, y in corners(sh))
        return f'<polygon class="n {cls}" points="{pts}"{op}>{t}</polygon>'
    if k == "disc":
        return (
            f'<circle class="n {cls}" cx="{sh["c"][0]:.3f}" cy="{-sh["c"][1]:.3f}" r="{sh["r"]:.3f}"{op}>{t}</circle>'
        )
    if k == "dot":
        return (
            f'<circle class="n {cls}" cx="{sh["c"][0]:.3f}" cy="{-sh["c"][1]:.3f}" r="{sh["r"]:.3f}"{op}>{t}</circle>'
        )
    if k == "seg":
        (ax, ay), (bx, by) = sh["a"], sh["b"]
        return (f'<line class="n {cls}" x1="{ax:.3f}" y1="{-ay:.3f}" x2="{bx:.3f}" y2="{-by:.3f}" '
                f'stroke-width="{2 * sh["w"]:.3f}" stroke-linecap="round"{op}>{t}</line>')  # fmt: skip
    if k == "robot":
        x, y, r, yaw = sh["c"][0], sh["c"][1], sh["r"], sh["yaw"]
        hx, hy = x + 1.6 * r * math.cos(yaw), y + 1.6 * r * math.sin(yaw)
        return (f'<g{op}><circle class="n {cls}" cx="{x:.3f}" cy="{-y:.3f}" r="{r:.3f}">{t}</circle>'
                f'<line class="heading" x1="{x:.3f}" y1="{-y:.3f}" x2="{hx:.3f}" y2="{-hy:.3f}"/></g>')  # fmt: skip
    return ""


STYLE_NODES = {"object": "fill:var(--object);fill-opacity:.85;stroke:var(--object)",
               "effector": "fill:var(--effector)", "robot": "fill:var(--robot);fill-opacity:.25;stroke:var(--robot)",
               "surface": "fill:var(--surface);fill-opacity:.35;stroke:var(--surface)",
               "furniture": "fill:var(--furniture);fill-opacity:.35;stroke:var(--furniture)",
               "region": "fill:var(--region);fill-opacity:.12;stroke:var(--region);stroke-dasharray:.04 .03"}  # fmt: skip


def edge_classes(e: dict, flips: set, bad: set, diff: set) -> str:
    k = edge_key(e)
    tags = ["e", REL.get(e["r"], "predicate")]
    hot = False
    if k in flips:
        tags.append("flip")
        hot = True
    wild = {(b[1], b[2]) for b in bad if b[0] == "*"}  # "nothing blocks the stance": every blocker is the miss
    if k in bad or ((e["r"], e.get("o")) in wild and e.get("v")):
        tags.append("disc")
        hot = True
    if k in diff:
        tags.append("diff")
        hot = True
    if not hot:
        tags.append("plain")
    if e.get("v") is None:
        tags.append("unk")
    return " ".join(tags)


def step_svg(rec: dict, meta: dict, vb_round: str, diff: set) -> str:
    ents = rec.get("entities") or {}
    sb, sa = rec["G_before"].get("states") or {}, rec["G_after"].get("states") or {}
    after, parts, labels = {}, [], []
    order = ["region", "surface", "furniture", "object", "effector", "robot"]
    shapes_step = []
    for typ in order:
        for eid, ent in ents.items():
            if ent.get("type") != typ or PAST.is_shell(meta, eid):
                continue
            b = shape_of(eid, ent, sb.get(eid), meta)
            a = shape_of(eid, ent, sa.get(eid), meta) or b
            if not a:
                continue
            after[eid] = a
            shapes_step.append(a)
            name = f"{eid}{T['lp']}{T['type'].get(typ, typ)}{T['rp']}" + afford_tip(ent)
            moving = typ in ("object", "effector", "robot")
            if moving and b and math.dist(centre(a), centre(b)) > 0.01:
                parts.append(svg_shape(b, typ, f"{name} {T['at_start']}", True))
                (x0, y0), (x1, y1) = centre(b), centre(a)
                parts.append(
                    f'<line class="move" x1="{x0:.3f}" y1="{-y0:.3f}" x2="{x1:.3f}" y2="{-y1:.3f}" marker-end="url(#arr)"/>'
                )
            parts.append(svg_shape(a, typ, name, False))
            drawn_of = (meta.get("surfaces") or {}).get(eid, {}).get("of_object") if typ == "surface" else None
            if typ in ("object", "surface", "furniture", "robot") and not (drawn_of and drawn_of in ents):
                cx, cy = centre(a)
                # objects above their box, surfaces and furniture under theirs, so a thing and what it stands on part
                dy = -0.05 if typ in ("object", "robot") else (max(float(a.get("half", (0, 0.1))[1]), 0.1) + 0.08)
                labels.append(f'<text class="lbl {typ}" x="{cx:.3f}" y="{-cy + dy:.3f}">{esc(eid)}</text>')
    flips = {tuple(e["edge"]) for e in rec.get("events") or []}
    bad = {tuple(d["edge"]) for d in rec.get("discrepancies") or []}
    tgt = (rec.get("roles") or {}).get("target")
    lines = []
    for e in rec["G_after"]["edges"]:
        r, s, o = e["r"], e["s"], e.get("o")
        cls = edge_classes(e, flips, bad, diff)
        tip = f"<title>{esc(s)} —{esc(r)}→ {esc(o)}{T['colon']}{esc(fmt_v(e))}{T['lp']}{esc(gloss(r))}{T['rp']}</title>"
        if r in DRAWN and s in after and o in after and (e.get("v") or e.get("v") is None):
            (x0, y0), (x1, y1) = centre(after[s]), centre(after[o])
            lines.append(f'<line class="{cls}" x1="{x0:.3f}" y1="{-y0:.3f}" x2="{x1:.3f}" y2="{-y1:.3f}">{tip}</line>')
        elif r in ("dist", "reach_vec", "at") and s in after and o in after:
            (x0, y0), (x1, y1) = centre(after[s]), centre(after[o])
            lines.append(
                f'<line class="{cls} arrow" x1="{x0:.3f}" y1="{-y0:.3f}" x2="{x1:.3f}" y2="{-y1:.3f}" marker-end="url(#arr)">{tip}</line>'
            )
        elif r == "above" and e.get("v") and tgt in after:
            cx, cy = centre(after[tgt])
            labels.append(
                f'<text class="{cls}" x="{cx + 0.05:.3f}" y="{-cy + 0.12:.3f}">{esc(T["above"].format(gap=e.get("gap_m", 0), s=s))}{tip}</text>'
            )
    vb_step = view_box(shapes_step)
    defs = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
            '<path d="M0,0L10,5L0,10z" fill="currentColor"/></marker></defs>')  # fmt: skip
    style = "".join(f".n.{k}{{{v}}}" for k, v in STYLE_NODES.items())
    style += (".n{stroke-width:.012}.heading{stroke:var(--robot);stroke-width:.02}.move{stroke:var(--dim);stroke-width:.012;color:var(--dim)}"
              "text.lbl.surface,text.lbl.furniture{fill:var(--dim);font-size:.06px}text{font-size:.07px;fill:var(--fg);paint-order:stroke;stroke:var(--bg);stroke-width:.012}"
              ".e{stroke-width:.018;fill:none}.e.unk{stroke-dasharray:.03 .02;stroke:var(--predicate)!important}"
              ".e.flip{stroke-width:.04}.e.disc{stroke:var(--warn)!important;stroke-width:.028}.e.diff{stroke:var(--diff)!important}"
              ".arrow{color:currentColor}"
              + "".join(f".e.{c}{{stroke:var(--{c});color:var(--{c})}}text.e.{c}{{fill:var(--{c});stroke:var(--bg)}}" for c in CLASSES))  # fmt: skip
    return (
        f'<svg class="map" viewBox="{vb_round}" data-round="{vb_round}" data-step="{vb_step}" '
        f'xmlns="http://www.w3.org/2000/svg"><style>{style}</style>{defs}'
        + "".join(parts)
        + "".join(lines)
        + "".join(labels)
        + "</svg>"
    )


# -- the relation list ------------------------------------------------------------------------------------------------
def fmt_v(e: dict | None) -> str:
    if e is None:
        return "-"
    v = e.get("v")
    if v is None:
        return T["unknown"]
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v, ensure_ascii=False)


def nums(e: dict) -> str:
    return T["sep"].join(f"{k} {v}" for k, v in e.items() if k not in ("s", "r", "o", "v", "src"))


def relation_table(rec: dict, diff: set) -> str:
    before = {edge_key(e): e for e in rec["G_before"]["edges"]}
    after = {edge_key(e): e for e in rec["G_after"]["edges"]}
    exp = {tuple(x["edge"]): x for x in rec.get("expected") or []}
    bad = {tuple(d["edge"]) for d in rec.get("discrepancies") or []}
    flips = {tuple(e["edge"]) for e in rec.get("events") or []}
    rows = {}
    for k in set(before) | set(after):
        b, a = before.get(k), after.get(k)
        e = a or b
        cls = REL.get(k[1], "predicate")
        mark = ""
        if k in exp:
            mark = f'<span class="mark {"bad" if k in bad else "ok"}">{T["expect"]} {fmt_v(dict(v=exp[k]["to"]))} {"✗" if k in bad else "✓"}</span>'
        hot = k in bad or k in flips or k in diff
        rc = " ".join(x for x in ("disc" if k in bad else "", "diff" if k in diff else "", "" if hot else "plain") if x)
        gl = gloss(k[1])
        rows.setdefault(cls, []).append(
            f'<tr class="{rc}"><td>{esc(k[0])}</td><td><abbr title="{esc(gl)}">{esc(k[1])}</abbr></td><td>{esc(k[2] or "")}</td>'
            f"<td>{esc(fmt_v(b))} → {esc(fmt_v(a))}</td><td class='wrap'>{esc(nums(e))}</td>"
            f'<td>{esc(T["src"].get(e.get("src"), e.get("src")))}</td><td>{mark}</td></tr>'
        )  # fmt: skip
    for d in rec.get("discrepancies") or []:
        if tuple(d["edge"]) not in before and tuple(d["edge"]) not in after:
            s, r, o = d["edge"]
            rows.setdefault(REL.get(r, "predicate"), []).append(
                f'<tr class="disc"><td>{esc(s)}</td><td>{esc(r)}</td><td>{esc(o or "")}</td><td colspan="3">{T["observed"]} {esc(fmt_v(dict(v=d.get("observed"))))}</td>'
                f'<td><span class="mark bad">{T["expect"]} {fmt_v(dict(v=d["to"]))} ✗</span></td></tr>'
            )  # fmt: skip
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in T["head"]) + "</tr>"
    out = []
    for cls in CLASSES:
        if cls in rows:
            out.append(
                f'<h3 class="dim" style="font-size:13px;margin:10px 0 2px">{T["cls"][cls]}</h3><div class="tw"><table>{head}{"".join(sorted(rows[cls]))}</table></div>'
            )
    return "".join(out) or f'<p class="dim">{T["no_rel"]}</p>'


CHATTER_S = 0.3  # flips of one edge this close together are drawn as one line (contacts chatter while a jaw closes)


def merged_events(ev: list[dict]) -> list[dict]:
    """Consecutive flips of the same edge within CHATTER_S as one entry: first time, first from, last to, count."""
    out: list[dict] = []
    for e in ev:
        last = next((x for x in reversed(out) if x["edge"] == e["edge"]), None)
        if last is not None and e["t"] - last["t_end"] <= CHATTER_S and last is out[-1]:
            last.update(to=e["to"], t_end=e["t"], n=last["n"] + 1)
        else:
            out.append(dict(e, t_end=e["t"], n=1))
    return out


def events_html(rec: dict) -> str:
    ev = merged_events(rec.get("events") or [])
    joined = (rec.get("scope") or {}).get("added") or []
    if not ev and not joined:
        return f'<p class="dim">{T["no_ev"]}</p>'
    items = []
    for e in ev:
        when = f"{e['t']:.1f} s" if e["n"] == 1 else f"{e['t']:.1f}–{e['t_end']:.1f} s"
        flick = T["chatter"].format(n=e["n"]) if e["n"] > 1 else ""
        cm = f"{T['lp']}{e['cm']} cm{T['rp']}" if "cm" in e else ""
        items.append(f"<li>{when}{T['gap']}{esc(e['edge'][0])} —{esc(e['edge'][1])}→ {esc(e['edge'][2] or '')}{T['gap']}"
                     f"{esc(fmt_v(dict(v=e['from'])))} → {esc(fmt_v(dict(v=e['to'])))}{cm}{flick}</li>")  # fmt: skip
    items += [f"<li>{x['t']:.1f} s{T['gap']}{esc(T['joined'].format(id=x['id'], why=x['why']))}</li>" for x in joined]
    return f'<ul class="ev">{"".join(items)}</ul>'


def frames_html(fr: dict | None) -> str:
    if not fr:
        return ""
    figs = "".join(f'<figure><img alt="{esc(cap)}" src="data:image/jpeg;base64,{b64}"><figcaption>{esc(cap)}</figcaption></figure>'
                   for cap, b64 in fr)  # fmt: skip
    return f'<div class="frames">{figs}</div>'


# -- pages -----------------------------------------------------------------------------------------------------------
def step_head(r: dict) -> str:
    a = r.get("action") or {}
    k = r["kind"] + ("?" if r.get("kind_src") == "primitive" else "")
    roles = T["sep"].join(f"{k2}={v}" for k2, v in (r.get("roles") or {}).items() if v)
    why = f"{T['gap']}<span class='bad'>{esc(a['why'])}</span>" if a.get("why") else ""
    chk = (
        f"<div class='dim'>{T['check']} {esc(json.dumps(a['check'], ensure_ascii=False))}</div>"
        if a.get("check")
        else ""
    )
    stale = T["sep"].join(T["stale_item"].format(e=e, d=float(ent["affordance"].get("tilt_deg") or 0.0))
                          for e, ent in (r.get("entities") or {}).items() if (ent.get("affordance") or {}).get("stale"))  # fmt: skip
    stale = f"<div class='bad'>{esc(T['stale'] + stale)}</div>" if stale else ""
    g = T["gap"]
    verdict = T["ok"] if a.get("ok") else f"<span class=bad>{T['fail']}</span>"
    return (f"<h2>{r['step_index']}.{r['attempt']} {esc(r['step'])} [{esc(k)}]{g}{a.get('t0', 0):.1f}–{a.get('t1', 0):.1f} s{g}"
            f"{verdict}{why}</h2><div class='dim'>{T['roles']}{T['colon']}{esc(roles)}</div>{chk}{stale}")  # fmt: skip


# -- affordances (schema 0.2) -----------------------------------------------------------------------------------------
def _cm(m) -> str:
    return f"{float(m) * 100:.1f}"


def affordances(recs: list[dict]) -> dict:
    """Each object entity's `affordance`, from the first record it appears in (as `past.py show`)."""
    seen: dict = {}
    for r in recs:
        for eid, ent in (r.get("entities") or {}).items():
            if ent.get("type") == "object" and ent.get("affordance") and eid not in seen:
                seen[eid] = ent["affordance"]
    return seen


def grasp_word(a: dict) -> str:
    g = a.get("graspable")
    return T["unknown"] if g is None else T["grasp"].get(str(g), str(g))


def afford_tip(ent: dict) -> str:
    """The map tooltip's tail for an object with an affordance: its grasp kind and mass."""
    a = ent.get("affordance") if ent.get("type") == "object" else None
    if not a:
        return ""
    mass = (a.get("hug") or {}).get("mass_kg")
    return f"{T['semi']}{grasp_word(a)}" + (f"{T['sep']}{mass} kg" if mass is not None else "")


def jaw_cell(a: dict) -> str:
    oh = a.get("one_hand")
    if not oh:
        return T["oh_error"].format(e=a["one_hand_error"]) if a.get("one_hand_error") else ""
    if oh.get("below_jaw"):
        parts = [T["jaw_below"]]
    elif oh.get("side_fits_deg"):
        parts = [T["jaw_fits"].format(runs=" ".join(f"[{lo:+.0f}, {hi:+.0f}]" for lo, hi in oh["side_fits_deg"]))]
    else:
        parts = [T["jaw_none"]]
    n = oh.get("narrowest")
    if n:
        deg = int(round(n["approach_deg"])) or 0  # no "-0"
        parts.append(T["narrowest"].format(w=_cm(n["width_m"]), deg=deg, m=float(n["margin_m"]) * 100))
    if (oh.get("rim") or {}).get("available"):
        parts.append(T["rim_pinch"].format(w=_cm(oh["rim"]["wall_m"])))
    if (oh.get("top_pinch") or {}).get("fits"):
        parts.append(T["top_pinch"].format(w=_cm(oh["top_pinch"]["width_m"])))
    if oh.get("grasp_z_above_origin_m") is not None:
        parts.append(T["grasp_z"].format(z=float(oh["grasp_z_above_origin_m"])))
    return T["sep"].join(parts)


def afford_row(key: str, a: dict) -> str:
    name = esc(key) + (f"<div class='dim'>{esc(a['phrase'])}</div>" if a.get("phrase") else "")
    if a.get("category"):
        name += f"<div class='dim'>[{esc(a['category'])}]</div>"
    h = a.get("hug")
    hug = ""
    if h:
        hug = T["hug"].format(x=_cm(h["span_x_m"]), y=_cm(h["span_y_m"]), h=_cm(h["height_m"]))
        hug += f"{T['sep']}{h['mass_kg']} kg" if h.get("mass_kg") is not None else ""
    opening = ""
    if a.get("opening_half_extent_m"):
        o = a["opening_half_extent_m"]
        opening = T["opening"].format(x=_cm(2 * o[0]), y=_cm(2 * o[1]))
        opening += T["sep"] + T["rim_h"].format(r=float(a["rim_height_m"])) if a.get("rim_height_m") is not None else ""
    height = "" if a.get("grasp_height_m") is None else f"{float(a['grasp_height_m']):.2f} m"
    if a.get("in_reach_band") is not None:
        height += (T["sep"] if height else "") + (T["in_band"] if a["in_reach_band"] else T["out_band"])
    src = "<br>".join(f"{esc(k)}={esc(v)}" for k, v in (a.get("src") or {}).items())
    cells = (name, esc(grasp_word(a)), esc(jaw_cell(a)), esc(hug), esc(opening), esc(height),
             esc(a.get("capability") or ""), f"<span class='dim'>{src}</span>")  # fmt: skip
    return "<tr>" + "".join(f"<td class='wrap'>{c}</td>" for c in cells) + "</tr>"


def afford_table(recs: list[dict]) -> str:
    """The round's object affordances as one card, or "" for records without them (schema 0.1)."""
    seen = affordances(recs)
    if not seen:
        return ""
    head = "".join(f"<th>{h}</th>" for h in T["afford_head"])
    rows = "".join(afford_row(k, a) for k, a in seen.items())
    return (f"<div class='card afford'><h2 style='margin-top:0'>{T['afford']}</h2><div class='dim'>{T['afford_note']}</div>"
            f"<div class='tw'><table><tr>{head}</tr>{rows}</table></div></div>")  # fmt: skip


def step_block(r: dict, meta: dict, vb: str, diff: set, fr) -> str:
    return (f"{step_head(r)}<div class='grid'><div class='card'>{step_svg(r, meta, vb, diff)}"
            f"<div class='bar'><button onclick='zoom(this)' data-in='{T['zoom_in']}' data-out='{T['zoom_out']}'>{T['zoom_in']}</button></div>"
            f"<h3 class='dim' style='font-size:13px;margin:6px 0 2px'>{T['events']}</h3>{events_html(r)}{frames_html(fr)}</div>"
            f"<div class='card'>{relation_table(r, diff)}</div></div>"
            f"<div class='dim' style='margin-top:6px'>{esc(T['evidence'].format(npz=r['evidence'].get('npz'), k0=r['evidence'].get('k0'), k1=r['evidence'].get('k1')))}</div>")  # fmt: skip


def legend() -> str:
    nodes = "".join(f"<span><i class='sw' style='background:var(--{k})'></i>{T['type'][k]}</span>" for k in TYPES)
    edges = "".join(f"<span><i class='sw' style='background:var(--{k})'></i>{T['cls'][k]}</span>" for k in CLASSES)
    return (f"<div class='legend dim'>{T['legend_ent']}{T['colon']}{nodes}</div><div class='legend dim'>{T['legend_rel']}{T['colon']}{edges}"
            f"<span>{T['leg_dash']}</span><span>{T['leg_bold']}</span><span style='color:var(--warn)'>{T['leg_red']}</span>"
            f"<span>{T['leg_faint']}</span></div>")  # fmt: skip


def page(title: str, sub: str, body: str) -> str:
    return (f"<!doctype html><html lang='{T['html_lang']}'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><h1>{esc(title)}</h1><div class='dim'>{sub}</div>"
            f"<div class='bar'><button onclick='only(this)'>{T['only']}</button><button onclick='theme(this)'>{T['theme']}</button></div>"
            f"{legend()}{body}<script>{JS}</script></body></html>")  # fmt: skip


def outcome_line(recs: list[dict]) -> str:
    o = (recs[-1].get("outcome") or {}) if recs else {}  # the last step carries the round's final verdict
    ok = o.get("round_ok")
    why = o.get("round_why") or ""
    fc = o.get("failure_class") or ""
    head = f"{T['task']} {esc(recs[0]['task'] if recs else '')}{T['gap']}{T['round_ok'] if ok else T['round_fail']}"
    return head + (f"{T['lp']}{esc(fc)}{T['colon']}{esc(why)}{T['rp']}" if not ok else "")


def single(path: Path, meta: dict, frames: dict | None = None) -> str:
    recs, notes = load(path)
    if not recs:
        msg = T["semi"].join(x.get("note") or f"error: {x.get('error')}" for x in notes) or T["no_records"]
        return page(Path(path).name, esc(msg), "")
    recs = [clean(r, meta) for r in recs]
    vb = view_box(round_shapes(recs, meta))
    bar = "".join(f"<button class='{'' if r['action'].get('ok') else 'fail'}' onclick=\"show('s',{i})\">"
                  f"{r['step_index']}.{r['attempt']} {esc(r['step'])}</button>" for i, r in enumerate(recs))  # fmt: skip
    steps = "".join(f"<div class='step{' on' if i == 0 else ''}'>{step_block(r, meta, vb, set(), (frames or {}).get(r['record_id']))}</div>"
                    for i, r in enumerate(recs))  # fmt: skip
    return page(
        recs[0]["label"],
        outcome_line(recs),
        f"{afford_table(recs)}<section id='s'><div class='bar steps'>{bar}</div>{steps}</section>",
    )


def lcs(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out: list = []
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append((i, None))
            i += 1
        else:
            out.append((None, j))
            j += 1
    return out + [(k, None) for k in range(i, n)] + [(None, k) for k in range(j, m)]


def differing(ra: dict, rb: dict) -> set:
    ea = {edge_key(e): e.get("v") for e in ra["G_after"]["edges"]}
    eb = {edge_key(e): e.get("v") for e in rb["G_after"]["edges"]}
    return {k for k in set(ea) | set(eb) if ea.get(k, "-") != eb.get(k, "-")}


def compare(a: Path, b: Path, meta: dict, frames: dict | None = None) -> str:
    ra, _ = load(a)
    rb, _ = load(b)
    ra, rb = [clean(r, meta) for r in ra], [clean(r, meta) for r in rb]
    va, vbb = view_box(round_shapes(ra, meta)), view_box(round_shapes(rb, meta))
    pairs = lcs([r["kind"] for r in ra], [r["kind"] for r in rb])
    blocks, bar = [], []
    for n, (i, j) in enumerate(pairs):
        x, y = (ra[i] if i is not None else None), (rb[j] if j is not None else None)
        diff = differing(x, y) if x and y else set()
        cell = lambda r, vb: (step_block(r, meta, vb, diff, (frames or {}).get(r["record_id"])) if r  # noqa: E731
                              else f"<p class='dim'>{T['no_pair']}</p>")  # fmt: skip
        kind = (x or y)["kind"]
        bar.append(
            f"<button onclick=\"show('c',{n})\">{n + 1}. {esc(kind)}{'' if x and y else T['one_side']}{T['n_diff'].format(n=len(diff)) if diff else ''}</button>"
        )
        blocks.append(f"<div class='step{' on' if n == 0 else ''}'><div class='pair'><div><h2 class='dim'>A{T['gap']}{esc(ra[0]['label'] if ra else a)}</h2>{cell(x, va)}</div>"
                      f"<div><h2 class='dim'>B{T['gap']}{esc(rb[0]['label'] if rb else b)}</h2>{cell(y, vbb)}</div></div></div>")  # fmt: skip
    sub = f"A{T['colon']}{outcome_line(ra)}<br>B{T['colon']}{outcome_line(rb)}<br>{T['cmp_note']}"
    afford = ""
    if affordances(ra) or affordances(rb):
        afford = f"<div class='pair'><div>{afford_table(ra)}</div><div>{afford_table(rb)}</div></div>"
    return page(
        f"{ra[0]['label'] if ra else a} vs {rb[0]['label'] if rb else b}",
        sub,
        f"{afford}<section id='c'><div class='bar steps'>{''.join(bar)}</div>{''.join(blocks)}</section>",
    )


# -- MuJoCo frames (optional) ------------------------------------------------------------------------------------------
CAM_DIST_M = 2.2  # the camera's distance from the target when nothing is in the way
CAM_MIN_M = 0.8  # never closer than this (the view is then mostly the object)
# (azimuth offset from the robot->target bearing, elevation), in preference order: side views first, then from behind
# the robot's shoulder, then steeper; the first that backs off CAM_DIST_M unobstructed wins
CAM_VIEWS = [(off, el) for el in (-28, -45, -65) for off in (90, -90, 135, -135, 45, -45, 180, 0)]
CAM_RAYS = ((0, 0), (15, 0), (-15, 0), (0, 10), (0, -8))  # the view's centre and four points towards its edges (deg)
STEEP_EL = -45  # at or below this elevation the apartment's shell (its ceiling) is hidden


def pick_view(sc, lookat, bearing: float) -> dict:
    """A free camera on `lookat` that walls and furniture do not cover (ga-past-b, 2026-11-03: a fixed side view at
    2.2 m sat inside the wall behind DishSoap, DisplayWolf and VaseSmall and showed only what touched the lens). A view
    is as clear as the shortest of a few rays from the target back towards the camera (Scene.clear_distance)."""
    best = None
    for off, el in CAM_VIEWS:
        az = bearing + off
        clear = min(sc.clear_distance(lookat, az + da, el + de) for da, de in CAM_RAYS)
        if clear >= CAM_DIST_M:
            return dict(lookat=list(lookat), distance=CAM_DIST_M, azimuth=az, elevation=el)
        if best is None or clear > best[0]:
            best = (clear, az, el)
    clear, az, el = best
    return dict(lookat=list(lookat), distance=max(CAM_MIN_M, min(CAM_DIST_M, clear - 0.1)), azimuth=az, elevation=el)


def render_frames(paths: list[Path], scene_dir: Path) -> dict:
    """{record_id: [(caption, base64 jpg), ...]}: each step's first and last tick, a camera behind the robot looking at
    the target. Needs mujoco and EGL."""
    import numpy as np
    from PIL import Image
    from visualize import Scene  # scripts/rsi/report/visualize.py

    sc = Scene(scene_dir)
    out = {}
    for p in paths:
        recs, _ = load(p)
        npz = side_file(p).with_name(side_file(p).name.replace(".past.jsonl", ".npz"))
        if not recs or not npz.exists():
            continue
        z = dict(np.load(npz, allow_pickle=True))
        keys = [k for k in z if k.startswith("obj_")]

        class R:  # what Scene.set reads
            pass

        rr = R()
        rr.z, rr.key = z, keys[0] if keys else None
        for r in recs:
            figs, cam = [], None
            for cap, k in ((T["f_start"], r["evidence"]["k0"]), (T["f_end"], r["evidence"]["k1"])):
                k = int(min(k, len(z["t"]) - 1))
                if rr.key:
                    sc.set(rr, k)
                if cam is None:  # one view per step, chosen at its start, so the two frames compare
                    pel = z["robot"][k][:3]
                    tgt = (r["G_after"]["states"].get((r.get("roles") or {}).get("target") or "", {}) or {}).get(
                        "p"
                    ) or pel
                    cam = pick_view(sc, [tgt[0], tgt[1], max(float(tgt[2]) - 0.1, 0.5)],
                                    math.degrees(math.atan2(tgt[1] - pel[1], tgt[0] - pel[0])))  # fmt: skip
                img = sc.render(cam, size=FRAME_SIZE, top_down=cam["elevation"] <= STEEP_EL)
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "JPEG", quality=68)
                figs.append(
                    (T["f_cap"].format(step=r["step"], which=cap, k=k), base64.b64encode(buf.getvalue()).decode())
                )
            out[r["record_id"]] = figs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", type=Path, help="<diag dir>/<label>, its .json or its .past.jsonl")
    ap.add_argument("--compare", type=Path, help="a second round, aligned step by step")
    ap.add_argument("--frames", action="store_true", help="add a MuJoCo frame of each step's start and end (EGL)")
    ap.add_argument("--scene", type=Path, default=SCENE, help="the scene directory (scene_meta.json)")
    ap.add_argument(
        "--out", type=Path, help=f"default {OUT.relative_to(REPO)}/<label>.html (<label>.en.html in English)"
    )
    ap.add_argument("--lang", choices=sorted(LANG), default="zh", help="the page's language (past_view_en.py: en)")
    a = ap.parse_args(argv)
    set_lang(a.lang)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    paths = [a.round] + ([a.compare] if a.compare else [])
    frames = render_frames(paths, a.scene) if a.frames else None
    text = compare(a.round, a.compare, meta, frames) if a.compare else single(a.round, meta, frames)
    label = side_file(a.round).name.replace(".past.jsonl", "")
    if a.compare:
        label += "__vs__" + side_file(a.compare).name.replace(".past.jsonl", "")
    out = a.out or OUT / f"{label}{'' if a.lang == 'zh' else '.' + a.lang}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"{out} ({len(text) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
