"""TODO 47.9: the search on the 128 SimReady objects, in the ADT apartment, with the incumbent skills package.

`build` writes a scene variant `outputs/scene_discovery/adt_apartment_mars_sr`: the apartment plus the 128 objects of
the self-model line's SimReady set (worktree locomanipulation-selfwm, outputs/reachwm_20261030/simready; the ones with
at least 10 validated Dex1 grasps) as free bodies. A round needs one of them, so they are parked on the ground plane
outside the apartment (resting, asleep) and each has a home surface in the apartment, one of the 16 surfaces the
pick_any objects rest on: a start puts the task's object on its home surface, as it does for the apartment's own
objects. A home is taken only when the object, dropped at a drawn spot there, stays (SETTLE_*). Meshes are copied
re-centred on the box centre, the resting attitude is the one each object comes to on flat ground, masses are the SimReady set's (volume x 400 kg/m^3, clipped), friction the apartment's.

    /venv/main/bin/python scripts/experiments/table_rsi_20261110/sr.py build     # scene + family pick_sr
    /venv/main/bin/python scripts/experiments/table_rsi_20261110/sr.py draw      # pools: train / held-out, by type
    $VENV_SIM_PY           scripts/experiments/table_rsi_20261110/sr.py base      # incumbent on both pools
    $VENV_SIM_PY           scripts/experiments/table_rsi_20261110/sr.py sweep     # feat.py's grid on the training pool
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feat as F  # noqa: E402
import table as T  # noqa: E402

SRC = T.REPO / "outputs/scene_discovery/adt_apartment_mars"
DST = T.REPO / "outputs/scene_discovery/adt_apartment_mars_sr"
SIMREADY = Path("/workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/simready")
FAMILY = "pick_sr"
FAMILIES = T.REPO / "scripts/rsi/families"
D = T.OUT / "sr"
MIN_HELD = 10
PARK_XY, PARK_STEP, PARK_COLS = (14.0, -3.0), 0.45, 12
FRICTION = "0.7 0.005 0.0001"
SETTLE_S, SETTLE_MOVED_M, SETTLE_TILT_DEG = 1.5, 0.03, 12.0
REST_S = 4.0
SMALL_FILES = ("colliders.json", "floorplan.png", "panels.json", "settle.json", "shadowed.json", "surface_heights.json",
               "upright.json")  # fmt: skip
STARTS_TRAIN, STARTS_TEST = 6, 6


def picks() -> list[dict]:
    out = []
    for o in json.loads((SIMREADY / "objects.json").read_text())["objects"]:
        held = json.loads((Path(o["scene"]) / "grasp_cache_summary.json").read_text())["held"]
        if held >= MIN_HELD:
            out.append(o)
    return out


def recentre(src: Path, dst: Path, c: np.ndarray) -> None:
    lines = []
    for ln in src.read_text().splitlines():
        if ln.startswith("v "):
            v = np.array(ln.split()[1:4], float) - c
            ln = f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
        lines.append(ln)
    dst.write_text("\n".join(lines) + "\n")


def verts(f: Path) -> np.ndarray:
    return np.array([ln.split()[1:4] for ln in f.read_text().splitlines() if ln.startswith("v ")], float)


def build() -> None:
    import mujoco

    sys.path.insert(0, str(T.REPO / "src"))
    from locomani.tasks.scene.start import footprint, place_on_surface

    meta = json.loads((SRC / "scene_meta.json").read_text())
    homes = sorted(
        {meta["objects"][t[5:]]["surface"] for t in json.loads((FAMILIES / "pick_any.json").read_text())["tasks"]}
    )
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "sr_mesh").mkdir(parents=True)
    tree = ET.parse(SRC / "scene_locomani.xml")
    root = tree.getroot()
    asset, world = root.find("asset"), root.find("worldbody")
    objs = picks()
    info = {}
    for i, o in enumerate(objs):
        key = f"sr_{o['asset']}"
        am = json.loads((SIMREADY / "mj" / o["asset"] / "meta.json").read_text())
        src = SIMREADY / "mj" / o["asset"]
        v = np.concatenate([verts(src / p) for p in am["pieces"]])
        lo, hi = v.min(0), v.max(0)
        c, he = (lo + hi) / 2, (hi - lo) / 2
        d = DST / "sr_mesh" / o["asset"]
        d.mkdir()
        for name in [*am["pieces"], "all.obj", *[x["obj"] for x in am["vis"]]]:
            recentre(src / name, d / name, c)
        park = np.array(
            [PARK_XY[0] + PARK_STEP * (i % PARK_COLS), PARK_XY[1] + PARK_STEP * (i // PARK_COLS), he[2] + 0.003]
        )
        body = ET.SubElement(
            world, "body", name=f"{key}_object", pos=" ".join(f"{x:.5f}" for x in park), quat="1 0 0 0"
        )
        ET.SubElement(body, "freejoint", name=f"{key}_free")
        for j, x in enumerate(am["vis"]):
            n = f"{key}_v{j}"
            ET.SubElement(asset, "mesh", name=n, file=str(d / x["obj"]))
            attr = dict(name=n, type="mesh", mesh=n, contype="0", conaffinity="0", group="1", density="0")
            if x.get("tex"):
                ET.SubElement(asset, "texture", name=f"{key}_t{j}", type="2d", file=str(src / x["tex"]))
                ET.SubElement(asset, "material", name=f"{key}_m{j}", texture=f"{key}_t{j}")
                attr["material"] = f"{key}_m{j}"
            elif x.get("color"):
                attr["rgba"] = " ".join(f"{float(u):.3f}" for u in [*x["color"][:3], 1.0])
            ET.SubElement(body, "geom", **attr)
        for j, p in enumerate(am["pieces"]):
            n = f"{key}_c{j}"
            ET.SubElement(asset, "mesh", name=n, file=str(d / p))
            ET.SubElement(body, "geom", name=n, type="mesh", mesh=n, friction=FRICTION, density="400", group="3", conaffinity="3")  # fmt: skip
        info[key] = dict(o=o, he=he, park=park, body=body, mesh=str(d / "all.obj"))
    tree.write(DST / "scene_locomani.xml")
    # pass 1: the bodies' inertia from their hulls, rescaled to the SimReady set's mass and written out explicitly
    m = mujoco.MjModel.from_xml_path(str(DST / "scene_locomani.xml"))
    for key, x in info.items():
        b = m.body(f"{key}_object").id
        s = float(x["o"]["mass"]) / float(m.body_mass[b])
        ET.SubElement(x["body"], "inertial", mass=f"{x['o']['mass']:.5f}", pos=" ".join(f"{u:.6f}" for u in m.body_ipos[b]),
                      quat=" ".join(f"{u:.6f}" for u in m.body_iquat[b]),
                      diaginertia=" ".join(f"{u * s:.4e}" for u in m.body_inertia[b]))  # fmt: skip
        for g in x["body"].iter("geom"):
            g.set("density", "0")
    tree.write(DST / "scene_locomani.xml")
    m = mujoco.MjModel.from_xml_path(str(DST / "scene_locomani.xml"))
    d = mujoco.MjData(m)
    # pass 2: the asset's own pose is not always a resting one (a pear rolls onto its side): let every parked object
    # come to rest on the ground, and take that attitude and height as its resting pose, in the lot and on its home
    for _ in range(int(REST_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for key, x in info.items():
        b = m.body(f"{key}_object").id
        x["park"], x["quat"], x["rest_z"] = d.xpos[b].copy(), d.xquat[b].copy(), float(d.xpos[b][2])
        x["body"].set("pos", " ".join(f"{u:.5f}" for u in x["park"]))
        x["body"].set("quat", " ".join(f"{u:.6f}" for u in x["quat"]))
    tree.write(DST / "scene_locomani.xml")
    m = mujoco.MjModel.from_xml_path(str(DST / "scene_locomani.xml"))
    d = mujoco.MjData(m)
    # a home surface per object: dropped at a drawn spot there, it must stay
    rng = np.random.default_rng(47)
    taken = [footprint(o, o["init_pos"], o["init_quat"]) for o in meta["objects"].values() if o.get("surface")]
    order, report, tasks = sorted(info, key=lambda k: (info[k]["o"]["type"], k)), {}, []
    for i, key in enumerate(order):
        x = info[key]
        o, he = x["o"], x["he"]
        entry = dict(
            surface=None, init_pos=None, init_quat=[float(u) for u in x["quat"]], height=float(2 * x["rest_z"]),
            box=dict(half_extent=[float(u) for u in he], centre=[0.0, 0.0, 0.0]), bimanual=False, container=False, free=True,
            semantic=dict(asset=f"simready/{o['asset']}", adt=dict(instance_name=key, prototype_name=o["type"], category=o["group"],
                                                                   motion_type="dynamic", rigidity="rigid"),
                          room=None, movable=True, upright_axis=[0.0, 0.0, 1.0], graspable="single", support=False,
                          phrase=f"the {o['type'].split('_')[-1]}", mass_kg=float(o["mass"]), friction=0.7, container={"is": False}),
            provenance=dict(asset="simready", graspable="selfwm grasp cache"), mesh=x["mesh"], body=f"{key}_object",
        )  # fmt: skip
        b = m.body(f"{key}_object").id
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        tried = []
        for h in homes[i % len(homes) :] + homes[: i % len(homes)]:
            surf = meta["surfaces"][h]
            z = float(surf["top"] + x["rest_z"] + 0.002)
            probe = dict(entry, surface=h, init_pos=[0.0, 0.0, z])
            spot = place_on_surface(dict(meta, objects={**meta["objects"], key: probe}), key, rng, taken)
            if spot is None:
                tried.append((h, "no free spot"))
                continue
            mujoco.mj_resetData(m, d)
            d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7] = spot[0], spot[1]
            mujoco.mj_forward(m, d)
            up0 = d.xmat[b].reshape(3, 3)[:, 2].copy()
            for _ in range(int(SETTLE_S / m.opt.timestep)):
                mujoco.mj_step(m, d)
            moved = float(np.linalg.norm(d.xpos[b] - spot[0]))
            tilt = float(np.degrees(np.arccos(np.clip(d.xmat[b].reshape(3, 3)[:, 2] @ up0, -1, 1))))
            if moved <= SETTLE_MOVED_M and tilt <= SETTLE_TILT_DEG:
                entry.update(surface=h, init_pos=[float(x["park"][0]), float(x["park"][1]), z])
                entry["semantic"]["room"] = surf["room"]
                entry["semantic"]["drop_test"] = dict(moved_m=round(moved, 4), tilt_deg=round(tilt, 1), ok=True)
                break
            tried.append((h, f"moved {moved:.3f} m, tilt {tilt:.0f} deg"))
        report[key] = dict(home=entry["surface"], tried=tried)
        if entry["surface"]:
            meta["objects"][key] = entry
            tasks.append(dict(id=f"pick_{key}", goal=[f"held({key})"], prompts=[], room=entry["semantic"]["room"]))
        else:  # no home: the body stays parked and is no task
            print("no home:", key, tried[-3:])
    mujoco.mj_resetData(m, d)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = max(float(np.linalg.norm(d.xpos[m.body(f"{k}_object").id] - x["park"])) for k, x in info.items())
    meta.update(
        scene_xml=str(DST / "scene_locomani.xml"),
        nbody=int(m.nbody),
        ngeom=int(m.ngeom),
        nq=int(m.nq),
        variant_of=str(SRC),
    )
    (DST / "scene_meta.json").write_text(json.dumps(meta))
    for name in SMALL_FILES:
        if (SRC / name).exists():
            shutil.copy2(SRC / name, DST / name)
    (DST / "mesh").symlink_to(SRC / "mesh")
    (DST / "sr_objects.json").write_text(json.dumps(report, indent=1))
    (FAMILIES / f"{FAMILY}_tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (FAMILIES / f"{FAMILY}.json").write_text(json.dumps(dict(
        name=FAMILY,
        about="TODO 47.9: held(X) of the 128 SimReady objects added to the ADT apartment (scripts/experiments/"
              "table_rsi_20261110/sr.py build); only for the table search's data, never a loop's family, no gate pools.",
        scene=str(DST.relative_to(T.REPO)), tasks_file=str((FAMILIES / f"{FAMILY}_tasks.jsonl").relative_to(T.REPO)),
        tasks=[t["id"] for t in tasks], spawn=dict(rooms="all", distance_m=[0.5, None]),
        pools=dict(transfer=dict(seed=880000, n=0))), indent=1) + "\n")  # fmt: skip
    homes_n = collections.Counter(v["home"] for v in report.values())
    print("objects", len(info), "with a home", len(tasks), "| parked bodies drift over 2 s:", round(drift, 4), "m")
    print("homes:", dict(homes_n))
    print("model: nbody", m.nbody, "ngeom", m.ngeom, "nq", m.nq)


def split() -> tuple[list, list]:
    """Whole SimReady types to one side (seeded), about a third held out."""
    fam = json.loads((FAMILIES / f"{FAMILY}.json").read_text())
    meta = json.loads((DST / "scene_meta.json").read_text())
    groups = collections.defaultdict(list)
    for t in fam["tasks"]:
        groups[meta["objects"][t[5:]]["semantic"]["adt"]["prototype_name"]].append(t)
    names = sorted(groups)
    random.Random(47).shuffle(names)
    train, test = [], []
    for g in names:
        (test if 2 * len(test) < len(train) else train).extend(sorted(groups[g]))
    return sorted(train), sorted(test)


def draw() -> None:
    sys.path[:0] = [str(T.REPO / "scripts/rsi"), str(T.REPO / "src")]
    import family as Fam

    fam, meta = Fam.load(FAMILY), json.loads((DST / "scene_meta.json").read_text())
    train, test = split()
    D.mkdir(exist_ok=True)
    (D / "split.json").write_text(json.dumps(dict(train=train, test=test), indent=1))
    for name, tasks, seed, per in (
        ("pool_sr_train", train, 880000, STARTS_TRAIN),
        ("pool_sr_test", test, 890000, STARTS_TEST),
    ):
        out = D / name
        if not (out / "_pool.json").exists():
            r = Fam.draw(fam, "transfer", out, seed=seed, n=per * len(tasks), tasks=tasks)
            print(name, {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()})
        bad = 0
        for f in sorted(
            out.glob("pick_*.json")
        ):  # a start whose object stayed parked (no spot on its home) is no start
            st = json.loads(f.read_text())
            key = st["task"][5:]
            surf = meta["surfaces"][meta["objects"][key]["surface"]]
            if np.linalg.norm(np.asarray(st["start"]["placements"][key][0][:2]) - np.asarray(surf["xy"])) > 3.0:
                f.unlink()
                bad += 1
        n = collections.Counter(T.task_of(f.stem) for f in out.glob("pick_*.json"))
        print(name, "objects", len(n), "starts", sum(n.values()), "dropped (object stayed parked)", bad)


def job(name: str, pool: Path, variants: list, memo: bool) -> Path:
    f = D / f"{name}.json"
    f.write_text(json.dumps(dict(family=FAMILY, pool=str(pool), memo=memo, starts=sorted(p.stem for p in pool.glob("pick_*.json")),
                                 variants=variants)))  # fmt: skip
    return f


def base(procs: int, gpus: str) -> None:
    import fast

    for name in ("pool_sr_train", "pool_sr_test"):
        out = D / "rounds" / name / "base"
        fast.run(job(f"base_{name}", D / name, [dict(name="base", skills=str(T.SKILLS), set=dict(T.BASE), out=str(out))], False), procs, gpus)  # fmt: skip
        rs = T.rounds(out)
        cls = collections.Counter("won" if r["ok"] else str(r.get("failure_class")) for r in rs.values())
        text = f"{name}: incumbent {sum(r['ok'] for r in rs.values())} of {len(rs)} | {dict(cls.most_common())}"
        (D / f"base_{name}.txt").write_text(text + "\n")
        print(text)


def sweep(procs: int, gpus: str) -> None:
    import fast

    vs = [
        dict(name=n, skills=str(T.SKILLS), set={**T.BASE, **st}, out=str(D / "rounds/pool_sr_train" / n))
        for n, st in F.variants()
    ]
    print("sweep rc", fast.run(job("sweep", D / "pool_sr_train", vs, True), procs, gpus))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "draw", "base", "sweep"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd in ("build", "draw"):
        globals()[a.cmd]()
    else:
        globals()[a.cmd](a.procs, a.gpus)
