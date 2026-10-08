"""The scene file's Claude-estimated fields, and the rules that check them (plan decision 6, format §2-§4).

Claude (`claude -p`, JSON schema, no tools, every exchange logged) names objects, surfaces and rooms, estimates
mass and friction, and says which objects are open containers. Rules then bound what comes back and record
where each value came from: mass outside a plausible density for its box is kept but reported, friction is
clamped, a "container" too small to hold anything is overridden (provenance `rule`), and names repeated
within a room get a suffix. The drop test that turns `mass_kg` / `friction` provenance into `measured` runs
after the MuJoCo build (TODO 19.6.5).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

FRICTION = (0.1, 2.0)
DENSITY = (5.0, 3000.0)  # kg/m^3 over the box; a hollow mug is ~100-500, a full jar ~1000, a pan's box ~300
CONTAINER_MIN_OPENING = 0.06  # m; narrower openings hold nothing the robot can put in
CONTAINER_WALL = 0.01
SHELL_KEY = "ApartmentEnv"

OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "phrase": {"type": "string"},
                    "mass_kg": {"type": "number"},
                    "friction": {"type": "number"},
                    "container": {"type": "boolean"},
                },
                "required": ["key", "phrase", "mass_kg", "friction", "container"],
            },
        }
    },
    "required": ["objects"],
}
OBJECT_SYSTEM = """You annotate the objects of a scanned apartment for a humanoid-robot simulator.
For every object in the request return:
- phrase: the short English noun phrase a person would use in an instruction ("the white mug", "the blue
  laundry basket"), 2-5 words starting with "the", unique among the objects of the same room -- add colour,
  material or position words when two objects would otherwise share a phrase;
- mass_kg: your best estimate of its real mass;
- friction: a Coulomb friction coefficient for its outer surface (glass or glazed ceramic ~0.4, wood or
  hard plastic ~0.6, fabric or rubber ~1.0);
- container: true only if it is an open container another object could be placed into from above (bowl,
  basket, bin, open box, tray with rims, pot); false for closed boxes, lidded jars, bottles and furniture.
Use the category, the prototype name and the box size (metres, z vertical). Return every key you were given."""

SURFACE_SCHEMA = {
    "type": "object",
    "properties": {
        "surfaces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "phrase": {"type": "string"}},
                "required": ["key", "phrase"],
            },
        }
    },
    "required": ["surfaces"],
}
SURFACE_SYSTEM = """You name the support surfaces of a scanned apartment for robot instructions.
Each surface is a flat region (top height and size in metres) of a piece of furniture or of the apartment's
built-in fittings ("ApartmentEnv": kitchen counter, bathroom fittings). Return for every key an English
phrase starting with "the" that a person would say: "the kitchen island", "the top shelf of the white
bookcase", "the dining table", "the kitchen counter by the fridge". Phrases must be unique."""

ROOM_SCHEMA = {
    "type": "object",
    "properties": {
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "name": {"type": "string"}},
                "required": ["key", "name"],
            },
        }
    },
    "required": ["rooms"],
}
ROOM_SYSTEM = """You name the rooms of a scanned apartment from the furniture inside each one.
Return for every key a snake_case name ("bedroom", "living_kitchen"); an open-plan space that holds
several functions gets one name joining them. Names must be unique."""


def ask(prompt: dict, schema: dict, system: str, model: str, log_path: Path | None, timeout_s: float = 600.0) -> dict:
    """One `claude -p` exchange with structured output; raises on failure (a property pass is not optional).
    The prompt goes on stdin: a task proposal with the checker's verdicts passes the 128 KB argv limit."""
    argv = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
            "--tools", "", "--model", model, "--system-prompt", system]  # fmt: skip
    t0 = time.time()
    proc = subprocess.run(
        argv, input=json.dumps(prompt), capture_output=True, text=True, timeout=timeout_s, check=False
    )
    reply = json.loads(proc.stdout) if proc.stdout.strip() else {}
    out = reply.get("structured_output")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps({"elapsed_s": round(time.time() - t0, 1), "model": model, "prompt": prompt,
                                "reply": out, "cost_usd": reply.get("total_cost_usd"), "exit": proc.returncode}) + "\n")  # fmt: skip
    if proc.returncode != 0 or not isinstance(out, dict):
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-400:]}")
    return out


def full_size(obj: dict) -> list[float]:
    return [round(2 * v, 3) for v in obj["box"]["half_extent"]]


def object_rows(meta: dict) -> list[dict]:
    rows = []
    for key, o in sorted(meta["objects"].items()):
        if key == SHELL_KEY:
            continue
        sem = o["semantic"]
        rows.append({"key": key, "category": sem["adt"]["category"], "prototype": sem["adt"]["prototype_name"],
                     "size_m": full_size(o), "room": sem.get("room"), "fixed": not sem["movable"]})  # fmt: skip
    return rows


def surface_rows(meta: dict, object_phrase: dict) -> list[dict]:
    rows = []
    for key, s in sorted(meta["surfaces"].items()):
        of = s.get("of_object")
        rows.append({"key": key, "of": object_phrase.get(of, of), "category": s["name"].split(" ")[0],
                     "top_m": round(s["top"], 2), "size_m": [round(2 * v, 2) for v in s["half_extent"]],
                     "room": s.get("room")})  # fmt: skip
    return rows


def room_rows(meta: dict) -> list[dict]:
    rows = []
    for key in sorted(meta["room"]["rooms"]):
        furniture = sorted(
            {
                o["semantic"]["adt"]["category"]
                for o in meta["objects"].values()
                if o["semantic"].get("room") == key and not o["semantic"]["movable"]
            }  # fmt: skip
        )
        poly = np.asarray(meta["room"]["rooms"][key]["polygon"], dtype=float)
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        rows.append({"key": key, "area_m2": round(area, 1), "furniture": furniture})
    return rows


def apply_objects(meta: dict, estimates: list[dict]) -> list[str]:
    """Write phrase / mass / friction / container into `semantic` with provenance; return what the rules changed."""
    notes = []
    got = {e["key"]: e for e in estimates}
    missing = sorted(k for k in meta["objects"] if k != SHELL_KEY and k not in got)
    if missing:
        notes.append(f"no estimate for {len(missing)} objects: {missing[:8]}")
    for key, e in got.items():
        o = meta["objects"].get(key)
        if o is None:
            notes.append(f"estimate for unknown object {key!r} ignored")
            continue
        sem, prov = o["semantic"], o["provenance"]
        size = np.asarray(full_size(o), dtype=float)
        mass = float(e["mass_kg"])
        density = mass / max(float(np.prod(size)), 1e-6)
        if sem["movable"] and not DENSITY[0] <= density <= DENSITY[1]:
            notes.append(f"{key}: {mass:.3f} kg over a {size.tolist()} m box is {density:.0f} kg/m^3 (kept)")
        friction = float(np.clip(e["friction"], *FRICTION))
        if friction != e["friction"]:
            notes.append(f"{key}: friction {e['friction']} clamped to {friction}")
        opening = 2 * min(o["box"]["half_extent"][:2]) - 2 * CONTAINER_WALL
        is_container = bool(e["container"]) and opening >= CONTAINER_MIN_OPENING
        sem.update(phrase=str(e["phrase"]).strip(), mass_kg=round(max(mass, 1e-3), 4), friction=round(friction, 3))
        prov.update(phrase="claude", mass_kg="claude", friction="claude")
        if e["container"] and not is_container:
            notes.append(f"{key}: container refused, opening {opening:.3f} m < {CONTAINER_MIN_OPENING}")
        hx, hy = o["box"]["half_extent"][:2]
        sem["container"] = {"is": is_container}
        if is_container:
            sem["container"]["opening_half_extent"] = [round(hx - CONTAINER_WALL, 4), round(hy - CONTAINER_WALL, 4)]
            o["box"]["wall"] = CONTAINER_WALL
        prov["container"] = "claude" if is_container == bool(e["container"]) else "rule"
        o["container"] = is_container
    notes.extend(_unique_phrases(meta))
    return notes


def _unique_phrases(meta: dict) -> list[str]:
    notes, seen = [], {}
    for key in sorted(meta["objects"]):
        sem = meta["objects"][key]["semantic"]
        if "phrase" not in sem:
            continue
        slot = (sem.get("room"), sem["phrase"].lower())
        if slot in seen:
            seen[slot] += 1
            sem["phrase"] = f"{sem['phrase']} {seen[slot]}"
            meta["objects"][key]["provenance"]["phrase"] = "rule"
            notes.append(f"{key}: phrase repeated in its room, now {sem['phrase']!r}")
        else:
            seen[slot] = 1
    return notes


def apply_surfaces(meta: dict, names: list[dict]) -> list[str]:
    notes = []
    for e in names:
        s = meta["surfaces"].get(e["key"])
        if s is None:
            notes.append(f"name for unknown surface {e['key']!r} ignored")
            continue
        s["phrase"] = str(e["phrase"]).strip()
        s.setdefault("provenance", {})["phrase"] = "claude"
    missing = sorted(k for k, s in meta["surfaces"].items() if "phrase" not in s)
    if missing:
        notes.append(f"no name for {len(missing)} surfaces: {missing[:8]}")
    return notes


def rename_rooms(meta: dict, names: list[dict]) -> list[str]:
    """Rename rooms everywhere they are referenced; refuses names that collide."""
    rooms = meta["room"]["rooms"]
    mapping = {e["key"]: e["name"].strip() for e in names if e["key"] in rooms}
    if len(set(mapping.values())) != len(mapping) or set(rooms) - set(mapping):
        return [f"room names not applied (missing or repeated): {mapping}"]
    meta["room"]["rooms"] = {mapping[k]: dict(v, name_source="claude") for k, v in rooms.items()}
    meta["room"]["spawn"] = {mapping[k]: v for k, v in meta["room"].get("spawn", {}).items()}
    for group in ("objects", "surfaces"):
        for item in meta[group].values():
            holder = item["semantic"] if group == "objects" else item
            if holder.get("room") in mapping:
                holder["room"] = mapping[holder["room"]]
    rnd = meta.get("randomization") or {}
    rnd["spawn_rooms"] = [mapping.get(k, k) for k in rnd.get("spawn_rooms", [])]
    return []
