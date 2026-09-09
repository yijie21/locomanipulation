"""Per-episode appearance randomisation, applied to the LIVE model.

Everything the ego camera sees was fixed: one grey-ish room, one downward
light, one orange table, one blue-grey checkerboard. A model trained on that
has no reason to learn "the red can" rather than "that exact RGB triple in
that exact lighting", and nothing in the dataset would reveal the difference.

Applied at reset, in the sim process, by mutating mjModel fields directly --
no scene rebuild, so it costs nothing per episode. Measured effect on the ego
frame (mean absolute pixel change, 240x180, robot at a stand):

    groundplane texrepeat + rgba   34.3
    table material -> flat colour  21.7
    light dir + diffuse            20.5
    target geom_rgba                0.5   (small in frame at the stand)

Three mechanisms that LOOK available and are not. All three were tried here
and measured, because each one fails silently -- the field takes the value,
the render ignores it:

* `tex_data`: uploaded to the GPU when the render context is built and never
  re-read. Rewriting the ground checker's pixels moved the frame by 0.19.
  So the builtin checker and the sky gradient cannot be re-coloured this way,
  and the sky is left alone entirely.
* `mat_texid = -1`, to drop a texture and paint the material flat: same
  story. The model field reads back as -1 and the texture still renders. It
  looked like it worked at first because it was measured together with an
  rgba change, and the rgba alone accounts for the difference -- a "flat
  blue" table measured (121, 66, 20), i.e. still orange, just darker.
* `mat_rgba` on a TEXTURED material to re-hue it. It MULTIPLIES, so it can
  only darken and desaturate what the texture already is. The table's wood
  has almost no blue channel; no tint makes it a blue table.

What that leaves, all measured on the rendered pixels of the thing itself:

* the FLOOR really does re-hue, because the checker's own colours carry all
  three channels: (58, 94, 130) blue -> (53, 34, 34) warm -> (19, 80, 53)
  green. Plus check size and reflectance.
* the TARGETS really do re-hue -- they are primitives with no material, so
  `geom_rgba` is their colour outright: (161, 60, 45) -> (49, 83, 166) ->
  (49, 158, 61).
* LIGHTING: direction, diffuse and ambient.
* the mesh objects (table, counter, plate, fork, napkin, orange) vary in
  brightness and saturation only. Giving them different MATERIALS is a
  build-time change, so it belongs to a per-batch scene rebuild (~1 s, see
  the roadmap) rather than here.

TARGET COLOURS ARE NOT FREE. They are what the instruction names -- "pick up
the red can" -- so a target may only move WITHIN its family. That is not a
restriction so much as the point: sampling a range of reds teaches "red" as a
region of colour space instead of one triple, which is what makes the
reference robust. Two targets are never allowed to land on colours a person
would confuse, checked as an OKLab distance.

The randomisation is deliberately conservative: the result still looks like a
room someone could walk into. Wider domain randomisation is a different job
(sim-to-real transfer) and would make the frames hard for a person to audit,
which matters while the dataset is still being read by eye.
"""

from __future__ import annotations

import colorsys

import numpy as np

# Hue in degrees (a range may wrap through 0), saturation and value in [0, 1].
# The families are the colour words in OBJECT_PHRASE; keep them in step.
FAMILY = {
    "can": dict(h=(350.0, 15.0), s=(0.55, 0.85), v=(0.55, 0.85)),  # red
    "bottle": dict(h=(205.0, 240.0), s=(0.45, 0.80), v=(0.45, 0.80)),  # blue
    "cup": dict(h=(30.0, 45.0), s=(0.05, 0.20), v=(0.80, 0.96)),  # cream
    "jar": dict(h=(100.0, 140.0), s=(0.40, 0.75), v=(0.40, 0.75)),  # green
}
# OKLab distance (x100) below which two targets read as the same colour. 20 is
# comfortably above the ~8 that separates "adjacent steps of one ramp".
MIN_TARGET_SEPARATION = 20.0
MAX_COLOUR_TRIES = 60

MESH_MATERIALS = (
    "dining_table_material_0",
    "counter_material_0",
    "plate_material_0",
    "fork_material_0",
    "napkin_material_0",
    "orange_material_0",
)
# Textured, so this multiplies: brightness and saturation only. Kept narrow
# because a strong tint on wood just makes mud.
TINT_S = (0.00, 0.30)
TINT_V = (0.55, 1.05)

GROUND_TEXREPEAT = (3.0, 14.0)
GROUND_S = (0.05, 0.75)  # the checker carries all three channels, so a
GROUND_V = (0.45, 1.15)  # saturated tint here genuinely re-hues the floor
GROUND_REFLECT = (0.05, 0.35)

LIGHT_TILT_MAX = np.deg2rad(50.0)  # from straight down
LIGHT_DIFFUSE = (0.40, 1.00)
LIGHT_AMBIENT = (0.05, 0.35)


def _hsv(rng, h_range, s_range, v_range):
    lo, hi = h_range
    span = (hi - lo) % 360.0 or 360.0
    h = (lo + rng.uniform(0.0, span)) % 360.0
    return colorsys.hsv_to_rgb(h / 360.0, rng.uniform(*s_range), rng.uniform(*v_range))


def _oklab(rgb):
    """sRGB (0-1) -> OKLab. Only used for the "are these two the same colour"
    test, where a perceptually uniform space is the whole point."""
    r, g, b = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in rgb]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array(
        [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ]
    )


def _separated(colours):
    ok = [_oklab(c) * 100.0 for c in colours]
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            if float(np.linalg.norm(ok[i] - ok[j])) < MIN_TARGET_SEPARATION:
                return False
    return True


def sample(seed: int) -> dict:
    """Everything one episode's appearance needs, from one integer."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    for _ in range(MAX_COLOUR_TRIES):
        targets = {k: _hsv(rng, f["h"], f["s"], f["v"]) for k, f in FAMILY.items()}
        if _separated(list(targets.values())):
            break
    else:
        # Never block an episode on this: the families are far apart enough
        # that failing 60 times means the ranges were edited into overlap,
        # which is a bug to notice rather than a round to lose.
        print(
            "[scene_random] WARNING: could not separate target colours in "
            f"{MAX_COLOUR_TRIES} tries -- check FAMILY ranges"
        )

    mats = {
        name: list(colorsys.hsv_to_rgb(rng.uniform(0.0, 1.0), rng.uniform(*TINT_S), rng.uniform(*TINT_V)))
        for name in MESH_MATERIALS
    }

    tilt = rng.uniform(0.0, LIGHT_TILT_MAX)
    az = rng.uniform(0.0, 2.0 * np.pi)
    return dict(
        seed=int(seed),
        targets={k: list(v) for k, v in targets.items()},
        mats=mats,
        ground=dict(
            rgb=list(colorsys.hsv_to_rgb(rng.uniform(0.0, 1.0), rng.uniform(*GROUND_S), rng.uniform(*GROUND_V))),
            texrepeat=float(rng.uniform(*GROUND_TEXREPEAT)),
            reflectance=float(rng.uniform(*GROUND_REFLECT)),
        ),
        light=dict(
            dir=[float(np.sin(tilt) * np.cos(az)), float(np.sin(tilt) * np.sin(az)), float(-np.cos(tilt))],
            diffuse=float(rng.uniform(*LIGHT_DIFFUSE)),
            ambient=float(rng.uniform(*LIGHT_AMBIENT)),
        ),
    )


class SceneRandomizer:
    """Holds the pristine appearance so every apply() starts from it.

    Without the snapshot the tints would compound across episodes and the
    scene would walk to grey (or to whatever the tint hue happens to average
    to) over a long collection run.
    """

    def __init__(self, model, mujoco):
        self.model, self.mj = model, mujoco
        self.geom = {}
        for name in FAMILY:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g >= 0:
                self.geom[name] = (g, model.geom_rgba[g].copy())
        self.mat = {}
        for name in MESH_MATERIALS + ("groundplane",):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                self.mat[name] = (
                    i,
                    model.mat_rgba[i].copy(),
                    model.mat_texrepeat[i].copy(),
                    float(model.mat_reflectance[i]),
                )
        self.light = None
        if model.nlight:
            self.light = (model.light_dir[0].copy(), model.light_diffuse[0].copy(), model.light_ambient[0].copy())

    def restore(self) -> None:
        m = self.model
        for g, rgba in self.geom.values():
            m.geom_rgba[g] = rgba
        for i, rgba, rep, refl in self.mat.values():
            m.mat_rgba[i], m.mat_texrepeat[i] = rgba, rep
            m.mat_reflectance[i] = refl
        if self.light is not None:
            m.light_dir[0], m.light_diffuse[0], m.light_ambient[0] = self.light

    def apply(self, spec: dict) -> str:
        self.restore()
        m = self.model
        for name, rgb in spec["targets"].items():
            if name in self.geom:
                g = self.geom[name][0]
                m.geom_rgba[g] = [*rgb, 1.0]
        for name, rgb in spec["mats"].items():
            if name in self.mat:
                m.mat_rgba[self.mat[name][0]] = [*rgb, 1.0]
        if "groundplane" in self.mat:
            i = self.mat["groundplane"][0]
            g = spec["ground"]
            m.mat_rgba[i] = [*g["rgb"], 1.0]
            m.mat_texrepeat[i] = [g["texrepeat"], g["texrepeat"]]
            m.mat_reflectance[i] = g["reflectance"]
        if self.light is not None:
            L = spec["light"]
            m.light_dir[0] = L["dir"]
            m.light_diffuse[0] = [L["diffuse"]] * 3
            m.light_ambient[0] = [L["ambient"]] * 3
        L = spec["light"]
        return (
            f"seed {spec['seed']} light "
            f"[{L['dir'][0]:+.2f} {L['dir'][1]:+.2f} {L['dir'][2]:+.2f}] "
            f"diff {L['diffuse']:.2f} amb {L['ambient']:.2f} "
            f"ground rep {spec['ground']['texrepeat']:.1f}"
        )
