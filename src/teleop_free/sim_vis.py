"""Which debug visuals the simulator draws — one env var, `SIM_VIS`.

    SIM_VIS=full            # default: everything, identical to the old behaviour
    SIM_VIS=none            # a clean third-person video
    SIM_VIS=bbox,readout    # only the 6-D pose boxes and the numeric readout

The four items:

    reach    the reach-map cloud: pale green = graspable envelope, dark green
             core = the sweet zone.  ~5800 box geoms injected into the scene.
    ghost    the magenta ghost gripper parked at the planned grasp pose, plus
             the solid bead at its grip centre.
    bbox     oriented wireframe boxes on the bodies whose 6-D poses the dataset
             records (can, plate).
    readout  the pelvis-frame position text burned into the video.

None of this ever reached the training data: `reach` and `ghost` live in geom
group 4, which default MuJoCo visualisation options hide, so the ego camera and
the data exporter never saw them, and `bbox`/`readout` are drawn only into the
third-person mp4.  The switches are about a clean video and about cost — and
only `reach` has a real cost, because ~5800 geoms are paid at model-compile and
render time whether or not they are visible.  That is why `reach` is honoured
by *not injecting* the cloud rather than by hiding it, which in turn means it is
read once at sim start-up and cannot be changed for a running sim.
"""

from __future__ import annotations

import os

ITEMS = ("reach", "ghost", "bbox", "readout")
_ALL = {"full", "all", "1", ""}      # "" = exported but empty; treat as unset
_NONE = {"none", "off", "0"}


def _selected() -> frozenset:
    raw = os.environ.get("SIM_VIS", "full").strip().lower()
    if raw in _ALL:
        return frozenset(ITEMS)
    if raw in _NONE:
        return frozenset()
    want = {t.strip() for t in raw.split(",") if t.strip()}
    unknown = want - set(ITEMS)
    if unknown:
        # Loud, because the failure mode is silent: a typo would otherwise
        # just quietly drop a visual the user asked to keep.
        print(f"[sim_vis] ignoring unknown SIM_VIS item(s) {sorted(unknown)}; "
              f"valid items are {list(ITEMS)}")
    return frozenset(want & set(ITEMS))


SELECTED = _selected()


def enabled(item: str) -> bool:
    assert item in ITEMS, item
    return item in SELECTED


def describe() -> str:
    on = [i for i in ITEMS if i in SELECTED]
    return f"SIM_VIS={os.environ.get('SIM_VIS', 'full')} -> {on or ['nothing']}"


if __name__ == "__main__":
    print(describe())
