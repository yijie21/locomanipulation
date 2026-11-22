"""selfmodel_via_20261120, third step: does moving the via points against the predicted offset bring the arm onto the
sweep that was wanted?

    $PY via_comp_eval.py <model dir> <direct replay dir> <compensated replay dir>

Both replay datasets run the same paths from the same states (same seed and reset order; REACH_VIA_REPLAY, the second with
REACH_VIA_COMP), so reset k of one pairs with reset k of the other. Each needs all.npz, arm.npz (arm_spheres.py) and
table.npz (via_model.py build). Wanted: the sweep of the original command's reference plan -- the direct run's own
reference, and comp_ref0 saved by the compensated run. Error: via_model.py's swept-region error of the recorded ticks
against that wanted sweep, over the arm's 24 spheres and over the palm's 3.

Checks printed first (if they fail the features computed at run time are not the training's and nothing below counts):
  wanted sweep   comp_ref0 against the direct run's table reference, the same reset: largest difference
  prediction     comp_off0, predicted at run time for the original command, against via_point.py's prediction for the
                 direct run's row: largest difference
  plan           the reference the compensated command got when it was run (its table) against comp_ref, the plan made
                 for it beforehand
Then per kind of path (the route's name up to the first "_"; "detour" = every kind but FreshUnder): the medians of both
runs, the drop with a 95 % interval from resampling routes, the share of pairs that improved, how far the points were
moved, the palm's settled end against the wanted end, and how well the model predicted the compensated run itself.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

MD, DD, DC = (Path(x) for x in sys.argv[1:4])
M = torch.load(MD / "via_point.pt", map_location="cpu", weights_only=False)
Td, Tc = VP.load([DD]), VP.load([DC])
zc = dict(np.load(DC / "all.npz"))
names = json.load(open(DD / "routes.json"))["names"]
assert names == json.load(open(DC / "routes.json"))["names"], "the two runs replay different lists of paths"
kd, kc = Td["reset"] - Td["reset"].min(), Tc["reset"] - Tc["reset"].min()
common = np.intersect1d(kd, kc)
i_d, i_c = np.array([np.where(kd == k)[0][0] for k in common]), np.array([np.where(kc == k)[0][0] for k in common])
assert (Td["route"][i_d] == Tc["route"][i_c]).all() and (Td["walked"][i_d] == Tc["walked"][i_c]).all()
seg = Tc["seg"][i_c]
ref0, off0, refc, offc = (zc[k][seg].astype(float) for k in ("comp_ref0", "comp_off0", "comp_ref", "comp_off"))
print(f"pairs {len(common)} (direct {len(kd)}, compensated {len(kc)} paths; falls: direct {len(names) * 4 - len(kd)}, "
      f"compensated {len(names) * 4 - len(kc)} of {len(names) * 4})")  # fmt: skip
d_ref = np.abs(ref0 - Td["ref"][i_d]).max((1, 2, 3))
off_d = VP.offsets(M, Td, i_d)["direction"]
d_off = np.abs(off0 - off_d).max((1, 2, 3))
d_plan = np.abs(refc - Tc["ref"][i_c]).max((1, 2, 3))
print(f"check, wanted sweep at run time against the direct run's reference: median {1000 * np.median(d_ref):.2f} mm, largest "
      f"{1000 * d_ref.max():.2f} mm")  # fmt: skip
print(f"check, prediction at run time against the offline one: median {1000 * np.median(d_off):.2f} mm, largest "
      f"{1000 * d_off.max():.2f} mm")  # fmt: skip
print(f"check, the compensated command's plan beforehand against the one it ran with: median {1000 * np.median(d_plan):.2f} mm, "
      f"largest {1000 * d_plan.max():.2f} mm")  # fmt: skip

zero = np.zeros((len(common), VM.Q + 1, 24, 3))
ed, ehd = VP.errors([DD], Td, zero, i_d)  # the direct run against its own reference = the wanted sweep
want = dict(seg=Tc["seg"][i_c], a=Tc["a"][i_c], b=Tc["b"][i_c], ref=ref0.astype(np.float32), palm=Tc["palm"])
ec, ehc = VM.errors(DC, want, zero)  # the compensated run against the wanted sweep
ep, ehp = VP.errors([DC], Tc, offc, i_c)  # the compensated run against the sweep predicted for it
epd, _ = VP.errors([DD], Td, off_d, i_d)
Xd, Xc = VM._ARM[DD], VM._ARM[DC]
palm = int(np.where(Tc["palm"])[0][0])
end_d = np.array(
    [np.linalg.norm(Xd[Td["b"][j] - VM.TAIL : Td["b"][j], palm].mean(0) - Td["ref"][j][-1, palm]) for j in i_d]
)
end_c = np.array(
    [np.linalg.norm(Xc[Tc["b"][j] - VM.TAIL : Tc["b"][j], palm].mean(0) - ref0[n][-1, palm]) for n, j in enumerate(i_c)]
)
shift = np.linalg.norm(zc["comp_shift"][seg], axis=2)
kind = np.array([names[r].split("_")[0] for r in Td["route"][i_d]])
route = Td["route"][i_d]
rng = np.random.default_rng(0)


def drop(a, b, m):
    g = [np.where(m & (route == r))[0] for r in np.unique(route[m])]
    bs = []
    for _ in range(2000):
        j = np.concatenate([g[k] for k in rng.integers(len(g), size=len(g))])
        bs.append(1 - np.median(b[j]) / np.median(a[j]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{1 - np.median(b[m]) / np.median(a[m]):.2f} ({lo:.2f} to {hi:.2f})"


def show(tag, m):
    if m.sum() < 4:
        return
    print(f"{tag}: {len(np.unique(route[m]))} routes, {int(m.sum())} pairs")
    print(f"  palm, error against the wanted sweep: direct {100 * np.median(ehd[m]):.2f} cm, compensated {100 * np.median(ehc[m]):.2f} cm; "
          f"drop {drop(ehd, ehc, m)}; improved in {np.mean(ehc[m] < ehd[m]):.2f} of the pairs")  # fmt: skip
    print(f"  arm,  error against the wanted sweep: direct {100 * np.median(ed[m]):.2f} cm, compensated {100 * np.median(ec[m]):.2f} cm; "
          f"drop {drop(ed, ec, m)}; improved in {np.mean(ec[m] < ed[m]):.2f} of the pairs")  # fmt: skip
    print(f"  palm's settled end against the wanted end: direct {100 * np.median(end_d[m]):.2f} cm, compensated "
          f"{100 * np.median(end_c[m]):.2f} cm; drop {drop(end_d, end_c, m)}")  # fmt: skip
    print(f"  points moved: median {100 * np.median(shift[m][shift[m] > 0]):.2f} cm, largest {100 * shift[m].max():.2f} cm; the goal "
          f"{100 * np.median(shift[m][:, -1]):.2f} cm")  # fmt: skip
    print(f"  the model's own error (arm) on the run it predicted: direct {100 * np.median(epd[m]):.2f} cm, compensated "
          f"{100 * np.median(ep[m]):.2f} cm")  # fmt: skip


show("detour paths (over an edge, over a box)", kind != "FreshUnder")
for k in dict.fromkeys(kind.tolist()):
    show(k, kind == k)
show("all", np.ones(len(kind), bool))
show("first pass (from the stand)", Td["walked"][i_d] == 0)
