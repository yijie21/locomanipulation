"""selfmodel_faildiag_20261126: how many "furniture" counts in the existing LML entries are the robot's own pelvis body
`manipulator_mount` (robosuite's add_mobile_base puts the physical robot under it, no `robot0_` prefix, so
hg_run.Driver.arm_scene_contact -- shared by hardgrasp, lmlreal, lmlcmp, lmlcmp2, lmlcmp3, lmlroute -- counts a right
hand / wrist / elbow touch of it as furniture). Read only, records as they are on disk.

What the records hold: per episode `n_contacts` (rows in the episode's contact window with an arm-furniture contact;
one body per row, the first found in d.contact) and `first_contact` = [row, body] of the first such row; per reach
segment `furn_rows` (rows from the command to the label, a count, no body names). So per episode labelled
hit_furniture (n_contacts > 0 and no earlier reason): first body not the mount -> touched something else; first body
the mount and n_contacts == 1 -> the mount only; first body the mount and n_contacts > 1 -> not determinable from the
records. Segments with furn_rows > 0: not determinable; counted, with how many sit in an episode whose first contact
is the mount.

    python fd_mount_scan.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

Y = "/wuji-vefps-D/wuji-il/yijie"
ENTRIES = {
    "hardgrasp_20261125": [(f"{Y}/ws/code/loco/outputs/hardgrasp_20261125/locomanilab", "scalebfm")],
    "selfmodel_lmlreal_20261125": [(f"{Y}/loco_outputs/selfmodel_lmlreal_20261125/{b}", b) for b in ("scalebfm", "sonic", "heft")],
    "selfmodel_lmlcmp_20261126": [(f"{Y}/loco_outputs/selfmodel_lmlcmp_20261126/{b}", b) for b in ("scalebfm", "sonic", "heft")],
    "selfmodel_lmlcmp2_20261126": [(f"{Y}/loco_outputs/selfmodel_lmlcmp2_20261126/{b}", b) for b in ("scalebfm", "sonic", "heft")],
    "selfmodel_lmlcmp3_20261126": [(f"{Y}/loco_outputs/selfmodel_lmlcmp3_20261126/{b}", b) for b in ("scalebfm", "sonic", "heft")],
    "selfmodel_lmlroute_20261126": [(f"{Y}/loco_outputs/selfmodel_lmlroute_20261126/{b}", b) for b in ("scalebfm", "sonic", "heft")],
}
MOUNT = "manipulator_mount"


def main() -> None:
    for entry, dirs in ENTRIES.items():
        for d, be in dirs:
            eps = sorted(glob.glob(f"{d}/episodes/*/*/ep*.json") + glob.glob(f"{d}/episodes/*/ep*.json"))  # task/method/ or task/ (lmlreal)
            if not eps:
                print(f"{entry:28s} {be:8s} no episodes at {d}")
                continue
            c = defaultdict(int)
            first_bodies = defaultdict(int)
            mount_eps = set()
            for p in eps:
                r = json.load(open(p))
                fc = r.get("first_contact")
                body = fc[1] if isinstance(fc, list) and len(fc) > 1 else None
                n = int(r.get("n_contacts") or 0)
                c["episodes"] += 1
                c["with contact"] += n > 0
                if body == MOUNT:
                    mount_eps.add(os.path.relpath(p, f"{d}/episodes")[:-5])
                    c["first contact mount (any label)"] += 1
                if r.get("reason") == "hit_furniture":
                    c["hit_furniture"] += 1
                    first_bodies[str(body)[:24]] += 1
                    if body != MOUNT:
                        c["hf: first body other -> touched something else"] += 1
                    elif n == 1:
                        c["hf: mount only (n_contacts 1)"] += 1
                    else:
                        c["hf: mount first, n_contacts > 1 -> not determinable"] += 1
            segs = sorted(glob.glob(f"{d}/segs/*/*/ep*.npz") + glob.glob(f"{d}/segs/*/ep*.npz"))
            nseg = nfr = nfr_m = 0
            for s in segs:
                z = np.load(s)
                if "furn_rows" not in z.files:
                    continue
                fr = np.asarray(z["furn_rows"]).ravel()
                nseg += len(fr)
                k = int((fr > 0).sum())
                nfr += k
                if os.path.relpath(s, f"{d}/segs")[:-4] in mount_eps:
                    nfr_m += k
            seg_txt = (f"segments {nseg}, furn_rows > 0 {nfr} (not determinable; in episodes whose first contact is the mount {nfr_m})"
                       if nseg else "segments with furn_rows: none recorded")  # fmt: skip
            top = dict(sorted(first_bodies.items(), key=lambda kv: -kv[1])[:4])
            print(f"{entry:28s} {be:8s} " + "; ".join(f"{k} {v}" for k, v in c.items()) + f" | hit_furniture first bodies {top} | {seg_txt}")


if __name__ == "__main__":
    main()
