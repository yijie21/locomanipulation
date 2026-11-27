"""hadata revision 2c, before freezing eval_code2: hd_real.merge only (no model, no evaluation number), on HEFT's real
episodes, into a scratch dir. Checks:
  1 P&PBox merges (PASS) with exactly the 7 lost episodes' last segments left out; DoubleDesk PASS as before;
  2 every array of the new P&PBox all.npz == the failed run's all.npz with those 7 segments dropped (hd_touch's masks);
    the new DoubleDesk all.npz == the failed run's, key by key, byte for byte;
  3 data.load of the new P&PBox file: every derived array (hd_touch's DERIVED) byte-equal to data.load of the failed
    run's file restricted to the other segments; info rows == merged segments, same order;
  4 negative cases (copies of s229 in a scratch dir, one change each) still FAIL: the log's end not "lost"; the segment's
    lost flag cleared; the extra segment not the last one (an earlier chain pushed beyond the list).
    python hdreal_merge_test.py <dev dir with the new hd_real.py> <failed run's eval/real (stale)> <scratch>
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

dev, old, tmp = (Path(p) for p in sys.argv[1:4])
sys.path.insert(0, str(dev))
import hd_real as H  # noqa: E402

OH = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126")
HS = H.load_by_path("hr_smoke_stats", OH / "asrun_real/selfmodel_hacollect_20261126/hr_smoke_stats.py", "b4d65b92")
TOUCH = H.load_by_path("hd_touch", OH / "asrun_train/selfmodel_hacollect_20261126/hd_touch.py", "5526f05b")
coll_run = json.loads((OH / "heft/d3/run.json").read_text())
seeds = set(range(200, 240))
LOST = {208, 210, 211, 217, 225, 229, 231}
ok_all = True


def same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and np.asarray(a[k]).shape == np.asarray(b[k]).shape
                                    and np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)  # fmt: skip


# 1
res = {}
for t in ("ppbox", "doubledesk"):
    od, info, ok, lines = H.merge(OH / "real" / f"heft_{t}", tmp / "out", seeds, coll_run, HS, TOUCH)
    print("\n".join(lines))
    res[t] = (od, info, ok)
okp = res["ppbox"][2] and res["doubledesk"][2]
print(f"1 merges PASS: {okp}")
ok_all &= okp
# 2
newp = dict(np.load(res["ppbox"][0] / "data/all.npz"))
oldp = dict(np.load(old / "heft_ppbox/data/all.npz"))
rs, chs = oldp["reset"].astype(int), oldp["chain"].astype(int)
cut = np.zeros(len(oldp["n_steps"]), bool)
for s in LOST:
    lg = json.loads((OH / "real/heft_ppbox/log" / f"ppbox_s{s}_r0.json").read_text())
    m = (rs == int(lg["reset"])) & (chs == len(lg["reaches"]))
    assert m.sum() == 1, (s, m.sum())
    cut |= m
print(f"2 old P&PBox segments {len(cut)}, the 7 lost episodes' last reaches {int(cut.sum())}, new {len(newp['n_steps'])}")
e2 = same(newp, H.drop_segments(oldp, ~cut))
e2d = same(dict(np.load(res["doubledesk"][0] / "data/all.npz")), dict(np.load(old / "heft_doubledesk/data/all.npz")))
print(f"2 new P&PBox == old minus those 7, every key byte-equal: {e2}; DoubleDesk new == old: {e2d}")
ok_all &= e2 and e2d and int(cut.sum()) == 7
# 3
A, B = H.DT.load(str(old / "heft_ppbox/data/all.npz")), H.DT.load(str(res["ppbox"][0] / "data/all.npz"))
e3 = all(np.asarray(A[k])[~cut].tobytes() == np.asarray(B[k]).tobytes() for k in TOUCH.DERIVED)
info = res["ppbox"][1]
e3i = [(x["reset"], x["chain"]) for x in info] == list(zip(B["reset"].astype(int).tolist(), B["chain"].astype(int).tolist()))
print(f"3 data.load derived arrays ({len(TOUCH.DERIVED)}) byte-equal to the old file's other segments: {e3}; info rows == merged segments: {e3i}")
ok_all &= e3 and e3i
# 4
src = OH / "real/heft_ppbox"
for case in ("end not lost", "segment not lost", "not the last segment"):
    d = tmp / "neg" / case.replace(" ", "_") / "heft_ppbox"
    (d / "log").mkdir(parents=True)
    (d / "data").mkdir()
    lg = json.loads((src / "log/ppbox_s229_r0.json").read_text())
    z = dict(np.load(src / "data/ppbox_s229_r0.npz"))
    if case == "end not lost":
        lg["end"] = "reaches done"
    elif case == "segment not lost":
        z["lost"] = np.zeros_like(z["lost"])
    else:
        z["chain"] = np.asarray(z["chain"])[::-1].copy()  # the beyond-list chain first
    (d / "log/ppbox_s229_r0.json").write_text(json.dumps(lg))
    np.savez(d / "data/ppbox_s229_r0.npz", **z)
    _, _, okn, ln = H.merge(d, tmp / "neg_out" / case.replace(" ", "_"), {229}, coll_run, HS, TOUCH)
    print(f"4 negative '{case}': merge {'PASS' if okn else 'FAIL'} ({[x for x in ln if 'beyond' in x]})")
    ok_all &= not okn
print("MERGE TEST", "PASS" if ok_all else "FAIL")
