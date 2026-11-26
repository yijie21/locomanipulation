"""dev (heft_reanchor2x2_20261126 results): the lateral failures in the reading's cells (1.00, 1.50, 1.00 +-0.20 m), walk by
walk: asked / moved lateral (cm), lift-offs, re-issues used, final lateral error."""
import glob, json, sys
O = sys.argv[1]
rows = {}
for fn in glob.glob(O + "/heft/runs/*.jsonl"):
    for line in open(fn):
        r = json.loads(line)
        rows[(r["arm"], r["cid"], r["rep"])] = r
print("lateral failures in the L cells: arm cid rep | n_reissue | per walk: asked_lat / moved_lat cm, lifts | final lat err cm")
for k in sorted(rows):
    r = rows[k]
    if k[1] not in (5, 6, 9, 10) or r.get("success") or abs(r.get("err_lat_m", 0)) <= 0.03:
        continue
    ws = " ; ".join("%+.1f/%+.1f l%s" % (100 * w.get("asked_lat", 0), 100 * (w.get("moved_lat") or 0), w.get("lift")) for w in r["walks"])
    print("  %-4s c%d r%d | re %d | %s | lat %+.1f" % (k[0], k[1], k[2], r["n_reissue"], ws, 100 * r["err_lat_m"]))
