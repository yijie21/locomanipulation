"""view_scan.py <runs> <placement> <out json>: all views (az -60..60 step 10, el 35..55 step 5, d 2.0 2.4 2.8) at v3b's
panel moment (after3500ms) and at the close command, the v3a rule for each (both arms: head >= 80 %, hand >= 50 %,
vase >= 50 %, whole robot inside). Read only. notebook: skip figure render, not an experiment."""
import json
import sys
runs, pi, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], runs, pi, "lib", "/tmp"]
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import fig_final as FF  # noqa: E402
M, Z, st, _ = FF.load()
G = FF.FR.masks(M)
_, yaw, la = FF.main_cam(M, st, -30, 45, 2.0)
grid = [(az, el, d) for az in range(-60, 61, 10) for el in (35, 40, 45, 50, 55) for d in (2.0, 2.4, 2.8)]
res = {}
for tag in (FF.PANEL, FF.CLOSE):
    rows = FF.vis_main(M, G, st, tag, la, yaw, grid)
    res[tag] = rows
    print(f"@ {tag}: valid {sum(r['valid'] for r in rows)} of {len(rows)}")
    for el in (35, 40, 45, 50, 55):
        print(f"  el {el}: " + " ".join(f"{az:+d}" + "".join("o" if next(r for r in rows if r['az'] == az and r['el'] == el and r['d'] == d)['valid'] else "." for d in (2.0, 2.4, 2.8)) for az in range(-60, 61, 10)))
    for az in (-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60):
        r = next(r for r in rows if r['az'] == az and r['el'] == 45 and r['d'] == 2.0)
        print(f"    az {az:+d} el 45 d 2.0: " + " | ".join(f"{a} head {r[a]['head']:.2f} hand {r[a]['hand']:.2f} vase {r[a]['vase']:.2f}" for a in ("A", "R")))
json.dump(res, open(out, "w"), indent=1)
