"""build_index.py: picks the shown attempt per (label, version) from rows/ (the first that matches the loop's record, else the first),
copies it to <label>_v<k>.{mp4,jpg}, writes index.json (what_zh from what_zh.json {"<label>_v<k>": text}, written after looking at sheets)."""

import glob
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_lib as SL  # noqa: E402

OUT = SL.OUT
WZ = json.loads((OUT / "what_zh.json").read_text()) if (OUT / "what_zh.json").exists() else {}
groups = {}
for f in sorted(glob.glob(str(OUT / "rows/*.json"))):
    d = json.load(open(f))
    if d.get("bytes"):
        groups.setdefault((d["label"], d["version"]), []).append(d)
rows = []
for (label, k), L in sorted(groups.items()):
    L.sort(key=lambda d: d["attempt"])
    shown = next((d for d in L if d["match"]), L[0])
    stem = f"{label}_v{k}"
    src = f"{label}_v{k}.a{shown['attempt']}"
    for ext in ("mp4", "jpg"):
        shutil.copy(OUT / "attempts" / f"{src}.{ext}", OUT / f"{stem}.{ext}")
    o = lambda d: dict(ok=d["new_ok"], failure_class=d["new_class"] or "", t_end=round(d["new_t_end"] or 0, 2))  # noqa: E731
    rows.append(dict(loop="simple_art", task=label.rsplit("_s", 1)[0], label=label, role="story", version=k, hash=shown["hash"],
        attempt=shown["attempt"], attempts_run=len(L), attempt_matches={str(d["attempt"]): d["match"] for d in L}, video_outcome=o(shown),
        loop_outcome=dict(ok=shown["old_ok"], failure_class=shown["old_class"] or "", t_end=round(shown["old_t_end"] or 0, 2)),
        matches=shown["match"], sim_seconds=shown["sim_s"],
        files={"main": dict(mp4=f"{stem}.mp4", mp4_bytes=os.path.getsize(OUT / f"{stem}.mp4"), jpg=f"{stem}.jpg", jpg_bytes=os.path.getsize(OUT / f"{stem}.jpg"))},
        other_attempts=[dict(attempt=d["attempt"], **o(d)) for d in L if d is not shown], what_zh=WZ.get(stem, "")))  # fmt: skip
(OUT / "index.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
for r in rows:
    print(
        r["label"],
        "v",
        r["version"],
        "a",
        r["attempt"],
        r["video_outcome"],
        "| loop",
        r["loop_outcome"]["failure_class"] or "ok",
        r["matches"],
        r["files"]["main"]["mp4_bytes"],
    )
