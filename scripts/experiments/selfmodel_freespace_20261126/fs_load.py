"""selfmodel_freespace_20261126: read one controller's records (fs_run's runs/*.jsonl and segs/*.npz) for fs_check and
fs_eval; read-only."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def runs(base: Path) -> tuple[dict, list]:
    """{(cid, ep, method): record}; plus the duplicate keys (a job run twice)."""
    out, dup = {}, []
    for p in sorted(glob.glob(str(base / "runs" / "*.jsonl"))):
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            k = (r["cid"], r["ep"], r["method"])
            if k in out:
                dup.append(k)
            out[k] = r
    return out, dup


def segs(base: Path, derive: bool = True) -> tuple[dict, dict] | tuple[None, None]:
    """All segment chunks concatenated (data.py's derive applied when asked) and the arm arrays. Only chunks whose
    runs/<chunk>.jsonl exists (fs_run writes the npz first: a process killed in between leaves an npz nobody counts)."""
    names = sorted(p for p in glob.glob(str(base / "segs" / "*.npz")) if (base / "runs" / f"{Path(p).stem}.jsonl").exists())
    if not names:
        return None, None
    zs = [dict(np.load(p)) for p in names]
    keep = ("radius", "link")
    z = {k: (zs[0][k] if k in keep else np.concatenate([q[k] for q in zs])) for k in zs[0]}
    a = [dict(np.load(Path(p).parent / "arm" / Path(p).name)) for p in names]
    A = {k: (a[0][k] if k in keep else np.concatenate([q[k] for q in a])) for k in a[0]}
    if derive:
        import lml_tasks as LT

        DT, _ = LT.reachwm()
        z.setdefault("walk", np.zeros((len(z["n_steps"]), 3), np.float32))
        z.setdefault("target_b", z["target"])
        z["files"] = names
        DT.derive(z)
    return z, A
