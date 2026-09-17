"""How much of a round an exporter session actually holds.

Split out of the converter so it can be imported without pandas/pyarrow, the
same reason `verification_tail` is its own module.
"""

import json
from pathlib import Path

# A healthy session records about 1.1 rows per privileged sequence unit: 1.10 to
# 1.15 across every episode of an 8-round session on an idle box (2026-09-17
# 05:31). Under load the exporter silently keeps a fraction and says nothing --
# the same day, three sessions came back holding 0.04 to 0.44, one of them a 37 s
# round with 1.2 s of frames in it. The danger is not the crash; it is that such
# a stub passes every other check whenever its training boundary happens to land
# inside the range it did record, and enters the dataset as 22 frames of
# "demonstration". Only one of those five stubs failed loudly, and only by luck.
#
# The floor sits well under any healthy reading and well over the broken ones.
# A rejection prints the measurements so the call can be judged rather than
# trusted; re-collect instead of relaxing it.
MIN_RECORDING_DENSITY = 0.60


def recording_density(seqs, n_rows):
    """Rows kept per privileged sequence unit; None when the span cannot say."""
    span = seqs[-1] - seqs[0] if len(seqs) >= 2 else 0
    return n_rows / span if span > 0 else None


def check_recording_density(sess, src_eps):
    """Refuse episodes that hold a fraction of the round they claim to record."""
    thin = []
    for ep in src_eps:
        sidecar = Path(sess) / f"priv_seq/episode_{ep:06d}.json"
        if not sidecar.exists():
            continue
        side = json.loads(sidecar.read_text())
        seqs = side["priv_seq"]
        rows = side.get("n_rows", len(seqs))
        density = recording_density(seqs, rows)
        if density is not None and density < MIN_RECORDING_DENSITY:
            thin.append((ep, rows, seqs[-1] - seqs[0], density))
    if thin:
        detail = "\n".join(
            f"    episode {ep}: {rows} rows over {span} sequence units ({d:.2f} rows/unit)"
            for ep, rows, span, d in thin
        )
        raise SystemExit(
            f"[recording] {Path(sess).name}: {len(thin)} of {len(src_eps)} episodes hold far "
            f"less than the round they record (floor {MIN_RECORDING_DENSITY}, healthy is ~1.1):\n"
            f"{detail}\n"
            f"The exporter dropped most of these frames, so they are stubs rather than "
            f"demonstrations. Re-collect this session; converting it puts near-empty "
            f"episodes into the dataset that nothing downstream will flag."
        )
