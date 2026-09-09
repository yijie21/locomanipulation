"""Audit explicit async chunk provenance and token evidence in WBC debug.

A payload match is evidence of consumption of that value, not a unique wire
acknowledgement. Repeated values retain all candidate sends; never pick one.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def candidates(token, sent_tokens, sent_times, wall, max_age=0.5):
    ix = np.flatnonzero((sent_times <= wall) & (sent_times >= wall - max_age))
    if len(ix) == 0:
        return []
    return ix[np.max(np.abs(sent_tokens[ix] - np.asarray(token, dtype=np.float32)), axis=1) < 1e-7].tolist()


def audit(root):
    sent = [json.loads(s) for s in (root / "sent.jsonl").read_text().splitlines()]
    if not sent:
        raise ValueError("capture has no sent actions")
    status_path = root / "capture_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("dropped") or status.get("errors") or status.get("writer_alive"):
            raise ValueError("incomplete capture; absence of a send cannot be inferred")
    times = np.array([s["wall"] for s in sent])
    tokens = np.array([s["token"] for s in sent], dtype=np.float32)
    seen = set()
    ambiguous = 0
    debug_count = 0
    body = []
    with (root / "debug.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            token = row["state"].get("token_state", [])
            q = row["state"].get("body_q", [])
            if len(q) >= 10:
                body.append((row["wall"], float(np.mean(np.asarray(q)[[3, 9]]))))
            if len(token) != tokens.shape[1]:
                continue
            debug_count += 1
            ids = candidates(token, tokens, times, row["wall"])
            if len(ids) == 1:
                seen.add(ids[0])
            elif len(ids) > 1:
                ambiguous += 1
    chunks = {}
    errors = []
    for i, row in enumerate(sent):
        source = row.get("source") or {}
        cid = source.get("chunk_id")
        j = source.get("action_index")
        if cid is None or j is None:
            continue
        if cid not in chunks:
            with np.load(root / "chunks" / f"{cid:06d}.npz") as z:
                predicted_token = z["token"]
            chunks[cid] = dict(
                token=predicted_token, indices=set(), sent=0, unique_payload_evidence=0, first_send_wall=row["wall"]
            )
        c = chunks[cid]
        if not 0 <= j < len(c["token"]) or not np.allclose(c["token"][j], row["token"], rtol=0, atol=1e-7):
            errors.append(dict(send=i, chunk_id=cid, index=j))
            continue
        c["indices"].add(j)
        c["sent"] += 1
        c["unique_payload_evidence"] += i in seen
    for c in chunks.values():
        t = c["first_send_wall"]
        before = [q for wall, q in body if wall <= t]
        after = [q for wall, q in body if t < wall <= t + 1.0]
        c["body_knee_change_next_1s_rad"] = max(after) - before[-1] if before and after else None
    return dict(
        chunks={
            k: dict(
                indices=sorted(v["indices"]),
                sent=v["sent"],
                unique_payload_evidence=v["unique_payload_evidence"],
                body_knee_change_next_1s_rad=v["body_knee_change_next_1s_rad"],
            )
            for k, v in chunks.items()
        },
        provenance_status=("mismatch" if errors else "available" if chunks else "missing"),
        provenance_mismatches=errors,
        debug_frames=debug_count,
        ambiguous_debug_frames=ambiguous,
        receipt_limitation="payload evidence only; deploy does not echo input IDs; body response is temporal association, not causal attribution",
        unattributed_sends=sum((s.get("source") or {}).get("chunk_id") is None for s in sent),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = audit(a.capture)
    a.out.write_text(json.dumps(result, indent=2))
    if result["provenance_mismatches"]:
        raise SystemExit("Chunk/send payload mismatch")
