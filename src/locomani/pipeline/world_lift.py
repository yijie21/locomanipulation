"""World-frame object height for converted episodes, read from the run's state trace.

The exporter's `observation.<object>_pose` is in the PELVIS frame and no root
position is recorded, so a rise judged from that column under-reads whenever
the pelvis itself rises (standing up out of a crouch, the whole-body lift). The
sim's state trace (`SIM_STATE_TRACE_DIR/<label>_state_trace.npz`: `seq`, `qpos`)
has the object's world pose at every control frame, and the exporter's
`priv_seq/episode_N.json` names the seq each parquet row used, so the world z
of every converted row can be looked up exactly -- no interpolation, no guess.

Layout this reads (the crate pilot's; the single-arm pilot has no per-round
label in its manifest yet):

    <run>/inputs.json          sessions[] and manifests[], same order
    <run>/manifest_NN.jsonl    one row per attempt; `produced` rows are the
                               session's episodes in order; `label` names the trace
    <run>/trace/<label>_state_trace.npz
    <run>/<scene>/scene_meta.json   target_object -> objects[...]["qpos_adr"]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC_FPS, DST_FPS = 50, 25  # keep in step with convert_to_hf_format (tested)
EDGE_TICKS = 2  # exporter rows allowed to lead/trail the trace, mapped to the edge sample


def resample_selection(n_source: int) -> np.ndarray:
    """The converter's 50 -> 25 Hz nearest-row pick, row indices into the source."""
    n_out = int(np.floor((n_source - 1) * DST_FPS / SRC_FPS)) + 1
    return np.clip(np.round(np.arange(n_out) * SRC_FPS / DST_FPS).astype(int), 0, n_source - 1)


class RunTraces:
    def __init__(self, run_dir):
        self.run = Path(run_dir)
        inputs = json.loads((self.run / "inputs.json").read_text())
        self.sessions = [str(Path(s)) for s in inputs["sessions"]]
        self.manifests = [Path(m) for m in inputs["manifests"]]
        metas = [p for p in self.run.glob("*/scene_meta.json") if "dataset" not in p.parts]
        if len(metas) != 1:
            raise ValueError(f"expected one scene copy under {self.run}, found {[str(m) for m in metas]}")
        meta = json.loads(metas[0].read_text())
        target = meta["target_object"]
        self.object = target
        self.qpos_adr = int(meta["objects"][target]["qpos_adr"])
        self.trace_dir = self.run / "trace"

    def label(self, session: str, source_episode: int) -> str:
        index = self.sessions.index(str(Path(session)))
        produced = [
            json.loads(line)
            for line in self.manifests[index].read_text().splitlines()
            if line.strip() and json.loads(line).get("produced")
        ]
        return produced[source_episode]["label"]

    def world_z(self, session: str, source_episode: int, kept_rows: int | None = None) -> np.ndarray:
        """World z of the target object for every converted row of one episode."""
        priv = json.loads((Path(session) / "priv_seq" / f"episode_{source_episode:06d}.json").read_text())["priv_seq"]
        priv = np.asarray(priv[: kept_rows if kept_rows is not None else len(priv)], dtype=np.int64)
        seqs = priv[resample_selection(len(priv))]
        trace = np.load(self.trace_dir / f"{self.label(session, source_episode)}_state_trace.npz")
        tseq = np.asarray(trace["seq"], dtype=np.int64)
        # The exporter's 'c' key and the sim's trace start travel on different
        # sockets, so the first exporter row can precede the trace by a tick or
        # two (2/209 episodes on 2026-09-14). Those edge rows take the edge
        # sample; anything else missing is a real join error.
        clipped = np.clip(seqs, tseq[0], tseq[-1])
        if np.any(seqs < tseq[0] - EDGE_TICKS) or np.any(seqs > tseq[-1] + EDGE_TICKS):
            raise ValueError("exporter priv_seq rows are not all present in the state trace")
        idx = np.searchsorted(tseq, clipped)
        if not np.array_equal(tseq[idx], clipped):
            raise ValueError("exporter priv_seq rows are not all present in the state trace")
        return np.asarray(trace["qpos"])[idx, self.qpos_adr + 2].astype(float)
