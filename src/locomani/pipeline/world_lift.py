"""World-frame object height for converted episodes, read from the run's state trace.

The exporter's `observation.<object>_pose` is in the PELVIS frame and no root
position is recorded, so a rise judged from that column under-reads whenever
the pelvis itself rises (standing up out of a crouch, the whole-body lift). The
sim's state trace (`SIM_STATE_TRACE_DIR/<label>_state_trace.npz`: `seq`, `qpos`)
has the object's world pose at every control frame, and the exporter's
`priv_seq/episode_N.json` names the seq each parquet row used, so the world z
of every converted row can be looked up exactly -- no interpolation, no guess.

Two layouts are read. The crate runs name each round's trace in the manifest:

    <run>/inputs.json          sessions[] and manifests[], same order
    <run>/manifest_NN.jsonl    one row per attempt; `produced` rows are the
                               session's episodes in order; `label` names the trace
    <run>/trace/<label>_state_trace.npz
    <run>/<scene>/scene_meta.json   target_object -> objects[...]["qpos_adr"]

The single-arm stage layout carries no label, so the episode is joined to the
trace whose sequence window contains its rows:

    <run>/collect/inputs.json, <run>/collect/manifest_NN.jsonl
    <run>/collect/state_trace_NN/<gym_id>_state_trace.npz
    <run>/scene/scene_meta.json
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
        inputs_path = self.run / "collect" / "inputs.json"  # stage layout
        if not inputs_path.exists():
            inputs_path = self.run / "inputs.json"  # crate pilot layout
        inputs = json.loads(inputs_path.read_text())
        self.sessions = [str(Path(s)) for s in inputs["sessions"]]
        self.manifests = [Path(m) for m in inputs["manifests"]]
        self._ranges: dict[Path, list[tuple[Path, int, int]]] = {}
        metas = [p for p in self.run.glob("*/scene_meta.json") if "dataset" not in p.parts]
        if len(metas) != 1:
            raise ValueError(f"expected one scene copy under {self.run}, found {[str(m) for m in metas]}")
        self.scene_dir = metas[0].parent
        meta = json.loads(metas[0].read_text())
        target = meta["target_object"]
        self.object = target
        self.qpos_adr = int(meta["objects"][target]["qpos_adr"])
        self.box = meta["objects"][target].get("box")  # half_extent + centre in the object frame
        self.trace_dir = self.run / "trace"

    def label(self, session: str, source_episode: int) -> str | None:
        index = self.sessions.index(str(Path(session)))
        produced = [
            json.loads(line)
            for line in self.manifests[index].read_text().splitlines()
            if line.strip() and json.loads(line).get("produced")
        ]
        return produced[source_episode].get("label")

    def _seq_ranges(self, directory: Path) -> list[tuple[Path, int, int]]:
        """[(trace, first seq, last seq)] for one batch's traces; read once per directory."""
        if directory not in self._ranges:
            ranges = []
            for p in sorted(directory.glob("*_state_trace.npz")):
                seq = np.asarray(np.load(p)["seq"], dtype=np.int64)
                ranges.append((p, int(seq[0]), int(seq[-1])))
            self._ranges[directory] = ranges
        return self._ranges[directory]

    def _trace_path(self, session: str, source_episode: int, seqs: np.ndarray) -> Path:
        """The trace of one episode: by label where the expert writes one, else by seq coverage.

        The single-arm manifests carry no per-round label, and a batch's traces are not one per
        produced episode (a round that fails before the first control tick writes none), so the
        episode is joined to the trace whose sequence window contains its rows -- which also
        proves the join instead of assuming an order.
        """
        label = self.label(session, source_episode)
        if label:
            return self.trace_dir / f"{label}_state_trace.npz"
        batch = self.manifests[self.sessions.index(str(Path(session)))].stem.rsplit("_", 1)[-1]
        # One directory per run since 2026-09-21 (the stack outlives a batch, so SIM_STATE_TRACE_DIR
        # is fixed when the sim starts); runs collected before that have one per batch. Try the
        # current layout first and fall back, so an old run stays auditable. "Current" means it
        # holds traces, not merely exists: a later stack pointed at an old run creates the
        # directory empty (bottle_simple_200_20260916, 2026-09-22), and every join then failed.
        directory = self.run / "collect" / "state_trace"
        if not any(directory.glob("*_state_trace.npz")):
            directory = self.run / "collect" / f"state_trace_{batch}"
        lo, hi = int(seqs[0]), int(seqs[-1])
        hits = [
            p for p, first, last in self._seq_ranges(directory) if first - EDGE_TICKS <= lo <= hi <= last + EDGE_TICKS
        ]
        if len(hits) != 1:
            raise ValueError(
                f"episode {source_episode} of {session}: seq [{lo}..{hi}] matches {len(hits)} traces in {directory}"
            )
        return hits[0]

    def priv_seq(self, session: str, source_episode: int) -> np.ndarray:
        """The sim seq every exporter row of one episode used, one per parquet row."""
        priv = json.loads((Path(session) / "priv_seq" / f"episode_{source_episode:06d}.json").read_text())["priv_seq"]
        return np.asarray(priv, dtype=np.int64)

    def trace_rows(self, session: str, source_episode: int, seqs: np.ndarray, *, hold_tail: bool = False):
        """(trace, index of each seq in it) for exporter rows of one episode; raises on a join error.

        `hold_tail`: rows past the trace's last seq take its last sample. The exporter keeps recording ~0.5 s
        after the sim stops tracing (the stop travels on another socket; ~25 rows per crate episode,
        2026-09-24), and convert trims those rows with the verification tail -- but every row still needs a
        frame in the session's videos."""
        trace = np.load(self._trace_path(session, source_episode, seqs))
        tseq = np.asarray(trace["seq"], dtype=np.int64)
        if hold_tail:
            seqs = np.minimum(seqs, tseq[-1])
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
        return trace, idx

    def world_z(self, session: str, source_episode: int, kept_rows: int | None = None) -> np.ndarray:
        """World z of the target object for every converted row of one episode."""
        priv = self.priv_seq(session, source_episode)
        priv = priv[: kept_rows if kept_rows is not None else len(priv)]
        seqs = priv[resample_selection(len(priv))]
        trace, idx = self.trace_rows(session, source_episode, seqs)
        return np.asarray(trace["qpos"])[idx, self.qpos_adr + 2].astype(float)
