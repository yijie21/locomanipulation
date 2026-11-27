"""selfmodel_hardgrasp_20261125 (BiGym): m2.py's online row against bigym_data.py build --end early's table on recorded
bench J episodes (sm_real: the films and the replay_early table built from them), and its timing.

    source <bigym2>/env.sh; python scripts/experiments/hardgrasp_20261125/m2_check.py \
        outputs/hardgrasp_20261125/bigym/m2 outputs/bigym_20261124/sm_real [n]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2 as M2mod  # noqa: E402

TASK_ID = {"reach_target_multi_modal": 0, "drawer_top_open": 1}


def main() -> None:
    m2dir, real = Path(sys.argv[1]), Path(sys.argv[2])
    n_want = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    m = M2mod.M2(m2dir / "m2_weights.npz", m2dir / "band.npz")
    T = np.load(real / "replay_early" / "table.npz")
    rows = {}
    for f in sorted(real.glob("rows_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[(TASK_ID[r["task"]], int(r["seed"]))] = r
    picks = []
    for tid in (0, 1):  # both tasks
        idx = np.where(T["task"] == tid)[0]
        picks += list(idx[: (n_want + 1) // 2 if tid == 0 else n_want // 2])
    worst = dict(x=0.0, ref=0.0, label=0.0)
    for j in picks:
        tid, seed = int(T["task"][j]), int(T["seed"][j])
        r = rows[(tid, seed)]
        fp = real / "films" / f"{r['task']}_{seed}_J{'_stance' if r.get('stance') else ''}.npz"
        z = dict(np.load(fp))
        ph = z["phase"]
        i_cmd = int(np.where(ph == "reach")[0][0]) - 1
        n_reach = int((ph[i_cmd + 1 :] == "reach").sum())
        refs, ctrl0 = z["sent"][i_cmd + 1 : i_cmd + 1 + n_reach], z["sent"][i_cmd]
        dur = float(r["reach"]["dur_s"])
        h = 0.62 if (r["task"] == "drawer_top_open" and r.get("stance")) else -1.0
        t0 = time.time()
        out = m.predict(z, i_cmd, refs, ctrl0, dur, h)
        t_pred = time.time() - t0
        lab = m.label(z, i_cmd, n_reach, dur)
        n29, n7, _ = m.out_of_band(z, i_cmd)
        dx = float(np.abs(out["x"] - T["x"][j]).max())
        dref = float(np.abs(out["ref"] - T["ref"][j]).max())
        off_t = T["off"][j][M2mod.Q][m.palm].mean(0)
        dlab = float(np.abs(lab - off_t).max())
        worst = {k: max(worst[k], v) for k, v in (("x", dx), ("ref", dref), ("label", dlab))}
        print(f"{r['task']} {seed}: i_cmd {i_cmd}, n_reach {n_reach}, dur {dur}; |x - table| max {dx:.2e}, |ref - table| "
              f"max {dref:.2e} m, |label - table off[Q] palm| max {dlab:.2e} m; p_head {np.round(100 * out['p_head'], 2)} "
              f"cm, label {np.round(100 * lab, 2)} cm, |p - label| {100 * np.linalg.norm(out['p_head'] - lab):.2f} cm; "
              f"out of band 29: {n29}, arm 7: {n7}; predict {1000 * t_pred:.1f} ms")  # fmt: skip
    t0 = time.time()
    for _ in range(10):
        m.predict(z, i_cmd, refs, ctrl0, dur, h)
    print(f"worst: x {worst['x']:.2e}, ref {worst['ref']:.2e} m, label {worst['label']:.2e} m; predict "
          f"{100 * (time.time() - t0):.1f} ms a call (mean of 10, warm)")  # fmt: skip


if __name__ == "__main__":
    main()
