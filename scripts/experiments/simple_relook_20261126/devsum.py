"""simple_relook_20261126 development: one line per look of simple_bench.py --relook rows, and a summary per file.
    python3 devsum.py <rows.jsonl> ...
"""
import json
import sys

import numpy as np

for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f)]
    pe, ee, ed, pd, steps, first, last, done, told = [], [], [], [], [], [], [], [], []
    for r in R:
        rk = r.get("relook") or {}
        L = rk.get("looks", [])
        if not L:
            continue
        steps.append(rk["steps"])
        first.append(np.linalg.norm(L[0]["stance_err_true_cm"]))
        last.append(np.linalg.norm(L[-1]["stance_err_true_cm"]))
        for x in L:
            pe.append(x["prior_err_true_cm"]); pd.append(abs(x["prior_err_true_deg"]))
            ee.append(float(np.linalg.norm(x["est_err_true_cm"]))); ed.append(abs(x["est_err_true_deg"]))
            if "step_done_true_cm" in x:
                done.append(x["step_done_true_cm"]); told.append(x["step_h_cm"])
        if "-v" in sys.argv[0:1] or True:
            print(f"  ep {r['ep']} t_end {rk.get('t_end')} steps {rk['steps']} stance |.| " + " -> ".join(
                f"{np.linalg.norm(x['stance_err_true_cm']):.1f}" for x in L) + " | off " + " -> ".join(f"{x['off_n_cm']:.1f}" for x in L)
                + " | prior err " + " ".join(f"{x['prior_err_true_cm']:.2f}/{x['prior_err_true_deg']:.2f}" for x in L)
                + " | est err " + " ".join(f"{np.linalg.norm(x['est_err_true_cm']):.2f}/{x['est_err_true_deg']:.2f}" for x in L)
                + " | step told/done " + " ".join(f"{x.get('step_h_cm')}/{x.get('step_done_true_cm')}" for x in L if 'step' in x))
    q = lambda v: f"median {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} max {np.max(v):.2f} (n {len(v)})" if len(v) else "-"
    print(f"{f}: episodes {len(steps)}, steps {np.bincount(steps).tolist()}")
    print(f"   prior (odometry) error cm {q(pe)}, deg {q(pd)}")
    print(f"   estimate error      cm {q(ee)}, deg {q(ed)}")
    print(f"   stance error before {q(first)}; after {q(last)}")
    if done:
        d, t = np.array(done), np.array(told)
        print(f"   steps: told |.| {q(np.linalg.norm(t, axis=1))}; done-told |.| {q(np.linalg.norm(d - t, axis=1))}; "
              f"along told: {q((d * t).sum(1) / np.maximum(np.linalg.norm(t, axis=1) ** 2, 1e-9))}")
