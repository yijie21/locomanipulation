"""Run the existing evaluation with independent observation/action capture.

Usage: diag_transition_live.py CAPTURE_DIR [eval_closed_loop.py arguments...]
Requires the same stack/policy server as eval_closed_loop.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

from locomani import paths

sys.path.insert(0, paths.SONIC)
from locomani.rtc.transition_capture import Capture


def main():
    root = Path(sys.argv.pop(1)).resolve()
    if root.exists():
        raise SystemExit(f"Refusing to overwrite capture: {root}")
    cap = Capture(root)
    cap.instrument()
    from locomani.tasks.bottle import grasp_gym

    original_round = grasp_gym._run_round

    def measured_round(*args, **kwargs):
        cap.event("round_start")
        result = original_round(*args, **kwargs)
        cap.event("round_end", result=result)
        return result

    grasp_gym._run_round = measured_round
    # Diagnostic intervention only. Privileged distance chooses WHEN to
    # change cadence, never an action. The normal evaluation remains intact.
    intervention = None
    if "--near-replan" in sys.argv:
        i = sys.argv.index("--near-replan")
        interval = float(sys.argv[i + 1])
        intervention = dict(
            near_replan_seconds=interval, trigger_distance=0.45, privileged_trigger=True, benchmark_compatible=False
        )
        del sys.argv[i : i + 2]
        from locomani.backends.sonic import vla_round

        original_factory = vla_round.make_vla_policy_async

        class NearInterval:
            active = False

            def __sub__(self, elapsed):
                state = cap.latest_priv
                if not self.active and state is not None:
                    obj = state["objects"]["water_bottle"]["pos"]
                    pel = state["pelvis"]["pos"]
                    if np.linalg.norm(np.array(obj[:2]) - pel[:2]) < 0.45:
                        self.active = True
                        cap.event("near_replan_activated", seconds=interval, state=state)
                return (interval if self.active else 0.4) - elapsed

        def factory(*args, **kwargs):
            cadence = NearInterval()
            kwargs["inference_interval"] = cadence
            policy = original_factory(*args, **kwargs)

            def run(*args, **kwargs):
                cadence.active = False
                return policy(*args, **kwargs)

            return run

        vla_round.make_vla_policy_async = factory
        cap.event(
            "diagnostic_intervention",
            near_replan=interval,
            trigger_distance=0.45,
            note="Uses privileged trigger; not a deployable policy or benchmark arm.",
        )
    try:
        from locomani.pipeline import eval_closed_loop

        cap.event("evaluation_start", argv=sys.argv)
        eval_closed_loop.main()
        if intervention is not None and "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            record = json.loads(out.read_text())
            record["diagnostic_intervention"] = intervention
            out.write_text(json.dumps(record, indent=2))
    finally:
        cap.event("evaluation_end")
        cap.close()


if __name__ == "__main__":
    main()
