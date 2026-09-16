"""Stack control from any shell: `python -m locomani.stack.cli up|down|status|restart-deploy|restart-exporter`.

    python -m locomani.stack.cli up --instance 0 --gpu 1 --scene outputs/scene/sim_lowcom --prompt "pick up the bottle"
    python -m locomani.stack.cli status --instance 0
    python -m locomani.stack.cli restart-deploy --instance 0      # what the gym's deploy self-heal calls
    python -m locomani.stack.cli down --instance 0

`up` blocks until the stack is ready and then returns, leaving the children running (their state is
in `<session_root>/stack.json`); `down`, `status` and the restarts attach to that state from any process.
"""

from __future__ import annotations

import argparse
import json
import sys

from locomani import paths
from locomani.stack.manager import Stack, StackSpec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("up", "down", "status", "restart-deploy", "restart-exporter"))
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--scene", default=paths.SIM_DIR, help="frozen scene directory")
    ap.add_argument("--prompt", default="walk forward")
    ap.add_argument("--no-record", action="store_true", help="no exporter")
    ap.add_argument("--policy", default=paths.SONIC_POLICY)
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--lockstep", action="store_true")
    args = ap.parse_args(argv)
    if args.verb == "up":
        spec = StackSpec(
            instance=args.instance,
            gpu=args.gpu,
            scene_dir=args.scene,
            prompt=args.prompt,
            record=not args.no_record,
            policy=args.policy,
            encoder=not args.no_encoder,
            lockstep=args.lockstep,
            auto_start=args.lockstep,
            default_motion="neutral_kick_R_001__A543" if args.lockstep else None,
        )
        stack = Stack(spec)
        stack.up()
        print(json.dumps(dict(instance=args.instance, pids=stack.pids, ports=spec.ports(), root=str(stack.root))))
        return
    try:
        stack = Stack.attach(args.instance)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if args.verb == "down":
        stack.down()
    elif args.verb == "status":
        print(json.dumps(dict(pids=stack.pids, **stack.health()), indent=1))
    elif args.verb == "restart-deploy":
        sys.exit(0 if stack.restart_deploy() else 1)
    elif args.verb == "restart-exporter":
        stack.restart_exporter()


if __name__ == "__main__":
    main()
