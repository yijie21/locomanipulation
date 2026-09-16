"""The SONIC stack (sim, deploy, exporter) as supervised child processes of one Python object.

Decision 3 of docs/handoff/infra-plan-20260916.md replaces `scripts/stack/start_stack.sh`,
`stop_instance.sh`, `stop_stack.sh`, `restart_deploy.sh` and `restart_exporter.sh` with
`locomani.stack.Stack`: every child is started in its own session, logged to its own file, waited on
by an explicit readiness signal, and stopped by process group; the live stack's pids, ports and
paths are written to `<session_root>/stack.json` so any process on the box can attach to it.
"""

from locomani.stack.manager import Stack, StackSpec, session_root

__all__ = ["Stack", "StackSpec", "session_root"]
