"""The instance: its port block, and whether its exporter is up.

`scripts/stack/start_stack.sh` derives the whole port block from `INSTANCE`
(base + 10*i, and DDS domain i) and exports the results, so a process the stack
itself launches inherits the right numbers. A process started from a different
shell -- an ad-hoc gym run, a diagnostic, a runner that only exported
`INSTANCE` -- inherits nothing, and every call site used to fall back to its own
hard-coded instance-0 number:

    PRIVILEGED_PORT = int(os.environ.get("TF_PRIVILEGED_PORT", "5581"))

With `INSTANCE=1` the stack then sits on 5565/5566/5591 while the gym binds
:5556 and waits on :5581, and the only symptom is `no privileged state on :5581
after 10s` ten seconds later (2026-09-10, cost a stack cycle to read). Deriving
the fallback from `INSTANCE` with the same formula the shell uses removes the
second source of truth: an exported TF_* still wins, so nothing about the
stack's own launch path changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bases are instance 0, which is byte-identical to the historical single-stack
# layout. Keep in step with start_stack.sh's block.
_BASES = {
    "TF_CAMERA_PORT": 5555,
    "TF_SONIC_PORT": 5556,
    "TF_OUT_PORT": 5557,
    "TF_KEYBOARD_PORT": 5580,
    "TF_PRIVILEGED_PORT": 5581,
    "TF_EPISODE_PORT": 5582,
    # The Isaac mirror's request port (sim/isaac/mirror.py). Residue 8: the 555x and 558x series above
    # use 5/6/7 and 0/1/2 (and lockstep's XSUB 4), so no instance's mirror lands on another's block.
    "TF_ISAAC_MIRROR_PORT": 5558,
}


def instance_ports(index: int) -> dict[str, int]:
    """Resolve the complete port block before crossing a process boundary."""
    if type(index) is not int or index < 0:
        raise ValueError("instance must be a non-negative integer")
    return {name: base + 10 * index for name, base in _BASES.items()}


def instance() -> int:
    try:
        return int(os.environ.get("INSTANCE", "0"))
    except ValueError:
        return 0


def port(name: str) -> int:
    """The port for `name`: an exported TF_* wins, else base + 10*INSTANCE."""
    if name not in _BASES:
        raise KeyError(f"{name} is not an instance port ({sorted(_BASES)})")
    value = os.environ.get(name)
    if value:
        return int(value)
    return _BASES[name] + 10 * instance()


def camera() -> int:
    return port("TF_CAMERA_PORT")


def sonic() -> int:
    return port("TF_SONIC_PORT")


def out() -> int:
    return port("TF_OUT_PORT")


def keyboard() -> int:
    return port("TF_KEYBOARD_PORT")


def privileged() -> int:
    return port("TF_PRIVILEGED_PORT")


def episode() -> int:
    return port("TF_EPISODE_PORT")


def isaac_mirror() -> int:
    return port("TF_ISAAC_MIRROR_PORT")


# Every entry point start_stack.sh can launch as the exporter. The default is
# gear_sonic's run_data_exporter.py; the pipeline overrides TF_EXPORTER_SCRIPT
# to our run_instance_exporter.py (the collect stage), so
# matching only the default name refuses every pipeline collect on a stack whose
# exporter is up (caught by session 65 on a live gate run, 2026-09-10). A new
# entry point has to be added here.
_EXPORTER_ENTRY_POINTS = (b"run_data_exporter", b"run_instance_exporter")


def _is_exporter_cmdline(cmdline: bytes) -> bool:
    return any(name in cmdline for name in _EXPORTER_ENTRY_POINTS)


def exporter_pid() -> int | None:
    """PID of THIS instance's data exporter, or None.

    Scoped by the exporter's own TF_CAMERA_PORT rather than by name alone: the
    entry point is shared across instances, and matching on the command line
    alone also catches any shell whose arguments merely mention it (a guard of
    mine false-positived on two-day-old zsh wrappers, 2026-09-10).
    """
    want = f"TF_CAMERA_PORT={camera()}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if not _is_exporter_cmdline(entry.joinpath("cmdline").read_bytes()):
                continue
            if want in entry.joinpath("environ").read_bytes().split(b"\0"):
                return int(entry.name)
        except OSError:
            continue  # the process exited while we looked at it
    return None


def require_exporter(why: str = "--record") -> None:
    """Refuse before a batch that could not possibly write anything.

    `start_stack.sh` starts the exporter only on instance 0 -- on any other
    instance it prints "[3/3] data exporter - skipped (instance N records no
    episodes)" and carries on -- so a recording run there completes every round
    and produces no session at all, silently (2026-09-10).
    """
    if exporter_pid() is not None:
        return
    raise SystemExit(
        f"{why} needs the data exporter, and none is running for instance {instance()} "
        f"(camera port {camera()}).\n"
        "  start_stack.sh starts the exporter on instance 0, or on another instance only with\n"
        "  RECORD_INSTANCE=1 (the collect stage sets it when recording); otherwise it prints\n"
        '  "[3/3] data exporter - skipped (instance N records no episodes)" and records nothing.'
    )
