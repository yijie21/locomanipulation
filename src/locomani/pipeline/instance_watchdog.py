#!/usr/bin/env python3
"""Watch one stack instance and say when it stops being healthy.

Four things can go wrong quietly, and each has cost a run:

  process   the sim died and everything downstream blocked on a socket that
            will never answer
  port      the sim is up but the camera port never opened, so every round
            fails identically with "camera/state streams never became ready"
  stream    the port is open and the frames are arriving at a fraction of the
            target rate -- measured 13-15 Hz against a 30 Hz target while
            closed-loop numbers were being produced from it
  progress  the manifest stopped growing: the gym is wedged in a futex while
            sim and deploy carry on looking perfectly healthy (2026-08-25,
            eight hours)

**It reports and does not act.** No kill, no restart. Whether a wedged
instance should be restarted depends on what is being measured -- mid-A/B a
restart silently changes the conditions the arm was recorded under -- so that
decision belongs to the driver script or to a person. This process only makes
the state legible, one line per transition, in the same shape as
frame_rate_guard's --watch so a driver can parse both with one rule.

    instance_watchdog.py --instance 0 \
        --manifest outputs/eval/seq_ens/a.manifest.jsonl

    # thresholds via flags, so a driver can set them once for all instances
    instance_watchdog.py --stream-hz 25 --stall-s 600 --instance 1 ...

Output lines:

    [wd i0] DOWN process      sim process is gone
    [wd i0] OK process        sim process is back
    [wd i0] DOWN port         :5555 not listening
    [wd i0] DOWN stream       median 13.2 Hz over last 10 (threshold 25.0)
    [wd i0] DOWN progress     manifest unchanged for 612 s (threshold 600)

Exit status is 0 on Ctrl-C. Nothing here fails a build.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

FREQ_RE = re.compile(r"Image publish frequency:\s+([0-9.]+)")


def sim_alive(instance: int) -> tuple[bool, bool]:
    """-> (alive, precise).

    `start_stack.sh` passes `--instance N` to the sim purely so its cmdline
    names which stack it belongs to (the sim strips it before tyro). When the
    running sim carries that flag the answer is exact. A sim started before
    that landed does not carry it, and there is no other way to tell two of
    them apart -- the instance lives in the DDS domain and the log path, not
    on the cmdline -- so the check falls back to "is any sim running", says so
    via `precise=False`, and does NOT report a healthy old-cmdline sim as
    down. A watchdog that cries wolf gets turned off.

    Matches on argv ELEMENTS, not on a substring of the joined cmdline. The
    substring version is wrong in a way that is easy to miss and was caught
    here only because it fired: a shell running

        cp src/locomani/sim/run_sim_locomani.py /tmp/x && ... --instance 7

    carries both needles inside its single `-c` argument, so a substring test
    counts that shell as a tagged sim and every real sim then looks absent.
    The same class of bug has cost this project hours before (a `pkill -f`
    whose pattern matched the shell that invoked it). Scanning /proc instead
    of shelling out to pgrep does NOT fix it -- the string is in an ancestor's
    cmdline either way. Requiring `run_sim_locomani.py` to BE an argument, and
    `--instance` to be an argument immediately followed by the number, does:
    a shell has the whole command in one element and cannot match.
    """
    script = "%s_%s.py" % ("run_sim", "locomani")
    flag = "--%s" % "instance"
    me = os.getpid()
    found_any = found_tagged = tagged_exists = False
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if not any(a == script or a.endswith("/" + script) for a in argv):
            continue
        found_any = True
        if flag in argv:
            tagged_exists = True
            k = argv.index(flag)
            if k + 1 < len(argv) and argv[k + 1] == str(instance):
                found_tagged = True
    if tagged_exists:
        return found_tagged, True
    return found_any, False


def port_listening(port: int) -> bool:
    try:
        out = subprocess.run(["ss", "-lnt"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return False
    return f":{port} " in out


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch one stack instance.")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--sim-log", help="default: $LOCOMANI_TMP/p03[_i<N>]/sim.log")
    ap.add_argument("--manifest", help="growth is the liveness signal")
    ap.add_argument("--camera-port", type=int, help="default: 5555 + 10 * instance")
    # Thresholds. Argparse flags, same names/defaults as the old
    # WATCHDOG_*/LOCOMANI_TMP env vars (decision 8: a diagnostic tool's knobs
    # are CLI flags, not config-dataclass fields) -- a driver still sets them
    # once for every instance it spawns, just as flags instead of env.
    ap.add_argument("--stream-hz", type=float, default=25.0)
    ap.add_argument("--stream-window", type=int, default=10)
    ap.add_argument("--stall-s", type=float, default=600.0)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--tmp", help="default: $LOCOMANI_WORKSPACE/tmp")
    args = ap.parse_args()

    i = args.instance
    STREAM_HZ, STREAM_WINDOW, STALL_S, POLL_S = args.stream_hz, args.stream_window, args.stall_s, args.poll_s
    tmp = args.tmp or os.path.join(os.environ.get("LOCOMANI_WORKSPACE", "/workspace"), "tmp")
    sim_log = args.sim_log or os.path.join(tmp, "p03" if i == 0 else f"p03_i{i}", "sim.log")
    port = args.camera_port if args.camera_port is not None else 5555 + 10 * i
    tag = f"[wd i{i}]"

    def say(state: str, what: str, detail: str = "") -> None:
        print(f"{tag} {state:4s} {what:9s} {detail}".rstrip(), flush=True)

    say("OK", "start", f"log={sim_log} port={port} stream<{STREAM_HZ}Hz stall>{STALL_S:.0f}s")

    fh = None
    recent: list[float] = []
    # None = not yet determined, so the first observation always prints.
    health: dict[str, bool | None] = {"process": None, "port": None, "stream": None, "progress": None}
    last_size, last_change = None, time.time()

    def transition(key: str, ok: bool, detail: str = "") -> None:
        if health[key] is ok:
            return
        health[key] = ok
        say("OK" if ok else "DOWN", key, detail)

    try:
        while True:
            alive, precise = sim_alive(i)
            transition(
                "process",
                alive,
                ("" if alive else "sim process is gone")
                + (
                    ""
                    if precise
                    else "  (no --instance on the cmdline: this is 'any sim',"
                    " not 'this instance' -- restart the stack to get"
                    " the precise check)"
                ),
            )
            listening = port_listening(port)
            transition("port", listening, "" if listening else f":{port} not listening")

            # stream: read whatever is new since last poll
            if fh is None and os.path.exists(sim_log):
                fh = open(sim_log, errors="ignore")
                fh.seek(0, os.SEEK_END)
            if fh is not None:
                for line in fh:
                    m = FREQ_RE.search(line)
                    if m:
                        recent.append(float(m.group(1)))
                        del recent[:-STREAM_WINDOW]
                if len(recent) >= STREAM_WINDOW:
                    med = statistics.median(recent)
                    transition(
                        "stream",
                        med >= STREAM_HZ,
                        f"median {med:.1f} Hz over last {STREAM_WINDOW} (threshold {STREAM_HZ})",
                    )

            # progress: the manifest is the only thing that only grows when
            # rounds actually complete
            if args.manifest:
                try:
                    size = os.path.getsize(args.manifest)
                except OSError:
                    size = None
                if size is not None:
                    if size != last_size:
                        last_size, last_change = size, time.time()
                    quiet = time.time() - last_change
                    transition(
                        "progress", quiet <= STALL_S, f"manifest unchanged for {quiet:.0f} s (threshold {STALL_S:.0f})"
                    )
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        say("OK", "stop")
        sys.exit(0)


if __name__ == "__main__":
    main()
