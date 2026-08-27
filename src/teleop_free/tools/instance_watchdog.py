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

    # thresholds via env, so a driver can set them once for all instances
    WATCHDOG_STREAM_HZ=25 WATCHDOG_STALL_S=600 instance_watchdog.py --instance 1 ...

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

# Thresholds. Environment rather than flags so a driver can set them once for
# every instance it spawns.
STREAM_HZ = float(os.environ.get("WATCHDOG_STREAM_HZ", "25"))
STREAM_WINDOW = int(os.environ.get("WATCHDOG_STREAM_WINDOW", "10"))
STALL_S = float(os.environ.get("WATCHDOG_STALL_S", "600"))
POLL_S = float(os.environ.get("WATCHDOG_POLL_S", "10"))


def sim_alive(instance: int) -> bool:
    """The sim for this instance, identified by its own cmdline. The pattern is
    assembled at run time: written as a literal it would appear in this
    process's own /proc/*/cmdline and match itself."""
    pat = "%s_%s.py" % ("run_sim", "locomani")
    try:
        out = subprocess.run(["pgrep", "-af", pat], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:                                          # noqa: BLE001
        return False
    for line in out.splitlines():
        if str(os.getpid()) in line.split()[:1]:
            continue
        # One sim per instance; the instance shows up in its log path or in
        # SONIC_DDS_DOMAIN, neither of which is on the cmdline. With a single
        # stack this is unambiguous; with two, presence of ANY sim is the
        # weaker claim this check can honestly make, so it is what it claims.
        return True
    return False


def port_listening(port: int) -> bool:
    try:
        out = subprocess.run(["ss", "-lnt"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:                                          # noqa: BLE001
        return False
    return f":{port} " in out


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch one stack instance.")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--sim-log", help="default: $LOCOMANI_TMP/p03[_i<N>]/sim.log")
    ap.add_argument("--manifest", help="growth is the liveness signal")
    ap.add_argument("--camera-port", type=int,
                    help="default: 5555 + 10 * instance")
    args = ap.parse_args()

    i = args.instance
    tmp = os.environ.get("LOCOMANI_TMP") or os.path.join(
        os.environ.get("LOCOMANI_WORKSPACE", "/workspace"), "tmp")
    sim_log = args.sim_log or os.path.join(
        tmp, "p03" if i == 0 else f"p03_i{i}", "sim.log")
    port = args.camera_port if args.camera_port is not None else 5555 + 10 * i
    tag = f"[wd i{i}]"

    def say(state: str, what: str, detail: str = "") -> None:
        print(f"{tag} {state:4s} {what:9s} {detail}".rstrip(), flush=True)

    say("OK", "start", f"log={sim_log} port={port} "
        f"stream<{STREAM_HZ}Hz stall>{STALL_S:.0f}s")

    fh = None
    recent: list[float] = []
    # None = not yet determined, so the first observation always prints.
    health: dict[str, bool | None] = {
        "process": None, "port": None, "stream": None, "progress": None}
    last_size, last_change = None, time.time()

    def transition(key: str, ok: bool, detail: str = "") -> None:
        if health[key] is ok:
            return
        health[key] = ok
        say("OK" if ok else "DOWN", key, detail)

    try:
        while True:
            transition("process", sim_alive(i),
                       "" if sim_alive(i) else "sim process is gone")
            transition("port", port_listening(port),
                       "" if port_listening(port) else f":{port} not listening")

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
                    transition("stream", med >= STREAM_HZ,
                               f"median {med:.1f} Hz over last {STREAM_WINDOW} "
                               f"(threshold {STREAM_HZ})")

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
                    transition("progress", quiet <= STALL_S,
                               f"manifest unchanged for {quiet:.0f} s "
                               f"(threshold {STALL_S:.0f})")
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        say("OK", "stop")
        sys.exit(0)


if __name__ == "__main__":
    main()
