"""selfmodel_simplesonicdata_20261126: our processes' CPU use on H20b, measured as their CPU time (the coordinator's
quota: the two SIMPLE cells together <= 140 cores, us <= 160 on the machine); read only, stdlib only.

    python3 ssd_cpu.py <interval s> <log> PATTERN [PATTERN ...]

Every interval: the processes whose command line holds a pattern (this sampler left out), the CPU time (utime + stime)
each spent since the last sample (a process born since then: all of its CPU time), summed per pattern and in all, over
the interval's wall time = cores; the number of processes and their resident memory (GB). One line a sample."""
import os
import sys
import time

TCK = os.sysconf("SC_CLK_TCK")
PG = os.sysconf("SC_PAGE_SIZE")
me = os.getpid()


def scan(pats):
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            hit = [p for p in pats if p in cmd]
            if not hit:
                continue
            f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
            out[int(d)] = (hit[0], (int(f[11]) + int(f[12])) / TCK, int(f[19]), int(f[21]) * PG)  # utime+stime s, starttime, rss
        except (OSError, IndexError, ValueError):
            continue
    return out


def main():
    dt, log, pats = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
    prev, t0 = scan(pats), time.time()
    while True:
        time.sleep(dt)
        cur, t1 = scan(pats), time.time()
        per = {p: [0.0, 0, 0.0] for p in pats}
        for pid, (p, c, st, rss) in cur.items():
            old = prev.get(pid)
            d = c - old[1] if old and old[2] == st else c
            per[p][0] += max(d, 0.0)
            per[p][1] += 1
            per[p][2] += rss / 1e9
        w = t1 - t0
        tot = sum(v[0] for v in per.values()) / w
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)) + f" all {tot:6.1f} cores, {sum(v[1] for v in per.values())} procs | " + " | ".join(
            f"{p.split('/')[-1]}: {v[0] / w:6.1f} cores, {v[1]} procs, {v[2]:.1f} GB" for p, v in per.items())
        with open(log, "a") as f:
            f.write(line + "\n")
        prev, t0 = cur, t1


if __name__ == "__main__":
    main()
