"""selfmodel_lmlroute_20261126 lrt_cpu.py [seconds] [pattern]: cores used by every process visible in this container (utime + stime deltas over an interval), not loadavg;
prints "<all processes> <processes whose command line contains pattern>"."""
import os, sys, time
dt = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
pat = sys.argv[2] if len(sys.argv) > 2 else None
hz = os.sysconf("SC_CLK_TCK")
def snap():
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            s = open(f"/proc/{p}/stat").read()
            f = s[s.rfind(")") + 2:].split()
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            out[p] = (int(f[11]) + int(f[12]), cmd)
        except Exception:
            pass
    return out
a = snap(); t0 = time.time(); time.sleep(dt); b = snap(); el = time.time() - t0
tot = sub = 0.0
for p, (t, cmd) in b.items():
    if p in a:
        d = (t - a[p][0]) / hz / el
        tot += d
        if pat and pat in cmd:
            sub += d
print(f"{tot:.1f} {sub:.1f}")
