"""Check actual returned RTC prefixes against previous captured predictions."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', type=Path)
    a = ap.parse_args()
    previous = None
    rows = []
    for file in sorted((a.capture/'chunks').glob('*.npz')):
        with np.load(file) as data:
            current = {k: data[k].copy() for k in data.files}
        if current.get('rtc.reset', 0): previous = None
        if previous is not None and 'rtc.elapsed' in current:
            elapsed = int(current['rtc.elapsed'])
            horizon = len(previous['token'])
            if 0 < elapsed < horizon-1:
                commit = bool(current.get('rtc.commit', False))
                frozen = min(int(current['rtc.frozen']), horizon-elapsed-(0 if commit else 1))
                old = previous['token'][elapsed:elapsed+frozen]
                new = current['token'][:frozen]
                token_equal = np.all(np.isclose(old, new, rtol=0, atol=1e-7), axis=1)
                grip_error = float(np.max(abs(previous['grip'][elapsed:elapsed+frozen]-current['grip'][:frozen]))) if frozen else 0.
                rows.append(dict(file=file.name, commit=commit, frozen=frozen,
                                 equal_token_frames=int(token_equal.sum()), grip_max_error=grip_error))
        previous = current if 'rtc.elapsed' in current else None
    summary = dict(requests=len(rows), committed_requests=sum(r['commit'] for r in rows),
                   frozen_frames=sum(r['frozen'] for r in rows),
                   equal_token_frames=sum(r['equal_token_frames'] for r in rows),
                   grip_max_error=max((r['grip_max_error'] for r in rows), default=0.), rows=rows)
    (a.capture/'rtc_prefix_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))
    if not rows: raise SystemExit('No overlapping RTC requests to verify')
    if summary['frozen_frames'] != summary['equal_token_frames'] or summary['grip_max_error'] > 1e-5:
        raise SystemExit('Returned RTC prefix did not preserve the previous prediction')


if __name__ == '__main__': main()
