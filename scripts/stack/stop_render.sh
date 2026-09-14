#!/bin/bash
# Stop one managed render job, selected by a unique output-directory substring.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
exec python3 "$root/src/locomani/pipeline/render_job.py" stop "$@"
