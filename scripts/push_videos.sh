#!/bin/bash
# Refresh videos/ with the latest attempt's episode videos and push.
#
# Usage:
#   bash push_videos.sh <file-or-dir> [more files...] [-m "commit message"]
#
# Example (after a batch):
#   bash push_videos.sh outputs/videos/pbatch2_ok -m "pbatch2: 9/21 OK"
#
# Clears videos/ first (the folder always holds ONLY the newest attempt),
# copies the given .mp4 files (or every .mp4 in a given dir), skips anything
# over 95 MB (GitHub hard limit is 100 MB), commits and pushes.
set -euo pipefail
cd "$(dirname "$0")/.."

MSG=""
SRCS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MSG="$2"; shift 2 ;;
        *)  SRCS+=("$1"); shift ;;
    esac
done
[ ${#SRCS[@]} -gt 0 ] || { echo "usage: bash push_videos.sh <file-or-dir>... [-m msg]"; exit 1; }

rm -rf videos && mkdir -p videos
for s in "${SRCS[@]}"; do
    if [ -d "$s" ]; then FILES=("$s"/*.mp4); else FILES=("$s"); fi
    for f in "${FILES[@]}"; do
        [ -f "$f" ] || continue
        if [ "$(stat -c%s "$f")" -gt 99614720 ]; then
            echo "skip (>95MB, GitHub limit): $f"
            continue
        fi
        cp "$f" videos/
    done
done
ls -la videos/

git add -A videos
git commit -m "${MSG:-videos: latest attempt}" || { echo "nothing to commit"; exit 0; }
git push origin HEAD
