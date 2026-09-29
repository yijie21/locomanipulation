#!/bin/bash
# Stop the Unreal editor scripts/start_unreal.sh started, and its guard. SIGKILL: the editor segfaults on SIGTERM
# at shutdown anyway (harmless), and a level or import is saved when it is built, not at exit.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
for p in $(pgrep -f "^bash $LOCOMANI_ROOT/scripts/unreal/editor_guard.sh") $(pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/"); do
    kill -KILL "$p" 2>/dev/null
done
echo "unreal editor stopped"
