#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# Freeze the hatask code (revision 2 section 5): the development copy dev_code copied to asrun/ (no __pycache__), the md5
# of every regular file (the links repo/assets and repo/third_party are listed, not followed), FROZEN_UTC, read-only.
#   ht_freeze.sh          -> asrun/ (refuses when it exists)
#   ht_freeze.sh check    -> md5sum -c of asrun/ against its MD5 (every run from asrun starts with it)
set -euo pipefail
OT=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hatask_20261126
S=$OT/dev_code F=$OT/asrun
if [ "${1:-}" = check ]; then
  (cd $F && md5sum -c --quiet MD5) && echo "asrun = MD5 ($(cat $F/FROZEN_UTC))"
  exit
fi
[ -e $F ] && { echo "$F exists"; exit 1; }
cp -a $S $F
find $F -name __pycache__ -type d -prune -exec rm -rf {} +
(cd $F && find . -type f ! -name MD5 | LC_ALL=C sort | xargs md5sum > MD5)
(cd $F && find . -type l | LC_ALL=C sort | while read -r l; do echo "link $l -> $(readlink $l)"; done > LINKS)
date -u +%FT%TZ > $F/FROZEN_UTC
chmod -R a-w $F
echo "frozen $(cat $F/FROZEN_UTC): $(wc -l < $F/MD5) files, md5 of MD5 $(md5sum < $F/MD5 | cut -c1-8)"
cat $F/LINKS
