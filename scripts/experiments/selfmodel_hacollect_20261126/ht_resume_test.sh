#!/usr/bin/env bash
# notebook: selfmodel_hatask_20261126
# Revision 2 section 7, condition 2: the interrupt test of asrun2's resume rule, on a dev seed (results not read):
#   ht_resume_test.sh <gpu> <task> <seed> <repeat> <out>
# (1) asrun2 = its MD5; (2) one pair A,R at 1x from asrun2; (3) R half way: CUT_S seconds after A's end line (default
# half of A's wall time) and once R has written a look, the ht_task.py process is killed by pid (SIGKILL; its bash
# wrapper then ends); (4) the same command again;
# (5) the checks: the second process moved this pair's records of both arms aside (A's finished ones, R's partial ones),
# ran A and R, and printed "prefix A == R: True"; stale/ holds the first process's records.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1  # coordinator 11-26: nothing written into the frozen copy
GPU=$1 T=$2 SD=$3 RP=$4 O=$5
OT=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hatask_20261126
F2=$OT/asrun2
LOG=$O.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $O.test; }
(cd $F2 && md5sum -c --quiet MD5) || { say "asrun2 differs from its MD5: stop"; exit 1; }
say "asrun2 = MD5 ($(cat $F2/FROZEN_UTC)); $T s$SD r$RP on card $GPU"
run() { bash $F2/selfmodel_hacollect_20261126/ht_run.sh $GPU scalebfm $T $SD $RP A,R $O --budget-x 1 >> $LOG 2>&1; }
run &
W=$!
name=${T}_s${SD}_r${RP}
until grep -a -q "ht_task scalebfm $T A $name: end=" $LOG 2>/dev/null; do kill -0 $W 2>/dev/null || { say "first process ended before A finished"; exit 1; }; sleep 5; done
wa=$(grep -a "ht_task scalebfm $T A $name: end=" $LOG | head -1 | grep -o "wall=[0-9]*" | cut -d= -f2)
CUT_S=${CUT_S:-$((wa / 2))}
say "A ended (wall $wa s); R running in the same process; cut after ${CUT_S} s"
sleep $CUT_S
until ls $O/R/looks/${name}_* > /dev/null 2>&1; do kill -0 $W 2>/dev/null || { say "first process ended before R wrote a look"; exit 1; }; sleep 2; done
P=$(pgrep -f "[h]t_task.py .*--out $O( |$)" | head -1)
[ -n "$P" ] || { say "no ht_task process found to cut"; exit 1; }
[ -e $O/R/log/$name.json ] && { say "R finished before the cut: raise CUT_S"; exit 1; }
say "kill -9 $P (ht_task.py of this test); R's partial records: $(ls $O/R/looks 2>/dev/null | wc -l) looks, $(ls $O/R/data/$name.* 2>/dev/null | wc -l) data"
kill -9 $P
wait $W
say "first process gone; A log $(ls $O/A/log/$name.json 2>/dev/null | wc -l), R log $(ls $O/R/log/$name.json 2>/dev/null | wc -l); second process"
echo "=== second process ===" >> $LOG
run
say "second process ended"
n2=$(sed -n '/=== second process ===/,$p' $LOG)
ma=$(echo "$n2" | grep -a -c "$name: A: .* records from an earlier process moved")
mr=$(echo "$n2" | grep -a -c "$name: R: .* records from an earlier process moved")
ea=$(echo "$n2" | grep -a -c "ht_task scalebfm $T A $name: end=")
er=$(echo "$n2" | grep -a -c "ht_task scalebfm $T R $name: end=")
pf=$(echo "$n2" | grep -a "prefix A == R" | sed 's/.*prefix/prefix/')
say "second process: moved A $ma, moved R $mr; ran A $ea, R $er; $pf"
say "stale: A $(ls $O/A/stale/ 2>/dev/null | tr '\n' ' ')($(ls $O/A/stale/*/ 2>/dev/null | grep -c .) files), R $(ls $O/R/stale/ 2>/dev/null | tr '\n' ' ')($(ls $O/R/stale/*/ 2>/dev/null | grep -c .) files)"
pc=$(find $F2 -name __pycache__ | wc -l); (cd $F2 && md5sum -c --quiet MD5) && m2=same || m2=DIFFERS
say "asrun2 after the test: MD5 $m2, __pycache__ dirs $pc"
if [ $m2 = same ] && [ $pc = 0 ] && [ $ma = 1 ] && [ $mr = 1 ] && [ $ea = 1 ] && [ $er = 1 ] && echo "$pf" | grep -q "True"; then say "RESUME TEST PASS"; else say "RESUME TEST FAIL"; fi
