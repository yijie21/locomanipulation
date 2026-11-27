# notebook: skip render for paper Figure 1, not an experiment
# fig1_replay.sh <task> <ep>: replay one lmlroute ScaleBFM episode (A then R, one process) from the private copy of the
# frozen code, environment as dev/lrt_queue.sh's workers (H20 form: LOCOMANI_WORKSPACE), CPU only, 1 thread, nice 19.
set -e
TASK=$1 EP=$2 TAG=${3:-$1_$2}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
F=/wuji-vefps-D/wuji-il/yijie/loco_outputs/fig1_teaser_20261126
C0=$F/code
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
(cd $C0 && md5sum -c --quiet $O/asrun/CODE_MD5) || { echo "copy differs from asrun/CODE_MD5"; exit 1; }
md5sum -c --quiet $O/asrun/MD5 && md5sum -c --quiet $O/asrun/LOOKS_MD5 || { echo "inputs differ from asrun"; exit 1; }
echo "code copy and inputs: same as asrun"
mkdir -p $F/out $F/state/$TAG $F/logs
cd $G
export PYTHONPATH=$G:$C0/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export LC_CODE_ROOT=$C0 LC_OUT=$F/out LRT_ROUTES=$F/out/routes LRT_LOOKS=$O/hg/looks
export LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws PYTHONDONTWRITEBYTECODE=1 FIG_STATE=$F/state/$TAG
exec nice -n 19 $PY $F/scripts/fig1_replay.py $TASK $EP alone,reachable
