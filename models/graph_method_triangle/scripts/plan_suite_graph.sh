#!/bin/bash
# Per-case MPPI planning suite (plan_eval_graph.py) for one checkpoint over several test sets,
# run inside pytorchserver:latest with this repo at /workspace and Lie-Shape-NTFields read-only at /lieshape.
#   bash models/graph_method_triangle/scripts/plan_suite_graph.sh <gpu> <kind> <checkpoint (host path or /workspace/...)> <tag> [set ...]
# Results: results/graph_method_cusp/plan/<tag>/<set>.json (+ .log). Existing JSONs are skipped.
GPU=$1; KIND=$2; CKPT=$3; TAG=$4; shift 4
SETS=${@:-"Tshape3d_env1 Tshape3d_env1_bigangle Tshape3d_env1_rotheavy Tshape3d_env1_transonly"}
HOST=/media/jeffrey/G-Shape-NTFields
LIE=/media/jeffrey/Lie-Shape-NTFields
OUT=$HOST/results/graph_method_cusp/plan/$TAG
mkdir -p $OUT
case $CKPT in
  /workspace/*) ;;
  /lieshape/*) ;;
  $HOST/*) CKPT=/workspace${CKPT#$HOST} ;;
  $LIE/*) CKPT=/lieshape${CKPT#$LIE} ;;
  Experiments/*) CKPT=/workspace/$CKPT ;;
esac
WORKERS=${WORKERS:-8}
for s in $SETS; do
  [ -f $LIE/testing_data/3dshape/$s/sampled_points.npy ] || { echo "skip $s (missing)"; continue; }
  [ -f $OUT/$s.json ] && { echo "skip $TAG/$s (done)"; continue; }
  docker run --rm --gpus "\"device=$GPU\"" -u $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
    -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -v $HOST:/workspace -v $LIE:/lieshape:ro -w /lieshape pytorchserver:latest \
    python -u /workspace/models/graph_method_triangle/scripts/plan_eval_graph.py --kind $KIND --checkpoint $CKPT \
      --dataPath testing_data/3dshape/$s --out /workspace/results/graph_method_cusp/plan/$TAG/$s.json \
      --device cuda:0 --batch 125 --workers $WORKERS --seed 0 $PLAN_EXTRA 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -3 > $OUT/$s.log
  echo "$TAG $s: $(tail -1 $OUT/$s.log | cut -c1-300)"
done
