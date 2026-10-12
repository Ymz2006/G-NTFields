#!/bin/bash
# Evaluate every finished run under $1 (Experiments root) into $2 (results root); skips done ones.
# Usage (inside container): bash models/graph_method/scripts/eval_queue.sh Experiments/graph_method_final results/graph_method_final [n_plan]
EXP=$1; OUT=$2; NPLAN=${3:-1000}; PAT=${4:-*}
for d in $EXP/$PAT/; do
  n=$(basename $d)
  [ -f $d/train_summary.json ] || continue
  [ -f $OUT/$n/metrics.json ] && continue
  echo "[eval] $n"
  python -m models.graph_method.evaluate --ckpt $d/best.pt --out $OUT/$n --n_plan $NPLAN --device cuda:0 > /dev/null 2>&1 || echo "[eval] FAILED $n"
done
