#!/bin/bash
# Poll: score every finished stage-E run under $1 into $2 with eval_shape.sh. Several daemons can
# run side by side; each run is claimed atomically with mkdir. N_ITERS polls of 60 s.
#   bash models/graph_method_triangle/scripts/eval_shape_daemon.sh Experiments/graph_method_triangle_stageE results/graph_method_triangle_stageE
EXP=$1; OUT=$2; N_ITERS=${3:-600}
mkdir -p $OUT/logs
for it in $(seq 1 $N_ITERS); do
  for d in $EXP/*/; do
    n=$(basename $d)
    [ -f $d/train_summary.json ] || continue
    [ -f $OUT/${n}_c1000/success_rate.txt ] && continue
    mkdir $OUT/${n}_c1000.claim 2>/dev/null || continue
    echo "[evalE] $n $(date)"
    bash models/graph_method_triangle/scripts/eval_shape.sh $EXP $n $OUT > $OUT/logs/eval_$n.log 2>&1 || echo "[evalE] FAILED $n"
  done
  sleep 60
done
