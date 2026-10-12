#!/bin/bash
# Poll: evaluate finished runs matching PAT until all runs matching PAT have metrics (or N_ITERS polls).
EXP=$1; OUT=$2; NPLAN=${3:-1000}; PAT=${4:-*}; N_ITERS=${5:-240}
for it in $(seq 1 $N_ITERS); do
  bash models/graph_method_triangle/scripts/eval_queue.sh $EXP $OUT $NPLAN "$PAT"
  todo=0
  for d in $EXP/$PAT/; do n=$(basename $d); [ -f $OUT/$n/metrics.json ] || todo=1; done
  [ $todo -eq 0 ] && { echo "[daemon] all done"; exit 0; }
  sleep 60
done
