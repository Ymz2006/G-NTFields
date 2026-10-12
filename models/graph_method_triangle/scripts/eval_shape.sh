#!/bin/bash
# Score a stage-E run with the unmodified Shape-NTFields MPPI evaluator on all 1000 test cases.
# Inside the container, from the repo root, with Shape-NTFields mounted at /shape and
# Lie-Shape-NTFields at /lieshape (only the latter has the L / A environment meshes;
# the evaluator code path for the default chart sampler is identical, EXPERIMENTS.md E14):
#   bash models/graph_method_triangle/scripts/eval_shape.sh <Experiments root> <run name> <results root>
EXP=$1; NAME=$2; OUT=$3
DATA=$(python -c "import json; print(json.load(open('$EXP/$NAME/config.json'))['data_path'].rstrip('/').split('/')[-1])")
case $DATA in
  Tshape3d_env1) ROOT=/shape ;;
  *) ROOT=/lieshape ;;
esac
python models/graph_method_triangle/scripts/run_shape_eval.py $ROOT /workspace \
  --dataPath testing_data/3dshape/$DATA --no-viser --cases 0 \
  --checkpoint /workspace/$EXP/$NAME/latest.pt --out /workspace/$OUT/${NAME}_c1000
