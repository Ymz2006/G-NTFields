#!/bin/bash
# Synthetic exact-truth benchmark (synth_fit.py): one container per (problem, seed), all heads in sequence.
#   bash models/graph_branch/scripts/run_synth.sh <gpu> <kind> <A> <seed> [heads]
GPU=$1; KIND=$2; A=$3; SEED=$4; HEADS=${5:-ggram,gmetric,br1,br4,br4P,br4L,bc4}
HOST=/media/jeffrey/G-Shape-NTFields
TAG=${KIND}_A${A}
mkdir -p $HOST/results/graph_branch/synth/$TAG
docker run -d --rm --gpus "\"device=$GPU\"" -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 \
  -v $HOST:/workspace -w /workspace --name gbs_${TAG}_s${SEED} pytorchserver:latest \
  /bin/bash -c "python -u models/graph_branch/scripts/synth_fit.py --kind $KIND --A $A --heads $HEADS --seed $SEED \
     --out results/graph_branch/synth/$TAG > results/graph_branch/synth/$TAG/run_s${SEED}.out 2>&1"
