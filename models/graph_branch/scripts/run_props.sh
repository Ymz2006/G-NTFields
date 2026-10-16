#!/bin/bash
# Property measurements (decoder_properties.py) for a list of runs, in one container.
#   bash models/graph_branch/scripts/run_props.sh <gpu> <scene> <run> [<run> ...]   (run = Experiments path or name)
GPU=$1; SCENE=$2; shift 2
HOST=/media/jeffrey/G-Shape-NTFields
CMD=""
for r in "$@"; do
  case $r in
    Experiments/*) ck=$r/latest.pt; name=ref_$(basename $r) ;;
    *) ck=Experiments/graph_branch/$r/latest.pt; name=$r ;;
  esac
  [ -f $HOST/results/graph_branch/properties/$name.json ] && continue
  CMD="$CMD python -u models/graph_branch/scripts/decoder_properties.py --scene $SCENE --out results/graph_branch/properties/$name.json $ck > results/graph_branch/properties/$name.log 2>&1;"
done
[ -z "$CMD" ] && exit 0
docker run -d --rm --gpus "\"device=$GPU\"" -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 \
  -v $HOST:/workspace -w /workspace --name gbp_${SCENE}_$(date +%s) pytorchserver:latest /bin/bash -c "$CMD"
