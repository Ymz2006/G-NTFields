#!/bin/bash
# Dump the metric-baseline paths of every evaluated shapes3d cell (results/shapes3d/paths/<cell>.{json,npz}),
# same planner flags as the pipeline's eval job, so the json matches results/shapes3d/plan/<cell>_metric_s1.json.
# Loops until every cell with an eval rc has its npz; runs one cell at a time on the GPU with most free memory.
#     bash dataprocessing/shapes3d_dump_all.sh
cd "$(dirname "$0")/.."
mkdir -p results/shapes3d/paths results/shapes3d/logs
while true; do
    todo=""
    for rc in results/shapes3d/jobs/eval_*.rc; do
        [ "$(cat $rc)" = "0" ] || continue
        c=$(basename $rc .rc); c=${c#eval_}
        [ -f results/shapes3d/paths/$c.npz ] || { todo=$c; break; }
    done
    if [ -z "$todo" ]; then
        pgrep -f "shapes3d_pipeline.py" >/dev/null || { echo "all cells dumped"; exit 0; }
        sleep 120; continue
    fi
    gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', ' '$2 ~ /^[0-9]+$/ {print $3-$2, $1}' | sort -n | tail -1 | cut -d' ' -f2)
    ck=$(ls Experiments/shapes3d_metric_baseline/${todo}_s1/Model_Epoch_04000_*.pt)
    echo "$(date +%T) dump $todo on gpu $gpu"
    docker run --rm --gpus "\"device=$gpu\"" -u $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD":/workspace -v /media/jeffrey/Lie-Shape-NTFields:/lieshape:ro \
        -w /lieshape pytorchserver:latest bash -c "OMP_NUM_THREADS=1 python -u /workspace/dataprocessing/shapes3d_dump_paths.py \
        --kind metric --checkpoint /workspace/$ck --dataPath /workspace/testing_data/3dshape/$todo \
        --out /workspace/results/shapes3d/paths/$todo.json --device cuda:0 --batch 125 --workers 8 --seed 0" \
        > results/shapes3d/logs/dump_$todo.log 2>&1 || { echo "dump $todo FAILED"; sleep 60; }
done
