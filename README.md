## About
This is a minimal example. 

## Setup
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. run `pip install scipy` inside the container to install the KD-tree dependency
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training.

## SE(3) smooth graph travel-time fields (`models/graph_method`)

Implementation of `SE3_Smooth_Graph_Implementation.pdf`: a smooth latent
(a, b) predicted by the G-Shape encoder + an explicit analytic decoder
T = sqrt(a/s_t^2 + phi(b)/s_r^2), trained with intrinsic endpoint Eikonal and
Bellman losses on SE(3). See `models/graph_method/README.md` (what is
implemented, results, how to reproduce) and `models/graph_method/EXPERIMENTS.md`
(log of every experiment). Entry points: `train/train_graph.py`,
`python -m models.graph_method.evaluate`, `python -m unittest tests.test_graph_method`,
configs in `configs/graph_method/`.

## Triangle inequality by construction (`models/graph_method_triangle`)

A copy of `models/graph_method` with pair heads whose decoded time is a pseudometric
for every parameter value. A shared encoder gives each pose (u(x), V(x)) with V in SO(3),
a = |u_x - u_y|^2 (or an exact seminorm m(x, y)^2) and b = |V_x - V_y|_F^2 / 8 =
(3 - tr V_x^T V_y) / 4, decoded by the unchanged analytic decoder. See
`models/graph_method_triangle/README.md` (findings, results, how to reproduce) and
`EXPERIMENTS.md` (log T0-T9; results dashboard https://claude.ai/artifact/LHyUtVzc8h9u4KgKKvz4ts). Entry points: `train/train_graph_triangle.py`,
`python -m models.graph_method_triangle.evaluate`, `python -m unittest tests.test_graph_method_triangle`,
`models/graph_method_triangle/scripts/triangle_check.py`, configs in `configs/graph_method_triangle/`.
