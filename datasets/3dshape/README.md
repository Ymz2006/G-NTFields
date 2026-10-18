# `datasets/3dshape`: meshes and generated training cells

* `obj/`  the shape and environment meshes (OBJ): `Tshape3d`, `Lshape3d`, `Ashape3d`, `env1`, `env3`, `env4`,
  the wall-with-hole rooms `hole90`, `hole75`, `hole66` (`models/graph_homotopy/scripts/make_wall_envs.py`), and the
  other shapes / scenes used by earlier campaigns (`Corozal`, `eastville`, `rect_small`, `rect_medium`).
* `data/` the generated data: one directory per training cell `<shape>_<env>` (`sampled_points.npy` (N, 12) start |
  goal in the chart (t, rotvec / 2 pi), `speed.npy`, `speed_dists.npy`, `speed_angles.npy`, `normal.npy`, `trans_n.npy`,
  `rot_n.npy`, `env.npy`, `meta.json`; 800k pairs for the standard cells, 200k for the hole cells made by
  `models/graph_homotopy/scripts/preprocess_hole.py`), the obstacle-free control `free_se3`, the bench sets of
  `models/graph_lie/scripts/bench` and `models/stress_bench` in `bench/`, and the preprocessing logs in `logs/`.

Scripts take `--dataPath datasets/3dshape/data/<cell>`; `meta.json` records the meshes by absolute path and every
evaluator also resolves them by basename in `datasets/3dshape/obj` (this repository) or `datasets/3dshape` of the
Lie-Shape-NTFields repository (mounted at `/lieshape`). Test sets live in `testing_data/3dshape` (its README).
