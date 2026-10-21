# `datasets/3dshape`: meshes and generated training cells

* `obj/`  the shape and environment meshes (OBJ): `Tshape3d`, `Lshape3d`, `Ashape3d`, `env1`, `env3`, `env4`, the five
  shapes of `dataprocessing/make_shapes3d.py` (`tripod3d`, `crank3d`, `loop3d`, `twistI3d`, `chair3d`) and its rooms
  `env5`..`env9` (env3's shell with one partition wall and a 50 / 60 / 70 window: env5 / env7 / env8; two crossing
  walls with 50 / 70 windows: env6 / env9), `cube`, the planar sofa scene of `make_sofa2d.py` (`Usofa2d`,
  `2dhall30/34/44`),
  the wall-with-hole rooms `hole90`, `hole75`, `hole66` (`models/graph_homotopy/scripts/make_wall_envs.py`), and the
  other shapes / scenes used by earlier campaigns (`Corozal`, `eastville`, `rect_small`, `rect_medium`).
* `data/` the generated data: one directory per training cell `<shape>_<env>` (`sampled_points.npy` (N, 12) start |
  goal in the chart (t, rotvec / 2 pi), `speed.npy`, `speed_dists.npy`, `speed_angles.npy`, `normal.npy`, `trans_n.npy`,
  `rot_n.npy`, `env.npy`, `meta.json`; 800k pairs for the standard cells, 200k for the hole cells made by
  `models/graph_homotopy/scripts/preprocess_hole.py`), the 30 shape cells `<shape>_env{4..9}`, `cube_eastville`,
  `rect_small_eastville`, `Usofa2d_2dhall{30,34,44}` and `synth_sector_A{4,20}` (brought over from
  G-Shape-NTFields-old on 2026-10-17: same arrays, the shards dropped, `meta.json` mesh paths rewritten to
  `/workspace/datasets/3dshape/obj/<mesh>`), the T / L / A cells `Lshape3d_env1`, `Ashape3d_env1`, `Tshape3d_env3`,
  `Ashape3d_env3` (4 x 200k pairs, `dataprocessing/preprocess_sharded.py`, made by
  `models/graph_homotopy/scripts/run_shapes_queue.py`), the obstacle-free control `free_se3`, the bench sets of
  `models/graph_lie/scripts/bench` and `models/stress_bench` in `bench/`, and the preprocessing logs in `logs/`.

Scripts take `--dataPath datasets/3dshape/data/<cell>`; `meta.json` records the meshes by absolute path and every
evaluator also resolves them by basename in `datasets/3dshape/obj` (this repository) or `datasets/3dshape` of the
Lie-Shape-NTFields repository (mounted at `/lieshape`). Test sets live in `testing_data/3dshape` (its README).
