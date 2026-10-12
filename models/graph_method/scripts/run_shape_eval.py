"""Run Shape-NTFields' evaluate_training_3d_batched.py (unmodified) on a graph-method checkpoint.

    python models/graph_method/scripts/run_shape_eval.py <Shape-NTFields root> <this repo root> \
        --dataPath testing_data/3dshape/Tshape3d_env1 --checkpoint <abs ckpt> --out <abs dir> --no-viser

Our package is loaded as the top-level module `graph_method` and registered as
`models.graph_adapter.model_train_metric`, so the evaluator's
`--models graph_adapter` import resolves to the adapter without touching the
other repository.
"""
import importlib
import importlib.util
import os
import runpy
import sys
import types

shape_root, graph_root = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
eval_args = sys.argv[3:]
pkg_dir = os.path.join(graph_root, 'models', 'graph_method')
spec = importlib.util.spec_from_file_location('graph_method', os.path.join(pkg_dir, '__init__.py'),
                                              submodule_search_locations=[pkg_dir])
gm = importlib.util.module_from_spec(spec)
sys.modules['graph_method'] = gm
spec.loader.exec_module(gm)
adapter = importlib.import_module('graph_method.shape_adapter')
holder = types.ModuleType('models.graph_adapter')
holder.__path__ = []
sys.modules['models.graph_adapter'] = holder
sys.modules['models.graph_adapter.model_train_metric'] = adapter
os.chdir(shape_root)
sys.path.insert(0, shape_root)
script = os.path.join(shape_root, 'evaluate_training_3d_batched.py')
sys.argv = [script] + eval_args + ['--models', 'graph_adapter']
runpy.run_path(script, run_name='__main__')
