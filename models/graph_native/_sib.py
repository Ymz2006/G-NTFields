"""Locate the sibling package models/graph_method_triangle whether this package is imported as
``models.graph_native`` (training from the repo root) or loaded by path (evaluation from the
Lie-Shape-NTFields root, where ``models`` is that repo's package and the sibling was loaded by
path as ``graph_method_triangle``)."""
import importlib
import sys


def _base():
    if 'graph_method_triangle' in sys.modules:
        return 'graph_method_triangle'
    try:
        importlib.import_module('graph_method_triangle')
        return 'graph_method_triangle'
    except ImportError:
        return 'models.graph_method_triangle'


BASE = _base()
nets = importlib.import_module(BASE + '.model_network_graph')
mshape = importlib.import_module(BASE + '.model_function_shape')
se3 = importlib.import_module(BASE + '.geometry.se3')
tangents = importlib.import_module(BASE + '.geometry.tangents')
