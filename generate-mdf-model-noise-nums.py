import os
import numpy as np
from modeci_mdf.utils import load_mdf
from modeci_mdf.execution_engine import evaluate_onnx_expr

np.set_printoptions(precision=16)
model_file = os.path.join(os.path.dirname(__file__), "tests/json/model_integrators.json")
m = load_mdf(model_file)

for n in m.graphs[0].nodes:
    for f in n.functions:
        try:
            func = list(f.function.keys())[0]
        except AttributeError:
            func = f.function
        if func is not None and 'onnx::Random' in func:
            func = func.replace('onnx::', '').lower()
            print(n.id, evaluate_onnx_expr(f'onnx_ops.{func}', f.args, f.args))
