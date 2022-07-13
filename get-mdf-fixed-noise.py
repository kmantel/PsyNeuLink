from modeci_mdf.execution_engine import evaluate_onnx_expr
import numpy as np

ops = {
    'onnx_ops.randomuniform': {
        'A': {'low': -1.0, 'high': 1.0, 'seed': 0, 'shape': (1, 1)},
        'D': {'low': -0.5, 'high': 0.5, 'seed': 0, 'shape': (1, 1)},
        'E': {'low': -0.25, 'high': 0.5, 'seed': 0, 'shape': (1, 1)}
    },
    'onnx_ops.randomnormal': {
        'B': {'mean': -1.0, 'scale': 0.5, 'seed': 0, 'shape': (1, 1)},
        'C': {'mean': 0.0, 'scale': 0.25, 'seed': 0, 'shape': (1, 1)},
    }
}

np.set_printoptions(precision=16)

for func in ops:
    for node, args in ops[func].items():
        r = evaluate_onnx_expr(func, base_parameters=args, evaluated_parameters=args)
        print(r)
        print(f'{node}: {r}\t{r.dtype}')
