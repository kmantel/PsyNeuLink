# Based on

import sys

import numpy as np
import psyneulink as pnl

in_to_hidden_matrix = np.random.rand(2, 10)
hidden_to_out_matrix = np.random.rand(10, 1)

inp = pnl.TransferMechanism(name="Input", default_variable=np.zeros(2))

hidden = pnl.TransferMechanism(
    name="Hidden", default_variable=np.zeros(10), function=pnl.Logistic()
)

output = pnl.TransferMechanism(
    name="Output", default_variable=np.zeros(1), function=pnl.Logistic()
)

in_to_hidden = pnl.MappingProjection(
    name="Input Weights",
    matrix=in_to_hidden_matrix.copy(),
    sender=inp,
    receiver=hidden,
)

hidden_to_out = pnl.MappingProjection(
    name="Output Weights",
    matrix=hidden_to_out_matrix.copy(),
    sender=hidden,
    receiver=output,
)

xor_comp = pnl.Composition()

xor_comp.add_backpropagation_learning_pathway(
    [inp, in_to_hidden, hidden, hidden_to_out, output],
    learning_rate=10,
)
xor_inputs = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1]]
)
xor_comp.learn(
    inputs={inp: xor_inputs},
    animate={
        pnl.SHOW_LEARNING: True,
        pnl.MOVIE_DIR: sys.argv[1],
        pnl.MOVIE_NAME: "Composition_XOR_animation",
    },
)
