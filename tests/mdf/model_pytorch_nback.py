# Model example from
# https://github.com/ModECI/MDF/blob/nback/examples/PyTorch/PyTorch_MDF/nback/nback_pytorch_to_mdf.py

import torch
import numpy as np

# exp params (imported)
# context params
CDIM = 25

# layers
SDIM = 20

# 0 to deal with difficulty replicating randomness between pytorch and pnl
dropout_p = 0.0


class FFWM(torch.nn.Module):
    """
    Model used for Sternberg and N-back

    Source: https://github.com/andrebeu/nback-paper/blob/master/utilsWM.py

    Slight modification in which concatenation of input tensors was removed
    as the first op of the model and handled as pre-processing. This was
    done to allow execution in the MDF execution engine which currently does
    not seem to support Concatenating a list of tensors.
    """

    def __init__(self, indim, hiddim, outdim=2, bias=False):
        super().__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.hid1_layer = torch.nn.Linear(indim, indim, bias=bias)
        self.hid2_layer = torch.nn.Linear(indim, hiddim, bias=bias)
        self.out_layer = torch.nn.Linear(hiddim, outdim, bias=bias)
        self.drop2 = torch.nn.Dropout(p=dropout_p, inplace=False)
        bias_dim = indim
        max_num_bias_modes = 10
        self.embed_bias = torch.nn.Embedding(max_num_bias_modes, bias_dim)
        return None

    def forward(self, inputL, control_bias_int=0):
        """inputL is list of tensors"""
        hid1_act = self.hid1_layer(inputL).relu()
        control_bias = self.embed_bias(torch.tensor(control_bias_int))
        hid2_in = hid1_act + control_bias
        hid2_in = self.drop2(hid2_in)
        hid2_act = self.hid2_layer(hid2_in).relu()
        yhat_t = self.out_layer(hid2_act)
        return yhat_t


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

# init net
indim = 2 * (CDIM + SDIM)
hiddim = SDIM * 4
torch_ffwm_net = FFWM(indim, hiddim)
