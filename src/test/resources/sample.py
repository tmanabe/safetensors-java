#!/usr/bin/env python

from safetensors.torch import save_file

import torch

tensors = {
    "some_ints": torch.tensor([[-1, 0, 1, 2]]),
    "some_floats": torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]),
}

save_file(tensors, "sample.safetensors")
