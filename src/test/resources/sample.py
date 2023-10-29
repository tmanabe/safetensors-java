#!/usr/bin/env python

import os
import torch

from safetensors.torch import save
from sys import stdout

tensors = {
    "some_ints": torch.tensor([[-1, 0, 1, 2]]),
    "some_floats": torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]),
}

os.write(stdout.fileno(), save(tensors))
