#!/usr/bin/env python

from safetensors.torch import load_file

import torch

tensors = load_file("subject.safetensors")
assert torch.equal(tensors["some_ints"], torch.tensor([[-1, 0, 1, 2]]))
assert torch.equal(tensors["some_floats"], torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]]))
