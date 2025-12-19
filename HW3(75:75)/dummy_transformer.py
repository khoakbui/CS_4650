import torch
from torch import nn


class DummyTransformer(nn.Module):
    """
    A dummy transformer module that does not perform any computation.
    Used for unit tests and debugging.
    DO NOT MODIFY THIS CLASS, your local tests might fail unexpectedly.
    """
    def __init__(self, dummy_generator):
        super(DummyTransformer, self).__init__()
        self.generator = dummy_generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = src.unsqueeze(1)
        return src

    def encode(self, src, src_mask):
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask):
        memory = memory.unsqueeze(1)
        return memory


class DummyGenerator(nn.Module):
    """
    A dummy generator module that normalizes the input tensor.
    Used for unit tests and debugging.
    DO NOT MODIFY THIS CLASS, or your local tests might fail unexpectedly.
    """
    def __init__(self):
        super(DummyGenerator, self).__init__()
        pass

    def forward(self, x):
        x = x / x.sum(dim=-1, keepdim=True)
        x = torch.log(x)
        return x
