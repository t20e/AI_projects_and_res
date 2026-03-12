import copy
import torch.nn as nn

def clones(module, N):
    """Produce N identical layers of the EncoderLayer or DecoderLayer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])