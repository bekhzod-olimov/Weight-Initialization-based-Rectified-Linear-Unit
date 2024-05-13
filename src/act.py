import torch
from torch.nn import *

class WiBReLU(Module): 
    def __init__(self): super().__init__() 
    def forward(self, inp): return functional.relu(inp) - torch.mean(inp)