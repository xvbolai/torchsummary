import numpy as np
import torch.nn as nn 
import torch

def activation_pre_hook(self, input):
    if hasattr(self, "activation_pre_process"):
        self.activation_pre_process(input[0])

def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node, modules, new_module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)
    
    
class InputShape(nn.Module):
    
    def __init__(self):
        super(InputShape, self).__init__()
        
    def forward(self, x):
        self.input_shape = x.shape
        return x