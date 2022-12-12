import numpy as np

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
    
    
class InputShape:
    def __init__(self):
        self.input_shape = None
        
    def __call__(self, x):
        self.input_shape = x.shape