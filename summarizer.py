import copy
from loguru import logger
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fx
from torch.fx import Tracer, Graph
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule

from .utils import (replace_node_module, activation_pre_hook, 
                    activation_post_hook, InputShape)


class BrocolliTracer(Tracer):
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )
        
class PytorchQuantizer:
    def __init__(self, model, input_shape, concrete_args=None):
        super(PytorchQuantizer, self).__init__()
        self.model = model
        self.input_shape = input_shape
        if isinstance(input_shape, (tuple, list)) and all(
            isinstance(element, int) for element in input_shape
        ):
            self.input_shape = [input_shape]
        self.concrete_args = concrete_args
        self.qconfig = None
        self.qconfig_dict = {"": self.qconfig}

        self.graph_module = self.get_graph_module(self.model, self.concrete_args, False)
        self.modules = dict(self.graph_module.named_modules())
        self.print_tabular(self.graph_module)

    def get_graph_module(self, model, concrete_args, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        if isinstance(model, GraphModule):
            trace = BrocolliGraphModule(model.root, model.graph)
        elif isinstance(model, nn.Module):
            tracer = BrocolliTracer()
            graph = tracer.trace(model, concrete_args)
            trace = BrocolliGraphModule(tracer.root, graph)
        else:
            raise Exception("model must be a torch.nn.Module or a torch.fx.GraphModule")

        return trace

    def print_tabular(self, graph_module):
        nodes = list(graph_module.graph.nodes)
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in nodes]
        logger.debug(
            tabulate(
                node_specs,
                headers=["\nopcode", "\nname", "\ntarget", "\nargs", "\nkwargs"],
            )
        )

    def gen_input_tensor(self, shapes):
        input_tensor = []
        for shape in shapes:
            if isinstance(shape, (tuple, list)):
                if all(isinstance(element, int) for element in shape):
                    input_tensor.append(torch.rand(shape).to(torch.float32))
                else:
                    input_tensor.append(self.gen_input_tensor(shape))
            else:
                input_tensor.append(torch.rand(shape).to(torch.float32))

        return input_tensor

    def shape_inference(self):
        dummy_input = self.gen_input_tensor(self.input_shape)
        ShapeProp(self.trace).propagate(*dummy_input)
        
        def forward(self, model, input):
        output = model(*input)

        if isinstance(output, torch.Tensor):
            output = [output]

        return output
    

    def prepare(self):
        """
        Return:
            A GraphModule with observer (configured by qconfig_dict), ready for calibration
        """
        if hasattr(self, 'fused_model'):
            graph_module = copy.deepcopy(self.fused_model)
        else:
            graph_module = copy.deepcopy(self.graph_module)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "placeholder":
                pass
            elif node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    module.add_module(
                        "activation_pre_process", InputShape()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
                elif isinstance(module, (nn.MaxPool2d, nn.MaxPool1d, nn.AvgPool2d, nn.AvgPool1d)):
                    module.add_module(
                        "activation_pre_process", InputShape()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
                elif isinstance(module, nn.Linear):
                    module.add_module(
                        "activation_pre_process", InputShape()
                    )
                    module.register_forward_pre_hook(activation_pre_hook)
            elif node.op == "output":
                pass

        self.observed_model = torch.fx.GraphModule(graph_module, graph_module.graph)

    def calibrate(self, calibtraion_func):
        calibtraion_func(self.observed_model)

        logger.info("calibtraion finish")
        
    def concise(self):
        graph_module = copy.deepcopy(self.observed_model)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    print(module.activation_pre_process.input_shape)
                elif isinstance(module, nn.AvgPool2d):
                    print(module.activation_pre_process.input_shape)
                elif isinstance(module, nn.MaxPool2d):
                    print(module.activation_pre_process.input_shape)
                elif isinstance(module, nn.Linear):
                    print(module.activation_pre_process.input_shape)