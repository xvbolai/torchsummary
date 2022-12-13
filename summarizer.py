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

from .graph_modules import BrocolliGraphModule

from .utils import (activation_pre_hook, InputShape)


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
        
class SMA:
    def __init__(self, model, input_shape, concrete_args=None):
        super(SMA, self).__init__()
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
        
    def init(self):
        self.save_conv(code = "CNN,IH,IW,IC,OC,KH,KW,SH,SW,PT,PH,PW,DH,DW\n")
        self.save_pool(code = "CNN,IH,IW,IC,OC,KH,KW,SH,SW,PAT,PH,PW,DH,DW,POT\n")
        self.save_fullyconn(code = "CNN,IH,IW,IC,OC,KH,KW\n")

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
        
    def save_conv(self, code = None, filename = "./tmp/dataset_conv2d_large_cnn.csv"):
        with open(filename, "a") as f:
            f.write(code)
            
    def save_pool(self, code = None, filename = "./tmp/dataset_pooling_large_cnn.csv"):
        with open(filename, "a") as f:
            f.write(code)
            
    def save_fullyconn(self, code = None, filename = "./tmp/dataset_fully_connected_cnn.csv"):
        with open(filename, "a") as f:
            f.write(code)
            
    def get_param(self, module):
        if isinstance(module.padding, tuple):
            pad_h = module.padding[0]
            pad_w = module.padding[1]
        else:
            pad_h = module.padding
            pad_w = module.padding

        if isinstance(module.stride, tuple):
            stride_h = module.stride[0]
            stride_w = module.stride[1]
        else:
            stride_h = module.stride
            stride_w = module.stride

        if isinstance(module.kernel_size, tuple):
            kernel_h = module.kernel_size[0]
            kernel_w = module.kernel_size[1]
        else:
            kernel_h = module.kernel_size
            kernel_w = module.kernel_size
            
        if isinstance(module.dilation, tuple):
            d_h = module.dilation[0]
            d_w = module.dilation[1]
        else:
            d_h = module.dilation
            d_w = module.dilation
        return (pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, d_h, d_w)
        
    def concise(self, CNN = None):
        graph_module = copy.deepcopy(self.observed_model)
        modules = dict(graph_module.named_modules())
        for node in list(graph_module.graph.nodes):
            if node.op == "call_module":
                module = modules[node.target]
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    # print(module.activation_pre_process.input_shape)
                    
                    input_shape = module.activation_pre_process.input_shape
                    
                    (pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, d_h, d_w) = self.get_param(module=module)
                        # dilation
                    # CNN,IH,IW,IC,OC,KH,KW,SH,SW,PT,PH,PW,DH,DW
                    code = f"{CNN},{input_shape[2]},{input_shape[3]},{input_shape[1]},{module.out_channels},{kernel_h},{kernel_w},{stride_h},{stride_w},{1},{pad_h},{pad_w},{d_h},{d_w}\n"
                    self.save_conv(code = code)
                elif isinstance(module, nn.AvgPool2d):
                    # print(module.activation_pre_process.input_shape)
                    # CNN,IH,IW,IC,OC,KH,KW,SH,SW,PAT,PH,PW,DH,DW,POT
                    input_shape = module.activation_pre_process.input_shape
                    (pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, d_h, d_w) = self.get_param(module=module)
                    code = f"{CNN},{input_shape[2]},{input_shape[3]},{input_shape[1]},{input_shape[1]},{kernel_h},{kernel_w},{stride_h},{stride_w},{1},{pad_h},{pad_w},{d_h},{d_w},{1}\n"
                    self.save_pool(code=code)
                elif isinstance(module, nn.MaxPool2d):
                    # print(module.activation_pre_process.input_shape)
                    input_shape = module.activation_pre_process.input_shape
                    (pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, d_h, d_w) = self.get_param(module=module)
                    code = f"{CNN},{input_shape[2]},{input_shape[3]},{input_shape[1]},{input_shape[1]},{kernel_h},{kernel_w},{stride_h},{stride_w},{1},{pad_h},{pad_w},{d_h},{d_w},{2}\n"
                    self.save_pool(code=code)
                elif isinstance(module, nn.Linear):
                    # print(module.activation_pre_process.input_shape)
                    # CNN,IH,IW,IC,OC,KH,KW
                    # print(module.weight.shape)
                    weight_shape = module.weight.shape
                    code = f"{CNN},{1},{1},{weight_shape[1]},{weight_shape[0]},{1},{1}\n"
                    self.save_fullyconn(code=code)