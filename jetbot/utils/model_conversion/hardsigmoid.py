import tensorrt as trt
from torch2trt import tensorrt_converter
from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.functional.hardsigmoid')
# @tensorrt_converter('torch.nn.Hardsigmoid')
def convert_Hardswish(ctx):
    # h-sigmoid(x) : y=x for -3<=x<=3; y=0 for x<-3; y=1 for x>3
    input = ctx.method_args[0]
    output = ctx.method_return

    input_a_trt, input_b_trt, input_c_trt = add_missing_trt_tensors(ctx.network, [input, 3., 6., 1., 0.])
    input_a_trt, input_b_trt, input_c_trt = broadcast_trt_tensors(ctx.network,
                                                                  [input_a_trt, input_b_trt, input_c_trt],
                                                                  len(output.shape) - 1)
    up_bound, low_bound = add_missing_trt_tensors(ctx.network, [1., 0.])
    up_bound, low_bound = broadcast_trt_tensors(ctx.network, [up_bound, low_bound],
                                                len(output.shape) - 1)
    # (x+3)/6
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)
    layer = ctx.network.add_elementwise(layer.get_output(0), input_c_trt, trt.ElementWiseOperation.DIV)

    # min(1, (x+3)/6)
    layer = ctx.network.add_elementwise(layer.get_output(0), up_bound, trt.ElementWiseOperation.MIN)

    # max(0, min(1, (x+3)/6))
    layer = ctx.network.add_elementwise(layer.get_output(0), low_bound, trt.ElementWiseOperation.MAX)

    output._trt = layer.get_output(0)
