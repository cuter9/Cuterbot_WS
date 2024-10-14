import tensorrt as trt
from torch2trt import tensorrt_converter
from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.functional.hardswish')
# @tensorrt_converter('torch.nn.Hardswish')
def convert_Hardswish(ctx):
    # h-swish(x) = x * ReLU6(x+3)/6 # source: https://paperswithcode.com/method/hard-swish
    input = ctx.method_args[0]
    output = ctx.method_return

    input_a_trt, input_b_trt, input_c_trt, input_d_trt = add_missing_trt_tensors(ctx.network, [input, 6., 6., 3.])
    input_a_trt, input_b_trt, input_c_trt, input_d_trt = broadcast_trt_tensors(ctx.network,
                                                                               [input_a_trt, input_b_trt, input_c_trt,
                                                                                input_d_trt], len(output.shape) - 1)

    # ReLU6(x+3)
    layer = ctx.network.add_elementwise(input_a_trt, input_d_trt, trt.ElementWiseOperation.SUM)
    layer = ctx.network.add_activation(input=layer.get_output(0), type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(layer.get_output(0), input_b_trt, trt.ElementWiseOperation.MIN)

    # ReLU6(x+3)/6
    layer = ctx.network.add_elementwise(layer.get_output(0), input_c_trt, trt.ElementWiseOperation.DIV)

    # x*ReLU6(x+3)/6
    layer = ctx.network.add_elementwise(input_a_trt, layer.get_output(0), trt.ElementWiseOperation.PROD)

    output._trt = layer.get_output(0)