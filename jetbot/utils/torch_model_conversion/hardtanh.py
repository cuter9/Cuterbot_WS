import tensorrt as trt
from torch2trt import tensorrt_converter
from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.functional.hardtanh')
# @tensorrt_converter('torch.nn.functional.hardtanh_')
# @tensorrt_converter('torch.nn.Hardtanh')
def convert_hardtanh(ctx):
    # h-hardtanh(x) : y=x for -1<=x<=1; y=-1 for x<-1; y=1 for x>1
    input = get_arg(ctx, 'input', pos=0, default=None)
    low_bound = get_arg(ctx, 'min_val', pos=1, default=-1.0)
    up_bound = get_arg(ctx, 'max_val', pos=2, default=1.0)

    dtype_in = input.dtype
    low_bound = torch.tensor(low_bound, dtype=dtype_in)
    up_bound = torch.tensor(up_bound, dtype=dtype_in)

    output = ctx.method_return

    input_a_trt, up_bound_trt, low_bound_trt = add_missing_trt_tensors(ctx.network, [input, up_bound, low_bound])
    input_a_trt, up_bound_trt, low_bound_trt = broadcast_trt_tensors(ctx.network,
                                                                     [input_a_trt, up_bound_trt, low_bound_trt],
                                                                     len(output.shape))

    # min(1, x)
    layer = ctx.network.add_elementwise(input_a_trt, up_bound_trt, trt.ElementWiseOperation.MIN)

    # max(0, min(1, x))
    layer = ctx.network.add_elementwise(layer.get_output(0), low_bound_trt, trt.ElementWiseOperation.MAX)

    output._trt = layer.get_output(0)
