import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _triple

import math

import dconv_native._ops as _ops

# TODO: Adapt order of arguments to dconv variants to torchvision
# Torchvision has mask parameter at the end, this module uses it at position 3.


def _any_half_inputs(*inputs):
    return any(t is not None and t.dtype == torch.half for t in inputs)


def deform_conv1d(
        input, offset, alpha, weight, bias=None, stride=1,
        padding=0, dilation=1, n_weight_groups=1, n_offset_groups=1
):
    if _any_half_inputs(input, offset, alpha, weight, bias):
        input = input.float()
        offset = offset.float()
        alpha = alpha.float() if alpha is not None else None
        weight = weight.float()
        bias = bias.float() if bias is not None else None

    if bias is None:
        out_channels = weight.shape[0]
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    if alpha is None:
        alpha = torch.ones_like(offset)

    return _ops.deform_conv1d(
        input, offset, alpha, weight, bias, stride,
        padding, dilation, n_weight_groups, n_offset_groups
    )

def deform_conv3d(
        input,
        offset,
        alpha,
        weight,
        bias=None,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        n_weight_groups=1,
        n_offset_groups=1
):
    if _any_half_inputs(input, offset, alpha, weight, bias):
        input = input.float()
        offset = offset.float()
        alpha = alpha.float() if alpha is not None else None
        weight = weight.float()
        bias = bias.float() if bias is not None else None

    #if bias is None:
    #    out_channels = weight.shape[0]
    #    bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    #if alpha is None:
    #    shape = list(offset.shape)
    #    shape[1] //= 3
    #    alpha = offset.new_ones(shape)

    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dilation_d, dilation_h, dilation_w = _triple(dilation)

    return _ops.deform_conv3d(
        input,
        offset,
        alpha,
        weight,
        bias,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        n_weight_groups,
        n_offset_groups
    )


class DeformConvNd(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, offset_groups, bias
    ):
        super(DeformConvNd, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if in_channels % offset_groups != 0:
            raise ValueError("in_channels must be divisible by offset_groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.offset_groups = offset_groups

        self.weight = Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *kernel_size
            ),
            requires_grad=True
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.offset_groups != 1:
            s += ', offset_groups={offset_groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(**self.__dict__)


class DeformConv1d(DeformConvNd):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, offset_groups=1, bias=True
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(DeformConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, offset_groups, bias
        )

    def forward(self, input, offset, alpha):
        return _ops.deform_conv1d(
            input, offset, alpha, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.offset_groups
        )


class DeformConv3d(DeformConvNd):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            offset_groups=1,
            bias=True
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(DeformConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, offset_groups, bias
        )

    def forward(self, input, offset, alpha):
        return deform_conv3d(
            input,
            offset,
            alpha,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.offset_groups
        )
