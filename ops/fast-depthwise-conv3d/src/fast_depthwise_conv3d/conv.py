import torch.nn as nn
import torch.nn.functional as F

import fast_depthwise_conv3d._ops as _ops


class DepthwiseConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=None, bias=True, padding_mode="zeros"):
        super(DepthwiseConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, in_channels, bias, padding_mode
        )

    def forward(self, input):
        weight = self.weight.to(dtype=input.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        if self.padding_mode != 'zeros':
            return _ops.depthwise_separable_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride, (0,0,0), self.dilation)
        return _ops.depthwise_separable_conv3d(input, weight, bias, self.stride,
                        self.padding, self.dilation)
