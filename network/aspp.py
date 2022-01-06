from torch import Tensor
from typing import Callable, Optional, List, Tuple, Union

import torch
import torch.nn as nn


__all__ = [
    "Conv2dNorm", "Conv3dNorm",
    "Conv2dCS", "Conv3dCS",
    "ASPP2D", "ASPP3D"
]


# ----------------------------------------------------------------------------------------------------------------------


class _ConvNorm(nn.Sequential):
    def __init__(self, conv_type: Callable, norm_type: Callable, inplanes: int, outplanes: int, kernel_size: int ,
                 stride: Optional[int] = 1, padding: Optional[int] = 0, dilation: Optional[int] = 1,
                 groups: Optional[int] = 1, bias: Optional[bool] = False, padding_mode: Optional[str] = "zeros"):

        super(_ConvNorm, self).__init__(
            conv_type(inplanes, outplanes, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
            norm_type(outplanes),
            nn.ReLU(inplace=True)
        )


class Conv2dNorm(_ConvNorm):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", norm_type=nn.BatchNorm2d):
        super(Conv2dNorm, self).__init__(nn.Conv2d, norm_type, inplanes, outplanes, kernel_size, stride, padding,
                                         dilation, groups, bias, padding_mode)


class Conv3dNorm(_ConvNorm):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", norm_type=nn.BatchNorm3d):
        super(Conv3dNorm, self).__init__(nn.Conv3d, norm_type, inplanes, outplanes, kernel_size, stride, padding,
                                         dilation, groups, bias, padding_mode)


# ----------------------------------------------------------------------------------------------------------------------

class _ConvCS(nn.Sequential):
    def __init__(self, conv_type, norm_type, inplanes, outplanes, kernel_size, stride, padding, dilation,
                 norm_before_activation):
        assert kernel_size > 1

        def get_norm_activation(num_planes):
            if norm_before_activation:
                return norm_type(num_planes), nn.ReLU(inplace=True)
            else:
                return nn.ReLU(inplace=True), norm_type(num_planes)

        super(_ConvCS, self).__init__(
            conv_type(inplanes, inplanes, kernel_size, stride, padding, dilation, inplanes, False),
            *get_norm_activation(inplanes),
            conv_type(inplanes, outplanes, 1, bias=False),
            *get_norm_activation(outplanes)
        )


class Conv2dCS(_ConvCS):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=nn.BatchNorm2d,
                 norm_before_activation=True):
        super(Conv2dCS, self).__init__(nn.Conv2d, norm_type, inplanes, outplanes, kernel_size, stride, padding,
                                       dilation, norm_before_activation)


class Conv3dCS(_ConvCS):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=nn.BatchNorm3d,
                 norm_before_activation=True):
        super(Conv3dCS, self).__init__(nn.Conv3d, norm_type, inplanes, outplanes, kernel_size, stride, padding,
                                       dilation, norm_before_activation)


# ----------------------------------------------------------------------------------------------------------------------


class _ASPP(nn.Module):
    def __init__(self, conv_type, norm_type, inplanes, inter_planes, outplanes, dilation_rates):
        super().__init__()

        self.atrous_convs = nn.ModuleList([_ConvNorm(conv_type, norm_type, inplanes, inter_planes, 1, bias=False)])
        for rate in dilation_rates:
            self.atrous_convs.append(_ConvNorm(conv_type, norm_type, inplanes, inter_planes, 3, padding=rate,
                                               dilation=rate, bias=False))

        self.conv_out = _ConvNorm(conv_type, norm_type, inter_planes * (len(dilation_rates) + 1), outplanes, 1,
                                  bias=False)

    def forward(self, x):
        y = torch.cat([conv(x) for conv in self.atrous_convs], 1)
        return self.conv_out(y)


class ASPP2D(_ASPP):
    def __init__(self, inplanes, inter_planes, outplanes, dilation_rates, norm_type=nn.BatchNorm2d):
        super(ASPP2D, self).__init__(nn.Conv2d, norm_type, inplanes, inter_planes, outplanes, dilation_rates)


class ASPP3D(_ASPP):
    def __init__(self, inplanes, inter_planes, outplanes, dilation_rates, norm_type=nn.BatchNorm3d):
        super(ASPP3D, self).__init__(nn.Conv3d, norm_type, inplanes, inter_planes, outplanes, dilation_rates)
