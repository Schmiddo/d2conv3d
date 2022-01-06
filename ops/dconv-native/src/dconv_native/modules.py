import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from .deform_conv import DeformConv3d
from .grid import offsets_from_size_map


class _DConvBase(nn.Module):
    _activations = {
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
        "tanh": torch.tanh,
    }

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, offset_groups=1, bias=True,
                 activation="sigmoid", deform_param_op=nn.Conv3d):
        super(_DConvBase, self).__init__()
        kernel_size = _triple(kernel_size)

        self.num_kernel_points = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.offset_groups = offset_groups
        self.groups = groups

        self.deform_params = deform_param_op(
            in_channels,
            self._deform_channels(),
            kernel_size,
            stride,
            padding,
            dilation,
            1,
            bias
        )

        self.deform_conv = DeformConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            offset_groups,
            bias
        )

        self.activation = self._activations[activation] if isinstance(activation, str) else activation

        nn.init.zeros_(self.deform_params.weight)
        if bias:
            nn.init.zeros_(self.deform_params.bias)

    def _deform_channels(self):
        return 3 * self.offset_groups * self.num_kernel_points

    def _get_deformation(self, input):
        return self.deform_params(input), None

    def forward(self, input):
        deformation = self._get_deformation(input)
        return self.deform_conv(input, *deformation)


class DConv3d(_DConvBase):
    pass


class MDConv3d(_DConvBase):
    def _deform_channels(self):
        self.offset_channels = 3 * self.offset_groups * self.num_kernel_points
        self.alpha_channels = self.offset_groups * self.num_kernel_points
        return self.offset_channels + self.alpha_channels

    def _get_deformation(self, input):
        deformation = self.deform_params(input)
        offset, alpha = torch.split(
            deformation,
            [self.offset_channels, self.alpha_channels],
            dim=1
        )
        return offset, self.activation(alpha)


class SpatialDConv3d(DConv3d):
    def _get_deformation(self, input):
        offsets, alpha = super()._get_deformation(input)
        offsets = offsets.clone()
        offsets[:, ::3] = 0
        return offsets, alpha


class SpatialMDConv3d(MDConv3d):
    def _get_deformation(self, input):
        # activation function is applied in parent class
        offsets, alpha = super()._get_deformation(input)
        offsets = offsets.clone()
        offsets[:, ::3] = 0
        return offsets, alpha


class TemporalDConv3d(DConv3d):
    def _get_deformation(self, input):
        offsets, alpha = super()._get_deformation(input)
        offsets = offsets.clone()
        offsets[:, 1::3] = 0
        offsets[:, 2::3] = 0
        return offsets, alpha


class TemporalMDConv3d(MDConv3d):
    def _get_deformation(self, input):
        # activation function is applied in parent class
        offsets, alpha = super()._get_deformation(input)
        offsets = offsets.clone()
        offsets[:, 1::3] = 0
        offsets[:, 2::3] = 0
        return offsets, alpha


class SizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 1
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = F.elu(self.deform_params(input)) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class XYSizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 2
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = F.elu(self.deform_params(input)) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class XYZSizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 3
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = F.elu(self.deform_params(input)) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class SizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 1
    return self.offset_channels + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [1, self.alpha_channels],
      dim=1
    )
    size_map = F.elu(size_map) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class XYSizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 2
    return self.offset_channels + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [2, self.alpha_channels],
      dim=1
    )
    size_map = F.elu(size_map) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class XYZSizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 3
    return self.offset_channels + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [3, self.alpha_channels],
      dim=1
    )
    size_map = F.elu(size_map) + 1
    offsets = offsets_from_size_map(
      size_map, self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)
