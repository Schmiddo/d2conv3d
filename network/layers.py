import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.modules.utils import _triple
import einops
import math

from dconv_native import DeformConv3d
from utils.grid import (
  offsets_from_size_map, generate_offsets_from_size_and_flow, construct_3d_kernel_grid,
  grid_sample_with_kernel_offsets
)

from torch.utils.checkpoint import checkpoint


class _DConvBase(nn.Module):
  _activations = {
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
    "none": lambda x: x,
  }

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
               dilation=1, groups=1, offset_groups=1, bias=True,
               activation="sigmoid", init="zero", norm_type="none", norm_groups=1,
               size_activation="elu", deform_param_op=nn.Conv3d):
    super(_DConvBase, self).__init__()
    kernel_size = _triple(kernel_size)

    self.num_kernel_points = kernel_size[0] * kernel_size[1] * kernel_size[2]
    self.offset_groups = offset_groups
    self.groups = groups
    self.size_activation = size_activation

    if norm_type not in ("BatchNorm", "GroupNorm", "none"):
      raise ValueError(f"Unexpected norm_type: '{norm_type}'")
    if norm_type == "GroupNorm":
      self.norm = nn.GroupNorm(norm_groups, in_channels)
    elif norm_type == "BatchNorm":
      self.norm = nn.BatchNorm3d(in_channels)
    else:
      self.norm = None

    if isinstance(deform_param_op, nn.Module):
      self.deform_params = deform_param_op
    else:
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

    if not isinstance(deform_param_op, nn.Module):
      self._init_deform_params(activation, init)

  def _init_deform_params(self, activation, init):
    if init == "zero":
      # initialize offsets to zero
      nn.init.zeros_(self.deform_params.weight)
      if self.deform_params.bias is not None:
        nn.init.zeros_(self.deform_params.bias)
    else:
      raise ValueError(f"init scheme {init} not supported")

  def _deform_channels(self):
    return 3 * self.offset_groups * self.num_kernel_points

  def _get_deformation(self, input):
    return self.deform_params(input), None

  def forward(self, input):
    if self.norm is None:
      deform_input = input
    else:
      deform_input = self.norm(input)
    deformation = checkpoint(self._get_deformation, deform_input)
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

  def _init_deform_params(self, activation, init):
    if init == "zero":
      # initialize offsets to zero
      nn.init.zeros_(self.deform_params.weight)
    elif init == "random":
      if activation == "sigmoid" or activation == "tanh":
        nn.init.xavier_normal_(self.deform_params.weight[self.offset_channels:])
      else:
        nn.init.kaiming_normal_(self.deform_params.weight[self.offset_channels:])
    if self.deform_params.bias is not None:
      nn.init.zeros_(self.deform_params.bias)


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


def _apply_size_activation(activation_mode, size_map):
  if activation_mode == "elu":
    return F.elu(size_map) + 1
  elif activation_mode == "relu":
    return F.relu(size_map + 1)
  elif activation_mode == "relu_plus_one":
    return F.relu(size_map + 1e-3) + 1
  elif activation_mode == "linear":
    return size_map
  else:
    raise ValueError(f"Unexpected size activation mode '{activation_mode}'")


class SizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 1
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = _apply_size_activation(self.size_activation, self.deform_params(input))
    offsets = offsets_from_size_map(
      size_map, "x", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class XYSizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 2
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = _apply_size_activation(self.size_activation, self.deform_params(input))
    offsets = offsets_from_size_map(
      size_map, "xy", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class XXZSizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 2
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = _apply_size_activation(self.size_activation, self.deform_params(input))
    offsets = offsets_from_size_map(
      size_map, "xxz", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class XYZSizeConditionedDConv3d(DConv3d):
  def _deform_channels(self):
    self.offset_channels = 3
    return self.offset_channels

  def _get_deformation(self, input):
    size_map = _apply_size_activation(self.size_activation, self.deform_params(input))
    offsets = offsets_from_size_map(
      size_map, "xyz", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, None


class SizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 1
    return 1 + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [1, self.alpha_channels],
      dim=1
    )
    size_map = _apply_size_activation(self.size_activation, size_map)
    offsets = offsets_from_size_map(
      size_map, "x", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class XYSizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 2
    return 2 + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [2, self.alpha_channels],
      dim=1
    )
    size_map = _apply_size_activation(self.size_activation, size_map)
    offsets = offsets_from_size_map(
      size_map, "xy", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class XXZSizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 2
    return 2 + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [2, self.alpha_channels],
      dim=1
    )
    size_map = _apply_size_activation(self.size_activation, size_map)
    offsets = offsets_from_size_map(
      size_map, "xxz", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class XYZSizeConditionedMDConv3d(MDConv3d):
  def _deform_channels(self):
    self.alpha_channels = self.offset_groups * self.num_kernel_points
    self.offset_channels = 3
    return 3 + self.alpha_channels

  def _get_deformation(self, input):
    deformation = self.deform_params(input)
    size_map, alpha = torch.split(
      deformation,
      [3, self.alpha_channels],
      dim=1
    )
    size_map = _apply_size_activation(self.size_activation, size_map)
    offsets = offsets_from_size_map(
      size_map, "xyz", self.deform_conv.kernel_size, self.deform_conv.dilation
    )
    return offsets, self.activation(alpha)


class _ASPPImagePooler(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(_ASPPImagePooler, self).__init__()

    self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
    self.conv = nn.Conv3d(in_planes, out_planes, 1, bias=False)
    self.gn = nn.GroupNorm(16, out_planes)

  def forward(self, x):
    T, H, W = x.shape[-3:]
    x = self.pool(x)
    x = self.gn(F.relu(self.conv(x)))
    return F.interpolate(x, (T, H, W), mode='trilinear', align_corners=True)


class _ASPPConv(nn.Sequential):
  def __init__(self, in_planes, out_planes, dilation):
    super(_ASPPConv, self).__init__(
      nn.Conv3d(in_planes, in_planes, (1, 3, 3), padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation), groups=in_planes, bias=False),
      nn.ReLU(inplace=True),
      nn.GroupNorm(16, in_planes),
      nn.Conv3d(in_planes, out_planes, 1, bias=False),
      nn.ReLU(inplace=True),
      nn.GroupNorm(16, out_planes)
    )


class ASPPModule(nn.Module):
  def __init__(self, in_planes, out_planes, inter_planes=None):
    super(ASPPModule, self).__init__()

    if not inter_planes:
      inter_planes = int(out_planes / 4)

    self.pyramid_layers = nn.ModuleList([
      nn.Sequential(
        nn.Conv3d(in_planes, inter_planes, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.GroupNorm(16, inter_planes)
      ),
      _ASPPConv(in_planes, inter_planes, 3),
      _ASPPConv(in_planes, inter_planes, 6),
      _ASPPConv(in_planes, inter_planes, 9),
      _ASPPImagePooler(in_planes, inter_planes)
    ])

    self.conv = nn.Conv3d(inter_planes * 5, out_planes, 1, padding=0, bias=False)
    self.gn = nn.GroupNorm(32, out_planes)

    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

  def forward(self, x):
    x = [layer(x) for layer in self.pyramid_layers]
    x = torch.cat(x, 1)
    return self.gn(F.relu(self.conv(x)))


class ChannelSepConv3d(nn.Sequential):
  def __init__(self, inplanes, outplanes, n_groups=32):
    super(ChannelSepConv3d, self).__init__(
      nn.Conv3d(inplanes, inplanes, 3, padding=1, groups=outplanes, bias=False),
      nn.GroupNorm(n_groups, inplanes),
      nn.ReLU(inplace=True),
      nn.Conv3d(inplanes, outplanes, 1, bias=False),
      nn.GroupNorm(n_groups, outplanes),
      nn.ReLU(inplace=True)
    )


class FrozenBatchNorm(nn.Module):
  def __init__(self, num_features, epsilon=1e-5):
    super(FrozenBatchNorm, self).__init__()

    self.register_buffer("weight", torch.ones(num_features))
    self.register_buffer("bias", torch.zeros(num_features))
    self.register_buffer("running_mean", torch.zeros(num_features))
    self.register_buffer("running_var", torch.ones(num_features))
    self.epsilon = epsilon

  def forward(self, x):
    scale = self.weight * (self.running_var + self.epsilon).rsqrt()
    bias = self.bias - self.running_mean * scale

    reshape_args = [1, -1] + ([1] * (x.ndimension() - 2))
    scale = scale.reshape(*reshape_args)
    bias = bias.reshape(*reshape_args)
    return x * scale + bias


@torch.jit.script
def scaled_relu(x):
  # relu(x) / sqrt(0.5 * (1 - (1/pi)))
  return F.relu(x) * 1.71285855


class ScaledReLU(nn.Module):
  def __init__(self):
    super(ScaledReLU, self).__init__()

  def forward(self, x):
    return scaled_relu(x)


@torch.jit.script
def standardize(weight, eps: float=1e-4):
  fan_in = torch.prod(torch.as_tensor(weight.shape[1:]))
  flat_weight: torch.Tensor = weight.flatten(1)
  mean = torch.mean(flat_weight, dim=1, keepdim=True)
  var = torch.var(flat_weight, dim= 1, keepdim=True)
  scale = torch.rsqrt(var * fan_in + eps)
  shift = mean * scale
  flat_weight = flat_weight * scale - shift
  return flat_weight.view_as(weight)

