import torch
import torch.nn as nn
import torch.nn.functional as F

from network.backbone.resnet import BottleneckBlock3d
from network.layers import (
  DConv3d, MDConv3d, ScaledReLU, scaled_relu,
  XYZSizeConditionedMDConv3d,
  DeformConv3d
)
from network.aspp import ASPP3D

from utils.registry import register

from torch.utils.checkpoint import checkpoint

try:
  from fast_depthwise_conv3d import DepthwiseConv3d
except ImportError as e:
  print("Could not load fast depthwise 3d conv")
  print(e)
  DepthwiseConv3d = nn.Conv3d


@register("module")
class Refine3d(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2, ConvType=nn.Conv3d, NormType=nn.LayerNorm, ActivationType=nn.ReLU, extra_backbone_norm=False,
               interpolation_mode="trilinear", align_corners=False):
    super(Refine3d, self).__init__()
    self.interpolation_mode = interpolation_mode
    self.align_corners = align_corners

    if extra_backbone_norm:
      self.backbone_norm = NormType(inplanes)
    else:
      self.backbone_norm = None

    def _conv(inplanes, planes):
      return nn.Sequential(
        ConvType(inplanes, planes, kernel_size=3, padding=1),
        ActivationType(),
        NormType(planes),
      )
    self.convFS1 = _conv(inplanes, planes)
    self.convFS2 = _conv(planes, planes)
    self.convFS3 = _conv(planes, planes)
    self.convMM1 = _conv(planes, planes)
    self.convMM2 = _conv(planes, planes)
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    if self.backbone_norm is not None:
      f = self.backbone_norm(f)
    s = self.convFS1(f)
    sr = self.convFS2(s)
    sr = self.convFS3(sr)
    s = s + sr

    m = s + F.interpolate(pm, size=s.shape[-3:], mode=self.interpolation_mode, align_corners=self.align_corners)

    mr = self.convMM1(m)
    mr = self.convMM2(mr)
    m = m + mr
    return m

