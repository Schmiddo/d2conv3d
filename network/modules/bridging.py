import torch
import torch.nn as nn
import torch.nn.functional as F

from dconv_native import DeformConv3d
from network.layers import DConv3d, MDConv3d, SpatialDConv3d, SpatialMDConv3d
from network.layers import ASPPModule

from utils.registry import register


@register("module")
class GC3d(nn.Module):
  def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256, which_conv=nn.Conv3d):
    super(GC3d, self).__init__()
    self.conv_l1 = which_conv(inplanes, mdim, kernel_size=(1, kh, 1),
                             padding=(0, int(kh / 2), 0))
    self.conv_l2 = which_conv(mdim, planes, kernel_size=(1, 1, kw),
                             padding=(0, 0, int(kw / 2)))
    self.conv_r1 = which_conv(inplanes, mdim, kernel_size=(1, 1, kw),
                             padding=(0, 0, int(kw / 2)))
    self.conv_r2 = which_conv(mdim, planes, kernel_size=(1, kh, 1),
                             padding=(0, int(kh / 2), 0))

  def forward(self, x):
    x_l = self.conv_l2(self.conv_l1(x))
    x_r = self.conv_r2(self.conv_r1(x))
    x = x_l + x_r
    return x


@register("module")
class DeformableGC3d(GC3d):
  def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256):
    super(DeformableGC3d, self).__init__(inplanes, planes, kh, kw, mdim, DConv3d)


@register("module")
class ModulatedDeformableGC3d(GC3d):
  def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256):
    super(ModulatedDeformableGC3d, self).__init__(inplanes, planes, kh, kw, mdim, MDConv3d)


@register("module")
class SpatialDeformableGC3d(GC3d):
  def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256):
      super(SpatialDeformableGC3d, self).__init__(inplanes, planes, kh, kw, mdim, SpatialDConv3d)


@register("module")
class SpatialModulatedDeformableGC3d(GC3d):
  def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256):
    super(SpatialModulatedDeformableGC3d, self).__init__(inplanes, planes, kh, kw, mdim, SpatialMDConv3d)


@register("module")
class CoordDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(CoordDeformableC3D, self).__init__()
    self.deformation = CoordConv3d(inplanes, 81, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    offsets = self.deformation(x)
    return self.c3d(x, offsets, None)


@register("module")
class CoordModulatedDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(CoordModulatedDeformableC3D, self).__init__()
    self.deformation = CoordConv3d(inplanes, 108, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    offsets, alpha = torch.split(self.deformation(x), [81, 27], dim=1)
    alpha = F.sigmoid(alpha)
    return self.c3d(x, offsets, alpha)


@register("module")
class MultilayerDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(MultilayerDeformableC3D, self).__init__()
    mdim = 256
    self.layers = nn.Sequential(
      DConv3d(inplanes, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      DConv3d(mdim, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      DConv3d(mdim, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )
    self.deformation = nn.Conv3d(mdim, 81, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    offsets = self.deformation(self.layers(x))
    return self.c3d(x, offsets, None)


@register("module")
class MultilayerModulatedDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(MultilayerModulatedDeformableC3D, self).__init__()
    mdim = 256
    self.layers = nn.Sequential(
      MDConv3d(inplanes, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      MDConv3d(mdim, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      MDConv3d(mdim, mdim, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )
    self.deformation = nn.Conv3d(mdim, 108, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    deformation = self.deformation(self.layers(x))
    offsets, alpha = torch.split(deformation, [81, 27], dim=1)
    alpha = F.sigmoid(alpha)
    return self.c3d(x, offsets, alpha)


@register("module")
class ASPPDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(ASPPDeformableC3D, self).__init__()
    self.aspp = ASPPModule(inplanes, 256, 256)
    self.deformation = nn.Conv3d(256, 81, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    pyramid = self.aspp(x)
    offsets = self.deformation(pyramid)
    return self.c3d(x, offsets, None)


@register("module")
class ASPPModulatedDeformableC3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(ASPPModulatedDeformableC3D, self).__init__()
    self.aspp = ASPPModule(inplanes, 256, 256)
    self.deformation = nn.Conv3d(256, 108, kernel_size=3, padding=1)
    self.c3d = DeformConv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    pyramid = self.aspp(x)
    offsets, alpha = torch.split(self.deformation(pyramid), [81, 27], dim=1)
    alpha = F.sigmoid(alpha)
    return self.c3d(x, offsets, alpha)


@register("module")
class C3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(C3D, self).__init__()
    self.c3d = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.c3d(x)
    return x


@register("module")
class DeformableC3D(C3D):
  def __init__(self, inplanes, planes):
    super(DeformableC3D, self).__init__(inplanes, planes)
    self.c3d = DConv3d(inplanes, planes, kernel_size=3, padding=1)


@register("module")
class ModulatedDeformableC3D(C3D):
  def __init__(self, inplanes, planes):
    super(ModulatedDeformableC3D, self).__init__(inplanes, planes)
    self.c3d = MDConv3d(inplanes, planes, kernel_size=3, padding=1)

