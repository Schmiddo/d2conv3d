import torch
from torch import nn as nn
from torch.nn import functional as F

from network.layers import ChannelSepConv3d, scaled_relu
from network.aspp import ASPP3D

from utils.construct import build_backbone, get_module_fn
from utils.replace import add_weight_standardisation, add_deformable_conv
from utils.registry import register


@register("network")
class EncoderDecoderNetwork(nn.Module):
  def __init__(self, cfg):
    super(EncoderDecoderNetwork, self).__init__()
    backbone = build_backbone(cfg.backbone)

    self.register_buffer("mean", torch.as_tensor(cfg.mean)[:, None, None, None])
    self.register_buffer("std", torch.as_tensor(cfg.std)[:, None, None, None])

    self.encoder = Encoder(backbone)
    decoder = get_module_fn(cfg.decoder.name)(cfg.decoder, self.encoder.layer_widths)
    if cfg.use_ws:
      decoder, _ = add_weight_standardisation(decoder)
    self.decoder = decoder

  def forward(self, x):
    x = x.sub(self.mean).div(self.std)
    features = self.encoder(x)
    output = self.decoder(*features)
    return output


###############################################################################
#
# Encoder
#
###############################################################################


class EncoderBase(nn.Module):
  def __init__(self):
    super(EncoderBase, self).__init__()


class Encoder(EncoderBase):
  def __init__(self, backbone):
    super(Encoder, self).__init__()
    self.stem = backbone.get_stem()
    if hasattr(self.stem, "maxpool"):
      self.stem.maxpool = nn.MaxPool3d(
        kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
      )
    self.layer1 = backbone.layer1
    self.layer2 = backbone.layer2
    self.layer3 = backbone.layer3
    self.layer4 = backbone.layer4
    self.layer_widths = backbone.layer_widths[1:]

  def forward(self, x):
    f = self.stem(x)
    e1 = self.layer1(f)
    e2 = self.layer2(e1)
    e3 = self.layer3(e2)
    e4 = self.layer4(e3)

    return e1, e2, e3, e4


###############################################################################
#
# Decoder
#
###############################################################################


def _make_aspp(inplanes, planes, kernel_size, padding):
  if inplanes > 256:
    dilation_rates = [(1, 3, 3), (1, 6, 6), (1, 9, 9)] if inplanes > 512 else [(1, 6, 6), (1, 12, 12), (1, 18, 18)]
    return ASPP3D(inplanes, planes, planes, dilation_rates)
  else:
    return nn.Conv3d(inplanes, planes, kernel_size, padding=padding)


_conv_types = {
  "Conv3d": nn.Conv3d,
  "ASPP": _make_aspp,
}

_norm_layers = {
  None: nn.Identity,
  "none": nn.Identity,
  "LayerNorm": nn.LayerNorm,
  "InstanceNorm": nn.InstanceNorm3d,
  "BatchNorm": nn.BatchNorm3d,
}


_activations = {
  "relu": nn.ReLU,
  "leaky_relu": nn.LeakyReLU,
}


class DecoderBase(nn.Module):
  def __init__(self, cfg, layer_widths):
    super(DecoderBase, self).__init__()
    inter_block = get_module_fn(cfg.inter_block)
    refine_block = get_module_fn(cfg.refine_block)

    self.gc = inter_block(layer_widths[-1], cfg.mdim)
    self.convG1 = nn.Conv3d(cfg.mdim, cfg.mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(cfg.mdim, cfg.mdim, kernel_size=3, padding=1)

    ConvType = _conv_types[cfg.conv_type]
    if cfg.norm_type == "GroupNorm":
      NormType = lambda planes: nn.GroupNorm(cfg.norm_groups, planes)
    else:
      NormType = _norm_layers[cfg.norm_type]
    ActivationType = _activations[cfg.activation]
    if cfg.interpolation_mode not in ("trilinear", "nearest"): raise ValueError(f"Unknown interpolation mode '{cfg.interpolation_mode}'")
    align_corners = cfg.align_corners if cfg.interpolation_mode == "trilinear" else None

    extra_args = {"ConvType": ConvType, "NormType": NormType, "ActivationType": ActivationType,
                    "extra_backbone_norm": cfg.extra_backbone_norm, "interpolation_mode": cfg.interpolation_mode,
                    "align_corners": align_corners}

    self.rf3 = refine_block(layer_widths[-2], cfg.mdim, **extra_args)
    self.rf2 = refine_block(layer_widths[-3], cfg.mdim, **extra_args)
    self.rf1 = refine_block(layer_widths[-4], cfg.mdim, **extra_args)

    if cfg.deformable:
      total_conversions = 0
      for key in cfg.deformable.layers:
        layer = self.__getattr__(key)
        layer, n_conversions = add_deformable_conv(layer, **cfg.deformable)
        self.__setattr__(key, layer)
        total_conversions += n_conversions
      if total_conversions:
        print(f"[DECODER] Replaced {total_conversions} convs with {cfg.deformable.version}")

  def forward(self, e1, e2, e3, e4):
    x = self.gc(e4)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m4 = x + r
    m3 = self.rf3(e3, m4)
    m2 = self.rf2(e2, m3)
    m1 = self.rf1(e1, m2)

    return m1, m2, m3, m4


@register("module", "SaliencyDecoder")
class Decoder(DecoderBase):
  def __init__(self, cfg, layer_widths):
    super(Decoder, self).__init__(cfg, layer_widths)
    self.pred = nn.Conv3d(cfg.mdim, cfg.num_classes, kernel_size=3, padding=1)

  def forward(self, e1, e2, e3, e4):
    m1, _, _, _ = super(Decoder, self).forward(e1, e2, e3, e4)
    p1 = self.pred(F.relu(m1))
    return F.interpolate(p1, scale_factor=(1, 4, 4), mode="trilinear")

