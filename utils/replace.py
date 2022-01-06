import torch.nn as nn
from network.layers import (
  DConv3d, MDConv3d, SpatialDConv3d, SpatialMDConv3d, TemporalDConv3d, TemporalMDConv3d,
  SizeConditionedDConv3d, SizeConditionedMDConv3d,
  XYSizeConditionedDConv3d, XXZSizeConditionedDConv3d, XYZSizeConditionedDConv3d,
  XYSizeConditionedMDConv3d, XXZSizeConditionedMDConv3d, XYZSizeConditionedMDConv3d,
  FrozenBatchNorm,
)


def replace_module(replacement_func, replaced_cls, module, **kwargs):
  n_conversions = 0
  module_output = module

  if isinstance(module, replaced_cls):
    module_output = replacement_func(module, **kwargs)
    n_conversions += 1 if module_output is not module else 0

  for name, child in module.named_children():
    converted_module, conversions = replace_module(
      replacement_func, replaced_cls, child, **kwargs
    )
    module_output.add_module(name, converted_module)
    n_conversions += conversions

  del module

  return module_output, n_conversions

_dconv_versions = {
  "DCNv1": DConv3d,
  "DCNv2": MDConv3d,
  "SpatialDCNv1": SpatialDConv3d,
  "SpatialDCNv2": SpatialMDConv3d,
  "TemporalDCNv1": TemporalDConv3d,
  "TemporalDCNv2": TemporalMDConv3d,
  "ConditionedDCNv1": SizeConditionedDConv3d,
  "SizeConditionedDCNv1": SizeConditionedDConv3d,
  "XYConditionedDCNv1": XYSizeConditionedDConv3d,
  "XXZConditionedDCNv1": XXZSizeConditionedDConv3d,
  "XYZConditionedDCNv1": XYZSizeConditionedDConv3d,
  "ConditionedDCNv2": SizeConditionedMDConv3d,
  "SizeConditionedDCNv2": SizeConditionedMDConv3d,
  "XYConditionedDCNv2": XYSizeConditionedMDConv3d,
  "XXZConditionedDCNv2": XXZSizeConditionedMDConv3d,
  "XYZConditionedDCNv2": XYZSizeConditionedMDConv3d,
}
def dconv_replace_layer(conv_module, **kwargs):
  if not kwargs.get("replace_1x1_kernels") and conv_module.kernel_size == (1, 1, 1):
    return conv_module
  num_offset_groups = kwargs.get("offset_groups", 1)
  activation = kwargs.get("activation", "sigmoid")
  size_activation = kwargs.get("size_activation", "elu")
  init = kwargs.get("init", "zero")
  norm_type = kwargs.get("norm_type", "none")
  norm_groups = kwargs.get("norm_groups", 1)

  DC = _dconv_versions[kwargs["version"]]
  dconv = DC(
    conv_module.in_channels,
    conv_module.out_channels,
    conv_module.kernel_size,
    conv_module.stride,
    conv_module.padding,
    conv_module.dilation,
    conv_module.groups,
    num_offset_groups,
    conv_module.bias is not None,
    activation,
    init,
    norm_type,
    norm_groups,
    size_activation,
  )
  dconv.deform_conv.weight.data = conv_module.weight.data.detach()
  if conv_module.bias is not None:
    dconv.deform_conv.bias.data = conv_module.bias.data.detach()
  return dconv


def add_deformable_conv(module, **kwargs):
  return replace_module(dconv_replace_layer, nn.Conv3d, module, **kwargs)


def conv_to_convws(conv_module: nn.Conv3d):
  # TODO: also for linear layers/1d/2d
  conv = Conv3dWS(
    conv_module.in_channels,
    conv_module.out_channels,
    conv_module.kernel_size,
    conv_module.stride,
    conv_module.padding,
    conv_module.dilation,
    conv_module.groups,
    conv_module.bias is not None
  )
  conv.weight.data = conv_module.weight.data.detach()
  if conv.bias is not None:
    conv.bias.data = conv_module.bias.data.detach()
  return conv


def add_weight_standardisation(module, **kwargs):
  _replaced_modules = nn.modules.conv._ConvNd
  return replace_module(conv_to_convws, _replaced_modules, module, **kwargs)


def bn_to_fbngn(bn_module):
  fbn = FrozenBatchNorm(
    num_features=bn_module.num_features,
    epsilon=bn_module.eps
  )
  fbn.running_mean.data = bn_module.running_mean.data.detach()
  fbn.running_var.data = bn_module.running_var.data.detach()
  fbn.num_batches_tracked = bn_module.num_batches_tracked
  if bn_module.affine:
    fbn.weight.data = bn_module.weight.data.detach()
    fbn.bias.data = bn_module.bias.data.detach()
  hn = nn.Sequential(fbn, nn.GroupNorm(32, bn_module.num_features))
  return fbn
  

def bn_to_frozen_bn(bn_module):
  fbn = FrozenBatchNorm(
    num_features=bn_module.num_features,
    epsilon=bn_module.eps
  )
  fbn.running_mean.data = bn_module.running_mean.data.detach()
  fbn.running_var.data = bn_module.running_var.data.detach()
  fbn.num_batches_tracked = bn_module.num_batches_tracked
  if bn_module.affine:
    fbn.weight.data = bn_module.weight.data.detach()
    fbn.bias.data = bn_module.bias.data.detach()
  return fbn


def bn_to_frozen_bn_with_stats(bn_module):
  fbn = nn.BatchNorm3d(
    num_features=bn_module.num_features,
    eps=bn_module.eps,
    momentum=bn_module.momentum,
    affine=bn_module.affine,
    track_running_stats=bn_module.track_running_stats
  )
  fbn.running_mean.data = bn_module.running_mean.data.detach()
  fbn.running_var.data = bn_module.running_var.data.detach()
  fbn.num_batches_tracked = bn_module.num_batches_tracked
  if bn_module.affine:
    fbn.weight.data = bn_module.weight.data.detach()
    fbn.bias.data = bn_module.bias.data.detach()
  fbn.requires_grad_(False)
  return fbn


def bn_to_gn(bn_module, num_groups):
  gn = nn.GroupNorm(
    num_groups=num_groups,
    num_channels=bn_module.num_features,
    eps=bn_module.eps,
    affine=bn_module.affine
  )
  if bn_module.affine:
    gn.weight.data = bn_module.weight.data.detach()
    gn.bias.data = bn_module.bias.data.detach()
  return gn


_replace_modes = {
  "FBNGN": bn_to_fbngn,
  "GroupNorm": bn_to_gn,
  "FrozenBatchNorm": bn_to_frozen_bn,
  "FrozenBatchNormWithStats": bn_to_frozen_bn_with_stats,
  "BatchNorm": lambda x: x
}


def convert_batchnorm(replace_mode, module, **kwargs):
  replacement_func = _replace_modes[replace_mode]
  return replace_module(replacement_func, nn.modules.batchnorm._BatchNorm, module, **kwargs)
