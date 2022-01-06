from .modules import *


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
    "SizeConditionedDCNv1": SizeConditionedDConv3d,
    "XYSizeConditionedDCNv1": XYSizeConditionedDConv3d,
    "XYZSizeConditionedDCNv1": XYZSizeConditionedDConv3d,
    "SizeConditionedDCNv2": SizeConditionedMDConv3d,
    "XYSizeConditionedDCNv2": XYSizeConditionedMDConv3d,
    "XYZSizeConditionedDCNv2": XYZSizeConditionedMDConv3d,
}
def dconv_replace_layer(conv_module, **kwargs):
    if not kwargs.get("replace_1x1_kernels") and conv_module.kernel_size == (1, 1, 1):
        return conv_module
    num_offset_groups = kwargs.get("offset_groups", 1)
    activation = kwargs.get("activation", "sigmoid")
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
        )
    dconv.deform_conv.weight.data = conv_module.weight.data.detach()
    if conv_module.bias is not None:
        dconv.deform_conv.bias.data = conv_module.bias.data.detach()
    return dconv


def add_deformable_conv(module, **kwargs):
    return replace_module(dconv_replace_layer, nn.Conv3d, module, **kwargs)
