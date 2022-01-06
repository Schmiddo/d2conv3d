#pragma once

#include <ATen/Tensor.h>
#include <tuple>

at::Tensor
deform_conv1d_forward_cuda(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        int stride,
        int padding,
        int dilation,
        int groups,
        int offset_groups
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv1d_backward_cuda(
        const at::Tensor& grad,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        int stride,
        int padding,
        int dilation,
        int groups,
        int offset_groups
);