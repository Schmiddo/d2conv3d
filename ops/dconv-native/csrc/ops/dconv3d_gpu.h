#pragma once

#include <torch/torch.h>

at::Tensor deform_conv3d_forward_cuda(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        std::tuple<int, int, int> stride,
        std::tuple<int, int, int> pad,
        std::tuple<int, int, int> dilation,
        int n_weight_groups,
        int n_offset_groups
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv3d_backward_cuda(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        std::tuple<int, int, int> stride,
        std::tuple<int, int, int> pad,
        std::tuple<int, int, int> dilation,
        int n_weight_groups,
        int n_offset_groups
);