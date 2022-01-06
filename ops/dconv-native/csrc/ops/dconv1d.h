#pragma once

#include <ATen/Tensor.h>
#include <torch/autograd.h>

#include <tuple>

#include "dconv1d_gpu.h"

at::Tensor deform_conv1d_forward(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const int stride,
        const int padding,
        const int dilation,
        const int groups,
        const int offset_groups
) {
    return deform_conv1d_forward_cuda(
            input.contiguous(),
            offset.contiguous(),
            alpha.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            stride,
            padding,
            dilation,
            groups,
            offset_groups
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv1d_backward(
        const at::Tensor& grad,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const int stride,
        const int padding,
        const int dilation,
        const int groups,
        const int offset_groups
) {
    return deform_conv1d_backward_cuda(
            grad.contiguous(),
            input.contiguous(),
            offset.contiguous(),
            alpha.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            stride,
            padding,
            dilation,
            groups,
            offset_groups
    );
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DeformConv1dFunction
        : public torch::autograd::Function<DeformConv1dFunction> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable input,
            Variable offset,
            Variable alpha,
            Variable weight,
            Variable bias,
            int64_t stride,
            int64_t pad,
            int64_t dilation,
            int64_t groups,
            int64_t offset_groups
    ) {
        auto output = deform_conv1d_forward(
                input,
                offset,
                alpha,
                weight,
                bias,
                stride,
                pad,
                dilation,
                groups,
                offset_groups
        );

        ctx->save_for_backward({input, offset, alpha, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["pad"] = pad;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["offset_groups"] = offset_groups;

        return {output};
    }

    static variable_list backward(
            AutogradContext* ctx,
            variable_list grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto offset = saved[1];
        auto alpha = saved[2];
        auto weight = saved[3];
        auto bias = saved[4];

        auto stride = ctx->saved_data["stride"].toInt();
        auto pad = ctx->saved_data["pad"].toInt();
        auto dilation = ctx->saved_data["dilation"].toInt();
        auto groups = ctx->saved_data["groups"].toInt();
        auto offset_groups = ctx->saved_data["offset_groups"].toInt();

        auto grads = deform_conv1d_backward(
                grad_output[0],
                input,
                offset,
                alpha,
                weight,
                bias,
                stride,
                pad,
                dilation,
                groups,
                offset_groups
        );

        auto grad_input = std::get<0>(grads);
        auto grad_offset = std::get<1>(grads);
        auto grad_alpha = std::get<2>(grads);
        auto grad_weight = std::get<3>(grads);
        auto grad_bias = std::get<4>(grads);


        return {
                grad_input,
                grad_offset,
                grad_alpha,
                grad_weight,
                grad_bias,
                Variable(),
                Variable(),
                Variable(),
                Variable(),
                Variable()
        };
    }
};

at::Tensor deform_conv1d(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        int64_t stride,
        int64_t pad,
        int64_t dilation,
        int64_t groups,
        int64_t offset_groups
) {
    auto result = DeformConv1dFunction::apply(
            input,
            offset,
            alpha,
            weight,
            bias,
            stride,
            pad,
            dilation,
            groups,
            offset_groups
    );

    return result[0];
}