#pragma once

#include "torch/torch.h"
#include <iostream>

at::Tensor conv_depthwise3d_cuda(
        const at::Tensor& input,
        const at::Tensor& weight,
        at::IntArrayRef kernel_size,
        const at::Tensor& bias,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation);

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_depthwise3d_backward_cuda(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& weight,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        const std::array<bool, 3> output_mask);

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DepthwiseConv3dFunction
: public torch::autograd::Function<DepthwiseConv3dFunction> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            const Variable& input,
            const Variable& weight,
            const at::optional<Variable>& bias,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation
    ) {
        auto _bias = bias ? bias.value() : at::Tensor();
        auto output = conv_depthwise3d_cuda(
                input,
                weight,
                weight.sizes().slice(2),
                _bias,
                stride,
                padding,
                dilation
                );
        ctx->save_for_backward({input, weight, _bias});
        ctx->saved_data["stride_d"] = stride[0];
        ctx->saved_data["stride_h"] = stride[1];
        ctx->saved_data["stride_w"] = stride[2];
        ctx->saved_data["padding_d"] = padding[0];
        ctx->saved_data["padding_h"] = padding[1];
        ctx->saved_data["padding_w"] = padding[2];
        ctx->saved_data["dilation_d"] = dilation[0];
        ctx->saved_data["dilation_h"] = dilation[1];
        ctx->saved_data["dilation_w"] = dilation[2];

        return {output};
    }

    static variable_list backward(
            AutogradContext* ctx,
            variable_list grad_output
            ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto stride_d = ctx->saved_data["stride_d"].toInt();
        auto stride_h = ctx->saved_data["stride_h"].toInt();
        auto stride_w = ctx->saved_data["stride_w"].toInt();
        auto padding_d = ctx->saved_data["padding_d"].toInt();
        auto padding_h = ctx->saved_data["padding_h"].toInt();
        auto padding_w = ctx->saved_data["padding_w"].toInt();
        auto dilation_d = ctx->saved_data["dilation_d"].toInt();
        auto dilation_h = ctx->saved_data["dilation_h"].toInt();
        auto dilation_w = ctx->saved_data["dilation_w"].toInt();

        auto gradients = conv_depthwise3d_backward_cuda(
                grad_output[0],
                input,
                weight,
                weight.sizes().slice(2),
                {stride_d, stride_h, stride_w},
                {padding_d, padding_h, padding_w},
                {dilation_d, dilation_h, dilation_w},
                {input.requires_grad(), weight.requires_grad(), bias.requires_grad()}
                );
        auto grad_input = std::get<0>(gradients);
        auto grad_weight = std::get<1>(gradients);
        auto grad_bias = std::get<2>(gradients);

        return {
                grad_input,
                grad_weight,
                grad_bias,
                Variable(),
                Variable(),
                Variable()
        };
    }
};

at::Tensor depthwise_conv3d(
        const at::Tensor& input,
        const at::Tensor& weight,
        at::optional<at::Tensor> bias,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation
        ) {
    return DepthwiseConv3dFunction::apply(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation
            )[0];
}