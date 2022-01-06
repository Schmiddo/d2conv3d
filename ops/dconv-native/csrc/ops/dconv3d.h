#pragma once

//#include <torch/extension.h>
#include <tuple>

#include "dconv3d_gpu.h"

size_t get_max_intermediate_elements();
void set_max_intermediate_elements(size_t m);

at::Tensor deform_conv3d_forward(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const std::tuple<int, int, int>& stride,
        const std::tuple<int, int, int>& padding,
        const std::tuple<int, int, int>& dilation,
        const int groups,
        const int offset_groups
) {
    return deform_conv3d_forward_cuda(
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
deform_conv3d_backward(
        const at::Tensor& grad,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const std::tuple<int, int, int>& stride,
        const std::tuple<int, int, int>& padding,
        const std::tuple<int, int, int>& dilation,
        const int groups,
        const int offset_groups
) {
    return deform_conv3d_backward_cuda(
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

class DeformConv3dFunction
: public torch::autograd::Function<DeformConv3dFunction> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            const Variable& input,
            const Variable& offset,
            const at::optional<Variable>& alpha,
            const Variable& weight,
            const at::optional<Variable>& bias,
            int64_t stride_d,
            int64_t stride_h,
            int64_t stride_w,
            int64_t pad_d,
            int64_t pad_h,
            int64_t pad_w,
            int64_t dilation_d,
            int64_t dilation_h,
            int64_t dilation_w,
            int64_t groups,
            int64_t offset_groups
    ) {
        auto _alpha = alpha ? alpha.value() : at::Tensor();
        auto _bias = bias ? bias.value() : at::Tensor();
        auto output = deform_conv3d_forward(
                input,
                offset,
                _alpha,
                weight,
                _bias,
                {stride_d, stride_h, stride_w},
                {pad_d, pad_h, pad_w},
                {dilation_d, dilation_h, dilation_w},
                groups,
                offset_groups
        );

        ctx->save_for_backward({input, offset, _alpha, weight, _bias});
        ctx->saved_data["stride_d"] = stride_d;
        ctx->saved_data["stride_h"] = stride_h;
        ctx->saved_data["stride_w"] = stride_w;
        ctx->saved_data["pad_d"] = pad_d;
        ctx->saved_data["pad_h"] = pad_h;
        ctx->saved_data["pad_w"] = pad_w;
        ctx->saved_data["dilation_d"] = dilation_d;
        ctx->saved_data["dilation_h"] = dilation_h;
        ctx->saved_data["dilation_w"] = dilation_w;
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

        auto stride_d = ctx->saved_data["stride_d"].toInt();
        auto stride_h = ctx->saved_data["stride_h"].toInt();
        auto stride_w = ctx->saved_data["stride_w"].toInt();
        auto pad_d = ctx->saved_data["pad_d"].toInt();
        auto pad_h = ctx->saved_data["pad_h"].toInt();
        auto pad_w = ctx->saved_data["pad_w"].toInt();
        auto dilation_d = ctx->saved_data["dilation_d"].toInt();
        auto dilation_h = ctx->saved_data["dilation_h"].toInt();
        auto dilation_w = ctx->saved_data["dilation_w"].toInt();
        auto groups = ctx->saved_data["groups"].toInt();
        auto offset_groups = ctx->saved_data["offset_groups"].toInt();

        auto grads = deform_conv3d_backward(
                grad_output[0],
                input,
                offset,
                alpha,
                weight,
                bias,
                {stride_d, stride_h, stride_w},
                {pad_d, pad_h, pad_w},
                {dilation_d, dilation_h, dilation_w},
                groups,
                offset_groups
        );

        auto grad_input = std::get<0>(grads);
        auto grad_offset = std::get<1>(grads);
        auto grad_alpha = std::get<2>(grads);
        auto grad_weight = std::get<3>(grads);
        auto grad_bias = std::get<4>(grads);

        // TODO: don't compute gradients for undefined inputs
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
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable()
        };
    }
};

at::Tensor deform_conv3d(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::optional<at::Tensor>& alpha,
        const at::Tensor& weight,
        const at::optional<at::Tensor>& bias,
        int64_t stride_d,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_d,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_d,
        int64_t dilation_h,
        int64_t dilation_w,
        int64_t groups,
        int64_t offset_groups
) {
    auto result = DeformConv3dFunction::apply(
            input,
            offset,
            alpha,
            weight,
            bias,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dilation_d,
            dilation_h,
            dilation_w,
            groups,
            offset_groups
    );

    return result[0];
}