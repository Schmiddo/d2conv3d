#include <torch/torch.h>

#include <tuple>

#include "dconv_gpu_common.h"
#include "dconv1d_gpu.h"

using torch::RestrictPtrTraits;
template<typename scalar_t, int num_dimensions>
using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dimensions, RestrictPtrTraits>;

template<typename T>
__device__ inline T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

template<typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
        const int num_points,
        const Accessor<scalar_t, 3> input,
        const Accessor<scalar_t, 3> offset,
        const Accessor<scalar_t, 3> alpha,
        const int in_channels,
        const int input_size,
        const int kernel_size,
        const int pad,
        const int stride,
        const int dilation,
        const int output_size,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        Accessor<scalar_t, 3> columns
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        const int out_x = index % output_size;
        const int b = (index / output_size) % batch_size;
        const int c = index / (output_size * batch_size);

        const int offset_group = c / channels_per_offset_group;

        for(int i = 0; i < kernel_size; i++) {
            const scalar_t o = offset[b][offset_group * kernel_size + i][out_x];
            const scalar_t a = alpha[b][offset_group * kernel_size + i][out_x];
            const scalar_t x = out_x * stride - pad + i * dilation + o;

            const int x0 = floor(x);
            const int x1 = x0 + 1;
            const scalar_t v0 = x0 >= 0 && x0 < input_size ? input[b][c][x0] : 0;
            const scalar_t v1 = x1 >= 0 && x1 < input_size ? input[b][c][x1] : 0;

            columns[b][c * kernel_size + i][out_x] = a * lerp(v0, v1, x - x0);
        }
    }
}

static void deformable_im2col(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        int in_channels,
        int input_size,
        int kernel_size,
        int pad,
        int stride,
        int dilation,
        int output_size,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        at::Tensor& columns
) {
    int num_points = in_channels * output_size * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "deformable_im2col_gpu", ([&] {
            deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_points), CUDA_NUM_THREADS>>>(
                    num_points,
                    input.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                    offset.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                    alpha.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                    in_channels,
                    input_size,
                    kernel_size,
                    pad, stride, dilation,
                    output_size,
                    batch_size,
                    n_weight_groups,
                    n_offset_groups,
                    columns.packed_accessor32<scalar_t, 3, RestrictPtrTraits>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }
}

at::Tensor deform_conv1d_forward_cuda(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        int stride,
        int pad,
        int dilation,
        int n_weight_groups,
        int n_offset_groups
) {
    SANITY_CHECK_DCONV1D(input, offset, alpha, weight, bias)
    SANITY_CHECK_DCONV1D_GROUPS(input, offset, alpha, weight, n_weight_groups, n_offset_groups)

    at::DeviceGuard guard(input.device());

    int batch_size = input.size(0);
    int input_size = input.size(2);
    int in_channels = input.size(1);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int in_channels_per_group = in_channels / n_weight_groups;
    int out_channels_per_group = out_channels / n_weight_groups;

    int output_size = get_output_size(input_size, kernel_size, stride, pad, dilation);

    TORCH_CHECK(kernel_size > 0)
    TORCH_CHECK(stride > 0)
    TORCH_CHECK(pad >= 0)
    TORCH_CHECK(dilation > 0)
    TORCH_CHECK(output_size > 0)

    TORCH_CHECK(
            offset.size(2) == output_size,
            "offset output dims do not match computed dims. Got: ",
            offset.size(2), " but expected: ", output_size
    );
    TORCH_CHECK(
            alpha.size(2) == output_size,
            "mask output dims do not match computed dims. Got: ",
            alpha.size(2), " but expected: ", output_size
    );

    auto columns = torch::empty(
        {batch_size, in_channels * kernel_size, output_size},
            input.options()
    );
    auto out = torch::empty(
            {batch_size, out_channels, output_size},
            input.options()
    );

    deformable_im2col(
            input,
            offset,
            alpha,
            in_channels,
            input_size,
            kernel_size,
            pad, stride, dilation,
            output_size,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            columns
    );

    for(int weight_group = 0; weight_group < n_weight_groups; weight_group++) {
        int columns_index = weight_group * in_channels_per_group * kernel_size * output_size;
        int weight_index = weight_group * out_channels_per_group * in_channels_per_group * kernel_size;
        int out_index = weight_group * out_channels_per_group * output_size;

        AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "gemmStridedBatched", ([&] {
            gemmStridedBatched(
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    output_size,
                    out_channels_per_group,
                    in_channels_per_group * kernel_size,
                    scalar_t(1.0),
                    //
                    columns.data_ptr<scalar_t>() + columns_index,
                    output_size, output_size * in_channels * kernel_size,
                    //
                    weight.data_ptr<scalar_t>() + weight_index,
                    in_channels_per_group * kernel_size, 0,
                    scalar_t(0.0),
                    // out^T = (weight * col)^T = col^T * weight^T
                    out.data_ptr<scalar_t>() + out_index,
                    output_size, out_channels * output_size,
                    batch_size
            );
        })
        );
    }

    out.add_(bias.view({1, out_channels, 1}));

    return out;
}

template<typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
        const int num_points,
        const Accessor<scalar_t, 3> columns,
        const Accessor<scalar_t, 3> offset,
        const Accessor<scalar_t, 3> alpha,
        const int in_channels,
        const int input_size,
        const int kernel_size,
        const int pad,
        const int stride,
        const int dilation,
        const int batch_size,
        const int output_size,
        const int n_weight_groups,
        const int n_offset_groups,
        Accessor<scalar_t, 3> grad_input
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        const int out_x = index % output_size;
        const int b = (index / output_size) % batch_size;
        const int i = (index / (output_size * batch_size)) % kernel_size;
        const int c = (index / (output_size * batch_size * kernel_size));

        const int offset_group = c / channels_per_offset_group;

        const scalar_t a = alpha[b][offset_group * kernel_size + i][out_x];
        const scalar_t o = offset[b][offset_group * kernel_size + i][out_x];

        const scalar_t x = out_x * stride - pad + i * dilation + o;

        const scalar_t grad = a * columns[b][c * kernel_size + i][out_x];
        const int x0 = floor(x);
        const int x1 = x0 + 1;

        if(0 <= x0 && x0 < input_size) {
            scalar_t weight = (1 - (x - x0));
            atomicAdd(&grad_input[b][c][x0], weight * grad);
        }
        if(0 <= x1 && x1 < input_size && x1 - x < 1) {
            scalar_t weight = (x - x0);
            atomicAdd(&grad_input[b][c][x1], weight * grad);
        }
    }
}

static void compute_grad_input(
        const at::Tensor& columns,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const int in_channels,
        const int input_size,
        const int kernel_size,
        const int pad,
        const int stride,
        const int dilation,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        at::Tensor& grad_input
) {
    int output_size = get_output_size(input_size, kernel_size, stride, pad, dilation);

    int num_points = in_channels * kernel_size * output_size * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        columns.scalar_type(), "deformable_col2im_gpu", ([&] {
            deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_points), CUDA_NUM_THREADS>>>(
                num_points,
                columns.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                offset.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                alpha.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                in_channels,
                input_size,
                kernel_size,
                pad, stride, dilation,
                batch_size,
                output_size,
                n_weight_groups,
                n_offset_groups,
                grad_input.packed_accessor32<scalar_t, 3, RestrictPtrTraits>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in compute_grad_input: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(
        const int num_points,
        const Accessor<scalar_t, 3> columns,
        const Accessor<scalar_t, 3> input,
        const Accessor<scalar_t, 3> offset,
        const Accessor<scalar_t, 3> alpha,
        const int in_channels,
        const int input_size,
        const int kernel_size,
        const int pad,
        const int stride,
        const int dilation,
        const int batch_size,
        const int output_size,
        const int n_weight_groups,
        const int n_offset_groups,
        Accessor<scalar_t, 3> grad_offset,
        Accessor<scalar_t, 3> grad_alpha
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        scalar_t val = 0;
        scalar_t alpha_val = 0;

        const int out_x = index % output_size;
        const int i = (index / output_size) % kernel_size;
        const int g = (index / (output_size * kernel_size)) % n_offset_groups;
        const int b = (index / (output_size * kernel_size * n_offset_groups));

        const scalar_t o = offset[b][g * kernel_size + i][out_x];
        const scalar_t a = alpha[b][g * kernel_size + i][out_x];

        const scalar_t x = out_x * stride - pad + i * dilation + o;
        int x0 = floor(x);
        const int x1 = x0 + 1;
        // TODO: is this workaround sane?
        // linear interpolation is not differentiable at integers
        if(x - x0 < 1e-8) x0 -= 1;

        auto c_start = g * channels_per_offset_group;
        auto c_stop = c_start + channels_per_offset_group;
        for(int c = c_start; c < c_stop; c++) {
            const scalar_t v0 = 0 <= x0 && x0 < input_size ? input[b][c][x0] : 0;
            const scalar_t v1 = 0 <= x1 && x1 < input_size ? input[b][c][x1] : 0;

            const scalar_t col_value = columns[b][c * kernel_size + i][out_x];

            alpha_val += col_value * lerp(v0, v1, x - x0);
            val += (v1 - v0) * col_value * a;
        }

        grad_offset[b][g * kernel_size + i][out_x] = val;
        grad_alpha[b][g * kernel_size + i][out_x] = alpha_val;
    }
}

static void compute_grad_offset(
        const at::Tensor& columns,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const int in_channels,
        const int input_size,
        const int kernel_size,
        const int pad,
        const int stride,
        const int dilation,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        at::Tensor& grad_offset,
        at::Tensor& grad_alpha
) {
    const int output_size = get_output_size(input_size, kernel_size, stride, pad, dilation);
    const int num_points = output_size * n_offset_groups * kernel_size * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        columns.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
            deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_points), CUDA_NUM_THREADS>>>(
                num_points,
                columns.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                input.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                offset.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                alpha.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                in_channels,
                input_size,
                kernel_size,
                pad, stride, dilation,
                batch_size,
                output_size,
                n_weight_groups,
                n_offset_groups,
                grad_offset.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                grad_alpha.packed_accessor32<scalar_t, 3, RestrictPtrTraits>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in compute_grad_offset: %s\n", cudaGetErrorString(err));
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv1d_backward_cuda(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        const at::Tensor& bias,
        const int stride,
        const int pad,
        const int dilation,
        const int n_weight_groups,
        const int n_offset_groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_size = input.size(2);
    const int output_size = grad_out.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int in_channels_per_group = in_channels / n_weight_groups;
    int out_channels_per_group = out_channels / n_weight_groups;

    auto columns = torch::empty(
        {batch_size, in_channels * kernel_size, output_size},
        input.options()
    );

    auto grad_offset = torch::empty_like(offset);
    auto grad_alpha = torch::empty_like(alpha);
    auto grad_input = torch::zeros_like(input);
    //auto grad_weight = torch::zeros_like(weight);

    for(int weight_group = 0; weight_group < n_weight_groups; weight_group++) {
        auto grad_out_index = weight_group * out_channels_per_group * output_size;
        auto weight_index = weight_group * out_channels_per_group * in_channels_per_group * kernel_size;
        auto columns_index = weight_group * in_channels_per_group * kernel_size * output_size;

        AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "gemmStridedBatched", ([&] {
            gemmStridedBatched(
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    output_size,
                    in_channels_per_group * kernel_size,
                    out_channels_per_group,
                    scalar_t(1.0),
                    grad_out.data_ptr<scalar_t>() + grad_out_index, output_size, output_size * out_channels,
                    weight.data_ptr<scalar_t>() + weight_index, in_channels_per_group * kernel_size, 0,
                    scalar_t(0.0),
                    columns.data_ptr<scalar_t>() + columns_index, output_size, in_channels * kernel_size * output_size,
                    batch_size
            );
        })
        );
    }

    compute_grad_offset(
            columns,
            input,
            offset,
            alpha,
            in_channels,
            input_size,
            kernel_size,
            pad, stride, dilation,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            grad_offset,
            grad_alpha
    );

    compute_grad_input(
            columns,
            offset,
            alpha,
            in_channels,
            input_size,
            kernel_size,
            pad, stride, dilation,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            grad_input
    );

    // for grad_weight; overwrites columns
    deformable_im2col(
            input,
            offset,
            alpha,
            in_channels,
            input_size,
            kernel_size,
            pad, stride, dilation,
            output_size,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            columns
    );

    auto grad_weight = torch::empty({batch_size, out_channels, in_channels_per_group, kernel_size}, weight.options());

    for(int weight_group = 0; weight_group < n_weight_groups; weight_group++) {
        auto columns_index = weight_group * in_channels_per_group * kernel_size * output_size;
        auto grad_weight_index = weight_group * out_channels_per_group * in_channels_per_group * kernel_size;
        auto grad_out_index = weight_group * out_channels_per_group * output_size;

        AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "gemmStridedBatched", ([&] {
            gemmStridedBatched(
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    in_channels_per_group * kernel_size,
                    out_channels_per_group,
                    output_size,
                    scalar_t(1.0),
                    //
                    columns.data_ptr<scalar_t>() + columns_index,
                    output_size, output_size * in_channels * kernel_size,
                    //
                    grad_out.data_ptr<scalar_t>() + grad_out_index,
                    output_size, out_channels * output_size,
                    scalar_t(0.0),
                    //
                    grad_weight.data_ptr<scalar_t>() + grad_weight_index,
                    in_channels_per_group * kernel_size, out_channels * in_channels_per_group * kernel_size,
                    batch_size
            );
        })
        );
    }

    grad_weight = grad_weight.sum({0});

    auto grad_bias = grad_out.sum(c10::IntArrayRef{0, 2});

    return std::make_tuple(grad_input, grad_offset, grad_alpha, grad_weight, grad_bias);
}