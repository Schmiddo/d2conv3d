#include <torch/torch.h>
#include <ATen/AccumulateType.h>
#include <tuple>
#include "dconv_gpu_common.h"
#include "dconv3d_gpu.h"
#include <iostream>
using torch::RestrictPtrTraits;
template<typename scalar_t, int num_dimensions>
using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dimensions, RestrictPtrTraits>;
using at::acc_type;

static size_t MAX_INTERMEDIATE_ELEMENTS = 1ul << 32ul;
size_t get_max_intermediate_elements(){ return MAX_INTERMEDIATE_ELEMENTS; }
void set_max_intermediate_elements(size_t m) { MAX_INTERMEDIATE_ELEMENTS = m; }

inline int get_batch_channels(
        int in_channels,
        int kernel_size,
        int output_size,
        int n_weight_groups,
        int n_offset_groups
) {
    return 1;
    auto max_channels = MAX_INTERMEDIATE_ELEMENTS / (kernel_size * output_size);
    if(in_channels <= max_channels) {
        return in_channels;
    }
    if(max_channels == 0) {
        return 1;
    }

    auto batch_channels = max_channels;
    auto channels_per_weight_group = in_channels / n_weight_groups;
    auto channels_per_offset_group = in_channels / n_offset_groups;

    while(batch_channels > 1) {
        if(channels_per_weight_group % batch_channels == 0
            && channels_per_offset_group % batch_channels == 0) {
            break;
        } else {
            batch_channels--;
        }
    }

    return batch_channels;
}

template <typename scalar_t>
__device__ scalar_t trilinear_interpolate(
        const scalar_t* in,
        const int depth,
        const int height,
        const int width,
        scalar_t d,
        scalar_t h,
        scalar_t w
) {
    using accscalar_t = acc_type<scalar_t, true>;
    if (d <= -1 || depth <= d || h <= -1 || height <= h || w <= -1 || width <= w) {
        return 0;
    }

    int d0 = floor(d);
    int h0 = floor(h);
    int w0 = floor(w);
    int d1 = d0 + 1;
    int h1 = h0 + 1;
    int w1 = w0 + 1;

    scalar_t points[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if(d0 >= 0 && h0 >= 0 && w0 >= 0)
        points[0] = in[d0 * height * width + h0 * width + w0];
    if(d0 >= 0 && h0 >= 0 && w1 <= width - 1)
        points[1] = in[d0 * height * width + h0 * width + w1];
    if(d0 >= 0 && h1 <= height - 1 && w0 >= 0)
        points[2] = in[d0 * height * width + h1 * width + w0];
    if(d0 >= 0 && h1 <= height - 1 && w1 <= width - 1)
        points[3] = in[d0 * height * width + h1 * width + w1];
    if(d1 <= depth - 1 && h0 >= 0 && w0 >= 0)
        points[4] = in[d1 * height * width + h0 * width + w0];
    if(d1 <= depth - 1 && h0 >= 0 && w1 <= width - 1)
        points[5] = in[d1 * height * width + h0 * width + w1];
    if(d1 <= depth - 1 && h1 <= height - 1 && w0 >= 0)
        points[6] = in[d1 * height * width + h1 * width + w0];
    if(d1 <= depth - 1 && h1 <= height - 1 && w1 <= width - 1)
        points[7] = in[d1 * height * width + h1 * width + w1];

    scalar_t a000 = (d1 - d) * (h1 - h) * (w1 - w);
    scalar_t a001 = (d1 - d) * (h1 - h) * (w - w0);
    scalar_t a010 = (d1 - d) * (h - h0) * (w1 - w);
    scalar_t a011 = (d1 - d) * (h - h0) * (w - w0);
    scalar_t a100 = (d - d0) * (h1 - h) * (w1 - w);
    scalar_t a101 = (d - d0) * (h1 - h) * (w - w0);
    scalar_t a110 = (d - d0) * (h - h0) * (w1 - w);
    scalar_t a111 = (d - d0) * (h - h0) * (w - w0);

    accscalar_t res = 0;
    res = fma(points[0], a000, res);
    res = fma(points[1], a001, res);
    res = fma(points[2], a010, res);
    res = fma(points[3], a011, res);
    res = fma(points[4], a100, res);
    res = fma(points[5], a101, res);
    res = fma(points[6], a110, res);
    res = fma(points[7], a111, res);

    return res;

}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
        const int n,
        const scalar_t* input_base_ptr,
        const scalar_t* offset_base_ptr,
        const scalar_t* alpha_base_ptr,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int batch_size,
        const int in_channels,
        const int n_weight_groups,
        const int n_offset_groups,
        const int out_d,
        const int out_h,
        const int out_w,
        scalar_t* columns_base_ptr
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_z = (index / (out_w * out_h)) % out_d;
        const int in_c = (index / (out_w * out_h * out_d)) % in_channels;
        const int out_b = (index / (out_w * out_h * out_d * in_channels));

        const int grp_idx = in_c / channels_per_offset_group;

        auto columns_ptr = columns_base_ptr
                + out_b * in_channels * weight_d * weight_h * weight_w * out_d * out_h * out_w
                + in_c * weight_d * weight_h * weight_w * out_d * out_h * out_w
                + out_z * out_h * out_w
                + out_y * out_w
                + out_x;

        auto input_ptr = input_base_ptr
                + out_b * (in_channels * depth * height * width)
                + in_c * (depth * height * width);

        auto alpha_ptr = alpha_base_ptr
                ? alpha_base_ptr
                     + (out_b * n_offset_groups + grp_idx)
                         * weight_d * weight_h * weight_w
                         * out_d * out_h * out_w
                     + out_z * out_h * out_w
                     + out_y * out_w
                     + out_x
                : nullptr;

        auto offset_ptr = offset_base_ptr
                + (out_b * n_offset_groups + grp_idx)
                    * 3 * weight_d * weight_h * weight_w
                    * out_d * out_h * out_w
                + out_z * out_h * out_w
                + out_y * out_w
                + out_x;

        const int idx_stride = out_d * out_h * out_w;
        for(int i = 0; i < weight_d; i++) {
            for(int j = 0; j < weight_h; j++) {
                for(int k = 0; k < weight_w; k++) {
                    const int alpha_idx = ((i * weight_h + j) * weight_w + k);
                    const scalar_t alpha = alpha_ptr ? alpha_ptr[alpha_idx * idx_stride] : 1.0;

                    const int offset_idx = 3 * alpha_idx;
                    const scalar_t offset_d = offset_ptr[(offset_idx + 0) * idx_stride];
                    const scalar_t offset_h = offset_ptr[(offset_idx + 1) * idx_stride];
                    const scalar_t offset_w = offset_ptr[(offset_idx + 2) * idx_stride];

                    const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
                    const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
                    const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

                    *columns_ptr = alpha * trilinear_interpolate(input_ptr, depth, height, width, z, y, x);

                    columns_ptr += out_d * out_h * out_w;
                }
            }
        }
    }
}

static void deformable_im2col(
        const at::Tensor& input,
        const at::Tensor& data_offset,
        const at::Tensor& alpha,
        int in_channels,
        int depth,
        int height,
        int width,
        int weight_d,
        int weight_h,
        int weight_w,
        int pad_d,
        int pad_h,
        int pad_w,
        int stride_d,
        int stride_h,
        int stride_w,
        int dilation_d,
        int dilation_h,
        int dilation_w,
        int out_d,
        int out_h,
        int out_w,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        at::Tensor& columns
) {
    int num_kernels = in_channels * out_d * out_h * out_w * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "deformable_im2col_gpu", ([&] {
            deformable_im2col_gpu_kernel<<<
                GET_BLOCKS(num_kernels),
                CUDA_NUM_THREADS>>>(
                num_kernels,
                input.data_ptr<scalar_t>(),
                data_offset.data_ptr<scalar_t>(),
                alpha.defined() ? alpha.data_ptr<scalar_t>() : nullptr,
                depth, height, width,
                weight_d, weight_h, weight_w,
                pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w,
                dilation_d, dilation_h, dilation_w,
                batch_size,
                in_channels,
                n_weight_groups,
                n_offset_groups,
                out_d, out_h, out_w,
                columns.data_ptr<scalar_t>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }
}

void compute_forward_single_im2col(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        int in_channels,
        int out_channels,
        int in_d,
        int in_h,
        int in_w,
        int weight_d,
        int weight_h,
        int weight_w,
        int pad_d,
        int pad_h,
        int pad_w,
        int stride_d,
        int stride_h,
        int stride_w,
        int dilation_d,
        int dilation_h,
        int dilation_w,
        int out_d,
        int out_h,
        int out_w,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        at::Tensor& out
) {
    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    auto columns = torch::empty(
            {batch_size, in_channels * kernel_size, output_size},
            input.options()
    );

    deformable_im2col(
            input,
            offset,
            alpha,
            in_channels,
            in_d, in_h, in_w,
            weight_d, weight_h, weight_w,
            pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w,
            out_d, out_h, out_w,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            columns
    );

    auto in_channels_per_group = in_channels / n_weight_groups;
    auto out_channels_per_group = out_channels / n_weight_groups;
    for(int weight_group = 0; weight_group < n_weight_groups; weight_group++) {
        int columns_index = weight_group
                * in_channels_per_group
                * kernel_size
                * output_size;
        int weight_index = weight_group
                * out_channels_per_group
                * in_channels_per_group
                * kernel_size;
        int out_index = weight_group
                * out_channels_per_group
                * output_size;

        AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "gemmStridedBatched", ([&] {
            gemmStridedBatched(
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    output_size,
                    out_channels_per_group,
                    in_channels_per_group * kernel_size,
                    scalar_t(1.0),
                    columns.data_ptr<scalar_t>() + columns_index,
                    output_size, output_size * in_channels * kernel_size,
                    weight.data_ptr<scalar_t>() + weight_index,
                    in_channels_per_group * kernel_size, 0,
                    scalar_t(0.0),
                    out.data_ptr<scalar_t>() + out_index,
                    output_size, out_channels * output_size,
                    batch_size
            );
        })
        );
    }
}

void compute_forward_batched_im2col(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        int in_channels,
        int out_channels,
        int in_d,
        int in_h,
        int in_w,
        int weight_d,
        int weight_h,
        int weight_w,
        int pad_d,
        int pad_h,
        int pad_w,
        int stride_d,
        int stride_h,
        int stride_w,
        int dilation_d,
        int dilation_h,
        int dilation_w,
        int out_d,
        int out_h,
        int out_w,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        at::Tensor& out
) {
    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    // Split into multiple passes for large input data
    auto batch_channels = get_batch_channels(
            in_channels, kernel_size, output_size,
            n_weight_groups, n_offset_groups
    );

    auto in_channels_per_group = in_channels / n_weight_groups;
    auto out_channels_per_group = out_channels / n_weight_groups;
    auto in_channels_per_offset_group = in_channels / n_offset_groups;
    auto n_weight_groups_per_channel_batch = std::max(batch_channels / in_channels_per_group, 1);
    auto n_offset_groups_per_channel_batch = std::max(batch_channels / in_channels_per_offset_group, 1);
    auto columns = torch::empty(
            {batch_channels * kernel_size, output_size},
            input.options()
    );

    for(auto b = 0; b < batch_size; b++) {
        for (auto start_channel = 0; start_channel < in_channels; start_channel += batch_channels) {
            using namespace torch::indexing;
            auto end_channel = start_channel + batch_channels;
            auto start_alpha_channel = kernel_size
                    * (start_channel / in_channels_per_offset_group);
            auto end_alpha_channel = kernel_size
                    * (end_channel + in_channels_per_offset_group - 1) / in_channels_per_offset_group;

            auto sliced_input = input.index({b, Slice(start_channel, end_channel)});
            auto sliced_offset = offset.index({b, Slice(3 * start_alpha_channel, 3 * end_alpha_channel)});
            auto sliced_alpha = alpha.defined()
                    ? alpha.index({b, Slice(start_alpha_channel, end_alpha_channel)})
                    : at::Tensor();

            auto start_weight_group = start_channel / in_channels_per_group;
            auto end_weight_group = (end_channel + in_channels_per_group - 1) / in_channels_per_group;

            deformable_im2col(
                    sliced_input,
                    sliced_offset,
                    sliced_alpha,
                    batch_channels,
                    in_d, in_h, in_w,
                    weight_d, weight_h, weight_w,
                    pad_d, pad_h, pad_w,
                    stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w,
                    out_d, out_h, out_w,
                    1,
                    n_weight_groups_per_channel_batch,
                    n_offset_groups_per_channel_batch,
                    columns
            );

            for (int weight_group = start_weight_group; weight_group < end_weight_group; weight_group++) {
                auto out_channels_start = out_channels_per_group * weight_group;
                auto out_channels_end = out_channels_start + out_channels_per_group;

                auto column_start = 0;
                auto column_end = kernel_size * batch_channels;
                auto weight_start = start_channel % in_channels_per_group;
                auto weight_end = weight_start + batch_channels;

                auto sliced_out = out.index({b, Slice(out_channels_start, out_channels_end)}).flatten(1);
                auto sliced_columns = columns.index({Slice(column_start, column_end)});
                auto sliced_weight = weight.index({
                    Slice(out_channels_start, out_channels_end), Slice(weight_start, weight_end)
                });

                sliced_out.addmm_(sliced_weight.flatten(1), sliced_columns);
            }
        }
    }
}

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
) {
    SANITY_CHECK_DCONV3D(input, offset, alpha, weight, bias)
    SANITY_CHECK_DCONV3D_GROUPS(input, offset, alpha, weight, n_weight_groups, n_offset_groups)

    at::DeviceGuard guard(input.device());

    int batch_size = input.size(0);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    int in_channels = input.size(1);
    int out_channels = weight.size(0);

    int in_channels_per_group = in_channels / n_weight_groups;
    int out_channels_per_group = out_channels / n_weight_groups;

    int weight_d = weight.size(2);
    int weight_h = weight.size(3);
    int weight_w = weight.size(4);

    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int pad_d = std::get<0>(pad);
    int pad_h = std::get<1>(pad);
    int pad_w = std::get<2>(pad);

    int dilation_d = std::get<0>(dilation);
    int dilation_h = std::get<1>(dilation);
    int dilation_w = std::get<2>(dilation);

    int out_d = get_output_size(in_d, weight_d, stride_d, pad_d, dilation_d);
    int out_h = get_output_size(in_h, weight_h, stride_h, pad_h, dilation_h);
    int out_w = get_output_size(in_w, weight_w, stride_w, pad_w, dilation_w);

    CHECK_POSITIVE_3D(weight_d, weight_h, weight_w)
    CHECK_POSITIVE_3D(stride_d, stride_h, stride_w)
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0)
    CHECK_POSITIVE_3D(dilation_d, dilation_h, dilation_w)
    CHECK_POSITIVE_3D(out_d, out_h, out_w)

    TORCH_CHECK(
            offset.size(2) == out_d && offset.size(3) == out_h && offset.size(4) == out_w,
            "offset output dims do not match computed dims. Got: ",
            "(", offset.size(2), ", ", offset.size(3), ", ", offset.size(4), ")",
            " but expected: (", out_d, ", ", out_h, ", ", out_w, ")"
    );
    if(alpha.defined()) {
        TORCH_CHECK(
                alpha.size(2) == out_d && alpha.size(3) == out_h && alpha.size(4) == out_w,
                "mask output dims do not match computed dims. Got: ",
                "(", alpha.size(2), ", ", alpha.size(3), ", ", alpha.size(4), ")",
                " but expected: (", out_d, ", ", out_h, ", ", out_w, ")"
        );
    }

    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    auto out = torch::zeros(
            {batch_size, out_channels, out_d, out_h, out_w},
            input.options()
    );

    auto num_elements = batch_size * in_channels * kernel_size * output_size;
    if(num_elements > MAX_INTERMEDIATE_ELEMENTS) {
        compute_forward_batched_im2col(
                input, offset, alpha, weight, in_channels, out_channels,
                in_d, in_h, in_w, weight_d, weight_h, weight_w, pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                out_d, out_h, out_w, batch_size, n_weight_groups, n_offset_groups,
                out
        );
    } else {
        compute_forward_single_im2col(
                input, offset, alpha, weight, in_channels, out_channels,
                in_d, in_h, in_w, weight_d, weight_h, weight_w, pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                out_d, out_h, out_w, batch_size, n_weight_groups, n_offset_groups,
                out
        );
    }

    if(bias.defined()) {
        out.add_(bias.view({1, out_channels, 1, 1, 1}));
    }

    return out;
}

template<typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
        const int num_points,
        const Accessor<scalar_t, 3> columns,
        const Accessor<scalar_t, 5> offset,
        ///const Accessor<scalar_t, 5> alpha,
        const scalar_t* alpha,
        const int in_channels,
        const int depth,
        const int height,
        const int width,
        const int kernel_d,
        const int kernel_h,
        const int kernel_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        const int out_d,
        const int out_h,
        const int out_w,
        Accessor<scalar_t, 5> grad_input
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_z = (index / (out_h * out_w)) % out_d;
        const int b = (index / (out_d * out_h * out_w)) % batch_size;
        const int k = (index / (out_d * out_h * out_w * batch_size)) % kernel_w;
        const int j = (index / (out_d * out_h * out_w * batch_size * kernel_w)) % kernel_h;
        const int i = (index / (out_d * out_h * out_w * batch_size * kernel_w * kernel_h)) % kernel_d;
        const int c = (index / (out_d * out_h * out_w * batch_size * kernel_w * kernel_h * kernel_d));

        const int offset_group = c / channels_per_offset_group;

        const int kernel_idx = (i * kernel_h + j) * kernel_w + k;
        const int kernel_size = kernel_d * kernel_h * kernel_w;
        const int alpha_idx = offset_group * kernel_size + kernel_idx;
        //const scalar_t a = alpha.data() ? alpha[b][alpha_idx][out_z][out_y][out_x] : 1.0;
        const scalar_t a = alpha ? alpha[(((b * n_offset_groups * kernel_size + alpha_idx) * out_d  + out_z) * out_h  + out_y) * out_w + out_x] : 1.0;
        const scalar_t offset_d = offset[b][3 * alpha_idx + 0][out_z][out_y][out_x];
        const scalar_t offset_h = offset[b][3 * alpha_idx + 1][out_z][out_y][out_x];
        const scalar_t offset_w = offset[b][3 * alpha_idx + 2][out_z][out_y][out_x];

        const scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
        const scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
        const scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

        const int row_idx = c * kernel_size + kernel_idx;
        const int col_idx = (out_z * out_h + out_y) * out_w + out_x;
        const scalar_t base_grad = a * columns[b][row_idx][col_idx];

        for(int dz = 0; dz <= 1; dz++) {
            for(int dy = 0; dy <= 1; dy++) {
                for(int dx = 0; dx <= 1; dx++) {
                    int zp = floor(z) + dz;
                    int yp = floor(y) + dy;
                    int xp = floor(x) + dx;

                    if(
                        0 <= zp && zp < depth && 0 <= yp && yp < height && 0 <= xp && xp < width
                        && std::abs(z - zp) < 1 && std::abs(y - yp) < 1 && std::abs(x - xp) < 1
                    ) {
                        scalar_t weight = (1 - std::abs(z - zp))
                                        * (1 - std::abs(y - yp))
                                        * (1 - std::abs(x - xp));
                        atomicAdd(&grad_input[b][c][zp][yp][xp], weight * base_grad);
                    }
                }
            }
        }
    }
}

static void compute_grad_input(
        const at::Tensor& columns,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const int in_channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        at::Tensor& grad_input
) {
    int out_d = get_output_size(depth, weight_d, stride_d, pad_d, dilation_d);
    int out_h = get_output_size(height, weight_h, stride_h, pad_h, dilation_h);
    int out_w = get_output_size(width, weight_w, stride_w, pad_w, dilation_w);

    int num_points = in_channels
                     * weight_d * weight_h * weight_w
                     * out_d * out_h * out_w
                     * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        columns.scalar_type(), "deformable_col2im_gpu", ([&] {
            deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_points), CUDA_NUM_THREADS>>>(
                num_points,
                columns.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
                offset.packed_accessor32<scalar_t, 5, RestrictPtrTraits>(),
                //alpha.packed_accessor32<scalar_t, 5, RestrictPtrTraits>(),
                alpha.defined() ? alpha.data_ptr<scalar_t>() : nullptr,
                in_channels,
                depth, height, width,
                weight_d, weight_h, weight_w,
                pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w,
                dilation_d, dilation_h, dilation_w,
                batch_size,
                n_weight_groups,
                n_offset_groups,
                out_d, out_h, out_w,
                grad_input.packed_accessor32<scalar_t, 5, RestrictPtrTraits>()
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("error in compute_grad_input: %s\n", cudaGetErrorString(err));
    }
}

template<typename scalar_t>
__device__ void get_coordinate_weights(
        const scalar_t* im_data,
        const int depth,
        const int height,
        const int width,
        scalar_t z,
        scalar_t y,
        scalar_t x,
        scalar_t* out
) {
    int z0 = floor(z);
    int y0 = floor(y);
    int x0 = floor(x);
    int z1 = z0 + 1;
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    bool valid_z0 = 0 <= z0 && z0 < depth;
    bool valid_y0 = 0 <= y0 && y0 < height;
    bool valid_x0 = 0 <= x0 && x0 < width;
    bool valid_z1 = 0 <= z1 && z1 < depth;
    bool valid_y1 = 0 <= y1 && y1 < height;
    bool valid_x1 = 0 <= x1 && x1 < width;

    scalar_t zero = 0;
    scalar_t C000 = (valid_z0 && valid_y0 && valid_x0) ? im_data[(z0 * height + y0) * width + x0] : zero;
    scalar_t C001 = (valid_z0 && valid_y0 && valid_x1) ? im_data[(z0 * height + y0) * width + x1] : zero;
    scalar_t C010 = (valid_z0 && valid_y1 && valid_x0) ? im_data[(z0 * height + y1) * width + x0] : zero;
    scalar_t C011 = (valid_z0 && valid_y1 && valid_x1) ? im_data[(z0 * height + y1) * width + x1] : zero;
    scalar_t C100 = (valid_z1 && valid_y0 && valid_x0) ? im_data[(z1 * height + y0) * width + x0] : zero;
    scalar_t C101 = (valid_z1 && valid_y0 && valid_x1) ? im_data[(z1 * height + y0) * width + x1] : zero;
    scalar_t C110 = (valid_z1 && valid_y1 && valid_x0) ? im_data[(z1 * height + y1) * width + x0] : zero;
    scalar_t C111 = (valid_z1 && valid_y1 && valid_x1) ? im_data[(z1 * height + y1) * width + x1] : zero;

    // TODO: fix me?
    scalar_t dz = z - z0;
    scalar_t dy = y - y0;
    scalar_t dx = x - x0;

    out[0] = ((C100 - C000) * (1 - dx) + (C101 - C001) * dx) * (1 - dy)
            + ((C110 - C010) * (1 - dx) + (C111 - C011) * dx) * dy;
    out[1] = ((C010 - C000) * (1 - dz) + (C110 - C100) * dz) * (1 - dx)
            + ((C011 - C001) * (1 - dz) + (C111 - C101) * dz) * dx;
    out[2] = ((C001 - C000) * (1 - dz) + (C101 - C100) * dz) * (1 - dy)
            + ((C011 - C010) * (1 - dz) + (C111 - C110) * dz) * dy;
}

template<typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(
        const int num_points,
        const scalar_t* columns_base_ptr,
        const scalar_t* input_base_ptr,
        const scalar_t* offset_base_ptr,
        const scalar_t* alpha_base_ptr,
        const int in_channels,
        const int depth,
        const int height,
        const int width,
        const int weight_d,
        const int weight_h,
        const int weight_w,
        const int pad_d,
        const int pad_h,
        const int pad_w,
        const int stride_d,
        const int stride_h,
        const int stride_w,
        const int dilation_d,
        const int dilation_h,
        const int dilation_w,
        const int batch_size,
        const int n_weight_groups,
        const int n_offset_groups,
        const int out_d,
        const int out_h,
        const int out_w,
        scalar_t* grad_offset,
        scalar_t* grad_alpha
) {
    const int channels_per_offset_group = in_channels / n_offset_groups;
    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        scalar_t val[3] = {0, 0, 0};
        scalar_t alpha_val = 0;

        int out_x = index % out_w;
        int out_y = (index / out_w) % out_h;
        int out_z = (index / (out_h * out_w)) % out_d;
        int k = (index / (out_d * out_h * out_w)) % weight_w;
        int j = (index / (out_d * out_h * out_w * weight_w)) % weight_h;
        int i = (index / (out_d * out_h * out_w * weight_h * weight_w)) % weight_d;
        int g = (index / (out_d * out_h * out_w * weight_d * weight_h * weight_w)) % (n_offset_groups);
        int b = (index / (out_d * out_h * out_w * weight_d * weight_h * weight_w * n_offset_groups));

        const int offset_group = g;

        int c_per_offset_group = in_channels / n_offset_groups;

        auto kernel_idx = (i * weight_h + j) * weight_w + k;
        auto out_idx = (out_z * out_h + out_y) * out_w + out_x;
        auto col_ptr = columns_base_ptr
                + (b * in_channels + offset_group * c_per_offset_group) * kernel_size * output_size
                + kernel_idx * output_size
                + out_idx;
        auto input_ptr = input_base_ptr
                + (b * in_channels + offset_group * c_per_offset_group) * depth * height * width;
        auto offset_ptr = offset_base_ptr
                + 3 * (b * n_offset_groups + offset_group) * kernel_size * output_size
                + 3 * kernel_idx * output_size
                + out_idx;
        auto alpha_ptr = alpha_base_ptr
                    + (b * n_offset_groups + offset_group) * kernel_size * output_size
                    + kernel_idx * output_size
                    + out_idx;

        const scalar_t alpha = alpha_base_ptr ? *alpha_ptr : 1.0;
        const scalar_t offset_d = *offset_ptr;
        const scalar_t offset_h = *(offset_ptr + output_size);
        const scalar_t offset_w = *(offset_ptr + 2 * output_size);

        scalar_t z = (out_z * stride_d - pad_d) + i * dilation_d + offset_d;
        scalar_t y = (out_y * stride_h - pad_h) + j * dilation_h + offset_h;
        scalar_t x = (out_x * stride_w - pad_w) + k * dilation_w + offset_w;

        for(int c = 0; c < c_per_offset_group; c++) {
            const scalar_t col_value = *col_ptr;

            if(alpha_base_ptr) {
                alpha_val += col_value * trilinear_interpolate(input_ptr, depth, height, width, z, y, x);
            }

            scalar_t weights[3];
            get_coordinate_weights(
                    input_ptr,
                    depth, height, width,
                    z, y, x,
                    weights
            );
            val[0] += weights[0] * col_value;
            val[1] += weights[1] * col_value;
            val[2] += weights[2] * col_value;
            col_ptr += kernel_size * output_size;
            input_ptr += depth * height * width;
        }

        const int offset_idx = offset_ptr - offset_base_ptr;
        grad_offset[offset_idx] += alpha * val[0];
        grad_offset[offset_idx + output_size] += alpha * val[1];
        grad_offset[offset_idx + 2 * output_size] += alpha * val[2];

        if(alpha_base_ptr) {
            grad_alpha[index] += alpha_val;
        }
    }
}

static void compute_grad_offset(
    const at::Tensor& columns,
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& alpha,
    const int in_channels,
    const int depth,
    const int height,
    const int width,
    const int weight_d,
    const int weight_h,
    const int weight_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int batch_size,
    const int n_weight_groups,
    const int n_offset_groups,
    at::Tensor& grad_offset,
    at::Tensor& grad_alpha
) {
    int out_d = get_output_size(depth, weight_d, stride_d, pad_d, dilation_d);
    int out_h = get_output_size(height, weight_h, stride_h, pad_h, dilation_h);
    int out_w = get_output_size(width, weight_w, stride_w, pad_w, dilation_w);
    int num_points = out_d * out_h * out_w
                     * weight_d * weight_h * weight_w
                     * n_offset_groups
                     * batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        columns.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
            deformable_col2im_coord_gpu_kernel<<<
                GET_BLOCKS(num_points)*2,
                CUDA_NUM_THREADS/2>>>(
                num_points,
                columns.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                offset.data_ptr<scalar_t>(),
                alpha.defined() ? alpha.data_ptr<scalar_t>() : nullptr,
                in_channels,
                depth, height, width,
                weight_d, weight_h, weight_w,
                pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w,
                dilation_d, dilation_h, dilation_w,
                batch_size,
                n_weight_groups,
                n_offset_groups,
                out_d, out_h, out_w,
                grad_offset.data_ptr<scalar_t>(),
                alpha.defined() ? grad_alpha.data_ptr<scalar_t>() : nullptr
            );
        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in compute_grad_offset: %s\n", cudaGetErrorString(err));
    }
}

void compute_backward_single_im2col(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        int in_channels,
        int out_channels,
        int in_d,
        int in_h,
        int in_w,
        int weight_d,
        int weight_h,
        int weight_w,
        int pad_d,
        int pad_h,
        int pad_w,
        int stride_d,
        int stride_h,
        int stride_w,
        int dilation_d,
        int dilation_h,
        int dilation_w,
        int out_d,
        int out_h,
        int out_w,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        const at::Tensor& grad_out,
        at::Tensor& grad_input,
        at::Tensor& grad_offset,
        at::Tensor& grad_alpha,
        at::Tensor& grad_weight
) {
    const int in_channels_per_group = in_channels / n_weight_groups;
    const int out_channels_per_group = out_channels / n_weight_groups;
    const int kernel_size = weight_d * weight_h * weight_w;
    const int output_size = out_d * out_h * out_w;
    auto columns = torch::empty(
            {batch_size, in_channels * kernel_size, output_size},
            input.options()
    );

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
                    grad_out.data_ptr<scalar_t>() + grad_out_index,
                    output_size, output_size * out_channels,
                    weight.data_ptr<scalar_t>() + weight_index,
                    in_channels_per_group * kernel_size, 0,
                    scalar_t(0.0),
                    columns.data_ptr<scalar_t>() + columns_index,
                    output_size, in_channels * kernel_size * output_size,
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
            in_d, in_h, in_w,
            weight_d, weight_h, weight_w,
            pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w,
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
            in_d, in_h, in_w,
            weight_d, weight_h, weight_w,
            pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w,
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
            in_d, in_h, in_w,
            weight_d, weight_h, weight_w,
            pad_d, pad_h, pad_w,
            stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w,
            out_d, out_h, out_w,
            batch_size,
            n_weight_groups,
            n_offset_groups,
            columns
    );

    auto grad_weight_batches = torch::empty(
            {batch_size, out_channels, in_channels_per_group, weight_d, weight_h, weight_w},
            weight.options()
    );

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
                    grad_weight_batches.data_ptr<scalar_t>() + grad_weight_index,
                    in_channels_per_group * kernel_size, out_channels * in_channels_per_group * kernel_size,
                    batch_size
            );
        })
        );
    }

    grad_weight = grad_weight_batches.sum({0});
}

void compute_backward_batched_im2col(
        const at::Tensor& input,
        const at::Tensor& offset,
        const at::Tensor& alpha,
        const at::Tensor& weight,
        int in_channels,
        int out_channels,
        int in_d,
        int in_h,
        int in_w,
        int weight_d,
        int weight_h,
        int weight_w,
        int pad_d,
        int pad_h,
        int pad_w,
        int stride_d,
        int stride_h,
        int stride_w,
        int dilation_d,
        int dilation_h,
        int dilation_w,
        int out_d,
        int out_h,
        int out_w,
        int batch_size,
        int n_weight_groups,
        int n_offset_groups,
        const at::Tensor& grad_out,
        at::Tensor& grad_input,
        at::Tensor& grad_offset,
        at::Tensor& grad_alpha,
        at::Tensor& grad_weight
) {
    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    // Split into multiple passes for large input data
    auto batch_channels = get_batch_channels(
            in_channels, kernel_size, output_size,
            n_weight_groups, n_offset_groups
    );

    auto in_channels_per_group = in_channels / n_weight_groups;
    auto out_channels_per_group = out_channels / n_weight_groups;
    auto in_channels_per_offset_group = in_channels / n_offset_groups;
    auto n_weight_groups_per_channel_batch = std::max(batch_channels / in_channels_per_group, 1);
    auto n_offset_groups_per_channel_batch = std::max(batch_channels / in_channels_per_offset_group, 1);
    auto columns = torch::zeros(
            {batch_channels * kernel_size, output_size},
            input.options()
    );

    for(auto b = 0; b < batch_size; b++) {
        for (auto start_channel = 0; start_channel < in_channels; start_channel += batch_channels) {
            using namespace torch::indexing;
            auto end_channel = start_channel + batch_channels;
            auto start_alpha_channel = kernel_size
                    * (start_channel / in_channels_per_offset_group);
            auto end_alpha_channel = kernel_size
                    * (end_channel + in_channels_per_offset_group - 1) / in_channels_per_offset_group;

            auto sliced_input = input.index({Slice(b, b+1), Slice(start_channel, end_channel)});
            auto sliced_offset = offset.index({Slice(b, b+1), Slice(3 * start_alpha_channel, 3 * end_alpha_channel)});
            auto sliced_alpha = alpha.defined()
                    ? alpha.index({Slice(b, b+1), Slice(start_alpha_channel, end_alpha_channel)})
                    : at::Tensor();

            auto start_weight_group = start_channel / in_channels_per_group;
            auto end_weight_group = (end_channel + in_channels_per_group - 1) / in_channels_per_group;

            columns.zero_();

            for (int weight_group = start_weight_group; weight_group < end_weight_group; weight_group++) {
                auto out_channels_start = out_channels_per_group * weight_group;
                auto out_channels_end = out_channels_start + out_channels_per_group;

                auto column_start = 0;
                auto column_end = kernel_size * batch_channels;
                auto weight_start = start_channel % in_channels_per_group;
                auto weight_end = weight_start + batch_channels;

                auto sliced_columns = columns.index({Slice(column_start, column_end)});
                auto sliced_weight = weight.index({
                        Slice(out_channels_start, out_channels_end), Slice(weight_start, weight_end)
                }).flatten(1);
                auto sliced_grad_out = grad_out.index({b, Slice(out_channels_start, out_channels_end)}).flatten(1);
                sliced_columns.addmm_(sliced_weight.transpose(0, 1), sliced_grad_out);
            }

            auto sliced_grad_input = grad_input.index({Slice(b, b+1), Slice(start_channel, end_channel)});
            auto sliced_grad_offset = grad_offset.index({Slice(b, b+1), Slice(3 * start_alpha_channel, 3 * end_alpha_channel)});
            auto sliced_grad_alpha = alpha.defined() ?
                    grad_alpha.index({Slice(b, b+1), Slice(start_alpha_channel, end_alpha_channel)})
                    : at::Tensor();

            compute_grad_offset(
                    columns.unsqueeze(0),
                    sliced_input,
                    sliced_offset,
                    sliced_alpha,
                    batch_channels,
                    in_d, in_h, in_w,
                    weight_d, weight_h, weight_w,
                    pad_d, pad_h, pad_w,
                    stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w,
                    1,
                    n_weight_groups_per_channel_batch,
                    n_offset_groups_per_channel_batch,
                    sliced_grad_offset,
                    sliced_grad_alpha
            );

            compute_grad_input(
                    columns.unsqueeze(0),
                    sliced_offset,
                    sliced_alpha,
                    batch_channels,
                    in_d, in_h, in_w,
                    weight_d, weight_h, weight_w,
                    pad_d, pad_h, pad_w,
                    stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w,
                    1,
                    n_weight_groups_per_channel_batch,
                    n_offset_groups_per_channel_batch,
                    sliced_grad_input
            );

            // for grad_weight; overwrites columns
            deformable_im2col(
                    sliced_input,
                    sliced_offset,
                    sliced_alpha,
                    batch_channels,
                    in_d, in_h, in_w,
                    weight_d, weight_h, weight_w,
                    pad_d, pad_h, pad_w,
                    stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w,
                    out_d, out_h, out_w,
                    1,
                    n_weight_groups_per_channel_batch,
                    n_offset_groups_per_channel_batch,
                    columns
            );

            for (int weight_group = start_weight_group; weight_group < end_weight_group; weight_group++) {
                auto out_channels_start = out_channels_per_group * weight_group;
                auto out_channels_end = out_channels_start + out_channels_per_group;

                auto column_start = 0;
                auto column_end = kernel_size * batch_channels;
                auto weight_start = start_channel % in_channels_per_group;
                auto weight_end = weight_start + batch_channels;

                auto sliced_grad_weight = grad_weight.index({
                    Slice(out_channels_start, out_channels_end), Slice(weight_start, weight_end)
                }).flatten(1);
                auto sliced_columns = columns.index({Slice(column_start, column_end)});
                auto sliced_grad_out = grad_out.index({
                    b, Slice(out_channels_start, out_channels_end)
                }).flatten(1);

                sliced_grad_weight.addmm_(sliced_grad_out, sliced_columns.transpose(0, 1));
            }
        }
    }
}

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
        const int n_weight_groups,
        const int n_offset_groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);

    const int in_d = input.size(2);
    const int in_h = input.size(3);
    const int in_w = input.size(4);

    const int out_d = grad_out.size(2);
    const int out_h = grad_out.size(3);
    const int out_w = grad_out.size(4);

    const int weight_d = weight.size(2);
    const int weight_h = weight.size(3);
    const int weight_w = weight.size(4);

    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    int pad_d = std::get<0>(pad);
    int pad_h = std::get<1>(pad);
    int pad_w = std::get<2>(pad);

    int dilation_d = std::get<0>(dilation);
    int dilation_h = std::get<1>(dilation);
    int dilation_w = std::get<2>(dilation);

    auto kernel_size = weight_d * weight_h * weight_w;
    auto output_size = out_d * out_h * out_w;
    auto grad_input = torch::zeros_like(input);
    auto grad_offset = torch::zeros_like(offset);
    auto grad_alpha = alpha.defined() ? torch::zeros_like(alpha) : at::Tensor();
    auto grad_weight = torch::zeros_like(weight);

    auto num_elements = batch_size * in_channels * kernel_size * output_size;
    if(num_elements > MAX_INTERMEDIATE_ELEMENTS) {
        compute_backward_batched_im2col(
                input, offset, alpha, weight, in_channels, out_channels,
                in_d, in_h, in_w, weight_d, weight_h, weight_w, pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                out_d, out_h, out_w, batch_size, n_weight_groups, n_offset_groups,
                grad_out, grad_input, grad_offset, grad_alpha, grad_weight
        );
    } else {
        compute_backward_single_im2col(
                input, offset, alpha, weight, in_channels, out_channels,
                in_d, in_h, in_w, weight_d, weight_h, weight_w, pad_d, pad_h, pad_w,
                stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w,
                out_d, out_h, out_w, batch_size, n_weight_groups, n_offset_groups,
                grad_out, grad_input, grad_offset, grad_alpha, grad_weight
        );
    }

    auto grad_bias = bias.defined() ? grad_out.sum({0, 2, 3, 4}) : at::Tensor();

    return std::make_tuple(grad_input, grad_offset, grad_alpha, grad_weight, grad_bias);
}
