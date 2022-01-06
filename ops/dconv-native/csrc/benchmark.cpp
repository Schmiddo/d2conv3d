#include <torch/torch.h>
#include <torchvision/ops/deform_conv2d.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <iostream>
#include <chrono>

#include "ops/dconv3d.h"
#include "ops/dconv1d.h"


void time(const std::string& name, int repetitions, std::function<void(void)> f) {
    std::cout << "Timing start (" << name << "), "
        << repetitions << "x... " << std::flush;

    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < repetitions; i++){
        f();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "done. s/it: " << elapsed.count() / repetitions << std::endl;
}

std::vector<at::Tensor>
generate_data(
        int dim,
        int batch_size,
        int in_channels,
        int out_channels,
        int input_size,
        int kernel_size,
        int pad, int stride, int dilation,
        int num_weight_groups = 1,
        int num_offset_groups = 1
) {
    torch::TensorOptions opts(torch::TensorOptions(torch::kCUDA).requires_grad(true));

    int output_size = (input_size + 2 * pad - dilation * ((kernel_size - 1) + 1)) / stride + 1;
    int num_kernel_points = std::pow(kernel_size, dim);
    int num_offsets_per_point = dim * num_kernel_points;

    std::vector<long> sizes(size_t(2 + dim), input_size);
    sizes[0] = batch_size; sizes[1] = in_channels; sizes[2] = input_size;
    auto input = torch::randn(sizes, opts);

    std::fill(sizes.begin() + 2, sizes.end(), output_size);
    sizes[1] = num_offsets_per_point;
    auto offsets = torch::randn(sizes, opts);

    sizes[1] = num_kernel_points;
    auto alpha = torch::randn(sizes, opts);

    sizes[1] = out_channels;
    auto grad_out = torch::randn(sizes, opts);

    std::fill(sizes.begin() + 2, sizes.end(), kernel_size);
    sizes[0] = out_channels; sizes[1] = in_channels;
    auto weight = torch::randn(sizes, opts);
    auto bias = torch::randn({out_channels}, opts);

    return {input, offsets, alpha, weight, bias, grad_out};
}

int main(int argc, char** argv) {
    int kernel_size = 3;
    int pad = 0;
    int stride = 1;
    int dilation = 1;

    int batch_size = 8;
    int in_channels = 16;
    int out_channels = 32;
    int input_size = 800;
    int repetitions = 100;

    std::cout << "Generating data... ";
    auto tensors = generate_data(1, batch_size, in_channels, out_channels,
            input_size, kernel_size, pad, stride, dilation);
    std::cout << "Done." << std::endl;

    time("conv1d", repetitions, [&]{
        auto res = torch::conv1d(
                tensors[0],
                tensors[3],
                tensors[4],
                stride, pad, dilation, 1
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });
    time("dconv1d", repetitions, [&]{
        auto res = deform_conv1d(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            stride, pad, dilation, 1, 1
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });

    auto input_size_ = std::sqrt(input_size);
    std::cout << "Generating data... ";
    tensors = generate_data(2, batch_size, in_channels, out_channels,
            input_size_, kernel_size, pad, stride, dilation);
    std::cout << "Done." << std::endl;

    time("conv2d", repetitions, [&]{
        auto res = torch::conv2d(
                tensors[0],
                tensors[3],
                tensors[4],
                stride,
                pad,
                dilation,
                1
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });

    time("dconv2d", repetitions, [&]{
        auto res = vision::ops::deform_conv2d(
                tensors[0],
                tensors[3],
                tensors[1],
                tensors[2],
                tensors[4],
                stride, stride,
                pad, pad,
                dilation, dilation,
                1, 1, true
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[1], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });

    input_size_ = std::cbrt(input_size);
    std::cout << "Generating data... ";
    tensors = generate_data(3, batch_size, in_channels, out_channels,
            input_size_, kernel_size, pad, stride, dilation);
    std::cout << "Done." << std::endl;

    time("conv3d", repetitions, [&]{
        auto res = torch::conv3d(
                tensors[0],
                tensors[3],
                tensors[4],
                stride,
                pad,
                dilation,
                1
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });
    time("dconv3d", repetitions, [&]{
        auto res = deform_conv3d(
            tensors[0],
            tensors[1],
                {},
            tensors[3],
            tensors[4],
            stride, stride, stride,
            pad, pad, pad,
            dilation, dilation, dilation,
            1, 1
        );
        auto ignored = torch::autograd::grad(
                {res},
                {tensors[0], tensors[1], tensors[3], tensors[4]},
                {tensors[5]}
        );
        cudaDeviceSynchronize();
    });

    return 0;
}