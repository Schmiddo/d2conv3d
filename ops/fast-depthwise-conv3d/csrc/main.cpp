#include <iostream>
#include <torch/torch.h>

#include "grouped_conv3d.h"

int main(int argc, char** argv) {
    auto batch_size = 4;
    auto in_channels = 64;
    auto out_channels = 64;
    auto D = 8, H = 120, W = 216;
    auto mat = torch::randn({batch_size, in_channels, D, H, W}).cuda();
    auto weight = torch::randn({out_channels, 1, 3, 3, 3}).cuda();
    auto bias = torch::randn({out_channels}).cuda();

    auto result = depthwise_conv3d(
            mat, weight, bias,
            {1, 1, 1}, {1, 1, 1}, {1, 1, 1}
            );
    auto expected = torch::conv3d(mat, weight, bias, {1,1,1}, {1,1,1}, {1,1,1}, in_channels);

    std::cout << torch::allclose(result, expected) << (result - expected).abs().max() << std::endl;

    return 0;
}