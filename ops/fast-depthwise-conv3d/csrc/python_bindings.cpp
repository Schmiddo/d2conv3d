#include "grouped_conv3d.h"
#include <torch/torch.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_separable_conv3d", &depthwise_conv3d, "Depthwise Conv3d");
}
