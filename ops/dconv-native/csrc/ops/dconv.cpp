#include "dconv1d.h"
#include "dconv3d.h"

//class config {}; // dummy class

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deform_conv1d", &deform_conv1d, "DConv1d forward");
    m.def("deform_conv3d", &deform_conv3d, "DConv3d forward");
    py::module config = m.def_submodule("config");
    config.def("get_max_intermediate_elements", &get_max_intermediate_elements, "");
    config.def("set_max_intermediate_elements", &set_max_intermediate_elements, "");
}