#include "utils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_corners", &calculate_corners, "calculate_corners");
    m.def("calculate_wnp_iter", &calculate_wnp_iter, "calculate_wnp_iter");
}
