#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v5 {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(mamtorch_kernel_v5, m) {
    m.def("fullyconnected(Tensor a, Tensor b, int accblock_size) -> Tensor[]");
    m.def("fullyconnected_fast(Tensor a, Tensor b) -> Tensor");
    m.def("fullyconnected_backward(Tensor a, Tensor b, Tensor grad, Tensor argmax, Tensor argmin, int accblock_size) -> Tensor[]");
}

} // end namespace mamtorch