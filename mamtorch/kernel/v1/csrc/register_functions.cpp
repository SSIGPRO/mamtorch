#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v1 {

std::vector<at::Tensor> fullyconnected(
    at::Tensor A, 
    at::Tensor B);

std::vector<at::Tensor> fullyconnected_backward(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(mamtorch_kernel_v1, m) {
    m.def("fullyconnected(Tensor a, Tensor b) -> Tensor[]");
    m.def("fullyconnected_backward(Tensor a, Tensor b, Tensor grad, Tensor argmax, Tensor argmin) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mamtorch_kernel_v1, CPU, m) {
    m.impl("fullyconnected", &fullyconnected);
    m.impl("fullyconnected_backward", &fullyconnected_backward);
}

} // end namespace mamtorch