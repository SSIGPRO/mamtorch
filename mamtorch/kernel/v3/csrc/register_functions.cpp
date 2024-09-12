#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v3 {

std::vector<at::Tensor> fullyconnected(
    at::Tensor A,
    at::Tensor B,
    at::Tensor bias,
    double beta);

std::vector<at::Tensor> fullyconnected_backward(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    double beta);

at::Tensor dense(
    at::Tensor A,
    at::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(mamtorch_kernel_v3, m) {
    m.def("fullyconnected(Tensor a, Tensor b, Tensor bias, float beta) -> Tensor[]");
    m.def("fullyconnected_backward(Tensor a, Tensor b, Tensor grad, Tensor argmax, Tensor argmin, float beta) -> Tensor[]");
    m.def("dense(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mamtorch_kernel_v3, CPU, m) {
    m.impl("fullyconnected", &fullyconnected);
    m.impl("fullyconnected_backward", &fullyconnected_backward);
    m.impl("dense", &dense);
}

} // end namespace mamtorch
