#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v3 {

std::vector<at::Tensor> fullyconnected_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor bias,
    double beta);

at::Tensor fullyconnected_fast_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor bias,
    double beta);

std::vector<at::Tensor> fullyconnected_backward_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    double beta);

std::vector<at::Tensor> selection_count_cuda(
    at::Tensor A,
    at::Tensor Cargmax,
    at::Tensor Cargmin);

TORCH_LIBRARY_IMPL(mamtorch_kernel_v3, CUDA, m) {
    m.impl("fullyconnected", &fullyconnected_cuda);
    m.impl("fullyconnected_fast", &fullyconnected_fast_cuda);
    m.impl("fullyconnected_backward", &fullyconnected_backward_cuda);
    m.impl("selection_count", &selection_count_cuda);
}

} // end namespace mamtorch