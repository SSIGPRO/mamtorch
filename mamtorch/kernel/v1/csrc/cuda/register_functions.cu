#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v1 {

std::vector<at::Tensor> fullyconnected_cuda(
    at::Tensor A, 
    at::Tensor B);

std::vector<at::Tensor> fullyconnected_backward_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin);

TORCH_LIBRARY_IMPL(mamtorch_kernel_v1, CUDA, m) {
    m.impl("fullyconnected", &fullyconnected_cuda);
    m.impl("fullyconnected_backward", &fullyconnected_backward_cuda);
}

} // end namespace mamtorch