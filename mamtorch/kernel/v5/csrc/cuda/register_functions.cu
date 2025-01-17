#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_v5 {

std::vector<at::Tensor> fullyconnected_cuda(
    at::Tensor A,
    at::Tensor B,
    int64_t accblock_size);

at::Tensor fullyconnected_fast_cuda(
    at::Tensor A,
    at::Tensor B);

std::vector<at::Tensor> fullyconnected_backward_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    int64_t accblock_size);

TORCH_LIBRARY_IMPL(mamtorch_kernel_v5, CUDA, m) {
    m.impl("fullyconnected", &fullyconnected_cuda);
    m.impl("fullyconnected_fast", &fullyconnected_fast_cuda);
    m.impl("fullyconnected_backward", &fullyconnected_backward_cuda);
}

} // end namespace mamtorch