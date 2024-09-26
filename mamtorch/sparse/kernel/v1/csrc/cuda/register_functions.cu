#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_sparsev1 {

torch::Tensor unstructured_cusparse_cuda(
    at::Tensor A,
    at::Tensor B);

TORCH_LIBRARY_IMPL(mamtorch_kernel_sparsev1, SparseCUDA, m) {
    m.impl("unstructured_cusparse", &unstructured_cusparse_cuda);
}

} // end namespace mamtorch