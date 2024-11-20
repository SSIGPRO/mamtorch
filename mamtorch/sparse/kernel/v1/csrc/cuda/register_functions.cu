#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_sparsev1 {

torch::Tensor cusparsemm_coo(
    at::Tensor A,
    at::Tensor B);

torch::Tensor cusparsemm_csr(
    at::Tensor A,
    at::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(mamtorch_kernel_sparsev1, m) {
    m.def("cusparsemm(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mamtorch_kernel_sparsev1, SparseCUDA, m) {
    m.impl("cusparsemm", &cusparsemm_coo);
}

TORCH_LIBRARY_IMPL(mamtorch_kernel_sparsev1, SparseCsrCUDA, m) {
    m.impl("cusparsemm", &cusparsemm_csr);
}

} // end namespace mamtorch