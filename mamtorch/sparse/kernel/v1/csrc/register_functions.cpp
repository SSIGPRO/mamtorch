#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

namespace mamtorch_kernel_sparsev1 {

torch::Tensor unstructured_cusparse(
    torch::Tensor A,
    torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(mamtorch_kernel_sparsev1, m) {
    m.def("unstructured_cusparse(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mamtorch_kernel_sparsev1, CPU, m) {
    m.impl("unstructured_cusparse", &unstructured_cusparse);
}

} // end namespace mamtorch
