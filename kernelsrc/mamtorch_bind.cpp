#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> mamdense_forward(
    torch::Tensor A, 
    torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamdense_forward", &mamdense_forward, "MAMDense forward");
}