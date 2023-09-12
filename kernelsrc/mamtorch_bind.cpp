#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> mamdense_forward(
    torch::Tensor A, 
    torch::Tensor B);

std::vector<torch::Tensor> mamdense_backward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Cgrad,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamdense_forward", &mamdense_forward, "MAMDense forward");
    m.def("mamdense_backward", &mamdense_backward, "MAMDense backward");
}