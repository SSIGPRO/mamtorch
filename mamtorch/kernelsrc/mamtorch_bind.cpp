#include <torch/extension.h>

#include <vector>

/*** MAM Dense ***/

std::vector<torch::Tensor> mamdense_forward(
    torch::Tensor A, 
    torch::Tensor B);

std::vector<torch::Tensor> mamdense_backward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Cgrad,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin);

/*** MAM Conv1D ***/

std::vector<torch::Tensor> mamconv1d_forward(
    torch::Tensor X,
    torch::Tensor W,
    int stride);

std::vector<torch::Tensor> mamconv1d_backward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Ygrad,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    int stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamdense_forward", &mamdense_forward, "MAMDense forward");
    m.def("mamdense_backward", &mamdense_backward, "MAMDense backward");
    m.def("mamconv1d_forward", &mamconv1d_forward, "MAMConv1D forward");
    m.def("mamconv1d_backward", &mamconv1d_backward, "MAMConv1D backward");
}