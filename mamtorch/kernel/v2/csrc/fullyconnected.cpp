#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v2 {

std::vector<at::Tensor> fullyconnected(
    at::Tensor A,
    at::Tensor B,
    at::Tensor bias,
    double beta)
{    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;
    // generate output matrix
    auto CTcm = at::empty({A.size(0), B.size(1)}, A.options());
    auto CargmaxTcm = at::empty({A.size(0), B.size(1)}, A.options());
    CargmaxTcm = CargmaxTcm.to(torch::kInt32);
    auto CargminTcm = at::empty({A.size(0), B.size(1)}, A.options());
    CargminTcm = CargminTcm.to(torch::kInt32);
    
    // implementation...
    std::cout << "Warning: cpu implementation of fullyconnected is not available" << std::endl;
    
    // column-major to row-major + transpose
    auto C = CTcm;
    auto Cargmax = CargmaxTcm;
    auto Cargmin = CargminTcm;
    
    return {C, Cargmax, Cargmin};
}

} // end namespace mamtorch