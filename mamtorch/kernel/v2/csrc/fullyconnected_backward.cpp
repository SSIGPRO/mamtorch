#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v2 {

std::vector<at::Tensor> fullyconnected_backward(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    double beta)
{        
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto BTcm = B;
    const auto CgradTcm = Cgrad;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;
    // generate output matrix
    auto AgradTcm = at::empty({A.size(0), A.size(1)}, A.options());
    auto BgradTcm = at::empty({B.size(0), B.size(1)}, A.options());
    
    // implementation...
    // implementation...
    std::cout << "Warning: cpu implementation of fullyconnected_backward is not available" << std::endl;
    
    // column-major to row-major + transpose
    auto Agrad = AgradTcm;
    auto Bgrad = BgradTcm;
    auto biasgrad = Cgrad;
    
    return {Agrad, Bgrad};
}

} // end namespace mamtorch