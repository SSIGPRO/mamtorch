#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v3 {

std::vector<at::Tensor> selection_count(
    at::Tensor A,
    at::Tensor Cargmax,
    at::Tensor Cargmin)
{        
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;
    // generate output matrix
    auto Minselection = at::empty({A.size(0), A.size(1)}, A.options());
    auto Maxselection = at::empty({A.size(0), A.size(1)}, A.options());
    
    // implementation...
    // implementation...
    std::cout << "Warning: cpu implementation of selection_count is not available" << std::endl;
    
    
    return {Maxselection, Minselection};
}

} // end namespace mamtorch