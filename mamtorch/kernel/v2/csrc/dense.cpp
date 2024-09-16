#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v2 {

at::Tensor dense(
    at::Tensor A,
    at::Tensor B)
{    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;
    // generate output matrix
    auto CTcm = at::empty({A.size(0), B.size(1)}, A.options());
    
    // implementation...
    std::cout << "Warning: cpu implementation of dense is not available" << std::endl;
    
    // column-major to row-major + transpose
    auto C = CTcm;
    
    return C;
}

} // end namespace mamtorch