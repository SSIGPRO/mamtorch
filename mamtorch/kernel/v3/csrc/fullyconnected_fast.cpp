#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v3 {

at::Tensor fullyconnected_fast(
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
    
    // implementation...
    std::cout << "Warning: cpu implementation of fullyconnected_fast is not available" << std::endl;
    
    // column-major to row-major + transpose
    auto C = CTcm;
    
    return C;
}

} // end namespace mamtorch