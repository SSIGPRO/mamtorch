#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_v3 {

at::Tensor dense(
    at::Tensor A,
    at::Tensor B)
{    
    auto C = at::empty({A.size(0), B.size(1)}, A.options());
    
    // implementation...
    std::cout << "Warning: cpu implementation of dense is not available" << std::endl;
    
    return C;
}

} // end namespace mamtorch