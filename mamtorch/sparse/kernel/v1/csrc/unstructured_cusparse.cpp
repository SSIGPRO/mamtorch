#include <torch/extension.h>

#include <vector>

#include <iostream>

namespace mamtorch_kernel_sparsev1 {

torch::Tensor unstructured_cusparse(
    torch::Tensor A,
    torch::Tensor B)
{    
    auto C = at::empty({A.size(0), B.size(1)}, A.options());
    
    // implementation...
    std::cout << "Warning: cpu implementation of unstrucuted_cusparse is not available" << std::endl;
    
    return C;
}

} // end namespace mamtorch