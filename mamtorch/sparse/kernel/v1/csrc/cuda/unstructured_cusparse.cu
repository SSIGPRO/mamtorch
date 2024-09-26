#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

namespace mamtorch_kernel_sparsev1 {

torch::Tensor unstructured_cusparse_cuda(
    torch::Tensor A,
    torch::Tensor B)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    
    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);

    std::cout << A.values()[0] << std::endl;

    //torch::Tensor result = torch::sparse_mm(A, B);

    return C;
}

} // end namespace mamtorch