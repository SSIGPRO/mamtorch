#include <torch/extension.h>

#include <vector>

void mamdense_forward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin);

std::vector<torch::Tensor> mamdense_forward(
    torch::Tensor A,
    torch::Tensor B)
{    
    // Check devices
    torch::Device Adevice = A.device();
    torch::Device Bdevice = B.device();
    TORCH_CHECK(Adevice == Bdevice, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Adevice.str() + " and "
                                    + Bdevice.str() + "!");
    
    // Check sizes
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) is not "
                                        "the same as B.size(0).");
    
    // Check contiguity
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous.");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous.");
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;
    // generate output matrix
    auto CTcm = torch::empty({A.size(0), B.size(1)}, A.options());
    auto CargmaxTcm = torch::empty({A.size(0), B.size(1)}, A.options());
    CargmaxTcm = CargmaxTcm.to(torch::kInt32);
    auto CargminTcm = torch::empty({A.size(0), B.size(1)}, A.options());
    CargminTcm = CargminTcm.to(torch::kInt32);
    
    if(A.is_cuda())
    {
        // if A@B=C; B.T@A.T=C.T
        mamdense_forward_cuda(BTcm, ATcm, CTcm, CargmaxTcm, CargminTcm); 
    }
    else
    {
        TORCH_CHECK(0, "CPU implementation of mamdense_forward"
                       "has not been implemented.")
    }
    
    // column-major to row-major + transpose
    auto C = CTcm;
    auto Cargmax = CargmaxTcm;
    auto Cargmin = CargminTcm;
    
    return {C, Cargmax, Cargmin};
}