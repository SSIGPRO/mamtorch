#include <torch/extension.h>

#include <vector>

void mamdense_forward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin);

void mamdense_backward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Cgrad,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin,
    torch::Tensor Agrad,
    torch::Tensor Bgrad);

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
                                        "the same as B.size(0). A is "
                                        + A.size(0).str() + "x"
                                        + A.size(1).str() + ". B is: " 
                                        + B.size(0).str() + "x"
                                        + B.size(1).str());
    
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

std::vector<torch::Tensor> mamdense_backward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Cgrad,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin)
{    
    // Check devices
    torch::Device Adevice = A.device();
    torch::Device Bdevice = B.device();
    torch::Device Cgrad_device = Cgrad.device();
    torch::Device Cargmax_device = Cargmax.device();
    torch::Device Cargmin_device = Cargmin.device();
    TORCH_CHECK(Adevice == Bdevice, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Adevice.str() + " and "
                                    + Bdevice.str() + "!");
    TORCH_CHECK(Adevice == Cgrad_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Adevice.str() + " and "
                                    + Cgrad_device.str() + "!");
    TORCH_CHECK(Adevice == Cargmax_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Adevice.str() + " and "
                                    + Cargmax_device.str() + "!");
    TORCH_CHECK(Adevice == Cargmin_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Adevice.str() + " and "
                                    + Cargmin_device.str() + "!");
    
    // Check sizes
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) is not "
                                        "the same as B.size(0).");
    TORCH_CHECK(Cgrad.size(0) == A.size(0), "A.size(0) is not "
                                            "the same as Cgrad.size(0).");
    TORCH_CHECK(Cgrad.size(1) == B.size(1), "B.size(1) is not "
                                            "the same as Cgrad.size(1).");
    TORCH_CHECK(Cgrad.size(0) == Cargmax.size(0), "Cgrad.size(0) is not "
                                                  "the same as Cargmax.size(0).");
    TORCH_CHECK(Cgrad.size(1) == Cargmax.size(1), "Cgrad.size(1) is not "
                                                  "the same as Cargmax.size(1).");
    TORCH_CHECK(Cgrad.size(0) == Cargmin.size(0), "Cgrad.size(0) is not "
                                                  "the same as Cargmin.size(0).");
    TORCH_CHECK(Cgrad.size(1) == Cargmin.size(1), "Cgrad.size(1) is not "
                                                  "the same as Cargmin.size(1).");
    
    // Check contiguity
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous.");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous.");
    TORCH_CHECK(Cgrad.is_contiguous(), "Cgrad must be contiguous.");
    TORCH_CHECK(Cargmax.is_contiguous(), "Cargmax must be contiguous.");
    TORCH_CHECK(Cargmin.is_contiguous(), "Cargmin must be contiguous.");
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto BTcm = B;
    const auto CgradTcm = Cgrad;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;
    // generate output matrix
    auto AgradTcm = torch::empty({A.size(0), A.size(1)}, A.options());
    auto BgradTcm = torch::empty({B.size(0), B.size(1)}, A.options());
    
    if(A.is_cuda())
    {
        // if A@B=C; B.T@A.T=C.T
        mamdense_backward_cuda(BTcm, ATcm, CgradTcm, CargmaxTcm, CargminTcm, BgradTcm, AgradTcm); 
    }
    else
    {
        TORCH_CHECK(0, "CPU implementation of mamdense_backward"
                       "has not been implemented.")
    }
    
    // column-major to row-major + transpose
    auto Agrad = AgradTcm;
    auto Bgrad = BgradTcm;
    
    return {Agrad, Bgrad};
}