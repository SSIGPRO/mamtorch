#include <torch/extension.h>

#include <vector>
#include <string>

/***
* IMPLEMENTATION NOTE
* computational time due to permutation of all inputs and outputs is not
* negligible.
* in order to improve speed, it might be useful try a row-major
* implementation of the conv function.
***/

void mamconv2d_forward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Y,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    int hstride,
    int wstride);

void mamconv2d_backward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Ygrad,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    torch::Tensor Xgrad,
    torch::Tensor Wgrad,
    int hstride,
    int wstride);

std::vector<torch::Tensor> mamconv2d_forward(
    torch::Tensor X,
    torch::Tensor W,
    int hstride,
    int wstride)
{    
    // Check stride
    TORCH_CHECK(hstride > 0, "W stride value must be positive!");
    TORCH_CHECK(wstride > 0, "H stride value must be positive!");
    
    // Check devices
    torch::Device Xdevice = X.device();
    torch::Device Wdevice = W.device();
    TORCH_CHECK(Xdevice == Wdevice, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Xdevice.str() + " and "
                                    + Wdevice.str() + "!");
    
    // Check sizes
    TORCH_CHECK(X.size(1) == W.size(1), "X.size(1) (channels) is not "
                                        "the same as W.size(1) (channels).");
    TORCH_CHECK(X.size(2) >= W.size(2), "X.size(2) (input h size) must be equal "
                                        "or larger than W.size(2) "
                                        "(filter h size).");
    TORCH_CHECK(X.size(3) >= W.size(3), "X.size(3) (input w size) must be equal "
                                        "or larger than W.size(3) "
                                        "(filter w size).");
    
    // Check contiguity
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous.");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous.");
    
    // Evaluate output size
    const int output_h_size = (X.size(2)-W.size(2))/hstride+1;
    const int output_w_size = (X.size(3)-W.size(3))/wstride+1;
    
    // row-major to column-major
    const auto X_cm = X.permute({3,2,1,0}).contiguous();
    // row-major to column-major
    const auto W_cm = W.permute({3,2,1,0}).contiguous();
    // generate output matrix
    auto Y_cm = torch::empty({output_w_size, output_h_size, W.size(0), X.size(0)}, X.options());
    auto Yargmax_cm = torch::empty({output_w_size, output_h_size, W.size(0), X.size(0)}, X.options());
    Yargmax_cm = Yargmax_cm.to(torch::kInt32);
    auto Yargmin_cm = torch::empty({output_w_size, output_h_size, W.size(0), X.size(0)}, X.options());
    Yargmin_cm = Yargmin_cm.to(torch::kInt32);
    
    if(X.is_cuda())
    {
        mamconv2d_forward_cuda(X_cm,
                               W_cm,
                               Y_cm,
                               Yargmax_cm,
                               Yargmin_cm,
                               hstride,
                               wstride); 
    }
    else
    {
        TORCH_CHECK(0, "CPU implementation of mamconv2d_forward"
                       "has not been implemented.")
    }
    
    // column-major to row-major
    auto Y = Y_cm.permute({3,2,1,0}).contiguous();
    auto Yargmax = Yargmax_cm.permute({3,2,1,0}).contiguous();
    auto Yargmin = Yargmin_cm.permute({3,2,1,0}).contiguous();
    
    return {Y, Yargmax, Yargmin};
}

std::vector<torch::Tensor> mamconv2d_backward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Ygrad,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    int hstride,
    int wstride)
{    
    // Check stride
    TORCH_CHECK(hstride > 0, "W stride value must be positive!");
    TORCH_CHECK(wstride > 0, "H stride value must be positive!");
    
    // Check devices
    torch::Device Xdevice = X.device();
    torch::Device Wdevice = W.device();
    torch::Device Ygrad_device = Ygrad.device();
    torch::Device Yargmax_device = Yargmax.device();
    torch::Device Yargmin_device = Yargmin.device();
    TORCH_CHECK(Xdevice == Wdevice, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Xdevice.str() + " and "
                                    + Wdevice.str() + "!");
    TORCH_CHECK(Xdevice == Ygrad_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Xdevice.str() + " and "
                                    + Ygrad_device.str() + "!");
    TORCH_CHECK(Xdevice == Yargmax_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Xdevice.str() + " and "
                                    + Yargmax_device.str() + "!");
    TORCH_CHECK(Xdevice == Yargmin_device, "Expected all tensors to be on the same "
                                    "device, but found at least two devices, "
                                    + Xdevice.str() + " and "
                                    + Yargmin_device.str() + "!");
    
    // Check sizes
    TORCH_CHECK(X.size(1) == W.size(1), "X.size(1) (channels) is not "
                                        "the same as W.size(1) (channels).");
    TORCH_CHECK(X.size(2) >= W.size(2), "X.size(2) (input h size) must be equal "
                                        "or larger than W.size(2) "
                                        "(filter h size).");
    TORCH_CHECK(X.size(3) >= W.size(3), "X.size(3) (input w size) must be equal "
                                        "or larger than W.size(3) "
                                        "(filter w size).");
    TORCH_CHECK(Ygrad.size(0) == Yargmax.size(0), "Ygrad.size(0) is not "
                                                  "the same as Yargmax.size(0).");
    TORCH_CHECK(Ygrad.size(1) == Yargmax.size(1), "Ygrad.size(1) is not "
                                                  "the same as Yargmax.size(1).");
    TORCH_CHECK(Ygrad.size(0) == Yargmin.size(0), "Ygrad.size(0) is not "
                                                  "the same as Yargmin.size(0).");
    TORCH_CHECK(Ygrad.size(1) == Yargmin.size(1), "Ygrad.size(1) is not "
                                                  "the same as Yargmin.size(1).");
    
    // Evaluate output size
    const int output_h_size = (X.size(2)-W.size(2))/hstride+1;
    const int output_w_size = (X.size(3)-W.size(3))/wstride+1;
    
    // Check output sizes
    TORCH_CHECK(Ygrad.size(0) == X.size(0), "Ygrad.size(0) is not "
                                            "the same as X.size(0).");
    TORCH_CHECK(Ygrad.size(1) == W.size(0), "Ygrad.size(1) is not "
                                            "the same as W.size(0).");
    TORCH_CHECK(Ygrad.size(2) == output_h_size, "Ygrad.size(2) is not correct, "
                                                "given the output_h_size "
                                                + std::to_string(output_h_size) +
                                                "evaluated from X.size(2) and "
                                                "W.size(2).");
    TORCH_CHECK(Ygrad.size(3) == output_w_size, "Ygrad.size(3) is not correct, "
                                                "given the output_w_size "
                                                + std::to_string(output_w_size) +
                                                "evaluated from X.size(3) and "
                                                "W.size(3).");
    
    // Check contiguity
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous.");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous.");
    TORCH_CHECK(Ygrad.is_contiguous(), "Ygrad must be contiguous.");
    TORCH_CHECK(Yargmax.is_contiguous(), "Yargmax must be contiguous.");
    TORCH_CHECK(Yargmin.is_contiguous(), "Yargmin must be contiguous.");
    
    // row-major to column-major
    const auto X_cm = X.permute({3,2,1,0}).contiguous();
    const auto W_cm = W.permute({3,2,1,0}).contiguous();
    const auto Ygrad_cm = Ygrad.permute({3,2,1,0}).contiguous();
    const auto Yargmax_cm = Yargmax.permute({3,2,1,0}).contiguous();
    const auto Yargmin_cm = Yargmin.permute({3,2,1,0}).contiguous();
    
    // generate output matrix
    // THESE NEED TO BE GENERATED FROM THE INPUT SIZE
    auto Xgrad_cm = torch::empty_like(X_cm);
    auto Wgrad_cm = torch::empty_like(W_cm);
    
    if(X.is_cuda())
    {
        mamconv2d_backward_cuda(X_cm,
                                W_cm,
                                Ygrad_cm,
                                Yargmax_cm,
                                Yargmin_cm,
                                Xgrad_cm,
                                Wgrad_cm,
                                hstride,
                                wstride);
    }
    else
    {
        TORCH_CHECK(0, "CPU implementation of mamconv2d_backward"
                       "has not been implemented.")
    }
    
    // column-major to row-major
    auto Xgrad = Xgrad_cm.permute({3,2,1,0}).contiguous();
    auto Wgrad = Wgrad_cm.permute({3,2,1,0}).contiguous();
    
    return {Xgrad, Wgrad};
}