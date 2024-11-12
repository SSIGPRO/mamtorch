#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#define BSM 64 // block size along M
#define BSN BSM // block size along N
#define BSK 64 // block size along K
#define WPTM 4 // work per thread along M
#define WPTN WPTM // work per thread along N
#define RBSM (BSM/WPTM) // reduced block_size along M
#define RBSN (BSN/WPTN) // reduced block_size along N
#define LPTA ((BSK*BSM)/(RBSM*RBSN)) // loads-per-thread from global memory A
#define LPTB ((BSK*BSN)/(RBSM*RBSN)) // loads-per-thread from global memory B
#define LPTM BSM/RBSM
#define LPTK BSK/RBSN

/* OPTIMIZATION NOTES 
* - prefetch reduces performance due to the reduction of active thread for
*   each processor (avoided)
* - vectorization of data has not been tested
* - the use of transposition and padding introduce negligible delay
*/

namespace mamtorch_kernel_v4 {

__global__ void fullyconnected_cuda_kernel(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    float * __restrict__ C,
    int * __restrict__ Cargmax,
    int * __restrict__ Cargmin,
    int M,
    int K,
    int N);

std::vector<at::Tensor> fullyconnected_cuda(
    at::Tensor A,
    at::Tensor B)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;

    // cuda matrices (A and B are swapped)
    auto Acuda = BTcm;
    auto Bcuda = ATcm;
    
    const auto M = Acuda.size(1);
    const auto K = Acuda.size(0);
    const auto N = Bcuda.size(0);
    
    auto BT = Bcuda.transpose(0,1).contiguous();

    // declare padded tensors
    at::Tensor A_padded = Acuda;
    at::Tensor BT_padded = BT;
    
    // evaluate padding to have matrix size multiple of BSM, BN, BSK
    int M_rest = M%BSM;
    int N_rest = N%BSN;
    int K_rest = K%BSK;
    int M_padding = 0;
    int N_padding = 0;
    int K_padding = 0;
    int M_padded = M;
    int N_padded = N;
    int K_padded = K;
    if(M_rest)
    {
        M_padding = BSM - M_rest;
        M_padded = M + M_padding;
    }
    if(N_rest)
    {
        N_padding = BSN - N_rest;
        N_padded = N + N_padding;
    }
    if(K_rest)
    {
        K_padding = BSK - K_rest;
        K_padded = K + K_padding;
    }
    
    // pad matrix A
    if(M_rest || K_rest)
    {
        A_padded = at::pad(Acuda.unsqueeze(0),
                           at::IntList{0, M_padding, 0, K_padding},
                           "replicate").squeeze();
    }
    
    // pad matrix BT
    if(N_rest || K_rest)
    {
        BT_padded = at::pad(BT.unsqueeze(0),
                            at::IntList{0, N_padding, 0, K_padding},
                            "replicate").squeeze();
    }
    
    // generate padded output matrix
    if(M_rest || N_rest)
    {
        
    }   
    auto C_padded = at::zeros({A.size(0), B.size(1)}, A.options());
    auto Cargmax_padded = at::zeros({A.size(0), B.size(1)}, A.options());
    auto Cargmin_padded = at::zeros({A.size(0), B.size(1)}, A.options());
    Cargmax_padded = Cargmax_padded.to(torch::kInt32);
    Cargmin_padded = Cargmin_padded.to(torch::kInt32);
    
    const dim3 threads(RBSM,
                       RBSN,
                       1);    
    const dim3 blocks(M_padded/BSM,
                      N_padded/BSN,
                      1);
    
    fullyconnected_cuda_kernel<<<blocks, threads>>>(
        A_padded.data_ptr<float>(),
        BT_padded.data_ptr<float>(),
        C_padded.data_ptr<float>(),
        Cargmax_padded.data_ptr<int>(),
        Cargmin_padded.data_ptr<int>(),
        M_padded, K_padded, N_padded);

    if(M_rest || N_rest)
    {
        C_padded = C_padded.slice(0, 0, N).slice(1, 0, M);
        Cargmax_padded = Cargmax_padded.slice(0, 0, N).slice(1, 0, M);
        Cargmin_padded = Cargmin_padded.slice(0, 0, N).slice(1, 0, M);
    }

    // transposed column-major to row-major -> identity
    auto C = C_padded;
    auto Cargmax = torch::clamp(Cargmax_padded, 0, K-1);
    auto Cargmin = torch::clamp(Cargmin_padded, 0, K-1);
    // NOTE: clamping is fundamental for the approximated computing kernel
    // when the maximum/minimum value is the last one, since padding is
    // performed with "replicate" option

    return {C, Cargmax, Cargmin};
}

} // end namespace mamtorch_kernel