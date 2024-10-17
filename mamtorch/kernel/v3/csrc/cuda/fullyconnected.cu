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

namespace mamtorch_kernel_v3 {

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
    at::Tensor B,
    at::Tensor bias,
    double beta)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;
    // generate output matrix
    auto CTcm = at::zeros({A.size(0), B.size(1)}, A.options());
    auto CargmaxTcm = at::zeros({A.size(0), B.size(1)}, A.options());
    CargmaxTcm = CargmaxTcm.to(torch::kInt32);
    auto CargminTcm = at::zeros({A.size(0), B.size(1)}, A.options());
    CargminTcm = CargminTcm.to(torch::kInt32);

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
    at::Tensor C_padded = CTcm;
    at::Tensor Cargmax_padded = CargmaxTcm.to(torch::kInt32);
    at::Tensor Cargmin_padded = CargminTcm.to(torch::kInt32);
    
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
        C_padded = at::empty({N_padded, M_padded}, CTcm.options());
        Cargmax_padded = at::empty({N_padded, M_padded}, CargmaxTcm.options());
        Cargmin_padded = at::empty({N_padded, M_padded}, CargminTcm.options());
    }
    
    const dim3 threads(RBSM,
                       RBSN,
                       1);    
    const dim3 blocks(M_padded/BSM,
                      N_padded/BSN,
                      1);
    
    if(beta < 1)
    {
        fullyconnected_cuda_kernel<<<blocks, threads>>>(
            A_padded.data_ptr<float>(),
            BT_padded.data_ptr<float>(),
            C_padded.data_ptr<float>(),
            Cargmax_padded.data_ptr<int>(),
            Cargmin_padded.data_ptr<int>(),
            M_padded, K_padded, N_padded);

        if(M_rest || N_rest)
        {
            CTcm.copy_(C_padded.slice(0, 0, N).slice(1, 0, M));
            CargmaxTcm.copy_(Cargmax_padded.slice(0, 0, N).slice(1, 0, M));
            CargminTcm.copy_(Cargmin_padded.slice(0, 0, N).slice(1, 0, M));
        }
    }

    // transposed column-major to row-major
    auto C = CTcm;
    auto Cargmax = CargmaxTcm;
    auto Cargmin = CargminTcm;

    // perform affine combination with MAC contribution
    if(beta > 0)
    {
        C *= 1-beta;
        C += at::linalg_matmul(A, B)*beta;
    }
    // add bias
    C += bias;

    return {C, Cargmax, Cargmin};
}

} // end namespace mamtorch_kernel