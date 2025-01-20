#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#include <iostream>

#include <stdexcept>

#define BSM 32 // block size along M
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

namespace mamtorch_kernel_v5 {

__global__ void fullyconnected_backward_cuda_kernel(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

__global__ void fullyconnected_backward_cuda_kernel_acc4(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

__global__ void fullyconnected_backward_cuda_kernel_acc8(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

__global__ void fullyconnected_backward_cuda_kernel_acc16(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

__global__ void fullyconnected_backward_cuda_kernel_acc32(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

__global__ void fullyconnected_backward_cuda_kernel_acc64(
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
    int M,
    int K,
    int N);

std::vector<at::Tensor> fullyconnected_backward_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    int64_t accblock_size)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto BTcm = B;
    const auto CgradTcm = Cgrad;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;
    // generate output matrix
    auto AgradTcm = at::empty({A.size(0), A.size(1)}, A.options());
    auto BgradTcm = at::empty({B.size(0), B.size(1)}, B.options());

    // ##########################################
    // GRADIENT OF A
    // we perform Cgrad@B^T
    /*{
        // cuda-ready matrices
        auto Acuda = CgradTcm.transpose(0,1).contiguous();  // CTcm to Ccm, stored as matrix A
        auto Aargmax_cuda = CargmaxTcm.transpose(0,1).contiguous();  // CTcm to Ccm, stored as matrix A
        auto Aargmin_cuda = CargminTcm.transpose(0,1).contiguous();  // CTcm to Ccm, stored as matrix A
        auto BTcuda = BTcm; //BTcm stored as transpose of Bcm
        
        const auto M = Acuda.size(1);
        const auto K = Acuda.size(0);
        const auto N = BTcuda.size(0);
        
        auto Bcuda = BTcuda.transpose(0,1).contiguous();

        // declare padded tensors
        at::Tensor A_padded = Acuda;
        at::Tensor Aargmax_padded = Aargmax_cuda;
        at::Tensor Aargmin_padded = Aargmin_cuda;
        at::Tensor B_padded = Bcuda;
        
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
                            "constant").squeeze();
            Aargmax_padded = at::pad(Aargmax_cuda.unsqueeze(0),
                            at::IntList{0, M_padding, 0, K_padding},
                            "constant").squeeze();
            Aargmin_padded = at::pad(Aargmin_cuda.unsqueeze(0),
                            at::IntList{0, M_padding, 0, K_padding},
                            "constant").squeeze();
        }
        
        // pad matrix B
        if(N_rest || K_rest)
        {
            B_padded = at::pad(Bcuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
        }
        
        // generate padded output matrix
        auto Agrad_padded = at::zeros({N_padded, M_padded}, A.options());
        
        const dim3 threads(RBSM,
                        RBSN,
                        1);    
        const dim3 blocks(M_padded/BSM,
                        N_padded/BSN,
                        1);
                        
        switch(accblock_size)
        {
            case 1:
                fullyconnected_backward_argAlike_cuda_kernel<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    B_padded.data_ptr<float>(), //transposed of the transposed
                    Aargmax_padded.data_ptr<int>(),
                    Aargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            default:
                throw std::invalid_argument("Invalid size for accumulation blocks");
        }

        AgradTcm.copy_(Agrad_padded.transpose(0,1).contiguous().slice(0, 0, M).slice(1, 0, N));
    }*/

    {
        // cuda-ready matrices
        auto Acuda = BTcm.transpose(0,1).contiguous(); 
        auto Bcuda = CgradTcm; 
        auto Bargmax_cuda = CargmaxTcm;  
        auto Bargmin_cuda = CargminTcm;
        
        const auto M = Acuda.size(1);
        const auto K = Acuda.size(0);
        const auto N = Bcuda.size(0);

        auto BTcuda = Bcuda.transpose(0,1).contiguous();
        auto BTargmax_cuda = Bargmax_cuda.transpose(0,1).contiguous();
        auto BTargmin_cuda = Bargmin_cuda.transpose(0,1).contiguous();

        // declare padded tensors
        at::Tensor A_padded = Acuda;
        at::Tensor BT_padded = BTcuda;
        at::Tensor BTargmax_padded = BTargmax_cuda;
        at::Tensor BTargmin_padded = BTargmin_cuda;
        
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
                            "constant").squeeze();
        }
        
        // pad matrix B
        if(N_rest || K_rest)
        {
            BT_padded = at::pad(BTcuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
            BTargmax_padded = at::pad(BTargmax_cuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
            BTargmin_padded = at::pad(BTargmin_cuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
        }
        
        // generate padded output matrix
        auto Agrad_padded = at::zeros({N_padded, M_padded}, A.options());
        
        const dim3 threads(RBSM,
                        RBSN,
                        1);    
        const dim3 blocks(M_padded/BSM,
                        N_padded/BSN,
                        1);
                        
        switch(accblock_size)
        {
            case 1:
                fullyconnected_backward_cuda_kernel<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 4:
                fullyconnected_backward_cuda_kernel_acc4<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 8:
                fullyconnected_backward_cuda_kernel_acc8<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 16:
                fullyconnected_backward_cuda_kernel_acc16<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 32:
                fullyconnected_backward_cuda_kernel_acc32<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 64:
                fullyconnected_backward_cuda_kernel_acc64<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Agrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            default:
                throw std::invalid_argument("Invalid size for accumulation blocks");
        }

        AgradTcm.copy_(Agrad_padded.slice(0, 0, N).slice(1, 0, M));
    }

    // transposed column-major to row-major -> identity
    auto Agrad = AgradTcm;

    // ##########################################
    // GRADIENT OF B
    // we perform A^T@Cgrad
    {
        // cuda-ready matrices
        auto Acuda = ATcm;  // ATcm (Acm transposed) stored as matrix A
        auto BTcuda = CgradTcm; // CTcm stored as transpose of Ccm
        auto BTargmax_cuda = CargmaxTcm; // CTcm stored as transpose of Ccm
        auto BTargmin_cuda = CargminTcm; // CTcm stored as transpose of Ccm
        
        const auto M = Acuda.size(1);
        const auto K = Acuda.size(0);
        const auto N = BTcuda.size(1); //BT already transposed

        // declare padded tensors
        at::Tensor A_padded = Acuda;
        at::Tensor BT_padded = BTcuda;
        at::Tensor BTargmax_padded = BTargmax_cuda;
        at::Tensor BTargmin_padded = BTargmin_cuda;
        
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
                            "constant").squeeze();
        }
        
        // pad matrix BT
        if(N_rest || K_rest)
        {
            BT_padded = at::pad(BTcuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
            BTargmax_padded = at::pad(BTargmax_cuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
            BTargmin_padded = at::pad(BTargmin_cuda.unsqueeze(0),
                                at::IntList{0, N_padding, 0, K_padding},
                                "constant").squeeze();
        }
        
        // generate padded output matrix
        auto Bgrad_padded = at::zeros({N_padded, M_padded}, A.options());
        
        const dim3 threads(RBSM,
                        RBSN,
                        1);    
        const dim3 blocks(M_padded/BSM,
                        N_padded/BSN,
                        1);

        switch(accblock_size)
        {
            case 1:
                fullyconnected_backward_cuda_kernel<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 4:
                fullyconnected_backward_cuda_kernel_acc4<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 8:
                fullyconnected_backward_cuda_kernel_acc8<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 16:
                fullyconnected_backward_cuda_kernel_acc16<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 32:
                fullyconnected_backward_cuda_kernel_acc32<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            case 64:
                fullyconnected_backward_cuda_kernel_acc64<<<blocks, threads>>>(
                    A_padded.data_ptr<float>(),
                    BT_padded.data_ptr<float>(),
                    BTargmax_padded.data_ptr<int>(),
                    BTargmin_padded.data_ptr<int>(),
                    Bgrad_padded.data_ptr<float>(),
                    M_padded, K_padded, N_padded);
                break;
            default:
                throw std::invalid_argument("Invalid size for accumulation blocks");
        }

        BgradTcm.copy_(Bgrad_padded.transpose(0,1).contiguous().slice(0, 0, M).slice(1, 0, N));
    }

    // transposed column-major to row-major -> identity
    //BgradTcm = at::zeros_like(B);
    auto Bgrad = BgradTcm;

    return {Agrad, Bgrad};
}

} // end namespace mamtorch_kernel