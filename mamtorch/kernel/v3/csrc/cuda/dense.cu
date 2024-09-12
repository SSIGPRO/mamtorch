#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#include "sgemm.cuh"
#include <iostream>

// Settings for A100
#define NUM_THREADS 128
#define BN 128
#define BM 64
#define BK 16
#define WN 64
#define WM 32
#define WNITER 1
#define TN 4
#define TM 4
// Settings for A6000
// #define NUM_THREADS 128
// #define BN 128
// #define BM 128
// #define BK 16
// #define WN 64
// #define WM 64
// #define WNITER 4
// #define TN 4
// #define TM 8

namespace mamtorch_kernel_v3
{

void runSgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) 
{
    dim3 blockDim(NUM_THREADS);

    constexpr uint NUM_WARPS = NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((BN % WN == 0) and (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
    constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    // warpsubtile in warptile
    static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

    static_assert((NUM_THREADS * 4) % BK == 0,
                    "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of Bs during each iteraion)");
    static_assert((NUM_THREADS * 4) % BN == 0,
                    "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of As during each iteration)");
    static_assert(BN % (16 * TN) == 0, "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(BM % (16 * TM) == 0, "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0, "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((BN * BK) % (4 * NUM_THREADS) == 0, "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

at::Tensor dense_cuda(
    at::Tensor A,
    at::Tensor B)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    // generate output matrix
    auto C = at::empty({A.size(0), B.size(1)}, A.options());
    
    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);

    // declare padded tensors
    at::Tensor A_padded = A;
    at::Tensor B_padded = B;
    at::Tensor C_padded = C;
    
    // evaluate padding to have matrix size multiple of BSM, BN, BSK
    int M_rest = M%BM;
    int N_rest = N%BN;
    int K_rest = K%BK;
    int M_padding = 0;
    int N_padding = 0;
    int K_padding = 0;
    int M_padded = M;
    int N_padded = N;
    int K_padded = K;
    if(M_rest)
    {
        M_padding = BM - M_rest;
        M_padded = M + M_padding;
    }
    if(N_rest)
    {
        N_padding = BN - N_rest;
        N_padded = N + N_padding;
    }
    if(K_rest)
    {
        K_padding = BK - K_rest;
        K_padded = K + K_padding;
    }
    
    // pad matrix A
    if(M_rest || K_rest)
    {
        A_padded = at::pad(A.unsqueeze(0),
                           at::IntList{0, K_padding, 0, M_padding},
                           "constant", 0).squeeze();
    }
    
    // pad matrix B
    if(N_rest || K_rest)
    {
        B_padded = at::pad(B.unsqueeze(0),
                           at::IntList{0, N_padding, 0, K_padding},
                           "constant", 0).squeeze();
    }
    
    // generate padded output matrix
    if(M_rest || N_rest)
    {
        C_padded = at::empty({M_padded, N_padded}, C.options());
    }

    // std::cout << "M_padding " << M_padding << std::endl;
    // std::cout << "N_padding " << N_padding << std::endl;
    // std::cout << "K_padding " << K_padding << std::endl;
    // std::cout << "M_padded " << M_padded << std::endl;
    // std::cout << "N_padded " << N_padded << std::endl;
    // std::cout << "K_padded " << K_padded << std::endl;
    // std::cout << "A " << A.size(0) << " " << A.size(1) << std::endl;
    // std::cout << "B " << B.size(0) << " " << B.size(1) << std::endl;
    // std::cout << "C " << C.size(0) << " " << C.size(1) << std::endl;
    // std::cout << "A_padded " << A_padded.size(0) << " " << A_padded.size(1) << std::endl;
    // std::cout << "B_padded " << B_padded.size(0) << " " << B_padded.size(1) << std::endl;
    // std::cout << "C_padded " << C_padded.size(0) << " " << C_padded.size(1) << std::endl;
    
    runSgemm(
        M_padded, N_padded, K_padded,
        1.0,
        A_padded.data_ptr<float>(),
        B_padded.data_ptr<float>(),
        0.0,
        C_padded.data_ptr<float>());

    if(M_rest || N_rest)
    {
        C.copy_(C_padded.slice(0, 0, M).slice(1, 0, N));
    }

    return C;
}

} // end namespace mamtorch_kernel