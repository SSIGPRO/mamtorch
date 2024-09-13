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
* - BSM = 64 is approximately the best value
* - BSK = 32 is approximately the best value
* - the use of transposition and padding introduce negligible delay
*/

namespace mamtorch_kernel_v2 {

template <typename scalar_t>
__global__ void dense_cuda_kernel(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ BT,
    scalar_t * __restrict__ C,
    int M,
    int K,
    int N)
{   
    // get thread and block ids
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    
    // get row and column in each register group
    const int i_reg = ti;
    const int j_reg = tj;
    // get tile row and column offset
    const int i_tile_off = BSM*bi;
    const int j_tile_off = BSN*bj;
    
    // *** EXECUTION INITIALIZATION ***
    
    // declare shared input blocks
    __shared__ scalar_t Ablock[BSK][BSM];
    __shared__ scalar_t Bblock[BSN][BSK+2];
    
    // declare and initialize accumulators with the first value
    scalar_t Areg;
    scalar_t Breg[WPTN];
    scalar_t acc[WPTM][WPTN];
    for(int wi = 0; wi < WPTM; ++wi)
    {
        for(int wj = 0; wj < WPTN; ++wj)
        {
            acc[wi][wj] = 0.0;
        }
    }
    
    // for each different shared input blocks
    const int num_blocks = K/BSK;
    for(int bk = 0; bk < num_blocks; ++bk)
    {        
        // put data in the shared input blocks
        for(int la = 0; la < LPTA; ++la)
        {
            // evaluate linear offset of the register group
            const int index_reg_off = la*RBSM*RBSN;
            // evaluate linear position in the tile
            // = register group offset + position in the register group
            const int index_tile = index_reg_off + j_reg*RBSM + i_reg; 
            // split linear position in row and column indices
            const int i_tile = index_tile % BSM;
            const int j_tile = index_tile / BSM;
            // evaluate position in the input
            const int i_A = i_tile_off + i_tile;
            const int j_A = BSK*bk + j_tile;
            const int i_BT = j_tile_off + i_tile;
            const int j_BT = j_A;
            // load on local memory and un-trasnspose matrix BT
            Ablock[j_tile][i_tile] = A[j_A*M + i_A];
            Bblock[i_tile][j_tile] = BT[j_BT*N + i_BT];
        }
        
        __syncthreads();
            
        // evaluate partial result
        for(int k = 0; k < BSK; ++k)
        {
            // cache the values of Bblock in registers
            for(int wj = 0; wj < WPTN; ++wj)
            {
                // register group offset + position in the register group
                int j_block = wj*RBSN + j_reg;
                Breg[wj] = Bblock[j_block][k];
            }
            
            // perform operation
            for(int wi = 0; wi < WPTM; ++wi)
            {               
                // register group offset + position in the register group
                int i_block =  wi*RBSM + i_reg;
                Areg = Ablock[k][i_block];
                
                for(int wj = 0; wj < WPTN; ++wj)
                {
                    // get weighted inputs and add to the accumulator
                    scalar_t tmp = Areg * Breg[wj]; 
                    acc[wi][wj] += tmp;                
                }
            }
            
        }
        __syncthreads();
    }

    // *** STORE THE OUTPUTS ***
    
    for(int wi = 0; wi < WPTM; ++wi)
    {
        // tile offset + register group offset + position in the register group
        const int i_out = i_tile_off + wi*RBSM + i_reg;
        for(int wj = 0; wj < WPTN; ++wj)
        {
            // tile off. + register group off. + position in the register group
            const int j_out = j_tile_off + wj*RBSN + j_reg;
            
            C[j_out*M + i_out] = acc[wi][wj];
        }
    }
}

at::Tensor dense_cuda(
    at::Tensor A,
    at::Tensor B)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    // row-major to column-major + transpose
    const auto ATcm = A;
    // row-major to column-major + transpose
    const auto BTcm = B;
    // generate output matrix
    auto CTcm = at::empty({A.size(0), B.size(1)}, A.options());

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
                           "constant", 0).squeeze();
    }
    
    // pad matrix BT
    if(N_rest || K_rest)
    {
        BT_padded = at::pad(BT.unsqueeze(0),
                            at::IntList{0, N_padding, 0, K_padding},
                            "constant", 0).squeeze();
    }
    
    // generate padded output matrix
    if(M_rest || N_rest)
    {
        C_padded = at::empty({N_padded, M_padded}, CTcm.options());
    }
    
    const dim3 threads(RBSM,
                       RBSN,
                       1);    
    const dim3 blocks(M_padded/BSM,
                      N_padded/BSN,
                      1);
    
    dense_cuda_kernel<float><<<blocks, threads>>>(
        A_padded.data_ptr<float>(),
        BT_padded.data_ptr<float>(),
        C_padded.data_ptr<float>(),
        M_padded, K_padded, N_padded);

    if(M_rest || N_rest)
    {
        CTcm.copy_(C_padded.slice(0, 0, N).slice(1, 0, M));
    }

    // transposed column-major to row-major
    auto C = CTcm;

    return C;
}

} // end namespace mamtorch_kernel