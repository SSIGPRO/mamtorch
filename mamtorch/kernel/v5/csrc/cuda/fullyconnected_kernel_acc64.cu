#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>
#include <limits>

#define ACC 64 // accumulation block sizes

// Macro to concatenate tokens
#define CONCAT_2_EXPAND(A, B) A ## B
#define CONCAT_2(A, B) CONCAT_2_EXPAND(A, B)

// Macro to generate function names
#define FUNC_(NUM) CONCAT_2(fullyconnected_cuda_kernel_acc, NUM)

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

namespace mamtorch_kernel_v5 {

__global__ void FUNC_(ACC)(    
    const float * __restrict__ A,
    const float * __restrict__ BT,
    float * __restrict__ C,
    int * __restrict__ Cargmax,
    int * __restrict__ Cargmin,
    int M,
    int K,
    int N)
{   
    union floatint_t
    {
        float s;
        int32_t i;
        int16_t ih[2];
    };

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
    __shared__ float Ablock[BSK][BSM];
    __shared__ float Bblock[BSN][BSK+2];
    
    // declare and initialize accumulators with the first value
    float Areg;
    float Breg[WPTN];
    union floatint_t acc[WPTM][WPTN];
    union floatint_t accmax[WPTM][WPTN];
    union floatint_t accmin[WPTM][WPTN];
    
    for(int wi = 0; wi < WPTM; ++wi)
    {
        for(int wj = 0; wj < WPTN; ++wj)
        {
            acc[wi][wj].s = 0.0f;
            accmax[wi][wj].i = 0xff7fffff;//std::numeric_limits<float>::min();
            accmin[wi][wj].i = 0x7f7fffff;//std::numeric_limits<float>::max();
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
        const int num_accblocks = BSK/ACC;
        for(int kab = 0; kab < num_accblocks; ++kab)
        {
            int arg = num_accblocks*bk+kab; // new arg

            for(int k = 0; k < ACC; ++k)
            {
                int ktot = kab*ACC+k;
                // cache the values of Bblock in registers
                for(int wj = 0; wj < WPTN; ++wj)
                {
                    // register group offset + position in the register group
                    int j_block = wj*RBSN + j_reg;
                    Breg[wj] = Bblock[j_block][ktot];
                }
                
                // perform MAC operation
                for(int wi = 0; wi < WPTM; ++wi)
                {               
                    // register group offset + position in the register group
                    int i_block =  wi*RBSM + i_reg;
                    Areg = Ablock[ktot][i_block];
                    
                    for(int wj = 0; wj < WPTN; ++wj) 
                    {
                        acc[wi][wj].s += Areg * Breg[wj]; // actual MAC
                    }
                }
            }

            // get max/min of the accumulated values
            for(int wi = 0; wi < WPTM; ++wi)
            {                               
                for(int wj = 0; wj < WPTN; ++wj)
                {                    
                    // get current values
                    acc[wi][wj].ih[0] = arg; // new arg

                    accmax[wi][wj].s = max(acc[wi][wj].s, accmax[wi][wj].s);
                    accmin[wi][wj].s = min(acc[wi][wj].s, accmin[wi][wj].s);
                    // NOTE: when input value is close to the acc value, big error in 
                    // the evaluation of argmax or argmin might occur.
                    // When using padding with "replicate" option, this results in
                    // memory illegal accesses during backprop.
                    // SOLUTION: saturate argmax argmin values outside of the kernel
                    acc[wi][wj].s = 0.0f;
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
            
            Cargmax[j_out*M + i_out] = accmax[wi][wj].ih[0];
            Cargmin[j_out*M + i_out] = accmin[wi][wj].ih[0];
        }
    }

    for(int wi = 0; wi < WPTM; ++wi)
    {
        for(int wj = 0; wj < WPTN; ++wj)
        {
            accmax[wi][wj].s += accmin[wi][wj].s;
        }
    }

    for(int wi = 0; wi < WPTM; ++wi)
    {
        // tile offset + register group offset + position in the register group
        const int i_out = i_tile_off + wi*RBSM + i_reg;
        for(int wj = 0; wj < WPTN; ++wj)
        {
            // tile off. + register group off. + position in the register group
            const int j_out = j_tile_off + wj*RBSN + j_reg;
            
            C[j_out*M + i_out] = accmax[wi][wj].s;
        }
    }
}

} // end namespace mamtorch_kernel