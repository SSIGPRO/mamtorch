#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>
#include <limits>

#define BSM 32 // block size along M
#define BSN BSM // block size along N
#define BSK 32 // block size along K
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

__global__ void fullyconnected_backward_argAlike_cuda_kernel(    
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ Aargmax,
    const int * __restrict__ Aargmin,
    float * __restrict__ C,
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
    __shared__ float Ablock[BSK][BSM];
    __shared__ int Aargmax_block[BSK][BSM+1];
    __shared__ int Aargmin_block[BSK][BSM+2];
    __shared__ float Bblock[BSN][BSK+3];
    
    // declare and initialize accumulators with the first value
    float Areg;
    float Breg[WPTN];
    int Aargmax_reg;
    int Aargmin_reg;
    float acc[WPTM][WPTN];
    
    for(int wi = 0; wi < WPTM; ++wi)
    {
        for(int wj = 0; wj < WPTN; ++wj)
        {
            acc[wi][wj] = 0.0f;
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
            Aargmax_block[j_tile][i_tile] = Aargmax[j_A*M + i_A];
            Aargmin_block[j_tile][i_tile] = Aargmin[j_A*M + i_A];
            Bblock[i_tile][j_tile] = BT[j_BT*N + i_BT];
        }
        
        __syncthreads();
            
        int arg_offset = j_tile_off + j_reg; 
        
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
                Aargmax_reg = Aargmax_block[k][i_block];
                Aargmin_reg = Aargmin_block[k][i_block];
                
                for(int wj = 0; wj < WPTN; ++wj)
                {
                    int arg = arg_offset + wj*RBSN;
                    // get weighted inputs, check if max or min and substitute in the accumulators                
                    if(Aargmax_reg == arg) acc[wi][wj] += Areg*Breg[wj];
                    if(Aargmin_reg == arg) acc[wi][wj] += Areg*Breg[wj];
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

__global__ void fullyconnected_backward_argBTlike_cuda_kernel(    
    const float * __restrict__ A,
    const float * __restrict__ BT,
    const int * __restrict__ BTargmax,
    const int * __restrict__ BTargmin,
    float * __restrict__ C,
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
    __shared__ float Ablock[BSK][BSM];
    __shared__ int Bargmax_block[BSK][BSM+1];
    __shared__ int Bargmin_block[BSK][BSM+2];
    __shared__ float Bblock[BSN][BSK+3];
    
    // declare and initialize accumulators with the first value
    float Areg;
    float Breg[WPTN];
    int Bargmax_reg[WPTN];
    int Bargmin_reg[WPTN];
    float acc[WPTM][WPTN];
    
    for(int wi = 0; wi < WPTM; ++wi)
    {
        for(int wj = 0; wj < WPTN; ++wj)
        {
            acc[wi][wj] = 0.0f;
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
            Bargmax_block[i_tile][j_tile] = BTargmax[j_BT*N + i_BT];
            Bargmax_block[i_tile][j_tile] = BTargmin[j_BT*N + i_BT];
        }
        
        __syncthreads();
            
        int arg_offset = i_tile_off + i_reg;

        // evaluate partial result
        for(int k = 0; k < BSK; ++k)
        {
            // cache the values of Bblock in registers
            for(int wj = 0; wj < WPTN; ++wj)
            {
                // register group offset + position in the register group
                int j_block = wj*RBSN + j_reg;
                Breg[wj] = Bblock[j_block][k];
                Bargmax_reg[wj] = Bargmax_block[j_block][k];
                Bargmin_reg[wj] = Bargmin_block[j_block][k];
            }
            
            // perform operation
            for(int wi = 0; wi < WPTM; ++wi)
            {               
                // register group offset + position in the register group
                int i_block =  wi*RBSM + i_reg;
                Areg = Ablock[k][i_block];

                int arg = arg_offset + wi*RBSM;
                
                for(int wj = 0; wj < WPTN; ++wj)
                {
                    // get weighted inputs, check if max or min and substitute in the accumulators                
                    if(Bargmax_reg[wj] == arg) acc[wi][wj] += Areg*Breg[wj];
                    if(Bargmin_reg[wj] == arg) acc[wi][wj] += Areg*Breg[wj];
                    //acc[wi][wj] += Areg*Breg[wj];
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

} // end namespace mamtorch_kernel