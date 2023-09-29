#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#define BS 64 // block size along B or F
#define BSK 32 // block size along K
#define WPT 4 // work per thread along B or F
#define RBS (BS/WPT) // reduced block_size along B or F
#define LPT ((BS*BSK)/(RBS*RBS)) // loads-per-thread from global memory

/* OPTIMIZATION NOTES 
* - prefetch reduces performance due to the reduction of active thread for
*   each processor (avoided)
* - vectorization of data has not been tested
* - BSM = 64 is approximately the best value
* - BSK = 32 is approximately the best value
* - the use of transposition and padding introduce negligible delay
*/

template <typename scalar_t>
__global__ void mamconv1d_forward_cuda_kernel(
    const scalar_t * __restrict__ X,
    const scalar_t * __restrict__ W,
    scalar_t * __restrict__ Y,
    int * __restrict__ Yargmax,
    int * __restrict__ Yargmin,
    int B,
    int C,
    int M,
    int F,
    int Mf,
    int stride,
    int Cpad)
{    
    // get thread and block ids
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    
    const int filter_off_i = bz*stride;
    
    // *** EXECUTION INITIALIZATION ***
    
    // declare shared input blocks
    __shared__ scalar_t Xblock[BSK][BS];
    __shared__ scalar_t Wblock[BS][BSK+2];
    
    // declare and initialize accumulators with the first value
    scalar_t Xreg;
    scalar_t Wreg[WPT];
    scalar_t accmax[WPT][WPT];
    scalar_t accmin[WPT][WPT];
    int argmax[WPT][WPT];
    int argmin[WPT][WPT];
    for(int wi = 0; wi < WPT; ++wi)
    {
        for(int wj = 0; wj < WPT; ++wj)
        {
            accmax[wi][wj] = std::numeric_limits<scalar_t>::min();
            accmin[wi][wj] = std::numeric_limits<scalar_t>::max();
            argmax[wi][wj] = 0;
            argmin[wi][wj] = 0;
        }
    }
    
    // for each different shared input blocks
    const int num_blocks = Mf*C/BSK;
    for(int bk = 0; bk < num_blocks; ++bk)
    {        
        // put data in the shared input blocks
        for(int l = 0; l < LPT; ++l)
        {
            const int filter_i = bk/C;
            // find linear index inside the filter
            const int index = l*RBS*RBS + tj*RBS + ti;
            // split linear position in batch and filter indices
            const int block_i = index % BS;
            const int block_j = index / BS;
            // evaluate position in the input
            const int Xblock_i = bx*BS + block_i;
            const int Xblock_j = bk%C*BSK + block_j;
            const int Wblock_i = by*BS + block_i;
            const int Wblock_j = bk*BSK + block_j;
            // load on local memory
            Xblock[block_j][block_i] = X[Xblock_i + Xblock_j*B
                                       + (filter_off_i+filter_i)*B*C];
            Wblock[block_i][block_j] = W[Wblock_i + Wblock_j*F];
        }
        
        __syncthreads();
            
        // evaluate partial result
        for(int k = 0; k < BSK; ++k)
        {
            // cache the values of Bblock in registers
            for(int wj = 0; wj < WPT; ++wj)
            {
                // register group offset + position in the register group
                int j_block = wj*RBS + tj;
                Wreg[wj] = Wblock[j_block][k];
            }
            
            // perform operation
            for(int wi = 0; wi < WPT; ++wi)
            {               
                // register group offset + position in the register group
                int i_block =  wi*RBS + ti;
                Xreg = Xblock[k][i_block];
                
                for(int wj = 0; wj < WPT; ++wj)
                {
                    // get weighted inputs and add to the accumulator
                    scalar_t tmp = Xreg * Wreg[wj]; 
                    //int arg = BSK*bk+k - (BSK*bk+k)/C*Cpad;
                    accmax[wi][wj] += tmp;
                    /*
                    if(tmp > accmax[wi][wj])
                    {
                        accmax[wi][wj] = tmp;
                        argmax[wi][wj] = arg;
                    }
                    
                    if(tmp < accmin[wi][wj])
                    {
                        accmin[wi][wj] = tmp;
                        argmin[wi][wj] = arg;
                    } */                   
                }
            }
            
        }
        __syncthreads();
    }
    
    // Add together maximum and minimum
    /*
    for(int wi = 0; wi < WPT; ++wi)
    {
        for(int wj = 0; wj < WPT; ++wj)
        {
            accmax[wi][wj] += accmin[wi][wj];
        }
    }*/
    
    // *** STORE THE OUTPUTS ***
    
    const int k_out = bz;
    
    for(int wi = 0; wi < WPT; ++wi)
    {
        const int i_out = bx*BS + wi*RBS + ti;
        for(int wj = 0; wj < WPT; ++wj)
        {
            const int j_out = by*BS + wj*RBS + tj;
            
            Y[k_out*B*F + j_out*B + i_out] = accmax[wi][wj];
            //Yargmax[k_out*B*F + j_out*B + i_out] = argmax[wi][wj];
            //Yargmin[k_out*B*F + j_out*B + i_out] = argmin[wi][wj];
        }
    }
}

void mamconv1d_forward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Y,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    int stride)
{    
    const auto B = X.size(2);
    const auto C = X.size(1);
    const auto M = X.size(0);
    const auto F = W.size(2);
    const auto Mf = W.size(0);
    const auto R = Y.size(0);
    
    int Fpad = 0;
    int Bpad = 0;
    int Cpad = 0;
    
    if(F%BS) Fpad = BS-F%BS;
    if(B%BS) Bpad = BS-B%BS;
    if(C%BS) Cpad = BSK-C%BSK;
    
    auto Wpad = W;
    auto Xpad = X;
    
    if(Fpad || Cpad)
    { 
        Wpad = torch::pad(W.unsqueeze(0),
                          torch::IntList{0, Fpad, 0, Cpad},
                          "replicate").squeeze();
    }
    
    if(Bpad || Cpad)
    { 
        Xpad = torch::pad(X.unsqueeze(0),
                          torch::IntList{0, Bpad, 0, Cpad},
                          "replicate").squeeze();
    }
    
    
    // generate output matrices
    auto Ypad = Y;
    auto Yargmax_pad = Yargmax;
    auto Yargmin_pad = Yargmin;
    
    if(Fpad || Bpad)
    {
        Ypad = torch::empty({R, F+Fpad, B+Bpad}, Y.options());
        Yargmax_pad = torch::empty({R, F+Fpad, B+Bpad}, Yargmax.options());
        Yargmin_pad = torch::empty({R, F+Fpad, B+Bpad}, Yargmin.options());
    }
        
    const dim3 threads(RBS,
                       RBS,
                       1);    
    const dim3 blocks((B+Bpad)/BS,
                      (F+Fpad)/BS,
                      R);
    
    cudaSetDevice(X.get_device());
    
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(),
                               "mamconv1d_forward_cuda_kernel", ([&]{
    mamconv1d_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        Xpad.data_ptr<scalar_t>(),
        Wpad.data_ptr<scalar_t>(),
        Ypad.data_ptr<scalar_t>(),
        Yargmax_pad.data_ptr<int>(),
        Yargmin_pad.data_ptr<int>(),
        (B+Bpad), (C+Cpad), M, (F+Fpad), Mf, stride, Cpad);
    }));
    
    if(Fpad || Bpad)
    {
        Y.copy_(Ypad.slice(1, 0, F).slice(2, 0, B));
        Yargmax.copy_(Yargmax_pad.slice(1, 0, F).slice(2, 0, B));
        Yargmin.copy_(Yargmin_pad.slice(1, 0, F).slice(2, 0, B));
    }
}