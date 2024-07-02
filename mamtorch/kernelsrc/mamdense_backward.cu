#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#define BS 16 // block size 
#define WPT 16 // work per thread
#define RBS (BS/WPT) // reduced block_size

/* OPTIMIZATION NOTES
* - each thread performs sparse additions to the gradient matrices (atomic add)
* - each thread evaluates the gradients coming from a specific output block
* - evaluation time is 2 orders of magnitude smaller than forward: no further
*   optimizations needed
*/

template <typename scalar_t>
__global__ void mamdense_backward_cuda_kernel(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    const scalar_t * __restrict__ Cgrad,
    const int * __restrict__ Cargmax,
    const int * __restrict__ Cargmin,
    scalar_t * __restrict__ Agrad,
    scalar_t * __restrict__ Bgrad,
    int M,
    int K,
    int N)
{    
    // get thread and block ids
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    
    // get row and column on the output matrix
    const int Ci = bi*BS + ti*WPT;
    const int Cj = bj*BS + tj*WPT;
    
    // *** EXECUTION ***
    
    if((Ci < M) && (Cj < N))
    {
        const int i_loops = WPT < M-Ci ? WPT : M-Ci;
        for(int i = 0; i < i_loops; ++i)
        {
            const int j_loops = WPT < N-Cj ? WPT : N-Cj;
            for(int j = 0; j < j_loops; ++j)
            {
                int index = Ci+i + (Cj+j)*M;
                scalar_t Cgrad_val = Cgrad[index];
                int kmax = Cargmax[index];
                int kmin = Cargmin[index];
                
                scalar_t A_val;
                scalar_t B_val;
                
                // backprop through max
                A_val = A[Ci+i + kmax*M];
                B_val = B[kmax + (Cj+j)*K];
                atomicAdd(&Agrad[Ci+i + kmax*M], B_val*Cgrad_val);
                atomicAdd(&Bgrad[kmax + (Cj+j)*K], A_val*Cgrad_val);
                
                // backprop through min
                A_val = A[Ci+i + kmin*M];
                B_val = B[kmin + (Cj+j)*K];
                atomicAdd(&Agrad[Ci+i + kmin*M], B_val*Cgrad_val);
                atomicAdd(&Bgrad[kmin + (Cj+j)*K], A_val*Cgrad_val);
            }
        }
    }
}

void mamdense_backward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Cgrad,
    torch::Tensor Cargmax,
    torch::Tensor Cargmin,
    torch::Tensor Agrad,
    torch::Tensor Bgrad)
{    
    const auto M = A.size(1);
    const auto K = A.size(0);
    const auto N = B.size(0);
    
    // initialize output matrices
    Agrad.zero_();
    Bgrad.zero_();
    
    const dim3 threads(RBS,
                       RBS,
                       1);    
    const dim3 blocks((M-1)/BS+1,
                      (N-1)/BS+1,
                      1);  
    
    cudaSetDevice(A.get_device());
    
    /*
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "mamdense_backward_cuda_kernel", ([&]{
    mamdense_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        Cgrad.data_ptr<scalar_t>(),
        Cargmax.data_ptr<int>(),
        Cargmin.data_ptr<int>(),
        Agrad.data_ptr<scalar_t>(),
        Bgrad.data_ptr<scalar_t>(),
        M, K, N);
    }));
    */
    
    mamdense_backward_cuda_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        Cgrad.data_ptr<float>(),
        Cargmax.data_ptr<int>(),
        Cargmin.data_ptr<int>(),
        Agrad.data_ptr<float>(),
        Bgrad.data_ptr<float>(),
        M, K, N);
}