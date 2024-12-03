#include <torch/extension.h>
#include <math.h>

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

namespace mamtorch_kernel_v3 {

__global__ void selection_count_cuda_kernel(
    const int * __restrict__ Cargmax,
    const int * __restrict__ Cargmin,
    int * __restrict__ Maxselection,
    int * __restrict__ Minselection,
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
        const int i_loops = min(WPT, M-Ci);
        for(int i = 0; i < i_loops; ++i)
        {
            const int j_loops = min(WPT, N-Cj);
            for(int j = 0; j < j_loops; ++j)
            {
                int index = Ci + i + (Cj + j)*M;
                int kmax = Cargmax[index];
                int kmin = Cargmin[index];
                
                atomicAdd(&Minselection[Ci+i + kmin*M], 1);
                atomicAdd(&Maxselection[Ci+i + kmax*M], 1);
            }
        }
    }
}

std::vector<at::Tensor> selection_count_cuda(
    at::Tensor A,
    at::Tensor Cargmax,
    at::Tensor Cargmin
    )
{       
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;

    // generate output matrix
    auto Minselection = at::empty({ATcm.size(0), ATcm.size(1)}, ATcm.options().dtype(at::kInt));
    auto Maxselection = at::empty({ATcm.size(0), ATcm.size(1)}, ATcm.options().dtype(at::kInt));

    auto Acuda = ATcm;
    
    const auto M = Acuda.size(1);
    const auto K = Acuda.size(0);
    const auto N = CargmaxTcm.size(0);

    
    // initialize output matrices
    Minselection.zero_();
    Maxselection.zero_();
    
    const dim3 threads(RBS,
                       RBS,
                       1);    
    const dim3 blocks((M-1)/BS+1,
                      (N-1)/BS+1,
                      1);  
    
    cudaSetDevice(A.get_device());
    
    selection_count_cuda_kernel<<<blocks, threads>>>(
        CargmaxTcm.data_ptr<int>(),
        CargminTcm.data_ptr<int>(),
        Maxselection.data_ptr<int>(),
        Minselection.data_ptr<int>(),
        M, K, N);
    
    cudaDeviceSynchronize();

    return {Maxselection, Minselection};
}

} // end namespace mamtorch