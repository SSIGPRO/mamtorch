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

template <typename scalar_t>
__global__ void fullyconnected_backward_cuda_kernel(
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
        const int i_loops = min(WPT, M-Ci);
        for(int i = 0; i < i_loops; ++i)
        {
            const int j_loops = min(WPT, N-Cj);
            for(int j = 0; j < j_loops; ++j)
            {
                int index = Ci + i + (Cj + j) * M;
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

std::vector<at::Tensor> fullyconnected_backward_cuda(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Cgrad,
    at::Tensor Cargmax,
    at::Tensor Cargmin,
    double beta)
{       
    // row-major to column-major + transpose
    const auto ATcm = A;
    const auto BTcm = B;
    const auto CgradTcm = Cgrad;
    const auto CargmaxTcm = Cargmax;
    const auto CargminTcm = Cargmin;
    // generate output matrix
    auto AgradTcm = at::empty({ATcm.size(0), ATcm.size(1)}, ATcm.options());
    auto BgradTcm = at::empty({BTcm.size(0), BTcm.size(1)}, ATcm.options());

    // cuda matrices (A and B are swapped)
    auto Acuda = BTcm;
    auto Bcuda = ATcm;
    auto Agradcuda = BgradTcm;
    auto Bgradcuda = AgradTcm;
    
    const auto M = Acuda.size(1);
    const auto K = Acuda.size(0);
    const auto N = Bcuda.size(0);
    
    // initialize output matrices
    Agradcuda.zero_();
    Bgradcuda.zero_();
    
    const dim3 threads(RBS,
                       RBS,
                       1);    
    const dim3 blocks((M-1)/BS+1,
                      (N-1)/BS+1,
                      1);  
    
    cudaSetDevice(A.get_device());
    
    if(beta < 1)
    {
    fullyconnected_backward_cuda_kernel<float><<<blocks, threads>>>(
        Acuda.data_ptr<float>(),
        Bcuda.data_ptr<float>(),
        CgradTcm.data_ptr<float>(),
        CargmaxTcm.data_ptr<int>(),
        CargminTcm.data_ptr<int>(),
        Agradcuda.data_ptr<float>(),
        Bgradcuda.data_ptr<float>(),
        M, K, N);
    }

    // swap again A and B
    auto Agrad = Bgradcuda;
    auto Bgrad = Agradcuda;

    // perform backward affine combination with MAC contribution
    if(beta > 0)
    {
        Agrad *= 1-beta;
        Agrad += at::linalg_matmul(Cgrad, B.transpose(0,1))*beta;
        Bgrad *= 1-beta;
        Bgrad += at::linalg_matmul(A.transpose(0,1), Cgrad)*beta;
    }

    return {Agrad, Bgrad};
}

} // end namespace mamtorch