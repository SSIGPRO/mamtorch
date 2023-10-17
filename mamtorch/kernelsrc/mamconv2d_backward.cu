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
__global__ void mamconv2d_backward_cuda_kernel(
    const scalar_t * __restrict__ X,
    const scalar_t * __restrict__ W,
    const scalar_t * __restrict__ Ygrad,
    const int * __restrict__ Yargmax,
    const int * __restrict__ Yargmin,
    scalar_t * __restrict__ Xgrad,
    scalar_t * __restrict__ Wgrad,
    int B,
    int C,
    int M,
    int N,
    int F,
    int Mf,
    int Nf,
    int Mo,
    int No,
    int hstride,
    int wstride)
{
    // get thread and block ids
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    const int bk = blockIdx.z;
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    const int tk = threadIdx.z;
    
    // get row and column on the output matrix
    const int Y_batch = bi*BS + ti*WPT;
    const int Y_channel = bj*BS + tj*WPT;
    const int Y_element = bk*BS + tk*WPT;
    
    const int Y_element_j = Y_element % Mo;
    const int Y_element_i = Y_element / Mo;
    
    // *** EXECUTION ***
    
    if((Y_batch < B) && (Y_channel < F) && (Y_element < Mo*No))
    {
        const int batch_loops = WPT < B-Y_batch ? WPT : B-Y_batch;
        for(int batch = 0; batch < batch_loops; ++batch)
        {
            const int channel_loops = WPT < F-Y_channel ? WPT : F-Y_channel;
            for(int channel = 0; channel < channel_loops; ++channel)
            {
                const int element_loops = WPT < Mo*No-Y_element ? WPT : Mo*No-Y_element;
                for(int element = 0; element < element_loops; ++element)
                {
                    int element_j = (Y_element + element) % Mo;
                    int element_i = (Y_element + element) / Mo;
                    int index = Y_batch+batch
                              + (Y_channel+channel)*B
                              + element_j*B*F
                              + element_i*B*F*Mo;
                    scalar_t Ygrad_val = Ygrad[index];
                    int kmax = Yargmax[index];
                    int kmin = Yargmin[index];
                    int kmax_channel = kmax % C;
                    int kmax_element = kmax / C;
                    int kmax_element_j = kmax_element % Mf;
                    int kmax_element_i = kmax_element / Mf;
                    int kmin_channel = kmin % C;
                    int kmin_element = kmin / C;
                    int kmin_element_j = kmin_element % Mf;
                    int kmin_element_i = kmin_element / Mf;
                    
                    scalar_t X_val;
                    scalar_t W_val;
                    int X_index;
                    int W_index;
                    
                    // backprop through max
                    X_index = Y_batch+batch
                            + kmax_channel*B
                            + (kmax_element_j+element_j*hstride)*B*C
                            + (kmax_element_i+element_i*wstride)*B*C*M;
                    W_index = Y_channel+channel
                            + kmax_channel*F
                            + kmax_element_j*F*C
                            + kmax_element_i*F*C*Mf;
                    X_val = X[X_index];
                    W_val = W[W_index];
                    atomicAdd(&Xgrad[X_index], W_val*Ygrad_val);
                    atomicAdd(&Wgrad[W_index], X_val*Ygrad_val);
                    
                    // backprop through min
                    X_index = Y_batch+batch
                            + kmin_channel*B
                            + (kmin_element_j+element_j*hstride)*B*C
                            + (kmin_element_i+element_i*wstride)*B*C*M;
                    W_index = Y_channel+channel
                            + kmin_channel*F
                            + kmin_element_j*F*C
                            + kmin_element_i*F*C*Mf;
                    X_val = X[X_index];
                    W_val = W[W_index];
                    atomicAdd(&Xgrad[X_index], W_val*Ygrad_val);
                    atomicAdd(&Wgrad[W_index], X_val*Ygrad_val);
                }
            }
        }
    }
}

void mamconv2d_backward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Ygrad,
    torch::Tensor Yargmax,
    torch::Tensor Yargmin,
    torch::Tensor Xgrad,
    torch::Tensor Wgrad,
    int hstride,
    int wstride)
{    
    const auto B = X.size(3);
    const auto C = X.size(2);
    const auto M = X.size(1);
    const auto N = X.size(0);
    const auto F = W.size(3);
    const auto Mf = W.size(1);
    const auto Nf = W.size(0);
    const auto Mo = Ygrad.size(1);
    const auto No = Ygrad.size(0);
    
    // initialize output matrices
    Xgrad.zero_();
    Wgrad.zero_();
    
    const dim3 threads(RBS,
                       RBS,
                       RBS);    
    const dim3 blocks((Mo*No-1)/BS+1,
                      (F-1)/BS+1,
                      (B-1)/BS+1);  
    
    cudaSetDevice(X.get_device());
    
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(),
                               "mamconv2d_backward_cuda_kernel", ([&]{
    mamconv2d_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        X.data_ptr<scalar_t>(),
        W.data_ptr<scalar_t>(),
        Ygrad.data_ptr<scalar_t>(),
        Yargmax.data_ptr<int>(),
        Yargmin.data_ptr<int>(),
        Xgrad.data_ptr<scalar_t>(),
        Wgrad.data_ptr<scalar_t>(),
        B, C, M, N, F, Mf, Nf, Mo, No, hstride, wstride);
    }));
}