#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <vector>
#include <cstring>

#include "cusparse_handle.h"

namespace mamtorch_kernel_sparsev1 {

static CUSPARSEHandle cusparse_handle;

using namespace torch::indexing;

torch::Tensor cusparsemm_coo(
    torch::Tensor A,
    torch::Tensor B)
{   
    cudaSetDevice(A.get_device()); // set GPU number
    
    const auto N = A.size(0);
    const auto K = A.size(1);
    const auto M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    cusparseDnMatDescr_t    Bdense, Cdense;
    cusparseSpMatDescr_t    Asparse;
    void*                   dBuffer = NULL;
    size_t                  bufferSize = 0;

    // Create sparse matrix A in COO format
    cusparseCreateCoo(&Asparse, N, K, A._nnz(), A.indices().data_ptr<int64_t>(), A.indices().data_ptr<int64_t>()+A._nnz(), A.values().data_ptr<float>(),
                    CUSPARSE_INDEX_64I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_32F);

    // Convert dense matrices to cusparse format
    cusparseCreateDnMat(&Bdense, K, M, M, B.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&Cdense, N, M, M, C.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    auto algorithm = CUSPARSE_SPMM_COO_ALG4;

    // allocate external buffer if needed
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseSpMM_bufferSize(
        cusparse_handle.getHandle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Asparse, Bdense, &beta, Cdense, CUDA_R_32F,
        algorithm, &bufferSize
    );

    cudaMalloc(&dBuffer, bufferSize);

    // perform sparse matmul
    cusparseSpMM(
        cusparse_handle.getHandle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, Asparse, Bdense, &beta, Cdense, CUDA_R_32F,
        algorithm, dBuffer
    );

    // destroy sparse handlers
    cusparseDestroySpMat(Asparse);
    cusparseDestroyDnMat(Bdense);
    cusparseDestroyDnMat(Cdense);

    return C;
}

} // end namespace mamtorch