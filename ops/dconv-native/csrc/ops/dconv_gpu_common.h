#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>

class _debug {

};

inline int get_output_size(int input_size, int kernel_size, int stride, int pad, int dilation) {
    return (input_size + 2 * pad - (dilation * (kernel_size - 1) + 1)) / stride + 1;
}

#define CHECK_POSITIVE_3D(x, y, z) \
    TORCH_CHECK((x) > 0 && (y) > 0 && (z) > 0)
#define SANITY_CHECK_DCONV_GROUPS(N, INPUT, OFFSET, ALPHA, WEIGHT, N_WEIGHT_GROUPS, N_OFFSET_GROUPS, NUM_KERNEL_POINTS) \
    TORCH_CHECK(WEIGHT.size(1) * (N_WEIGHT_GROUPS) == INPUT.size(1))             \
    TORCH_CHECK(WEIGHT.size(0) % (N_WEIGHT_GROUPS) == 0)                         \
    TORCH_CHECK(WEIGHT.size(0) % (N_OFFSET_GROUPS) == 0)                         \
    TORCH_CHECK(INPUT.size(1) % (N_OFFSET_GROUPS) == 0)                          \
    TORCH_CHECK(OFFSET.size(1) == (N_OFFSET_GROUPS) * (N) * (NUM_KERNEL_POINTS)) \
    if(ALPHA.defined()) TORCH_CHECK(ALPHA.size(1) == (N_OFFSET_GROUPS) * (NUM_KERNEL_POINTS))
#define SANITY_CHECK_DCONV_INPUTS(N, INPUT, OFFSET, ALPHA, WEIGHT, BIAS) \
    TORCH_CHECK(INPUT.ndimension() == ((N) + 2))   \
    TORCH_CHECK(OFFSET.ndimension() == ((N) + 2))  \
    if(ALPHA.defined()) TORCH_CHECK(ALPHA.ndimension() == ((N) + 2))   \
    TORCH_CHECK(WEIGHT.ndimension() == ((N) + 2))  \
    if(BIAS.defined()) TORCH_CHECK(BIAS.ndimension() == 1)    \
    TORCH_CHECK(INPUT.device().is_cuda())          \
    TORCH_CHECK(OFFSET.device().is_cuda())         \
    if(ALPHA.defined()) TORCH_CHECK(ALPHA.device().is_cuda())          \
    TORCH_CHECK(WEIGHT.device().is_cuda())         \
    if(BIAS.defined()) TORCH_CHECK(BIAS.device().is_cuda())           \
    TORCH_CHECK(INPUT.is_contiguous())             \
    TORCH_CHECK(OFFSET.is_contiguous())            \
    if(ALPHA.defined()) TORCH_CHECK(ALPHA.is_contiguous())             \
    TORCH_CHECK(WEIGHT.is_contiguous())            \
    if(BIAS.defined()) TORCH_CHECK(BIAS.is_contiguous())              \
    TORCH_CHECK(OFFSET.size(0) == INPUT.size(0))   \
    if(ALPHA.defined()) TORCH_CHECK(ALPHA.size(0) == INPUT.size(0))
#define SANITY_CHECK_DCONV1D(INPUT, OFFSET, ALPHA, WEIGHT, BIAS) \
    SANITY_CHECK_DCONV_INPUTS(1, INPUT, OFFSET, ALPHA, WEIGHT, BIAS)
#define SANITY_CHECK_DCONV1D_GROUPS(INPUT, OFFSET, ALPHA, WEIGHT, N_WEIGHT_GROUPS, N_OFFSET_GROUPS) \
    SANITY_CHECK_DCONV_GROUPS(1, INPUT, OFFSET, ALPHA, WEIGHT, N_WEIGHT_GROUPS, N_OFFSET_GROUPS, WEIGHT.size(2))
#define SANITY_CHECK_DCONV3D(INPUT, OFFSET, ALPHA, WEIGHT, BIAS) \
    SANITY_CHECK_DCONV_INPUTS(3, INPUT, OFFSET, ALPHA, WEIGHT, BIAS)
#define SANITY_CHECK_DCONV3D_GROUPS(INPUT, OFFSET, ALPHA, WEIGHT, N_WEIGHT_GROUPS, N_OFFSET_GROUPS) \
    SANITY_CHECK_DCONV_GROUPS(3, INPUT, OFFSET, ALPHA, WEIGHT, N_WEIGHT_GROUPS, N_OFFSET_GROUPS, WEIGHT.size(2) * WEIGHT.size(3) * WEIGHT.size(4))

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__forceinline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(
                address_as_ull,
                assumed,
                __double_as_longlong(val + __longlong_as_double(assumed))
        );

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while(assumed != old);

    return __longlong_as_double(old);
}
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

const int CUDA_NUM_THREADS = 512;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

#define CUDABLAS_GEMM_STRIDED_BATCHED_ARGTYPES(Dtype)                                      \
      cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n,                      \
      int64_t k, Dtype alpha, const Dtype *a, int64_t lda, int64_t stridea, \
      const Dtype *b, int64_t ldb, int64_t strideb, Dtype beta, Dtype *c, \
      int64_t ldc, int64_t stridec, int64_t num_batches

template<typename T>
__forceinline__ void gemmStridedBatched(CUDABLAS_GEMM_STRIDED_BATCHED_ARGTYPES(T)) {
    AT_ERROR("gemmStridedBatched not implemented for ", typeid(T).name());
}

template<>
__forceinline__ void  gemmStridedBatched(CUDABLAS_GEMM_STRIDED_BATCHED_ARGTYPES(float)) {
    cublasSgemmStridedBatched(
            at::cuda::getCurrentCUDABlasHandle(),
            transa, transb,
            m, n, k,
            &alpha,
            a, lda, stridea,
            b, ldb, strideb,
            &beta,
            c, ldc, stridec,
            num_batches
    );
}

template<>
__forceinline__ void gemmStridedBatched(CUDABLAS_GEMM_STRIDED_BATCHED_ARGTYPES(double)) {
    cublasDgemmStridedBatched(
            at::cuda::getCurrentCUDABlasHandle(),
            transa, transb,
            m, n, k,
            &alpha,
            a, lda, stridea,
            b, ldb, strideb,
            &beta,
            c, ldc, stridec,
            num_batches
    );
}

template<>
__forceinline__ void gemmStridedBatched(CUDABLAS_GEMM_STRIDED_BATCHED_ARGTYPES(__half)) {
    cublasHgemmStridedBatched(
            at::cuda::getCurrentCUDABlasHandle(),
            transa, transb,
            m, n, k,
            &alpha,
            a, lda, stridea,
            b, ldb, strideb,
            &beta,
            c, ldc, stridec,
            num_batches
    );
}
