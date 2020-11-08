#ifndef CNN_CUDA_H
#define CNN_CUDA_H
#if HAVE_CUDA

#include <vector>
#include <cassert>
#include <utility>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cnn/except.h"

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw cnn::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t s