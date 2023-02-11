// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H

namespace Eigen {
namespace internal {


#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
// Full reducers for GPU, don't vectorize for now

// Reducer function that enables multiple cuda thread to safely accumulate at the same
// output address. It basically reads the current value of the output variable, and
// attempts to update it with the new value. If in the meantime another cuda thread
// updated the content of the output address it will try again.
template <typename T, typename R>
__device__ EIGEN_ALWAYS_INLINE void atomicReduce(T* output, T accum, R& reducer) {
#if __CUDA_ARCH__ >= 300
  if (sizeof(T) == 4)
  {
    unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
    unsigned int newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned int readback;
    while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else if (sizeof(T) == 8) {
    unsigned long long oldval = *reinterpret_cast<unsigned long long*>(output);
    unsigned long long newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned long long readback;
    while ((readback = atomicCAS((unsigned long long*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else {
    assert(0 && "Wordsize not supported");
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


#ifdef EIGEN_HAS_CUDA_FP16
template <template <typename T> class R>
__device__ inline void atomicReduce(half2* output, half2 accum, R<half>& reducer) {
#if __CUDA_ARCH__ >= 300
  unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
  unsigned int newval = oldval;
  reducer.reducePacket(accum, reinterpret_cast<half2*>(&newval));
  if (newval == oldval) {
    return;
  }
  unsigned int readback;
  while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
    oldval = readback;
    newval = oldval;
    reducer.reducePacket(accum, reinterpret_cast<half2*>(&newval));
    if (newval == oldval) {
      return;
    }
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}
#endif

template <>
__device__ inline void atomicReduce(float* output, float accum, SumReducer<float>&) {
#if __CUDA_ARCH__ >= 300
  atomicAdd(output, accum);
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


template <typename CoeffType, typename Index>
__global__ void ReductionInitKernel(const CoeffType val, Index num_preserved_coeffs, CoeffType* output) {
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const Index num_threads = blockDim.x * gridDim.x;
  for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
    output[i] = val;
  }
}


template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(Reducer reducer, const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output, unsigned int* semaphore) {
#if __CUDA_ARCH__ >= 300
  // Initialize the output value
  const Index first_index = blockIdx.x * BlockSize * NumPerThread + threadIdx.x;
  if (gridDim.x == 1) {
    if (first_index == 0) {
      *output = reducer.initialize();
    }
  }
  else {
    if (threadIdx.x == 0) {
      unsigned int block = atomicCAS(semaphore, 0u, 1u);
      if (block == 0) {
        // We're the first block to run, initialize the output value
        atomicExch(output, reducer.initialize());
        __threadfence();
        atomicExch(semaphore, 2u);
      }
      else {
        // Wait for the first block to initialize the output value.
        // Use atomicCAS here to ensure that the reads aren't cached
        unsigned int val;
        do {
          val = atomicCAS(semaphore, 2u, 2u);
        }
        while (val < 2u);
      }
    }
  }

  __syncthreads();

  eigen_assert(gridDim.x == 1 || *semaphore >= 2u);

  typename Self::CoeffReturnType accum = reducer.initialize();
  Index max_iter = numext::mini<Index>(num_coeffs - first_index, NumPerThread*BlockSize);
  for (Index i = 0; i < max_iter; i+=BlockSize) {
    const Index index = first_index + i;
    eigen_assert(index < num_coeffs);
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

#pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reduce(__shfl_down(accum, offset, warpSize), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }

  if (gridDim.x > 1 && threadIdx.x == 0) {
    // Let the last block reset the semaphore
    atomicInc(semaphore, gridDim.x + 1);
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


#ifdef EIGEN_HAS_CUDA_FP16
template <typename Self,
          typename Reducer, typename Index>
__global__ void ReductionInitFullReduxKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs, half2* scratch) {
  eigen_assert(blockDim.x == 1);
  eigen_assert(gridDim.x == 1);
  if (num_coeffs % 2 != 0) {
    half last = input.m_impl.coeff(num_coeffs-1);
    *scratch = __halves2half2(last, reducer.initialize());
  } else {
    *scratch = reducer.template initializePacket<half2>();
  }
}

template <typename Self,
          typename Reducer, typename Index>
__global__ void ReductionInitKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs, half* output) {
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const Index num_threads = blockDim.x * gridDim.x;
  const Index num_packets = num_coeffs / 2;
  for (Index i = thread_id; i < num_packets; i += num_threads) {
    ((half2*)output)[i] = reducer.template initializePacket<half2>();
  }

  if (thread_id == 0 && num_coeffs % 2 != 0) {
    output[num_coeffs-1] = reducer.initialize();
  }
}

template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs,
                                    half* output, half2* scratch) {
  eigen_assert(NumPerThread % 2 == 0);

  const Index first_index = blockIdx.x * BlockSize * NumPerThread + 2*threadIdx.x;

  // Initialize the output value if it wasn't initialized by the ReductionInitKernel
  if (gridDim.x == 1 && first_index == 0) {
    if (num_coeffs % 2 != 0) {
      half last = input.m_impl.coeff(num_coeffs-1);
      *scratch = __halves2half2(last, reducer.initialize());
    } else {
      *scratch = reducer.template initializePacket<half2>();
    }
    __syncthreads();
  }

  half2 accum = reducer.template initializePacket<half2>();
  const Index max_iter = numext::mini<Index>((num_coeffs - first_index) / 2, NumPerThread*BlockSize / 2);
  for (Index i = 0; i < max_iter; i += BlockSize) {
    const Index index = first_index + 2*i;
    eigen_assert(index + 1 < num_coeffs);
    half2 val = input.m_impl.template packet<Unaligned>(index);
    reducer.reducePacket(val, &accum);
  }

#pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reducePacket(__shfl_down(accum, offset, warpSize), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(scratch, accum, reducer);
  }

  __syncthreads();

  if (gridDim.x == 1 && first_index == 0) {
    half tmp = __low2half(*scratch);
    reducer.reduce(__high2half(*scratch), &tmp);
    *output = tmp;
  }
}

template <typename Op>
__global__ void ReductionCleanupKernelHalfFloat(Op& reducer, half* output, half2* scratch) {
  eigen_assert(threadIdx.x == 1);
  half tmp = __low2half(*scratch);
  reducer.reduce(__high2half(*scratch), &tmp);
  *output = tmp;
}

#endif


template <typename Self, typename Op, typename OutputType, bool PacketAccess>
struct FullReductionLauncher {
  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
    assert(false && "Should only be called on floats and half floats");
  }
};

template <typename Self, typename Op, bool PacketAccess>
struct FullReductionLauncher<Self, Op, float, PacketAccess> {
  static void run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs) {
    typedef typename Self::Index Index;
    typedef typename Self::CoeffReturnType Scalar;
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = divup<int>(num_coeffs, block_size * num_per_thread);

    unsigned int* semaphore = NULL;
    if (num_blocks > 1) {
      semaphore = device.semaphore();
    }

    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs, output, semaphore);
  }
};

#ifdef EIGEN_HAS_CUDA_FP16
template <t