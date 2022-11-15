// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H

namespace Eigen {

static const int kCudaScratchSize = 1024;

// This defines an interface that GPUDevice can take to use
// CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const cudaStream_t& stream() const = 0;
  virtual const cudaDeviceProp& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;

  // Return a scratchpad buffer of size 1k
  virtual void* scratchpad() const = 0;

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  virtual unsigned int* semaphore() const = 0;
};

static cudaDeviceProp* m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    if (!m_devicePropInitialized) {
      int num_devices;
      cudaError_t status = cudaGetDeviceCount(&num_devices);
      if (status != cudaSuccess) {
        std::cerr << "Failed to get the number of CUDA devices: "
                  << cudaGetErrorString(status)
                  << std::endl;
        assert(status == cudaSuccess);
      }
      m_deviceProperties = new cudaDeviceProp[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
        if (status != cudaSuccess) {
          std::cerr << "Failed to initialize CUDA device #"
                    << i
                    << ": "
                    << cudaGetErrorString(status)
                    << std::endl;
          assert(status == cudaSuccess);
        }
      }
      m_devicePropInitialized = true;
    }
  }
}

static const cudaStream_t default_stream = cudaStreamDefault;

class CudaStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
    cudaGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the 