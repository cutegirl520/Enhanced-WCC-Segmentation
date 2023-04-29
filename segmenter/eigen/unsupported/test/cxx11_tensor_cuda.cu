// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_cuda_elementwise_small() {
  Tensor<float, 1> in1(Eigen::array<int, 1>(2));
  Tensor<float, 1> in2(Eigen::array<int, 1>(2));
  Tensor<float, 1> out(Eigen::array<int, 1>(2));
  in1.setRandom();
  in2.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
      d_in1, Eigen::array<int, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
      d_in2, Eigen::array<int, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
      d_out, Eigen::array<int, 1>(2));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(
        out(Eigen::array<int, 1>(i)),
        in1(Eigen::array<int, 1>(i)) + in2(Eigen::array<int, 1>(i)));
  }

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}

void test_cuda_elementwise()
{
  Tensor<float, 3> in1(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> in2(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> in3(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> out(Eigen::array<int, 3>(72,53,97));
  in1.setRandom();
  in2.setRandom();
  in3.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t in3_bytes = in3.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_in3;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_in3), in3_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in3, in3.data(), in3_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<int, 3>(72,53,97));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 53; ++j) {
      for (int k = 0; k < 97; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<int, 3>(i,j,k)), in1(Eigen::array<int, 3>(i,j,k)) + in2(Eigen::array<int, 3>(i,j,k)) * in3(Eigen::array<int, 3>(i,j,k)));
      }
    }
  }

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_in3);
  cudaFree(d_out);
}

void test_cuda_props() {
  Tensor<float, 1> in1(200);
  Tensor<bool, 1> out(200);
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(bool);

  float* d_in1;
  bool* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
      d_in1, 200);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
      d_out, 200);

  gpu_out.device(gpu_device) = (gpu_in1.isnan)();

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 200; ++i) {
    VERIFY_IS_EQUAL(out(i), (std::isnan)(in1(i)));
  }

  cudaFree(d_in1);
  cudaFree(d_out);
}

void test_cuda_reduction()
{
  Tensor<float, 4> in1(72,53,97,113);
  Tensor<float, 2> out(72,97);
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  array<int, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      float expected = 0;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 113; ++l) {
          expected =
              std::max<float>(expected, in1(i, k, j, l));
        }
      }
      VERIFY_IS_APPROX(out(i,j), expected);
    }
  }

  cudaFree(d_in1);
  cudaFree(d_out);
}

template<int DataLayout>
void test_cuda_contraction()
{
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<float, 4, DataLayout> t_left(6, 50, 3, 31);
  Tensor<float, 5, DataLayout> t_right(Eigen::array<int, 5>(3, 31, 7, 20, 1));
  Tensor<float, 5, DataLayout> t_result(Eigen::array<int, 5>(6, 50, 7, 20, 1));

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(float);
  std::size_t t_right_bytes = t_right.size() * sizeof(float);
  std::size_t t_result_bytes = t_result.size() * sizeof(float);

  float* d_t_left;
  float* d_t_right;
  float* d_t_result;

  cudaMalloc((void**)(&d_t_left), t_left_bytes);
  cudaMalloc((void**)(&d_t_right), t_right_bytes);
  cudaMalloc((void**)(&d_t_result), t_result_bytes);

  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);

  typedef Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> > MapXf;
  MapXf m_left(t_left.data(), 300, 93);
  MapXf m_right(t_right.data(), 93, 140);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(300, 140);

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims;
  dims[0] = DimPair(2, 0);
  dims[1] = DimPair(3, 1);

  m_result = m_left * m_right;
  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);

  cudaMemcpy(t_result.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < t_result.dimensions().TotalSize(); i++) {
    if (fabs(t_result.data()[i] - m_result.data()[i]) >= 1e-4) {
      std::cout << "mismatch detected at index " << i << ": " << t_result.data()[i] << " vs " <<  m_result.data()[i] << std::endl;
      assert(false);
    }
  }

  cudaFree(d_t_left);
  cudaFree(d_t_right);
  cudaFree(d_t_result);
}

template<int DataLayout>
void test_cuda_convolution_1d()
{
  Tensor<float, 4, DataLayout> input(74,37