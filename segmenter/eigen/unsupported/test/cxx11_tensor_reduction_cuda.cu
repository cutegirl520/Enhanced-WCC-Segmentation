// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>


template<int DataLayout>
static void test_full_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<float, 2, DataLayout> in(num_rows, num_cols);
  in.setRandom();

  Tensor<float, 0, DataLayout> full_redux;
  full_redux = in.sum();

  std::size_t in_bytes = in.size() * sizeof(float);
  std::size_t out_bytes = full_redux.size() * sizeof(fl