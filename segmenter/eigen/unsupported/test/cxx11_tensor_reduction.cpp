// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <numeric>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_trivial_reductions() {
  {
    Tensor<float, 0, DataLayout> tensor;
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 0, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result(), tensor());
  }

  {
    Tensor<float, 1, DataLayout> tensor(7);
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 1, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result.dimension(0), 7);
    for (int i = 0; i < 7; ++i) {
      VERIFY_IS_EQUAL(result(i), tensor(i));
    }
  }

  {
    Tensor<float, 2, DataLayout> tensor(2, 3);
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 2, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result.dimension(0), 2);
    VERIFY_IS_EQUAL(result.dimension(1), 3);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        VERIFY_IS_EQUAL(result(i, j), tensor(i, j));
      }
    }
  }
}

template <int DataLayout>
static void test_simple_reductions() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();
  array<ptrdiff_t, 2> reduction_axis2;
  reduction_axis2[0] = 1;
  reduction_axis2[1] = 3;

  Tensor<float, 2, DataLayout> result = tensor.sum(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 5);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor(i, k, j, l);
        }
      }
      VERIFY_IS_APPROX(result(i, j), sum);
    }
  }

  {
    Tensor<float, 0, DataLayout> sum1 = tensor.sum();
    VERIFY_IS_EQUAL(sum1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<float, 0, DataLayout> sum2 = tensor.sum(reduction_axis4);
    VERIFY_IS_EQUAL(sum2.rank(), 0);

    VERIFY_IS_APPROX(sum1(), sum2());
  }

  reduction_axis2[0] = 0;
  reduction_axis2[1] = 2;
  result = tensor.prod(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 3);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      float prod = 1.0f;
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 5; ++l) {
          prod *= tensor(k, i, l, j);
        }
      }
      VERIFY_IS_APPROX(result(i, j), pr