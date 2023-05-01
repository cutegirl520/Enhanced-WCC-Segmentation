// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template<int DataLayout>
static void test_simple_patch()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> patch_dims;

  patch_dims[0] = 1;
  patch_dims[1] = 1;
  patch_dims[2] = 1;
  patch_dims[3] = 1;

  Tensor<float, 5, DataLayout> no_patch;
  no_patch = tensor.extract_patches(patch_dims);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(no_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), tensor.size());
  } else {
    VERIFY_IS_EQUAL(no_patch.dimension(0), tensor.size());
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), 1);
  }

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], no_patch.data()[i]);
  }

  patch_dims[0] =