// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/StdDeque>
#include <Eigen/Geometry>

EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Vector4f)

EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Matrix2f)
EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Matrix4f)
EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Matrix4d)

EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Affine3f)
EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Affine3d)

EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Quaternionf)
EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(Quaterniond)

template<typename MatrixType>
void check_stddeque_matrix(const MatrixType& m)
{
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType x = MatrixType::Random(rows,cols), y = MatrixType::Random(rows,cols);
  std::deque<MatrixType> v(10, MatrixType(rows,cols)), w(20, y);
  v[5] = x;
  w[6] = v[5];
  VERIFY_IS_APPROX(w[6], v[5]);
  v = w;
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], v[i]);
  }

  v.resize(21);
  v[20] = x;
  VERIFY_IS_APPROX(v[20], x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);

  // do a lot of push_back such that the deque gets internally resized
  // (with memory reallocation)
  MatrixType* ref = &w[0];
  for(int i=0; i<30 |