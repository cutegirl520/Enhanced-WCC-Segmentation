// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Storage>
void run_matrix_tests()
{
  typedef Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Storage> MatrixType;
  typedef typename MatrixType::Index Index;

  MatrixType m, n;

  // bou