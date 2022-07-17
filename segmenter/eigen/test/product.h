// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/QR>

template<typename Derived1, typename Derived2>
bool areNotApprox(const MatrixBase<Derived1>& m1, const MatrixBase<Derived2>& m2, typename Derived1::RealScalar epsilon = NumTraits<typename Derived1::RealScalar>::dummy_precision())
{
  return !((m1-m2).cwiseAbs2().maxCoeff() < epsilon * epsilon
                          * (std::max)(m1.cwiseAbs2().maxCoeff(), m2.cwiseAbs2().maxCoeff()));
}

template<typename MatrixType> void product(const MatrixType& 