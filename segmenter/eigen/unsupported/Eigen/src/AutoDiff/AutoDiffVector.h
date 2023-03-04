// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_AUTODIFF_VECTOR_H
#define EIGEN_AUTODIFF_VECTOR_H

namespace Eigen {

/* \class AutoDiffScalar
  * \brief A scalar type replacement with automatic differentation capability
  *
  * \param DerType the vector type used to s