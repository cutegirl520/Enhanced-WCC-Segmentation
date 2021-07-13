// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRIANGULAR_MATRIX_MATRIX_H
#define EIGEN_TRIANGULAR_MATRIX_MATRIX_H

namespace Eigen { 

namespace internal {

// template<typename Scalar, int mr, int StorageOrder, bool Conjugate, int Mode>
// struct gemm_pack_lhs_triangular
// {
//   Matrix<Scalar,mr,mr,
//   void operator()(Scalar* blockA, const EIGEN_RESTRICT Scalar* _lhs, int lhsStride, int depth, int rows)
//   {
//     conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
//     const_blas_data_mapper<Scalar, Stora