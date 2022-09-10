// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SVD_DEFAULT
#error a macro SVD_DEFAULT(MatrixType) must be defined prior to including svd_common.h
#endif

#ifndef SVD_FOR_MIN_NORM
#error a macro SVD_FOR_MIN_NORM(MatrixType) must be defined prior to including svd_common.h
#endif

#include "svd_fill.h"

// Check that the matrix m is properly reconstructed and that the U and V factors are unitary
// The SVD must have already been computed.
template<typename SvdType, typename MatrixType>
void svd_check_full(const MatrixType& m, const SvdType& svd)
{
  typedef typename MatrixType::Index Index;
  Index rows = m.rows();
  Index cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixUType;
  typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixVType;

  MatrixType sigma = MatrixType::Zero(rows,cols);
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  MatrixUType u = svd.matrixU();
  MatrixVType v = svd.matrixV();
  RealScalar scaling = m.cwiseAbs().maxCoeff();
  if(scaling<=(std::numeric_limits<RealScalar>::min)())
    scaling = RealScalar(1);
  VERIFY_IS_APPROX(m/scaling, u * (sigma/scaling) * v.adjoint());
  VERIFY_IS_UNITARY(u);
  VERIFY_IS_UNITARY(v);
}

// Compare partial SVD defined by computationOptions to a full SVD referenceSvd
template<typename SvdType, typename MatrixType>
void svd_compare_to_full(const MatrixType& m,
                         unsigned int computationOptions,
                         const SvdType& referenceSvd)
{
  typedef typename MatrixType::RealScalar RealScalar;
  Index rows = m.rows();
  Index cols = m.cols();
  Index diagSize = (std::min)(rows, cols);
  RealScalar prec = test_precision<RealScalar>();

  SvdType svd(m, computationOptions);

  VERIFY_IS_APPROX(svd.singularValues(), referenceSvd.singularValues());
  
  if(computationOptions & (ComputeFullV|ComputeThinV))
  {
    VERIFY( (svd.matrixV().adjoint()*svd.matrixV()).isIdentity(prec) );
    VERIFY_IS_APPROX( svd.matrixV().leftCols(diagSize) * svd.singularValues().asDiagonal() * svd.matrixV().leftCols(diagSize).adjoint(),
                      referenceSvd.matrixV().leftCols(diagSize) * referenceSvd.singularValues().asDiagonal(