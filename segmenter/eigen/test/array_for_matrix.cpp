// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void array_for_matrix(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType; 

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  ColVectorType cv1 = ColVectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols);
  
  Scalar  s1 = internal::random<Scalar>(),
          s2 = internal::random<Scalar>();
          
  // scalar addition
  VERIFY_IS_APPROX(m1.array() + s1, s1 + m1.array());
  VERIFY_IS_APPROX((m1.array() + s1).matrix(), MatrixType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX(((m1*Scalar(2)).array() - s2).matrix(), (m1+m1) - MatrixType::Constant(rows,cols,s2) );
  m3 = m1;
  m3.array() += s2;
  VERIFY_IS_APPROX(m3, (m1.array() + s2).matrix());
  m3 = m1;
  m3.array() -= s1;
  VERIFY_IS_APPROX(m3, (m1.array() - s1).matrix());

  // reductions
  VERIFY_IS_MUCH_SMALLER_THAN(m1.colwise().sum().sum() - m1.sum(), m1.squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.rowwise().sum().sum() - m1.sum(), m1.squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.colwise().sum() + m2.colwise().sum() - (m1+m2).colwise().sum(), (m1+m2).squaredNorm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.rowwise().sum() - m2.rowwise().sum() - (m1-m2).rowwise().sum(), (m1-m2).squaredNorm());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(internal::scalar_sum_op<Scalar,Scalar>()));

  // vector-wise ops
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() += cv1, m1.colwise() + cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() -= cv1, m1.colwise() - cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() += rv1, m1.rowwise() + rv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() -= rv1, m1.rowwise() - rv1);
  
  // empty objects
  VERIFY_IS_APPROX(m1.block(0,0,0