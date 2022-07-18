// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void product_extra(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic,
                         MatrixType::Flags&RowMajorBit> OtherMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows),
             res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols),
             res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows), vrres(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>(),
         s3 = internal::random<Scalar>();

  VERIFY_IS_APPROX(m3.noalias() = m1 * m2.adjoint(),                 m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * square.adjoint(),   m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * m2,                 m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (s1 * m1.adjoint()) * m2,          (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = ((s1 * m1).adjoint()) * m2,        (numext::conj(s1) * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (- m1.adjoint() * s1) * (s3 * m2), (- m1.adjoint()  * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX(m3.noalias() = (s2 * m1.adjoint() * s1) * m2,     (s2 * m1.adjoint()  * s1).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1*s2) * s1*m2.adjoint(),        (-m1*s2).eval() * (s1*m2.adjoint()).eval());

  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX( m1.adjoint() * (s1*m2).conjugate(), (m1.adjoint()).eval() * ((s1*m2).conjugate()).eval());


  // test all possible conjugate combinations for the four matrix-vector product cases:

  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()),
                   (-m1*s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2.conjugate()).eval());

  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval()