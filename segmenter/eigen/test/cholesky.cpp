// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NO_ASSERTION_CHECKING
#define EIGEN_NO_ASSERTION_CHECKING
#endif

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>

template<typename MatrixType, int UpLo>
typename MatrixType::RealScalar matrix_l1_norm(const MatrixType& m) {
  MatrixType symm = m.template selfadjointView<UpLo>();
  return symm.cwiseAbs().colwise().sum().maxCoeff();
}

template<typename MatrixType,template <typename,int> class CholType> void test_chol_update(const MatrixType& symm)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  MatrixType symmLo = symm.template triangularView<Lower>();
  MatrixType symmUp = symm.template triangularView<Upper>();
  MatrixType symmCpy = symm;

  CholType<MatrixType,Lower> chollo(symmLo);
  CholType<MatrixType,Upper> cholup(symmUp);

  for (int k=0; k<10; ++k)
  {
    VectorType vec = VectorType::Random(symm.rows());
    RealScalar sigma = internal::random<RealScalar>();
    symmCpy += sigma * vec * vec.adjoint();

    // we are doing some downdates, so it might be the case that the matrix is not SPD anymore
    CholType<MatrixType,Lower> chol(symmCpy);
    if(chol.info()!=Success)
      break;

    chollo.rankUpdate(vec, sigma);
    VERIFY_IS_APPROX(symmCpy, chollo.reconstructedMatrix());

    cholup.rankUpdate(vec, sigma);
    VERIFY_IS_APPROX(symmCpy, cholup.reconstructedMatrix());
  }
}

template<typename MatrixType> void cholesky(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  /* this test covers the following files:
     LLT.h LDLT.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  MatrixType a0 = MatrixType::Random(rows,cols);
  VectorType vecB = VectorType::Random(rows), vecX(rows);
  MatrixType matB = MatrixType::Random(rows,cols), matX(rows,cols);
  SquareMatrixType symm =  a0 * a0.adjoint();
  // let's make sure the matrix is not singular or near singular
  for (int k=0; k<3; ++k)
  {
    MatrixType a1 = MatrixType::Random(rows,cols);
    symm += a1 * a1.adjoint();
  }

  {
    SquareMatrixType symmUp = symm.template triangularView<Upper>();
    SquareMatrixType symmLo = symm.template triangularView<Lower>();

    LLT<SquareMatrixType,Lower> chollo(symmLo);
    VERIFY_IS_APPROX(symm, chollo.reconstructedMatrix());
    vecX = chollo.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = chollo.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    const MatrixType symmLo_inverse = chollo.solve(MatrixType::Identity(rows,cols));
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm<MatrixType, Lower>(symmLo)) /
                             matrix_l1_norm<MatrixType, Lower>(symmLo_inverse);
    RealScalar rcond_est = chollo.rcond();
    // Verify that the estimated condition number is within a factor of 10 of the
    // truth.
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);

    // test the upper mode
    LLT<SquareMatrixType,Upper> cholup(symmUp);
    VERIFY_IS_APPROX(symm, cholup.reconstructedMatrix());
    vecX = cholup.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = cholup.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    // Verify that the estimated condition number is within a factor of 10 of the
    // truth.
    const MatrixType symmUp_inverse = cholup.solve(MatrixType::Identity(rows,cols));
    rcond = (RealScalar(1) / matrix_l1_norm<MatrixType, Upper>(symmUp)) /
                             matrix_l1_norm<MatrixType, Upper>(symmUp_inverse);
    rcond_est = cholup.rcond();
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);


    MatrixType neg = -symmLo;
    chollo.compute(neg);
    VERIFY(chollo.info()==NumericalIssue);

    VERIFY_IS_APPROX(MatrixType(chollo.matrixL().transpose().conjugate()), MatrixType(chollo.matrixU()));
    VERIFY_IS_APPROX(MatrixType(chollo.matrixU().transpose().conjugate()), MatrixType(chollo.matrixL()));
    VERIFY_IS_APPROX(MatrixType(cholup.matrixL().transpose().conjugate()), MatrixType(cholup.matrixU()));
    VERIFY_IS_APPROX(MatrixType(cholup.matrixU().transpose().conjugate()), MatrixType(cholup.matrixL()));

    // test some special use cases of SelfCwiseBinaryOp:
    MatrixType m1 = MatrixType::Random(rows,cols), m2(rows,cols);
    m2 = m1;
    m2 += symmLo.template selfadjointView<Lower>().llt().solve(matB);
    VERIFY_IS_APPROX(m2, m1 + symmLo.template selfadjointView<Lower>().llt().solve(matB));
    m2 = m1;
    m2 -= symmLo.template selfadjointView<Lower>().llt().solve(matB);
    VERIFY_IS_APPROX(m2, m1 - symmLo.template selfadjointView<Lower>().llt().solve(matB));
    m2 = m1;
    m2.noalias() += symmLo.template selfadjointView<Lower>().llt().solve(matB);
    VERIFY_IS_APPROX(m2, m1 + symmLo.template selfadjointView<Lower>().llt().solve(matB));
    m2 = m1;
    m2.noalias() -= symmLo.template selfadjointView<Lower>().llt().solve(matB);
    VERIFY_IS_APPROX(m2, m1 - symmLo.template selfadjointView<Lower>().llt().solve(matB));
  }

  // LDLT
  {
    int sign = internal::random<int>()%2 ? 1 : -1;

    if(sign == -1)
    {
      symm = -symm; // test a negative matrix
    }

    SquareMatrixType symmUp = symm.template triangularView<Upper>();
    SquareMatrixType symmLo = symm.template triangularView<Lower>();

    LDLT<SquareMatrixType,Lower> ldltlo(symmLo);
    VERIFY_IS_APPROX(symm, ldltlo.reconstructedMatrix());
    vecX = ldltlo.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = ldltlo.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    const MatrixType symmLo_inverse = ldltlo.solve(MatrixType::Identity(rows,cols));
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm<MatrixType, Lower>(symmLo)) /
                             matrix_l1_norm<MatrixType, Lower>(symmLo_inverse);
    RealScalar rcond_est = ldltlo.rcond();
    // Verify that the estimated condition number is within a factor of 10 of the
    // truth.
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);


    LDLT<SquareMatrixType,Upper> ldltup(symmUp);
    VERIFY_IS_APPROX(symm, ldltup.reconstructedMatrix());
    vecX = ldltup.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = ldltup.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    // Verify that the estimated condition number is within a factor of 10 of the
    // truth.
    const MatrixType symmUp_inverse = ldltup.solve(MatrixType::Identity(rows,cols));
    rcond = (RealScalar(1) / matrix_l1_norm<MatrixType, Upper>(symmUp)) /
                             matrix_l1_norm<MatrixType, Upper>(symmUp_inverse);
    rcond_est = ldltup.rcond();
    VERIFY(rcond_est > rcond / 10 && rcond_est < rcond * 10);

    VERIFY_IS_APPROX(MatrixType(ldltlo.matrixL().transpose().conjugate()), MatrixType(ldltlo.matrixU()));
    VERIFY_IS_APPROX(MatrixType(ldltlo.matrixU().transpose().conjugate()), MatrixType(ldltlo.matrixL()));
    VERIFY_IS_APPROX(MatrixType(ldltup.matrixL().transpose().