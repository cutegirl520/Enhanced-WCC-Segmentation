// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_SQUARE_ROOT
#define EIGEN_MATRIX_SQUARE_ROOT

namespace Eigen { 

namespace internal {

// pre:  T.block(i,i,2,2) has complex conjugate eigenvalues
// post: sqrtT.block(i,i,2,2) is square root of T.block(i,i,2,2)
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_2x2_diagonal_block(const MatrixType& T, typename MatrixType::Index i, ResultType& sqrtT)
{
  // TODO: This case (2-by-2 blocks with complex conjugate eigenvalues) is probably hidden somewhere
  //       in EigenSolver. If we expose it, we could call it directly from here.
  typedef typename traits<MatrixType>::Scalar Scalar;
  Matrix<Scalar,2,2> block = T.template block<2,2>(i,i);
  EigenSolver<Matrix<Scalar,2,2> > es(block);
  sqrtT.template block<2,2>(i,i)
    = (es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal() * es.eigenvectors().inverse()).real();
}

// pre:  block structure of T is such that (i,j) is a 1x1 block,
//       all blocks of sqrtT to left of and below (i,j) are correct
// post: sqrtT(i,j) has the correct value
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_1x1_off_diagonal_block(const MatrixType& T, typename MatrixType::Index i, typename MatrixType::Index j, ResultType& sqrtT)
{
  typedef typename traits<MatrixType>::Scalar Scalar;
  Scalar tmp = (sqrtT.row(i).segment(i+1,j-i-1) * sqrtT.col(j).segment(i+1,j-i-1)).value();
  sqrtT.coeffRef(i,j) = (T.coeff(i,j) - tmp) / (sqrtT.coeff(i,i) + sqrtT.coeff(j,j));
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_1x2_off_diagonal_block(const MatrixType& T, typename MatrixType::Index i, typename MatrixType::Index j, ResultType& sqrtT)
{
  typedef typename traits<MatrixType>::Scalar Scalar;
  Matrix<Scalar,1,2> rhs = T.template block<1,2>(i,j);
  if (j-i > 1)
    rhs -= sqrtT.block(i, i+1, 1, j-i-1) * sqrtT.block(i+1, j, j-i-1, 2);
  Matrix<Scalar,2,2> A = sqrtT.coeff(i,i) * Matrix<Scalar,2,2>::Identity();
  A += sqrtT.template block<2,2>(j,j).transpose();
  sqrtT.template block<1,2>(i,j).transpose() = A.fullPivLu().solve(rhs.transpose());
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_2x1_off_diagonal_block(const MatrixType& T, typename MatrixType::Index i, typename MatrixType::Index j, ResultType& sqrtT)
{
  typedef typename traits<MatrixType>::Scalar Scalar;
  Matrix<Scalar,2,1> rhs = T.template block<2,1>(i,j);
  if (j-i > 2)
    rhs -= sqrtT.block(i, i+2, 2, j-i-2) * sqrtT.block(i+2, j, j-i-2, 1);
  Matrix<Scalar,2,2> A = sqrtT.coeff(j,j) * Matrix<Scalar,2,2>::Identity();
  A += sqrtT.template block<2,2>(i,i);
  sqrtT.template block<2,1>(i,j) = A.fullPivLu().solve(rhs);
}

// solves the equation A X + X B = C where all matrices are 2-by-2
template <typename MatrixType>
void matrix_sqrt_quasi_triangular_solve_auxiliary_equation(MatrixType& X, const MatrixType& A, const MatrixType& B, const MatrixType& C)
{
  typedef typename traits<MatrixType>::Scalar Scalar;
  Matrix<Scalar,4,4> coeffMatrix = Matrix<Scalar,4,4>::Zero();
  coeffMatrix.coeffRef(0,0) = A.coeff(0,0) + B.coeff(0,0);
  coeffMatrix.coeffRef(1,1) = A.coeff(0,0) + B.coeff(1,1);
  coeffMatrix.coeffRef(2,2) = A.coeff(1,1) + B.coeff(0,0);
  coeffMatrix.coeffRef(3,3) = A.coeff(1,1) + B.coeff(1,1);
  coeffMatrix.coeffRef(0,1) = B.coeff(1,0);
  coeffMatrix.coeffRef(0,2) = A.coeff(0,1);
  coeffMatrix.coeffRef(1,0) = B.coeff(0,1);
  coeffMatrix.coeffRef(1,3) = A.coeff(0,1);
  coeffMatrix.coeffRef(2,0) = A.coeff(1,0);
  coeffMatrix.coeffRef(2,3) = B.coeff(1,0);
  coeffMatrix.coeffRef(3,1) = A.coeff(1,0);
  coeffMatrix.coeffRef(3,2) = B.coeff(0,1);

  Matrix<Scalar,4,1> rhs;
  rhs.coeffRef(0) = C.coeff(0,0);
  rhs.coeffRef(1) = C.coeff(0,1);
  rhs.coeffRef(2) = C.coeff(1,0);
  rhs.coeffRef(3) = C.coeff(1,1);

  Matrix<Scalar,4,1> result;
  result = coeffMatrix.fullPivLu().solve(rhs);

  X.coeffRef(0,0) = result.coeff(0);
  X.coeffRef(0,1) = result.coeff(1);
  X.coeffRef(1,0) = result.coeff(2);
  X.coeffRef(1,1) = result.coeff(3);
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_2x2_off_diagonal_block(const MatrixType& T, typename MatrixType::Index i, typename MatrixType::Index j, ResultType& sqrtT)
{
  typedef typename traits<MatrixType>::Scalar Scalar;
  Matrix<Scalar,2,2> A = sqrtT.template block<2,2>(i,i);
  Matrix<Scalar,2,2> B = sqrtT.template block<2,2>(j,j);
  Matrix<Scalar,2,2> C = T.template block<2,2>(i,j);
  if (j-i > 2)
    C -= sqrtT.block(i, i+2, 2, j-i-2) * sqrtT.block(i+2, j, j-i-2, 2);
  Matrix<Scalar,2,2> X;
  matrix_sqrt_quasi_triangular_solve_auxiliary_equation(X, A, B, C);
  sqrtT.template block<2,2>(i,j) = X;
}

// pre:  T is quasi-upper-triangular and sqrtT is a zero matrix of the same size
// post: the diagonal blocks of sqrtT are the square roots of the diagonal blocks of T
template <typename MatrixType, typename ResultType>
void matrix_sqrt_quasi_triangular_diagonal(const MatrixType& T, ResultType& sqrtT)
{
  usi