// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_REAL_QZ_H
#define EIGEN_REAL_QZ_H

namespace Eigen {

  /** \eigenvalues_module \ingroup Eigenvalues_Module
   *
   *
   * \class RealQZ
   *
   * \brief Performs a real QZ decomposition of a pair of square matrices
   *
   * \tparam _MatrixType the type of the matrix of which we are computing the
   * real QZ decomposition; this is expected to be an instantiation of the
   * Matrix class template.
   *
   * Given a real square matrices A and B, this class computes the real QZ
   * decomposition: \f$ A = Q S Z \f$, \f$ B = Q T Z \f$ where Q and Z are
   * real orthogonal matrixes, T is upper-triangular matrix, and S is upper
   * quasi-triangular matrix. An orthogonal matrix is a matrix whose
   * inverse is equal to its transpose, \f$ U^{-1} = U^T \f$. A quasi-triangular
   * matrix is a block-triangular matrix whose diagonal consists of 1-by-1
   * blocks and 2-by-2 blocks where further reduction is impossible due to
   * complex eigenvalues. 
   *
   * The eigenvalues of the pencil \f$ A - z B \f$ can be obtained from
   * 1x1 and 2x2 blocks on the diagonals of S and T.
   *
   * Call the function compute() to compute the real QZ decomposition of a
   * given pair of matrices. Alternatively, you can use the 
   * RealQZ(const MatrixType& B, const MatrixType& B, bool computeQZ)
   * constructor which computes the real QZ decomposition at construction
   * time. Once the decomposition is computed, you can use the matrixS(),
   * matrixT(), matrixQ() and matrixZ() functions to retrieve the matrices
   * S, T, Q and Z in the decomposition. If computeQZ==false, some time
   * is saved by not computing matrices Q and Z.
   *
   * Example: \include RealQZ_compute.cpp
   * Output: \include RealQZ_compute.out
   *
   * \note The implementation is based on the algorithm in "Matrix Computations"
   * by Gene H. Golub and Charles F. Van Loan, and a paper "An algorithm for
   * generalized eigenvalue problems" by C.B.Moler and G.W.Stewart.
   *
   * \sa class RealSchur, class ComplexSchur, class EigenSolver, class ComplexEigenSolver
   */

  template<typename _MatrixType> class RealQZ
  {
    public:
      typedef _MatrixType MatrixType;
      enum {
        RowsAtCompileTime = MatrixType::RowsAtCompileTime,
        ColsAtCompileTime = MatrixType::ColsAtCompileTime,
        Options = MatrixType::Options,
        MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
      };
      typedef typename MatrixType::Scalar Scalar;
      typedef std::complex<typename NumTraits<Scalar>::Real> ComplexScalar;
      typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3

      typedef Matrix<ComplexScalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> EigenvalueType;
      typedef Matrix<Scalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> ColumnVectorType;

      /** \brief Default constructor.
       *
       * \param [in] size  Positive integer, size of the matrix whose QZ decomposition will be computed.
       *
       * The default constructor is useful in cases in which the user intends to
       * perform decompositions via compute().  The \p size parameter is only
       * used as a hint. It is not an error to give a wrong \p size, but it may
       * impair performance.
       *
       * \sa compute() for an example.
     