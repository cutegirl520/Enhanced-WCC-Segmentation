// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_EIGENSOLVER_H
#define EIGEN_EIGENSOLVER_H

#include "./RealSchur.h"

namespace Eigen { 

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \class EigenSolver
  *
  * \brief Computes eigenvalues and eigenvectors of general matrices
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the
  * eigendecomposition; this is expected to be an instantiation of the Matrix
  * class template. Currently, only real matrices are supported.
  *
  * The eigenvalues and eigenvectors of a matrix \f$ A \f$ are scalars
  * \f$ \lambda \f$ and vectors \f$ v \f$ such that \f$ Av = \lambda v \f$.  If
  * \f$ D \f$ is a diagonal matrix with the eigenvalues on the diagonal, and
  * \f$ V \f$ is a matrix with the eigenvectors as its columns, then \f$ A V =
  * V D \f$. The matrix \f$ V \f$ is almost always invertible, in which case we
  * have \f$ A = V D V^{-1} \f$. This is called the eigendecomposition.
  *
  * The eigenvalues and eigenvectors of a matrix may be complex, even when the
  * matrix is real. However, we can choose real matrices \f$ V \f$ and \f$ D
  * \f$ satisfying \f$ A V = V D \f$, just like the eigendecomposition, if the
  * matrix \f$ D \f$ is not required to be diagonal, but if it is allowed to
  * have blocks of the form
  * \f[ \begin{bmatrix} u & v \\ -v & u \end{bmatrix} \f]
  * (where \f$ u \f$ and \f$ v \f$ are real numbers) on the diagonal.  These
  * blocks correspond to complex eigenvalue pairs \f$ u \pm iv \f$. We call
  * this variant of the eigendecomposition the pseudo-eigendecomposition.
  *
  * Call the function compute() to compute the eigenvalues and eigenvectors of
  * a given matrix. Alternatively, you can use the 
  * EigenSolver(const MatrixType&, bool) constructor which computes the
  * eigenvalues and eigenvectors at construction time. Once the eigenvalue and
  * eigenvectors are computed, they can be retrieved with the eigenvalues() and
  * eigenvectors() functions. The pseudoEigenvalueMatrix() and
  * pseudoEigenvectors() methods allow the construction of the
  * pseudo-eigendecomposition.
  *
  * The documentation for EigenSolver(const MatrixType&, bool) contains an
  * example of the typical use of this class.
  *
  * \note The implementation is adapted from
  * <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a> (public domain).
  * Their code is based on EISPACK.
  *
  * \sa MatrixBase::eigenvalues(), class ComplexEigenSolver, class SelfAdjointEigenSolver
  */
template<typename _MatrixType> class EigenSolver
{
  public:

    /** \brief Synonym for the template parameter \p _MatrixType. */
    typedef _MatrixType MatrixType;

    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    /** \brief Scalar type for matrices of type #MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3

    /** \brief Complex scalar type for #MatrixType. 
      *
      * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
      * \c float or \c double) and just \c Scalar if #Scalar is
      * complex.
      */
    typedef std::complex<RealScalar> ComplexScalar;

    /** \brief Type for vector of eigenvalues as returned by eigenvalues(). 
      *
      * This is a column vector with entries of type #ComplexScalar.
      * The length of the vector is the size of #MatrixType.
      */
    typedef Matrix<ComplexScalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> EigenvalueType;

    /** \brief Type for matrix of eigenvectors as returned by eigenvectors(). 
      *
      * This is a square matrix with entries of type #ComplexScalar. 
      * The size is the same as the size of #MatrixType.
      */
    typedef Matrix<ComplexScalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> EigenvectorsType;

    /** \brief Default constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via EigenSolver::compute(const MatrixType&, bool).
      *
      * \sa compute() for an example.
      */
    EigenSolver() : m_eivec(), m_eivalues(), m_i