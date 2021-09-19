// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLETEORTHOGONALDECOMPOSITION_H
#define EIGEN_COMPLETEORTHOGONALDECOMPOSITION_H

namespace Eigen {

namespace internal {
template <typename _MatrixType>
struct traits<CompleteOrthogonalDecomposition<_MatrixType> >
    : traits<_MatrixType> {
  enum { Flags = 0 };
};

}  // end namespace internal

/** \ingroup QR_Module
  *
  * \class CompleteOrthogonalDecomposition
  *
  * \brief Complete orthogonal decomposition (COD) of a matrix.
  *
  * \param MatrixType the type of the matrix of which we are computing the COD.
  *
  * This class performs a rank-revealing complete ortogonal decomposition of a
  * matrix  \b A into matrices \b P, \b Q, \b T, and \b Z such that
  * \f[
  *  \mathbf{A} \, \mathbf{P} = \mathbf{Q} \, \begin{matrix} \mathbf{T} &
  *  \mathbf{0} \\ \mathbf{0} & \mathbf{0} \end{matrix} \, \mathbf{Z}
  * \f]
  * by using Householder transformations. Here, \b P is a permutation matrix,
  * \b Q and \b Z are unitary matrices and \b T an upper triangular matrix of
  * size rank-by-rank. \b A may be rank deficient.
  *
  * This class supports the \link InplaceDecomposition inplace decomposition \endlink mechanism.
  * 
  * \sa MatrixBase::completeOrthogonalDecomposition()
  */
template <typename _MatrixType>
class CompleteOrthogonalDecomposition {
 public:
  typedef _MatrixType MatrixType;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
  typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime>
      PermutationType;
  typedef typename internal::plain_row_type<MatrixType, Index>::type
      IntRowVectorType;
  typedef typename internal::plain_row_type<MatrixType>::type RowVectorType;
  typedef typename internal::plain_row_type<MatrixType, RealScalar>::type
      RealRowVectorType;
  typedef HouseholderSequence<
      MatrixType, typename internal::remove_all<
                      typename HCoeffsType::ConjugateReturnType>::type>
      HouseholderSequenceType;
  typedef typename MatrixType::PlainObject PlainObject;

 private:
  typedef typename PermutationType::Index PermIndexType;

 public:
  /**
   * \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via
   * \c CompleteOrthogonalDecomposition::compute(const* MatrixType&).
   */
  CompleteOrthogonalDecomposition() : m_cpqr(), m_zCoeffs(), m_temp() {}

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem \a size.
   * \sa CompleteOrthogonalDecomposition()
   */
  CompleteOrthogonalDecomposition(Index rows, Index cols)
      : m_cpqr(rows, cols), m_zCoeffs((std::min)(rows, cols)), m_temp(cols) {}

  /** \brief Constructs a complete orthogonal decomposition from a given
   * matrix.
   *
   * This constructor computes the complete orthogonal decomposition of the
   * matrix \a matrix by calling the method compute(). The default
   * threshold for rank determination will be used. It is a short cut for:
   *
   * \code
   * CompleteOrthogonalDecomposition<MatrixType> cod(matrix.rows(),
   *                                                 matrix.cols());
   * cod.setThreshold(Default);
   * cod.compute(matrix);
   * \endcode
   *
   * \sa compute()
   */
  template <typename InputType>
  explicit CompleteOrthogonalDecomposition(const EigenBase<InputType>& matrix)
      : m_cpqr(matrix.rows(), matrix.cols()),
        m_zCoeffs((std::min)(matrix.rows(), matrix.cols())),
        m_temp(matrix.cols())
  {
    compute(matrix.derived());
  }

  /** \brief Constructs a complete orthogonal decomposition from a given matrix
    *
    * This overloaded constructor is provided for \link InplaceDecomposition inplace decomposition \endlink when \c MatrixType is a Eigen::Ref.
    *
    * \sa CompleteOrthogonalDecomposition(const EigenBase&)
    */
  template<typename InputType>
  explicit CompleteOrthogonalDecomposition(EigenBase<InputType>& matrix)
    : m_cpqr(matrix.derived()),
      m_zCoeffs((std::min)(matrix.rows(), matrix.cols())),
      m_temp(matrix.cols())
  {
    computeInPlace();
  }


  /** This method computes the minimum-norm solution X to a least squares
   * problem \f[\mathrm{minimize} ||A X - B|| \f], where \b A is the matrix of
   * which \c *this is the complete orthogonal decomposition.
   *
   * \param B the right-hand sides of the problem to solve.
   *
   * \returns a solution.
   *
   */
  template <typename Rhs>
  inline const Solve<CompleteOrthogonalDecomposition, Rhs> solve(
      const MatrixBase<Rhs>& b) const {
    eigen_assert(m_cpqr.m_isInitialized &&
                 "CompleteOrthogonalDecomposition is not initialized.");
    return Solve<CompleteOrthogonalDecomposition, Rhs>(*this, b.derived());
  }

  HouseholderSequenceType householderQ(void) const;
  HouseholderSequenceType matrixQ(void) const { return m_cpqr.householderQ(); }

  /** \returns the matrix \b Z.
   */
  MatrixType matrixZ() const {
    MatrixType Z = MatrixType::Identity(m_cpqr.cols(), m_cpqr.cols());
    applyZAdjointOnTheLeftInPlace(Z);
    return Z.adjoint();
  }

  /** \returns a reference to the matrix where the complete orthogonal
   * decomposition is stored
   */
  const MatrixType& matrixQTZ() const { return m_cpqr.matrixQR(); }

  /** \returns a reference to the matrix where the complete orthogonal
   * decomposition is stored.
   * \warning The strict lower part and \code cols() - rank() \endcode right
   * columns of this matrix contains internal values.
   * Only the upper triangular part should be referenced. To get it, use
   * \code matrixT().template triangularView<Upper>() \endcode
   * For rank-deficient matrices, use
   * \code
   * matrixR().topLeftCorner(rank(), rank()).template triangularView<Upper>()
   * \endcode
   */
  const MatrixType& matrixT() const { return m_cpqr.matrixQR(); }

  template <typename InputType>
  CompleteOrthogonalDecomposition& compute(const EigenBase<InputType>& matrix) {
    // Compute the column pivoted QR factorization A P = Q R.
    m_cpqr.compute(matrix);
    computeInPlace();
    return *this;
  }

  /** \returns a const reference to the column permutation matrix */
  const PermutationType& colsPermutation() const {
    return m_cpqr.colsPermutation();
  }

  /** \returns the absolute value of the determinant of the matrix of which
   * *this is the complete orthogonal decomposition. It has only linear
   * complexity (that is, O(n) where n is the dimension of the square matrix)
   * as the complete orthogonal decomposition has already been computed.
   *
   * \note This is only for square matrices.
   *
   * \warning a determinant can be very big or small, so for matrices
   * of large enough dimension, there is a risk of overflow/underflow.
   * One way to work around that i