// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARTIALLU_H
#define EIGEN_PARTIALLU_H

namespace Eigen {

namespace internal {
template<typename _MatrixType> struct traits<PartialPivLU<_MatrixType> >
 : traits<_MatrixType>
{
  typedef MatrixXpr XprKind;
  typedef SolverStorage StorageKind;
  typedef traits<_MatrixType> BaseTraits;
  enum {
    Flags = BaseTraits::Flags & RowMajorBit,
    CoeffReadCost = Dynamic
  };
};

template<typename T,typename Derived>
struct enable_if_ref;
// {
//   typedef Derived type;
// };

template<typename T,typename Derived>
struct enable_if_ref<Ref<T>,Derived> {
  typedef Derived type;
};

} // end namespace internal

/** \ingroup LU_Module
  *
  * \class PartialPivLU
  *
  * \brief LU decomposition of a matrix with partial pivoting, and related features
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the LU decomposition
  *
  * This class represents a LU decomposition of a \b square \b invertible matrix, with partial pivoting: the matrix A
  * is decomposed as A = PLU where L is unit-lower-triangular, U is upper-triangular, and P
  * is a permutation matrix.
  *
  * Typically, partial pivoting LU decomposition is only considered numerically stable for square invertible
  * matrices. Thus LAPACK's dgesv and dgesvx require the matrix to be square and invertible. The present class
  * does the same. It will assert that the matrix is square, but it won't (actually it can't) check that the
  * matrix is invertible: it is your task to check that you only use this decomposition on invertible matrices.
  *
  * The guaranteed safe alternative, working for all matrices, is the full pivoting LU decomposition, provided
  * by class FullPivLU.
  *
  * This is \b not a rank-revealing LU decomposition. Many features are intentionally absent from this class,
  * such as rank computation. If you need these features, use class FullPivLU.
  *
  * This LU decomposition is suitable to invert invertible matrices. It is what MatrixBase::inverse() uses
  * in the general case.
  * On the other hand, it is \b not suitable to determine whether a given matrix is invertible.
  *
  * The data of the LU decomposition can be directly accessed through the methods matrixLU(), permutationP().
  *
  * This class supports the \link InplaceDecomposition inplace decomposition \endlink mechanism.
  * 
  * \sa MatrixBase::partialPivLu(), MatrixBase::determinant(), MatrixBase::inverse(), MatrixBase::computeInverse(), class FullPivLU
  */
template<typename _MatrixType> class PartialPivLU
  : public SolverBase<PartialPivLU<_MatrixType> >
{
  public:

    typedef _MatrixType MatrixType;
    typedef SolverBase<PartialPivLU> Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(PartialPivLU)
    // FIXME StorageIndex defined in EIGEN_GENERIC_PUBLIC_INTERFACE should be int
    enum {
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef PermutationMatrix<RowsAtCompileTime, MaxRowsAtCompileTime> PermutationType;
    typedef Transpositions<RowsAtCompileTime, MaxRowsAtCompileTime> TranspositionType;
    typedef typename MatrixType::PlainObject PlainObject;

    /**
      * \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via PartialPivLU::compute(const MatrixType&).
      */
    PartialPivLU();

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa PartialPivLU()
      */
    explicit PartialPivLU(Index size);

    /** Constructor.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class FullPivLU instead.
      */
    template<typename InputType>
    explicit PartialPivLU(const EigenBase<InputType>& matrix);

    /** Constructor for \link InplaceDecomposition inplace decomposition \endlink
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class FullPivLU instead.
      */
    template<typename InputType>
    explicit PartialPivLU(EigenBase<InputType>& matrix);

    template<typename InputType>
    PartialPivLU& compute(const EigenBase<InputType>& matrix) {
      m_lu = matrix.derived();
      compute();
      return *this;
    }

    /** \returns the LU decomposition matrix: the upper-triangular part is U, the
      * unit-lower-triangular part is L (at least for square matrices; in the non-square
      * case, special care is needed, see the documentation of class FullPivLU).
      *
      * \sa matrixL(), matrixU()
      */
    inline const MatrixType& matrixLU() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return m_lu;
    }

    /** \returns the permutation matrix P.
      */
    inline const PermutationType& permutationP() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return m_p;
    }

    /** This method returns the solution x to the equation Ax=b, where A is the matrix of which
      * *this is the LU decomposition.
      *
      * \param b the right-hand-side of the equation to solve. Can be a vector or a matrix,
      *          the only requirement in order for the equation to make sense is that
      *          b.rows()==A.rows(), where A is the matrix of which *this is the LU decomposition.
      *
      * \returns the solution.
      *
      * Example: \include PartialPivLU_solve.cpp
      * Output: \verbinclude PartialPivLU_solve.out
      *
      * Since this PartialPivLU class assumes anyway that the matrix A is invertible, the solution
      * theoretically exists and is unique regardless of b.
      *
      * \sa TriangularView::solve(), inverse(), computeInverse()
      */
    // FIXME this is a copy-paste of the base-class member to add the isInitialized assertion.
    template<typename Rhs>
    inline const Solve<PartialPivLU, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return Solve<PartialPivLU, Rhs>(*this, b.derived());
    }

    /** \returns an estimate of the reciprocal condition number of the matrix of which \c *this is
        the LU decomposition.
      */
    inline RealScalar rcond() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return internal::rcond_estimate_helper(m_l1_norm, *this);
    }

    /** \returns the inverse of the matrix of which *this is the LU decomposition.
      *
      * \warning The matrix being decomposed here is assumed to be invertible. If you need to check for
      *          invertibility, use class FullPivLU instead.
      *
      * \sa MatrixBase::inverse(), LU::inverse()
      */
    inline const Inverse<PartialPivLU> inverse() const
    {
      eigen_assert(m_isInitialized && "PartialPivLU is not initialized.");
      return Inverse<PartialPivLU>(*this);
    }

    /** \returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * \note For fixed-size matrices of size up to 4, MatrixBase::determinant() offers
      *       optimized paths.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      *
      * \sa MatrixBase::determinant()
      */
    Scalar determinant() const;

    MatrixType reconstructedMatrix() const;

    inline Index rows() const { return m_lu.rows(); }
    inline Index cols() const { return m_lu.cols(); }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl(const RhsType &rhs, DstType &dst) const {
     /* The decomposition PA = LU can be rewritten as A = P^{-1} L U.
      * So we proceed as follows:
      * Step 1: compute c = Pb.
      * Step 2: replace c by the solution x to Lx = c.
      * Step 3: replace c by the solution x to Ux = c.
      */

      eigen_assert(rhs.rows() == m_lu.rows());

      // Step 1
      dst = permutationP() * rhs;

      // Step 2
      m_lu.template triangularView<UnitLower>().solveInPlace(dst);

      // Step 3
      m_lu.template triangularView<Upper>().solveInPlace(dst);
    }

    template<bool Conjugate, typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl_transposed(const RhsType &rhs, DstType &dst) const {
     /* The decomposition PA = LU can be rewritten as A = P^{-1} L U.
      * So we proceed as follows:
      * Step 1: compute c = Pb.
      * Step 2: replace c by the solution x to Lx = c.
      * Step 3: replace c by the solution x to Ux = c.
      */

      eigen_assert(rhs.rows() == m_lu.cols());

      if (Conjugate) {
        // Step 1
        dst = m_lu.template triangularView<Upper>().adjoint().solve(rhs);
        // Step 2
        m_lu.template triangularView<UnitLower>().adjoint().solveInPlace(dst);
  