// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011 Timothy E. Holy <tim.holy@gmail.com >
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_LDLT_H
#define EIGEN_LDLT_H

namespace Eigen {

namespace internal {
  template<typename MatrixType, int UpLo> struct LDLT_Traits;

  // PositiveSemiDef means positive semi-definite and non-zero; same for NegativeSemiDef
  enum SignMatrix { PositiveSemiDef, NegativeSemiDef, ZeroSign, Indefinite };
}

/** \ingroup Cholesky_Module
  *
  * \class LDLT
  *
  * \brief Robust Cholesky decomposition of a matrix with pivoting
  *
  * \tparam _MatrixType the type of the matrix of which to compute the LDL^T Cholesky decomposition
  * \tparam _UpLo the triangular part that will be used for the decompositon: Lower (default) or Upper.
  *             The other triangular part won't be read.
  *
  * Perform a robust Cholesky decomposition of a positive semidefinite or negative semidefinite
  * matrix \f$ A \f$ such that \f$ A =  P^TLDL^*P \f$, where P is a permutation matrix, L
  * is lower triangular with a unit diagonal and D is a diagonal matrix.
  *
  * The decomposition uses pivoting to ensure stability, so that L will have
  * zeros in the bottom right rank(A) - n submatrix. Avoiding the square root
  * on D also stabilizes the computation.
  *
  * Remember that Cholesky decompositions are not rank-revealing. Also, do not use a Cholesky
  * decomposition to determine whether a system of equations has a solution.
  *
  * This class supports the \link InplaceDecomposition inplace decomposition \endlink mechanism.
  * 
  * \sa MatrixBase::ldlt(), SelfAdjointView::ldlt(), class LLT
  */
template<typename _MatrixType, int _UpLo> class LDLT
{
  public:
    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
      UpLo = _UpLo
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef Matrix<Scalar, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime, 1> TmpMatrixType;

    typedef Transpositions<RowsAtCompileTime, MaxRowsAtCompileTime> TranspositionType;
    typedef PermutationMatrix<RowsAtCompileTime, MaxRowsAtCompileTime> PermutationType;

    typedef internal::LDLT_Traits<MatrixType,UpLo> Traits;

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via LDLT::compute(const MatrixType&).
      */
    LDLT()
      : m_matrix(),
        m_transpositions(),
        m_sign(internal::ZeroSign),
        m_isInitialized(false)
    {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa LDLT()
      */
    explicit LDLT(Index size)
      : m_matrix(size, size),
        m_transpositions(size),
        m_temporary(size),
        m_sign(internal::ZeroSign),
        m_isInitialized(false)
    {}

    /** \brief Constructor with decomposition
      *
      * This calculates the decomposition for the input \a matrix.
      *
      * \sa LDLT(Index size)
      */
    template<typename InputType>
    explicit LDLT(const EigenBase<InputType>& matrix)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_transpositions(matrix.rows()),
        m_temporary(matrix.rows()),
        m_sign(internal::ZeroSign),
        m_isInitialized(false)
    {
      compute(matrix.derived());
    }

    /** \brief Constructs a LDLT factorization from a given matrix
      *
      * This overloaded constructor is provided for \link InplaceDecomposition inplace decomposition \endlink when \c MatrixType is a Eigen::Ref.
      *
      * \sa LDLT(const EigenBase&)
      */
    template<typename InputType>
    explicit LDLT(EigenBase<InputType>& matrix)
      : m_matrix(matrix.derived()),
        m_transpositions(matrix.rows()),
        m_temporary(matrix.rows()),
        m_sign(internal::ZeroSign),
        m_isInitialized(false)
    {
      compute(matrix.derived());
    }

    /** Clear any existing decomposition
     * \sa rankUpdate(w,sigma)
     */
    void setZero()
    {
      m_isInitialized = false;
    }

    /** \returns a view of the upper triangular matrix U */
    inline typename Traits::MatrixU matrixU() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return Traits::getU(m_matrix);
    }

    /** \returns a view of the lower triangular matrix L */
    inline typename Traits::MatrixL matrixL() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return Traits::getL(m_matrix);
    }

    /** \returns the permutation matrix P as a transposition sequence.
      */
    inline const TranspositionType& transpositionsP() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_transpositions;
    }

    /** \returns the coefficients of the diagonal matrix D */
    inline Diagonal<const MatrixType> vectorD() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix.diagonal();
    }

    /** \returns true if the matrix is positive (semidefinite) */
    inline bool isPositive() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_sign == internal::PositiveSemiDef || m_sign == internal::ZeroSign;
    }

    /** \returns true if the matrix is negative (semidefinite) */
    inline bool isNegative(void) const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_sign == internal::NegativeSemiDef || m_sign == internal::ZeroSign;
    }

    /** \returns a solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * This function also supports in-place solves using the syntax <tt>x = decompositionObject.solve(x)</tt> .
      *
      * \note_about_checking_solutions
      *
      * More precisely, this method solves \f$ A x = b \f$ using the decomposition \f$ A = P^T L D L^* P \f$
      * by solving the systems \f$ P^T y_1 = b \f$, \f$ L y_2 = y_1 \f$, \f$ D y_3 = y_2 \f$,
      * \f$ L^* y_4 = y_3 \f$ and \f$ P x = y_4 \f$ in succession. If the matrix \f$ A \f$ is singular, then
      * \f$ D \f$ will also be singular (all the other matrices are invertible). In that case, the
      * least-square solution of \f$ D y_3 = y_2 \f$ is computed. This does not mean that this function
      * computes the least-square solution of \f$ A x = b \f$ is \f$ A \f$ is singular.
      *
      * \sa MatrixBase::ldlt(), SelfAdjointView::ldlt()
      */
    template<typename Rhs>
    inline const Solve<LDLT, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      eigen_assert(m_matrix.rows()==b.rows()
                && "LDLT::solve(): invalid number of rows of the right hand side matrix b");
      return Solve<LDLT, Rhs>(*this, b.derived());
    }

    template<typename Derived>
    bool solveInPlace(MatrixBase<Derived> &bAndX) const;

    template<typename InputType>
    LDLT& compute(const EigenBase<InputType>& matrix);

    /** \returns an estimate of the reciprocal condition number of the matrix of
     *  which \c *this is the LDLT decomposition.
     */
    RealScalar rcond() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return internal::rcond_estimate_helper(m_l1_norm, *this);
    }

    template <typename Derived>
    LDLT& rankUpdate(const MatrixBase<Derived>& w, const RealScalar& alpha=1);

    /** \returns the internal LDLT decomposition matrix
      *
      * TODO: document the storage layout
      */
    inline const MatrixType& matrixLDLT() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix;
    }

    MatrixType reconstructedMatrix() const;

    /** \returns the adjoint of \c *this, that is, a const reference to the decomposition itself as the underlying matrix is self-adjoint.
      *
      * This method is provided for compatibility with other matrix decompositions, thus enabling generic code such as:
      * \code x = decomposition.adjoint().solve(b) \endcode
      */
    const LDLT& adjoint() const { return *this; };

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return Success;
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl(const RhsType &rhs, DstType &dst) const;
    #endif

  protected:

    static void check_template_parameters()
    {
      EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar);
    }

    /** \internal
      * Used to compute and store the Cholesky decomposition A = L D L^* = U^* D U.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
    RealScalar m_l1_norm;
    TranspositionType m_transpositions;
    TmpMatrixType m_temporary;
    internal::SignMatrix m_sign;
    bool m_isInitialized;
};