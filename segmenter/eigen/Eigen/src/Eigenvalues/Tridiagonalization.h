// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRIDIAGONALIZATION_H
#define EIGEN_TRIDIAGONALIZATION_H

namespace Eigen { 

namespace internal {
  
template<typename MatrixType> struct TridiagonalizationMatrixTReturnType;
template<typename MatrixType>
struct traits<TridiagonalizationMatrixTReturnType<MatrixType> >
  : public traits<typename MatrixType::PlainObject>
{
  typedef typename MatrixType::PlainObject ReturnType; // FIXME shall it be a BandMatrix?
  enum { Flags = 0 };
};

template<typename MatrixType, typename CoeffVectorType>
void tridiagonalization_inplace(MatrixType& matA, CoeffVectorType& hCoeffs);
}

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \class Tridiagonalization
  *
  * \brief Tridiagonal decomposition of a selfadjoint matrix
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the
  * tridiagonal decomposition; this is expected to be an instantiation of the
  * Matrix class template.
  *
  * This class performs a tridiagonal decomposition of a selfadjoint matrix \f$ A \f$ such that:
  * \f$ A = Q T Q^* \f$ where \f$ Q \f$ is unitary and \f$ T \f$ a real symmetric tridiagonal matrix.
  *
  * A tridiagonal matrix is a matrix which has nonzero elements only on the
  * main diagonal and the first diagonal below and above it. The Hessenberg
  * decomposition of a selfadjoint matrix is in fact a tridiagonal
  * decomposition. This class is used in SelfAdjointEigenSolver to compute the
  * eigenvalues and eigenvectors of a selfadjoint matrix.
  *
  * Call the function compute() to compute the tridiagonal decomposition of a
  * given matrix. Alternatively, you can use the Tridiagonalization(const MatrixType&)
  * constructor which computes the tridiagonal Schur decomposition at
  * construction time. Once the decomposition is computed, you can use the
  * matrixQ() and matrixT() functions to retrieve the matrices Q and T in the
  * decomposition.
  *
  * The documentation of Tridiagonalization(const MatrixType&) contains an
  * example of the typical use of this class.
  *
  * \sa class HessenbergDecomposition, class SelfAdjointEigenSolver
  */
template<typename _MatrixType> class Tridiagonalization
{
  public:

    /** \brief Synonym for the template parameter \p _MatrixType. */
    typedef _MatrixType MatrixType;

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3

    enum {
      Size = MatrixType::RowsAtCompileTime,
      SizeMinusOne = Size == Dynamic ? Dynamic : (Size > 1 ? Size - 1 : 1),
      Options = MatrixType::Options,
      MaxSize = MatrixType::MaxRowsAtCompileTime,
      MaxSizeMinusOne = MaxSize == Dynamic ? Dynamic : (MaxSize > 1 ? MaxSize - 1 : 1)
    };

    typedef Matrix<Scalar, SizeMinusOne, 1, Options & ~RowMajor, MaxSizeMinusOne, 1> CoeffVectorType;
    typedef typename internal::plain_col_type<MatrixType, RealScalar>::type DiagonalType;
    typedef Matrix<RealScalar, SizeMinusOne, 1, Options & ~RowMajor, MaxSizeMinusOne, 1> SubDiagonalType;
    typedef typename internal::remove_all<typename MatrixType::RealReturnType>::type MatrixTypeRealView;
    typedef internal::TridiagonalizationMatrixTReturnType<MatrixTypeRealView> MatrixTReturnType;

    typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
              typename internal::add_const_on_value_type<typename Diagonal<const MatrixType>::RealReturnType>::type,
              const Diagonal<const MatrixType>
            >::type DiagonalReturnType;

    typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
              typename internal::add_const_on_value_type<typename Diagonal<const MatrixType, -1>::RealReturnType>::type,
              const Diagonal<const MatrixType, -1>
            >::type SubDiagonalReturnType;

    /** \brief Return type of matrixQ() */
    typedef HouseholderSequence<MatrixType,typename internal::remove_all<typename CoeffVectorType::ConjugateReturnType>::type> HouseholderSequenceType;

    /** \brief Default constructor.
      *
      * \param [in]  size  Positive integer, size of the matrix whose tridiagonal
      * decomposition will be computed.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute().  The \p size parameter is only
      * used as a hint. It is not an error to give a wrong \p size, but it may
      * impair performance.
      *
      * \sa compute() for an example.
      */
    explicit Tridiagonalization(Index size = Size==Dynamic ? 2 : Size)
      : m_matrix(size,size),
        m_hCoeffs(size > 1 ? size-1 : 1),
        m_isInitialized(false)
    {}

    /** \brief Constructor; computes tridiagonal decomposition of given matrix.
      *
      * \param[in]  matrix  Selfadjoint matrix whose tridiagonal decomposition
      * is to be computed.
      *
      * This constructor calls compute() to compute the tridiagonal decomposition.
      *
      * Example: \include Tridiagonalization_Tridiagonalization_MatrixType.cpp
      * Output: \verbinclude Tridiagonalization_Tridiagonalization_MatrixType.out
      */
    template<typename InputType>
    explicit Tridiagonalization(const EigenBase<InputType>& matrix)
      : m_matrix(matrix.derived()),
        m_hCoeffs(matrix.cols() > 1 ? matrix.cols()-1 : 1),
        m_isInitialized(false)
    {
      internal::tridiagonalization_inplace(m_matrix, m_hCoeffs);
      m_isInitialized = true;
    }

    /** \brief Computes tridiagonal decomposition of given matrix.
      *
      * \param[in]  matrix  Selfadjoint matrix whose tridiagonal decomposition
      * is to be computed.
      * \returns    Reference to \c *this
      *
      * The tridiagonal decomposition is computed by bringing the columns of
      * the matrix successively in the required form using Householder
      * reflections. The cost is \f$ 4n^3/3 \f$ flops, where \f$ n \f$ denotes
      * the size of the given matrix.
      *
      * This method reuses of the allocated data in the Tridiagonalization
      * object, if the size of the matrix does not change.
      *
      * Example: \include Tridiagonalization_compute.cpp
      * Output: \verbinclude Tridiagonalization_compute.out
      */
    template<typename InputType>
    Tridiagonalization& compute(const EigenBase<InputType>& matrix)
    {
      m_matrix = matrix.derived();
      m_hCoeffs.resize(matrix.rows()-1, 1);
      internal::tridiagonalization_inplace(m_matrix, m_hCoeffs);
      m_isInitialized = true;
      return *this;
    }

    /** \brief Returns the Householder coefficients.
      *
      * \returns a const reference to the vector of Householder coefficients
      *
      * \pre Either the constructor Tridiagonalization(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called before
      * to compute the tridiagonal decomposition of a matrix.
      *
      * The Householder coefficients allow the reconstruction of the matrix
      * \f$ Q \f$ in the tridiagonal decomposition from the packed data.
      *
      * Example: \include Tridiagonalization_householderCoefficients.cpp
      * Output: \verbinclude Tridiagonalization_householderCoefficients.out
      *
      * \sa packedMatrix(), \ref Householder_Module "Householder module"
      */
    inline CoeffVectorType householderCoefficients() const
    {
      eigen_assert(m_isInitialized && "Tridiagonalization is not initialized.");
      return m_hCoeffs;
    }

    /** \brief Returns the internal representation of the decomposition
      *
      *	\returns a const reference to a matrix with the internal representation
      *	         of the decomposition.
      *
      * \pre Either the constructor Tridiagonalization(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called before
      * to compute the tridiagonal decomposition of a matrix.
      *
      * The returned matrix contains the following information:
      *  - the strict upper triangular part is equal to the input matrix A.
      *  - the diagonal and lower sub-diagonal represent the real tridiagonal
      *    symmetric matrix T.
      *  - the rest of the lower part contains the Householder vectors that,
      *    combined with Householder coefficients returned by
      *    householderCoefficients(), allows to reconstruct the matrix Q as
      *       \f$ Q = H_{N-1} \ldots H_1 H_0 \f$.
      *    Here, the matrices \f$ H_i \f$ are the Householder transformations
      *       \f$ H_i = (I - h_i v_i v_i^T) \f$
      *    where \f$ h_i \f$ is the \f$ i \f$th Householder coefficient and
      *    \f$ v_i \f$ is the Householder vector defined by
      *       \f$ v_i = [ 0, \ldots, 0, 1, M(i+2,i), \ldots, M(N-1,i) ]^T \f$
      *    with M the matrix returned by this function.
      *
      * See LAPACK for further details on this packed storage.
      *
      * Example: \include Tridiagonalization_packedMatrix.cpp
      * Output: \verbinclude Tridiagonalization_packedMatrix.out
      *
      * \sa householderCoefficients()
      */
    inline const MatrixType& packedMatrix() const
    {
      eigen_assert(m_isInitialized && "Tridiagonalization is not initialized.");
      return m_matrix;
    }

    /** \brief Returns the unitary matrix Q in the decomposition
      *
      * \returns object representing the matrix Q
      *
      * \pre Either the constructor Tridiagonalization(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called before
      * to compute the tridiagonal decomposition of a matrix.
      *
      * This function returns a light-weight object of template class
      * HouseholderSequence. You can either apply it directly to a matrix or
      * you can convert it to a matrix of type #MatrixType.
      *
      * \sa Tridiagonalization(const MatrixType&) for an example,
      *     matrixT(), class HouseholderSequence
      */
    HouseholderSequenceType matrixQ() const
    {
      eigen_assert(m_isInitialized && "Tridiagonalization is not initialized.");
      return HouseholderSequenceType(m_matrix, m_hCoeffs.conjugate())
             .setLength(m_matrix.rows() - 1)
             .setShift(1);
    }

    /** \brief Returns an expression of the tridiagonal matrix T in the decomposition
      *
      * \returns expression object representing the matrix T
      *
      * \pre Either the constructor Tridiagonalization(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called before
      * to compute the tridiagonal decomposition of a matrix.
      *
      * Currently, this function can be used to extract the matrix T from internal
      * data and copy it to a dense matrix object. In most cases, it may be
      * sufficient to directly use the packed matrix or the vector expressions
      * returned by diagonal() and subDiagonal() instead of creating a new
      * dense copy matrix with this function.
      *
      * \sa Tridiagonalization(const MatrixType&) for an example,
      * matrixQ(), packedMatrix(), diagonal(), subDiagonal()
      */
    MatrixTReturnType matrixT() const
    {
      eigen_assert(m_isInitialized && "Tridiagonalization is not initialized.");
      return MatrixTReturnType(m_matrix.real());
    }

    /** \brief Returns the diagonal of the tridiagonal matrix T in the decomposition.
      *
      * \returns expression representing the diagonal of T
      *
      * \pre Either the constructor Tridiagonalization(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called before
      * to compute the tridiagonal decomposition of a matrix.
      *
      * Example: \include Tridiagonalization_diagonal.cpp
      * Output: \verbinclude Tridiagonalization_diagonal.out
      *
      * \sa matrixT(), subDiagonal()
      */
    DiagonalReturnType diagonal() 