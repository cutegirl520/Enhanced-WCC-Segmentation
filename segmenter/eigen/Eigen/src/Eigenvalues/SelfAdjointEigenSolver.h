// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFADJOINTEIGENSOLVER_H
#define EIGEN_SELFADJOINTEIGENSOLVER_H

#include "./Tridiagonalization.h"

namespace Eigen { 

template<typename _MatrixType>
class GeneralizedSelfAdjointEigenSolver;

namespace internal {
template<typename SolverType,int Size,bool IsComplex> struct direct_selfadjoint_eigenvalues;
template<typename MatrixType, typename DiagType, typename SubDiagType>
ComputationInfo computeFromTridiagonal_impl(DiagType& diag, SubDiagType& subdiag, const Index maxIterations, bool computeEigenvectors, MatrixType& eivec);
}

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \class SelfAdjointEigenSolver
  *
  * \brief Computes eigenvalues and eigenvectors of selfadjoint matrices
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the
  * eigendecomposition; this is expected to be an instantiation of the Matrix
  * class template.
  *
  * A matrix \f$ A \f$ is selfadjoint if it equals its adjoint. For real
  * matrices, this means that the matrix is symmetric: it equals its
  * transpose. This class computes the eigenvalues and eigenvectors of a
  * selfadjoint matrix. These are the scalars \f$ \lambda \f$ and vectors
  * \f$ v \f$ such that \f$ Av = \lambda v \f$.  The eigenvalues of a
  * selfadjoint matrix are always real. If \f$ D \f$ is a diagonal matrix with
  * the eigenvalues on the diagonal, and \f$ V \f$ is a matrix with the
  * eigenvectors as its columns, then \f$ A = V D V^{-1} \f$ (for selfadjoint
  * matrices, the matrix \f$ V \f$ is always invertible). This is called the
  * eigendecomposition.
  *
  * The algorithm exploits the fact that the matrix is selfadjoint, making it
  * faster and more accurate than the general purpose eigenvalue algorithms
  * implemented in EigenSolver and ComplexEigenSolver.
  *
  * Only the \b lower \b triangular \b part of the input matrix is referenced.
  *
  * Call the function compute() to compute the eigenvalues and eigenvectors of
  * a given matrix. Alternatively, you can use the
  * SelfAdjointEigenSolver(const MatrixType&, int) constructor which computes
  * the eigenvalues and eigenvectors at construction time. Once the eigenvalue
  * and eigenvectors are computed, they can be retrieved with the eigenvalues()
  * and eigenvectors() functions.
  *
  * The documentation for SelfAdjointEigenSolver(const MatrixType&, int)
  * contains an example of the typical use of this class.
  *
  * To solve the \em generalized eigenvalue problem \f$ Av = \lambda Bv \f$ and
  * the likes, see the class GeneralizedSelfAdjointEigenSolver.
  *
  * \sa MatrixBase::eigenvalues(), class EigenSolver, class ComplexEigenSolver
  */
template<typename _MatrixType> class SelfAdjointEigenSolver
{
  public:

    typedef _MatrixType MatrixType;
    enum {
      Size = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    
    /** \brief Scalar type for matrices of type \p _MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef Eigen::Index Index; ///< \deprecated since Eigen 3.3
    
    typedef Matrix<Scalar,Size,Size,ColMajor,MaxColsAtCompileTime,MaxColsAtCompileTime> EigenvectorsType;

    /** \brief Real scalar type for \p _MatrixType.
      *
      * This is just \c Scalar if #Scalar is real (e.g., \c float or
      * \c double), and the type of the real part of \c Scalar if #Scalar is
      * complex.
      */
    typedef typename NumTraits<Scalar>::Real RealScalar;
    
    friend struct internal::direct_selfadjoint_eigenvalues<SelfAdjointEigenSolver,Size,NumTraits<Scalar>::IsComplex>;

    /** \brief Type for vector of eigenvalues as returned by eigenvalues().
      *
      * This is a column vector with entries of type #RealScalar.
      * The length of the vector is the size of \p _MatrixType.
      */
    typedef typename internal::plain_col_type<MatrixType, RealScalar>::type RealVectorType;
    typedef Tridiagonalization<MatrixType> TridiagonalizationType;
    typedef typename TridiagonalizationType::SubDiagonalType SubDiagonalType;

    /** \brief Default constructor for fixed-size matrices.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute(). This constructor
      * can only be used if \p _MatrixType is a fixed-size matrix; use
      * SelfAdjointEigenSolver(Index) for dynamic-size matrices.
      *
      * Example: \include SelfAdjointEigenSolver_SelfAdjointEigenSolver.cpp
      * Output: \verbinclude SelfAdjointEigenSolver_SelfAdjointEigenSolver.out
      */
    EIGEN_DEVICE_FUNC
    SelfAdjointEigenSolver()
        : m_eivec(),
          m_eivalues(),
          m_subdiag(),
          m_isInitialized(false)
    { }

    /** \brief Constructor, pre-allocates memory for dynamic-size matrices.
      *
      * \param [in]  size  Positive integer, size of the matrix whose
      * eigenvalues and eigenvectors will be computed.
      *
      * This constructor is useful for dynamic-size matrices, when the user
      * intends to perform decompositions via compute(). The \p size
      * parameter is only used as a hint. It is not an error to give a wrong
      * \p size, but it may impair performance.
      *
      * \sa compute() for an example
      */
    EIGEN_DEVICE_FUNC
    explicit SelfAdjointEigenSolver(Index size)
        : m_eivec(size, size),
          m_eivalues(size),
          m_subdiag(size > 1 ? size - 1 : 1),
          m_isInitialized(false)
    {}

    /** \brief Constructor; computes eigendecomposition of given matrix.
      *
      * \param[in]  matrix  Selfadjoint matrix whose eigendecomposition is to
      *    be computed. Only the lower triangular part of the matrix is referenced.
      * \param[in]  options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
      *
      * This constructor calls compute(const MatrixType&, int) to compute the
      * eigenvalues of the matrix \p matrix. The eigenvectors are computed if
      * \p options equals #ComputeEigenvectors.
      *
      * Example: \include SelfAdjointEigenSolver_SelfAdjointEigenSolver_MatrixType.cpp
      * Output: \verbinclude SelfAdjointEigenSolver_SelfAdjointEigenSolver_MatrixType.out
      *
      * \sa compute(const MatrixType&, int)
      */
    template<typename InputType>
    EIGEN_DEVICE_FUNC
    explicit SelfAdjointEigenSolver(const EigenBase<InputType>& matrix, int options = ComputeEigenvectors)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_eivalues(matrix.cols()),
        m_subdiag(matrix.rows() > 1 ? matrix.rows() - 1 : 1),
        m_isInitialized(false)
    {
      compute(matrix.derived(), options);
    }

    /** \brief Computes eigendecomposition of given matrix.
      *
      * \param[in]  matrix  Selfadjoint matrix whose eigendecomposition is to
      *    be computed. Only the lower triangular part of the matrix is referenced.
      * \param[in]  options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
      * \returns    Reference to \c *this
      *
      * This function computes the eigenvalues of \p matrix.  The eigenvalues()
      * function can be used to retrieve them.  If \p options equals #ComputeEigenvectors,
      * then the eigenvectors are also computed and can be retrieved by
      * calling eigenvectors().
      *
      * This implementation uses a symmetric QR algorithm. The matrix is first
      * reduced to tridiagonal form using the Tridiagonalization class. The
      * tridiagonal matrix is then brought to diagonal form with implicit
      * symmetric QR steps with Wilkinson shift. Details can be found in
      * Section 8.3 of Golub \& Van Loan, <i>%Matrix Computations</i>.
      *
      * The cost of the computation is about \f$ 9n^3 \f$ if the eigenvectors
      * are required and \f$ 4n^3/3 \f$ if they are not required.
      *
      * This method reuses the memory in the SelfAdjointEigenSolver object that
      * was allocated when the object was constructed, if the size of the
      * matrix does not change.
      *
      * Example: \include SelfAdjointEigenSolver_compute_MatrixType.cpp
      * Output: \verbinclude SelfAdjointEigenSolver_compute_MatrixType.out
      *
      * \sa SelfAdjointEigenSolver(const MatrixType&, int)
      */
    template<typename InputType>
    EIGEN_DEVICE_FUNC
    SelfAdjointEigenSolver& compute(const EigenBase<InputType>& matrix, int options = ComputeEigenvectors);
    
    /** \brief Computes eigendecomposition of given matrix using a closed-form algorithm
      *
      * This is a variant of compute(const MatrixType&, int options) which
      * directly solves the underlying polynomial equation.
      * 
      * Currently only 2x2 and 3x3 matrices for which the sizes are known at compile time are supported (e.g., Matrix3d).
      * 
      * This method is usually significantly faster than the QR iterative algorithm
      * but it might also be less accurate. It is also worth noting that
      * for 3x3 matrices it involves trigonometric operations which are
      * not necessarily available for all scalar types.
      * 
      * For the 3x3 case, we observed the following worst case relative error regarding the eigenvalues:
      *   - double: 1e-8
      *   - float:  1e-3
      *
      * \sa compute(const MatrixType&, int options)
      */
    EIGEN_DEVICE_FUNC
    SelfAdjointEigenSolver& computeDirect(const MatrixType& matrix, int options = ComputeEigenvectors);

    /**
      *\brief Computes the eigen decomposition from a tridiagonal symmetric matrix
      *
      * \param[in] diag The vector containing the diagonal of the matrix.
      * \param[in] subdiag The subdiagonal of the matrix.
      * \param[in] options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
      * \returns Reference to \c *this
      *
      * This function assumes that the matrix has been reduced to tridiagonal form.
      *
      * \sa compute(const MatrixType&, int) for more information
      */
    SelfAdjointEigenSolver& computeFromTridiagonal(const RealVectorType& diag, const SubDiagonalType& subdiag , int options=ComputeEigenvectors);

    /** \b