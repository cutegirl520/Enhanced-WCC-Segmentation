// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INCOMPLETE_CHOlESKY_H
#define EIGEN_INCOMPLETE_CHOlESKY_H

#include <vector>
#include <list>

namespace Eigen {  
/** 
  * \brief Modified Incomplete Cholesky with dual threshold
  *
  * References : C-J. Lin and J. J. Moré, Incomplete Cholesky Factorizations with
  *              Limited memory, SIAM J. Sci. Comput.  21(1), pp. 24-45, 1999
  *
  * \tparam Scalar the scalar type of the input matrices
  * \tparam _UpLo The triangular part that will be used for the computations. It can be Lower
    *               or Upper. Default is Lower.
  * \tparam _OrderingType The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<int>,
  *                       unless EIGEN_MPL2_ONLY is defined, in which case the default is NaturalOrdering<int>.
  *
  * \implsparsesolverconcept
  *
  * It performs the following incomplete factorization: \f$ S P A P' S \approx L L' \f$
  * where L is a lower triangular factor, S is a diagonal scaling matrix, and P is a
  * fill-in reducing permutation as computed by the ordering method.
  *
  * \b Shifting \b strategy: Let \f$ B = S P A P' S \f$  be the scaled matrix on which the factorization is carried out,
  * and \f$ \beta \f$ be the minimum value of the diagonal. If \f$ \beta > 0 \f$ then, the factorization is directly performed
  * on the matrix B. Otherwise, the factorization is performed on the shifted matrix \f$ B + (\sigma+|\beta| I \f$ where
  * \f$ \sigma \f$ is the initial shift value as returned and set by setInitialShift() method. The default value is \f$ \sigma = 10^{-3} \f$.
  * If the factorization fails, then the shift in doubled until it succeed or a maximum of ten attempts. If it still fails, as returned by
  * the info() method, then you can either increase the initial shift, or better use another preconditioning technique.
  *
  */
template <typename Scalar, int _UpLo = Lower, typename _OrderingType =
#ifndef EIGEN_MPL2_ONLY
AMDOrdering<int>
#else
NaturalOrdering<int>
#endif
>
class IncompleteCholesky : public SparseSolverBase<IncompleteCholesky<Scalar,_UpLo,_OrderingType> >
{
  protected:
    typedef SparseSolverBase<IncompleteCholesky<Scalar,_UpLo,_OrderingType> > Base;
    using Base::m_isInitialized;
  public:
    typedef typename NumTraits<Scalar>::Real RealScalar; 
    typedef _OrderingType OrderingType;
    typedef typename OrderingType::PermutationType PermutationType;
    typedef typename PermutationType::StorageIndex StorageIndex; 
    typedef SparseMatrix<Scalar,ColMajor,StorageIndex> FactorType;
    typedef Matrix<Scalar,Dynamic,1> VectorSx;
    typedef Matrix<RealScalar,Dynamic,1> VectorRx;
    typedef Matrix<StorageIndex,Dynamic, 1> VectorIx;
    typedef std::vector<std::list<StorageIndex> > VectorList; 
    enum { UpLo = _UpLo };
    enum {
      ColsAtCompileTime = Dynamic,
      MaxColsAtCompileTime = Dynamic
    };
  public:

    /** Default constructor leaving the object in a partly non-initialized stage.
      *
      * You must call compute() or the pair analyzePattern()/factorize() to make it valid.
      *
      * \sa IncompleteCholesky(const MatrixType&)
      */
    IncompleteCholesky() : m_initialShift(1e-3),m_factorizationIsOk(false) {}
    
    /** Constructor computing the incomplete factorization for the given matrix \a matrix.
      */
    template<typename MatrixType>
    IncompleteCholesky(const MatrixType& matrix) : m_initialShift(1e-3),m_factorizationIsOk(false)
    {
      compute(matrix);
    }
    
    /** \returns number of rows of the factored matrix */
    Index rows() const { return m_L.rows(); }
    
    /** \returns number of columns of the factored matrix */
    Index cols() const { return m_L.cols(); }
    

    /** \brief Reports whether previous computation was successful.
      *
      * It triggers an assertion if \c *this has not been initialized through the respective constructor,
      * or a call to compute() or analyzePattern().
      *
      * \returns \c Success if computation was successful,
      *          \c Numer