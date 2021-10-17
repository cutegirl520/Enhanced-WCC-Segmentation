// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_SPARSE_LU_H
#define EIGEN_SPARSE_LU_H

namespace Eigen {

template <typename _MatrixType, typename _OrderingType = COLAMDOrdering<typename _MatrixType::StorageIndex> > class SparseLU;
template <typename MappedSparseMatrixType> struct SparseLUMatrixLReturnType;
template <typename MatrixLType, typename MatrixUType> struct SparseLUMatrixUReturnType;

/** \ingroup SparseLU_Module
  * \class SparseLU
  * 
  * \brief Sparse supernodal LU factorization for general matrices
  * 
  * This class implements the supernodal LU factorization for general matrices.
  * It uses the main techniques from the sequential SuperLU package 
  * (http://crd-legacy.lbl.gov/~xiaoye/SuperLU/). It handles transparently real 
  * and complex arithmetics with single and double precision, depending on the 
  * scalar type of your input matrix. 
  * The code has been optimized to provide BLAS-3 operations during supernode-panel updates. 
  * It benefits directly from the built-in high-performant Eigen BLAS routines. 
  * Moreover, when the size of a supernode is very small, the BLAS calls are avoided to 
  * enable a better optimization from the compiler. For best performance, 
  * you should compile it with NDEBUG flag to avoid the numerous bounds checking on vectors. 
  * 
  * An important parameter of this class is the ordering method. It is used to reorder the columns 
  * (and eventually the rows) of the matrix to reduce the number of new elements that are created during 
  * numerical factorization. The cheapest method available is COLAMD. 
  * See  \link OrderingMethods_Module the OrderingMethods module \endlink for the list of 
  * built-in and external ordering methods. 
  *
  * Simple example with key steps 
  * \code
  * VectorXd x(n), b(n);
  * SparseMatrix<double, ColMajor> A;
  * SparseLU<SparseMatrix<scalar, ColMajor>, COLAMDOrdering<Index> >   solver;
  * // fill A and b;
  * // Compute the ordering permutation vector from the structural pattern of A
  * solver.analyzePattern(A); 
  * // Compute the numerical factorization 
  * solver.factorize(A); 
  * //Use the factors to solve the linear system 
  * x = solver.solve(b); 
  * \endcode
  * 
  * \warning The input matrix A should be in a \b compressed and \b column-major form.
  * Otherwise an expensive copy will be made. You can call the inexpensive makeCompressed() to get a compressed matrix.
  * 
  * \note Unlike the initial SuperLU implementation, there is no step to equilibrate the matrix. 
  * For badly scaled matrices, this step can be useful to reduce the pivoting during factorization. 
  * If this is the case for your matrices, you can try the basic scaling method at
  *  "unsupported/Eigen/src/IterativeSolvers/Scaling.h"
  * 
  * \tparam _MatrixType The type of the sparse matrix. It must be a column-major SparseMatrix<>
  * \tparam _OrderingType The ordering method to use, either AMD, COLAMD or METIS. Default is COLMAD
  *
  * \implsparsesolverconcept
  * 
  * \sa \ref TutorialSparseSolverConcept
  * \sa \ref OrderingMethods_Module
  */
template <typename _MatrixType, typename _OrderingType>
class SparseLU : public SparseSolverBase<SparseLU<_MatrixType,_OrderingType> >, public internal::SparseLUImpl<typename _MatrixType::Scalar, typename _MatrixType::StorageIndex>
{
  protected:
    typedef SparseSolverBase<SparseLU<_MatrixType,_OrderingType> > APIBase;
    using APIBase::m_isInitialized;
  public:
    using APIBase::_solve_impl;
    
    typedef _MatrixType MatrixType; 
    typedef _OrderingType OrderingType;
    typedef typename MatrixType::Scalar Scalar; 
    typedef typename MatrixType::RealScalar RealScalar; 
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar,ColMajor,StorageIndex> NCMatrix;
    typedef internal::MappedSuperNodalMatrix<Scalar, StorageIndex> SCMatrix;
    typedef Matrix<Scalar,Dynamic,1> ScalarVector;
    typedef Matrix<StorageIndex,Dynamic,1> IndexVector;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;
    typedef internal::SparseLUImpl<Scalar, StorageIndex> Base;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    
  public:
    SparseLU():m_lastError(""),m_Ustore(0,0,0,0,0,0),m_symmetricmode(false),m_diagpivotthresh(1.0),m_detPermR(1)
    {
      initperfvalues(); 
    }
    explicit SparseLU(const MatrixType& matrix)
      : m_lastError(""),m_Ustore(0,0,0,0,0,0),m_symmetricmode(false),m_diagpivotthresh(1.0),m_detPermR(1)
    {
      initperfvalues(); 
      compute(matrix);
    }
    
    ~SparseLU()
    {
      // Free all explicit dynamic pointers 
    }
    
    void analyzePattern (const MatrixType& matrix);
    void factorize (const MatrixType& matrix);
    void simplicialfactorize(const MatrixType& matrix);
    
    /**
      * Compute the symbolic and numeric factorization of the input sparse matrix.
      * The input matrix should be in column-major storage. 
      */
    void compute (const MatrixType& matrix)
    {
      // Analyze 
      analyzePattern(matrix); 
      //Factorize
      factorize(matrix);
    } 
    
    inline Index rows() const { return m_mat.rows(); }
    inline Index cols() const { return m_mat.cols(); }
    /** Indicate that the pattern of the input matrix is symmetric */
    void isSymmetric(bool sym)
    {
      m_symmetricmode = sym;
    }
    
    /** \returns an expression of the matrix L, internally stored as supernodes
      * The only operation available with this expression is the triangular solve
      * \code
      * y = b; matrixL().solveInPlace(y);
      * \endcode
      */
    SparseLUMatrixLReturnType<SCMatrix> matrixL() const
    {
      return SparseLUMatrixLReturnType<SCMatrix>(m_Lstore);
    }
    /** \returns an expression of the matrix U,
      * The only operation available with this expression is the triangular solve
      * \code
      * y = b; matrixU().solveInPlace(y);
      * \endcode
      */
    SparseLUMatrixUReturnType<SCMatrix,MappedSparseMatrix<Scalar,ColMajor,StorageIndex> > matrixU() const
    {
      return SparseLUMatrixUReturnType<SCMatrix, MappedSparseMatrix<Scalar,ColMajor,StorageIndex> >(m_Lstore, m_Ustore);
    }

    /**
      * \returns a reference to the row matrix permutation \f$ P_r \f$ such that \f$P_r A P_c^T = L U\f$
      * \sa colsPermutation()
      */
    inline const PermutationType& rowsPermutation() const
    {
      return m_perm_r;
    }
    /**
      * \returns a reference to the column matrix permutation\f$ P_c^T \f$ such that \f$P_r A P_c^T = L U\f$
      * \sa rowsPermutation()
      */
    inline const PermutationType& colsPermutation() const
    {
      return m_perm_c;
    }
    /** Set the threshold used for a diagonal entry to be an acceptable pivot. */
    void setPivotThreshold(const RealScalar& thresh)
    {
      m_diagpivotthresh = thresh; 
    }

#ifdef EIGEN_PARSED_BY_DOXYGEN
    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
      *
      * \warning the destination matrix X in X = this->solve(B) must be colmun-major.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const Solve<SparseLU, Rhs> solve(const MatrixBase<Rhs>& B) const;
#endif // EIGEN_PARSED_BY_DOXYGEN
    
    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the LU factorization reports a problem, zero diagonal for instance
      *          \c InvalidInput if the input matrix is invalid
      *
      * \sa iparm()          
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }
    
    /**
      * \returns A string describing the type of error
      */
    std::string lastErrorMessage() const
    {
      return m_lastError; 
    }

    template<typename Rhs, typename Dest>
    bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &X_base) const
    {
      Dest& X(X_base.derived());
      eigen_assert(m_factorizationIsOk && "The matrix should be factorized first");
      EIGEN_STATIC_ASSERT((Dest::Flags&RowMajorBit)==0,
                        THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
      
      // Permute the right hand side to form X = Pr*B
      // on return, X is overwritten by the computed solution
      X.resize(B.rows(),B.cols());

      // this ugly const_cast_derived() helps to detect aliasing when applying the permutations
      for(Index j = 0; j < B.cols(); ++j)
        X.col(j) = rowsPermutation() * B.const_cast_derived().col(j);
      
      //Forward substitution with L
      this->matrixL().solveInPlace(X);
      this->matrixU().solveInPlace(X);
      
      // Permute back the solution 
      for (Index j = 0; j < B.cols(); ++j)
        X.col(j) = colsPermutation().inverse() * X.col(j);
      
      return true; 
    }
    
    /**
      * \returns the absolute value of the determinant of the matrix of which
      * *this is the QR decomposition.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      * One way to work around that is to use logAbsDeterminant() instead.
      *
      * \sa logAbsDeterminant(), signDeterminant()
      */
    Scalar absDeterminant()
    {
      using std::abs;
      eigen_assert(m_factorizationIsOk && "The matrix should be factorized first.");
      // Initialize with the determinant of the row matrix
      Scalar det = Scalar(1.);
      // Note that the diagonal blocks of U are stored in supernodes,
      // which are available in the  L part :)
      for (Index j = 0; j < this->cols(); ++j)
      {
        for (typename SCMatrix::InnerIterator it(m_Lstore, j); it; ++it)
        {
          if(it.index() == j)
          {
            det *= abs(it.value());
            break;
          }
        }
      }
      return det;
    }

    /** \returns the natural log of the absolute value of the determinant of the matrix
      * of which **this is the QR decomposition
      *
      * \note This method is useful to work around the risk of overflow/underflow that's
      * inherent to the determinant computation.
      *
      * \sa absDeterminant(), signDeterminant()
      */
    Scalar logAbsDeterminant() const
    {
      using std::log;
      using std::abs;

      eigen_assert(m_factorizationIsOk && "The matrix should be factorized first.");
      Scalar det = Scalar(0.);
      for (Index j = 0; j < this->cols(); ++j)
      {
        for (typename SCMatrix::InnerIterator it(m_Lstore, j); it; ++it)
        {
          if(it.row() < j) continue;
          if(it.row() == j)
          {
            det += log(abs(it.value()));
            break;
          }
        }
      }
      return det;
    }

    /** \returns A number representing the sign of the determinant
      *
      * \sa absDeterminant(), logAbsDeterminant()
      */
    Scalar signDeterminant()
    {
      eigen_assert(m_factorizationIsOk && "The matrix should be factorized first.");
      // Initialize with the determinant of the row matrix
      Index det = 1;
      // Note that the diagonal blocks of U are stored in supernodes,
      // which are available in the  L part :)
      for (Index j = 0; j < this