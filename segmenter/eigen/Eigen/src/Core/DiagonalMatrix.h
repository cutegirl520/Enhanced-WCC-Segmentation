// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DIAGONALMATRIX_H
#define EIGEN_DIAGONALMATRIX_H

namespace Eigen { 

#ifndef EIGEN_PARSED_BY_DOXYGEN
template<typename Derived>
class DiagonalBase : public EigenBase<Derived>
{
  public:
    typedef typename internal::traits<Derived>::DiagonalVectorType DiagonalVectorType;
    typedef typename DiagonalVectorType::Scalar Scalar;
    typedef typename DiagonalVectorType::RealScalar RealScalar;
    typedef typename internal::traits<Derived>::StorageKind StorageKind;
    typedef typename internal::traits<Derived>::StorageIndex StorageIndex;

    enum {
      RowsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
      ColsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
      MaxRowsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
      MaxColsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
      IsVectorAtCompileTime = 0,
      Flags = NoPreferredStorageOrderBit
    };

    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime> DenseMatrixType;
    typedef DenseMatrixType DenseType;
    typedef DiagonalMatrix<Scalar,DiagonalVectorType::SizeAtCompileTime,DiagonalVectorType::MaxSizeAtCompileTime> PlainObject;

    EIGEN_DEVICE_FUNC
    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    EIGEN_DEVICE_FUNC
    inline Derived& derived() { return *static_cast<Derived*>(this); }

    EIGEN_DEVICE_FUNC
    DenseMatrixType toDenseMatrix() const { return derived(); }
    
    EIGEN_DEVICE_FUNC
    inline const DiagonalVectorType& diagonal() const { return derived().diagonal(); }
    EIGEN_DEVICE_FUNC
    inline DiagonalVectorType& diagonal() { return derived().diagonal(); }

    EIGEN_DEVICE_FUNC
    inline Index rows() const { return diagonal().size(); }
    EIGEN_DEVICE_FUNC
    inline Index cols() const { return diagonal().size(); }

    template<typename MatrixDerived>
    EIGEN_DEVICE_FUNC
    const Product<Derived,MatrixDerived,LazyProduct>
    operator*(const MatrixBase<MatrixDerived> &matrix) const
    {
      return Product<Derived, MatrixDerived, LazyProduct>(derived(),matrix.derived());
    }

    typedef DiagonalWrapper<const CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const DiagonalVectorType> > InverseReturnType;
    EIGEN_DEVICE_FUNC
    inline const InverseReturnType
    inverse() const
    {
      return InverseReturnType(diagonal().cwiseInverse());
    }
    
    EIGEN_DEVICE_FUNC
    inline const DiagonalWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DiagonalVectorType,Scalar,product) >
    operator*(const Scalar& scalar) const
    {
      return DiagonalWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(DiagonalVectorType,Scalar,product) >(diagonal() * scalar);
    }
    EIGEN_DEVICE_FUNC
    friend inline const DiagonalWrapper<const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar,DiagonalVectorType,product) >
    operator*(const Scalar& scalar, const DiagonalBase& other)
    {
      return DiagonalWrapper<const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar,DiagonalVectorType,product) >(scalar * other.diagonal());
    }
};

#endif

/** \class DiagonalMatrix
  * \ingroup Core_Module
  *
  * \brief Represents a diagonal matrix with its storage
  *
  * \param _Scalar the type of coefficients
  * \param SizeAtCompileTime the dimension of the matrix, or Dynamic
  * \param MaxSizeAtCompileTime the dimension of the matrix, or Dynamic. This parameter is optional and defaults
  *        to SizeAtCompileTime. Most of the time, you do not need to specify it.
  *
  * \sa class DiagonalWrapper
  */

namespace internal {
template<typename _Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
struct traits<DiagonalMatrix<_Scalar,SizeAtCompileTime,MaxSizeAtCompileTime> >
 : traits<Matrix<_Scalar,SizeAtCompileTime,SizeAtCompileTime,0,MaxSizeAtCompileTime,MaxSizeAtCompileTime> >
{
  typedef Matrix<_Scalar,SizeAtCompileTime,1,0,MaxSizeAtCompileTime,1> DiagonalVectorType;
  typedef DiagonalShape StorageKind;
  enum {
    Flags = LvalueBit | NoPreferredStorageOrderBit
  };
};
}
template<typename _Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
class DiagonalMatrix
  : public DiagonalBase<DiagonalMatrix<_Scalar,SizeAtCompileTime,MaxSizeAtCompileTime> >
{
  public:
    #ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename internal::traits<DiagonalMatrix>::DiagonalVectorType DiagonalVectorType;
    typedef const DiagonalMatrix& Nested;
    typedef _Scalar Scalar;
    typedef typename internal::traits<DiagonalMatrix>::StorageKind StorageKind;
    typedef typename internal::traits<DiagonalMatrix>::StorageIndex StorageIndex;
    #endif

  protected:

    DiagonalVectorType m_diagonal;

  public:

    /** const version of diagonal(). */
    EIGEN_DEVICE_FUNC
    inline const DiagonalVectorType& diagonal() const { return m_diagonal; }
    /** \returns a reference to the stored vector of diagonal coefficients. */
    EIGEN_DEVICE_FUNC
    inline DiagonalVectorType& diagonal() { return m_diagonal; }

    /** Default constructor without initialization */
    EIGEN_DEVICE_FUNC
    inline DiagonalMatrix() {}

    /** Constructs a diagonal matrix with given dimension  */
    EIGEN_DEVICE_FUNC
    explicit inline DiagonalMatrix(Index dim) : m_diagonal(dim) {}

    /** 2D constructor. */
    EIGEN_DEVICE_FUNC
    inline DiagonalMatrix(const Scalar& x, const Scalar& y) : m_diagonal(x,y) {}

    /** 3D constructor. */
    EIGEN_DEVICE_FUNC
    inline DiagonalMatrix(const Scalar& x, const Scalar& y, const Scalar& z) : m_diagonal(x,y,z) {}

    /** Copy constructor. */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    inline DiagonalMatrix(const DiagonalBase<OtherDerived>& other) : m_diagonal(other.diagonal()) {}

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** copy constructor. prevent a default copy constructor from hiding the other templated constructor */
    inline DiagonalMatrix(const DiagonalMatrix& other) : m_diagonal(other.diagonal()) {}
    #endif

    /** generic constructor from expression of the diagonal coefficients */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    explicit inline DiagonalMatrix(const MatrixBase<OtherDerived>& other) : m_diagonal(other)
    {}

    /** Copy operator. */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    DiagonalMatrix& operator=(const DiagonalBase<OtherDerived>& other)
    {
      m_diagonal = other.diagonal();
      return *this;
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    EIGEN_DEVICE_FUNC
    DiagonalMatrix& operator=(const DiagonalMatrix& other)
    {
      m_diagonal = other.diagonal();
      return *this;
    }
    #endif

    /** Resizes to given size. */
    EIGEN_DEVICE_FUNC
    inline void resize(Index size) { m_diagonal.resize(size); }
    /** Sets all coefficients to zero. */
    EIGEN_DEVICE_FUNC
    inline void setZero() { m_diagonal.setZero(); }
    /** Resizes and sets all coefficients to zero. */
    EIGEN_DEVICE_FUNC
    inline void setZero(Index size) { m_diagonal.setZero(size); }
    /** Sets this matrix to be the identity matrix of the current size. */
    EIGEN_DEVICE_FUNC
    inline void setIdentity() { m_diagonal.setOnes(); }
    /** Sets this matrix to be the identity matrix of the given size. */
    EIGEN_DEVICE_FUNC
    inline void setIdentity(Index size) { m_diagonal.setOnes(size); }
};

/** \class DiagonalWrapper
  * \ingroup Core_Module
  *
  * \brief Expression of a diagonal matrix
  *
  * \param _DiagonalVectorType the type of the vector of diagonal coefficients
  *
  * This class is an expression of a diagonal matrix, but not storing its own vector of diagonal coefficients,
  * instead wrapping an existing vector expression. It is the ret