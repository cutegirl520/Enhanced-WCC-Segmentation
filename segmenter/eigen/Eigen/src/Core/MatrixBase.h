// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXBASE_H
#define EIGEN_MATRIXBASE_H

namespace Eigen {

/** \class MatrixBase
  * \ingroup Core_Module
  *
  * \brief Base class for all dense matrices, vectors, and expressions
  *
  * This class is the base that is inherited by all matrix, vector, and related expression
  * types. Most of the Eigen API is contained in this class, and its base classes. Other important
  * classes for the Eigen API are Matrix, and VectorwiseOp.
  *
  * Note that some methods are defined in other modules such as the \ref LU_Module LU module
  * for all functions related to matrix inversions.
  *
  * \tparam Derived is the derived type, e.g. a matrix type, or an expression, etc.
  *
  * When writing a function taking Eigen objects as argument, if you want your function
  * to take as argument any matrix, vector, or expression, just let it take a
  * MatrixBase argument. As an example, here is a function printFirstRow which, given
  * a matrix, vector, or expression \a x, prints the first row of \a x.
  *
  * \code
    template<typename Derived>
    void printFirstRow(const Eigen::MatrixBase<Derived>& x)
    {
      cout << x.row(0) << endl;
    }
  * \endcode
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_MATRIXBASE_PLUGIN.
  *
  * \sa \blank \ref TopicClassHierarchy
  */
template<typename Derived> class MatrixBase
  : public DenseBase<Derived>
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef MatrixBase StorageBaseType;
    typedef typename internal::traits<Derived>::StorageKind StorageKind;
    typedef typename internal::traits<Derived>::StorageIndex StorageIndex;
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::packet_traits<Scalar>::type PacketScalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    typedef DenseBase<Derived> Base;
    using Base::RowsAtCompileTime;
    using Base::ColsAtCompileTime;
    using Base::SizeAtCompileTime;
    using Base::MaxRowsAtCompileTime;
    using Base::MaxColsAtCompileTime;
    using Base::MaxSizeAtCompileTime;
    using Base::IsVectorAtCompileTime;
    using Base::Flags;

    using Base::derived;
    using Base::const_cast_derived;
    using Base::rows;
    using Base::cols;
    using Base::size;
    using Base::coeff;
    using Base::coeffRef;
    using Base::lazyAssign;
    using Base::eval;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;

    typedef typename Base::CoeffReturnType CoeffReturnType;
    typedef typename Base::ConstTransposeReturnType ConstTransposeReturnType;
    typedef typename Base::RowXpr RowXpr;
    typedef typename Base::ColXpr ColXpr;
#endif // not EIGEN_PARSED_BY_DOXYGEN



#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** type of the equivalent square matrix */
    typedef Matrix<Scalar,EIGEN_SIZE_MAX(RowsAtCompileTime,ColsAtCompileTime),
                          EIGEN_SIZE_MAX(RowsAtCompileTime,ColsAtCompileTime)> SquareMatrixType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    /** \returns the size of the main diagonal, which is min(rows(),cols()).
      * \sa rows(), cols(), SizeAtCompileTime. */
    EIGEN_DEVICE_FUNC
    inline Index diagonalSize() const { return (std::min)(rows(),cols()); }

    typedef typename Base::PlainObject PlainObject;

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<internal::scalar_constant_op<Scalar>,PlainObject> ConstantReturnType;
    /** \internal the return type of MatrixBase::adjoint() */
    typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
                        CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, ConstTransposeReturnType>,
                        ConstTransposeReturnType
                     >::type AdjointReturnType;
    /** \internal Return type of eigenvalues() */
    typedef Matrix<std::complex<RealScalar>, internal::traits<Derived>::ColsAtCompileTime, 1, ColMajor> EigenvaluesReturnType;
    /** \internal the return type of identity */
    typedef CwiseNullaryOp<internal::scalar_identity_op<Scalar>,PlainObject> IdentityReturnType;
    /** \internal the return type of unit vectors */
    typedef Block<const CwiseNullaryOp<internal::scalar_identity_op<Scalar>, SquareMatrixType>,
                  internal::traits<Derived>::RowsAtCompileTime,
                  internal::traits<Derived>::ColsAtCompileTime> BasisReturnType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

#define EIGEN_CURRENT_STORAGE_BASE_CLASS Eigen::MatrixBase
#   include "../plugins/CommonCwiseUnaryOps.h"
#   include "../plugins/CommonCwiseBinaryOps.h"
#   include "../plugins/MatrixCwiseUnaryOps.h"
#   include "../plugins/MatrixCwiseBinaryOps.h"
#   ifdef EIGEN_MATRIXBASE_PLUGIN
#     include EIGEN_MATRIXBASE_PLUGIN
#   endif
#undef EIGEN_CURRENT_STORAGE_BASE_CLASS

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator=(const MatrixBase& other);

    // We cannot inherit here via Base::operator= since it is causing
    // trouble with MSVC.

    template <typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator=(const DenseBase<OtherDerived>& other);

    template <typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator=(const EigenBase<OtherDerived>& other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator=(const ReturnByValue<OtherDerived>& other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator+=(const MatrixBase<OtherDerived>& other);
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator-=(const MatrixBase<OtherDerived>& other);

#ifdef __CUDACC__
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    const Product<Derived,OtherDerived,LazyProduct>
    operator*(const MatrixBase<OtherDerived> &other) const
    { return this->lazyProduct(other); }
#else

    template<typename OtherDerived>
    const Product<Derived,OtherDerived>
    operator*(const MatrixBase<OtherDerived> &other) const;

#endif

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    const Product<Derived,OtherDerived,LazyProduct>
    lazyProduct(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    Derived& operator*=(const EigenBase<OtherDerived>& other);

    template<typename OtherDerived>
    void applyOnTheLeft(const EigenBase<OtherDerived>& other);

    template<typename OtherDerived>
    void applyOnTheRight(const EigenBase<OtherDerived>& other);

    template<typename DiagonalDerived>
    EIGEN_DEVICE_FUNC
    const Product<Derived, DiagonalDerived, LazyProduct>
    operator*(const DiagonalBase<DiagonalDerived> &diagonal) const;

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    typename ScalarBinaryOpTraits<typename internal::traits<Derived>::Scalar,typename internal::traits<OtherDerived>::Scalar>::ReturnType
    dot(const MatrixBase<OtherDerived>& other) const;

    EIGEN_DEVICE_FUNC RealScalar squaredNorm() const;
    EIGEN_DEVICE_FUNC RealScalar norm() const;
    RealScalar stableNorm() const;
    RealScalar blueNorm() const;
    RealScalar hypotNorm() const;
    EIGEN_DEVICE_FUNC const PlainObject normalized() const;
    EIGEN_DEVICE_FUNC const PlainObject stableNormalized() const;
    EIGEN_DEVICE_FUNC void normalize();
    EIGEN_DEVICE_FUNC void stableNormalize();

    EIGEN_DEVICE_FUNC const AdjointReturnType adjoint() const;
    EIGEN_DEVICE_FUNC void adjointInPlace();

    typedef Diagonal<Derived> DiagonalReturnType;
    EIGEN_DEVICE_FUNC
    DiagonalReturnType diagonal();

    typed