// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_PADDING_H
#define EIGEN_CXX11_TENSOR_TENSOR_PADDING_H

namespace Eigen {

/** \class TensorPadding
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor padding class.
  * At the moment only padding with a constant value is supported.
  *
  */
namespace internal {
template<typename PaddingDimensions, typename XprType>
struct traits<TensorPaddingOp<PaddingDimensions, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename PaddingDimensions, typename XprType>
struct eval<TensorPaddingOp<PaddingDimensions, XprType>, Eigen::Dense>
{
  typedef const TensorPaddingOp<PaddingDimensions, XprType>& type;
};

template<typename PaddingDimensions, typename XprType>
struct nested<TensorPaddingOp<PaddingDimensions, XprType>, 1, typename eval<TensorPaddingOp<PaddingDimensions, XprType> >::type>
{
  typedef TensorPaddingOp<PaddingDimensions, XprType> type;
};

}  // end namespace internal



template<typename PaddingDimensions, typename XprType>
class TensorPaddingOp : public TensorBase<TensorPaddingOp<PaddingDimensions, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorPaddingOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorPaddingOp>::type Nested;
  typedef typename