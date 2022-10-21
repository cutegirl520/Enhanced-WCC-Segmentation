// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
#define EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H

namespace Eigen {

/** \class TensorKChippingReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A chip is a thin slice, corresponding to a column or a row in a 2-d tensor.
  *
  *
  */

namespace internal {
template<DenseIndex DimId, typename XprType>
struct traits<TensorChippingOp<DimId, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions - 1;
  static const int Layout = XprTraits::Layout;
};

template<DenseIndex DimId, typename XprType>
struct eval<TensorChippingOp<DimId, XprType>, Eigen::Dense>
{
  typedef const TensorChippingOp<DimId, XprType>& type;
};

template<DenseIndex DimId, typename XprType>
struct nested<TensorChippingOp<DimId, XprType>, 1, typename eval<TensorChippingOp<DimId, XprType> >::type>
{
  typedef TensorChippingOp<DimId, XprType> type;
};

template <DenseIndex DimId>
struct DimensionId
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DimensionId(DenseIndex dim) {
    eigen_assert(dim == DimId);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return DimId;
  }
};
template <>
struct DimensionId<Dynamic>
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DimensionId(DenseIndex dim) : actual_dim(dim) {
    eigen_assert(dim >= 0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return actual_dim;
  }
 private:
  const DenseIndex actual_dim;
};


}  // end namespace internal



template<DenseIndex DimId, typename XprType>
class TensorChippingOp : public TensorBase<TensorChippingOp<DimId, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorChippingOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorChippingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorChippingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorChippingOp(const XprType& expr, const Index offset, const Index dim)
      : m_xpr(expr), m_offset(offset), m_dim(dim) {
  }

  EIGEN_DEVICE_FUNC
  const Index offset() const { return m_offset; }
  EIGEN_DEVICE_FUNC
  const Index dim() const { return m_dim.actualDim(); }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename XprType::Nested>::type&
  expression() const { return m_xpr; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const TensorChippingOp& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const TensorChippingOp> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  template<typename OtherDerived>
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const OtherDerived& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const OtherDerived> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  protected:
    typename XprType::Nested m_xpr;
    const Index m_offset;
    const internal::DimensionId<DimId> m_dim;
};


// Eval as rvalue
template<DenseIndex DimId, typename ArgType, typename Device>
struct TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims 