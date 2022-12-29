// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H

namespace Eigen {

/** \class TensorEvaluator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor evaluator classes.
  *
  * These classes are responsible for the evaluation of the tensor expression.
  *
  * TODO: add support for more types of expressions, in particular expressions
  * leading to lvalues (slicing, reshaping, etc...)
  */

// Generic evaluator
template<typename Derived, typename Device>
struct TensorEvaluator
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::NumDimensions : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = Derived::Layout,
    CoordAccess = NumCoords > 0,
    RawAccess = true
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device& device)
      : m_data(const_cast<Scalar*>(m.data())), m_dims(m.dimensions()), m_device(device)
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* dest) {
    if (dest) {
      m_device.memcpy((void*)dest, m_data, sizeof(Scalar) * m_dims.TotalSize());
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data + index);
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    return internal::pstoret<Scalar, PacketReturnType, StoreMode>(m_data + index, x);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<DenseIndex, NumCoords>& coords) const {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<DenseIndex, NumCoords>& coords) {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return m_data; }

 protected:
  Scalar* m_data;
  Dimensions m_dims;
  const Device& m_device;
};

namespace {
template <typename T> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T loadConstant(const T* address) {
  return *address;
}
// Use the texture cache on CUDA devices whenever possible
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float loadConstant(const float* address) {
  return __ldg(address);
}
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double loadConstant(const double* address) {
  return __ldg(address);
}
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
Eigen::half loadConstant(const Eigen::half* address) {
  return Eigen::half(internal::raw_uint16_to_half(__ldg(&address->x)));
}
#endif
}


// Default evaluator for rvalues
template<typename Derived, typename Device>
struct TensorEvaluator<const Derived, Device>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::Num