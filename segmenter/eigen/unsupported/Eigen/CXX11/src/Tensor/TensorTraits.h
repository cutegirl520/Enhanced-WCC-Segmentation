
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_TRAITS_H
#define EIGEN_CXX11_TENSOR_TENSOR_TRAITS_H

namespace Eigen {
namespace internal {


template<typename Scalar, int Options>
class compute_tensor_flags
{
  enum {
    is_dynamic_size_storage = 1,

    is_aligned =
    (
        ((Options&DontAlign)==0) && (
#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
            (!is_dynamic_size_storage)
#else
            0
#endif
            ||
#if EIGEN_MAX_ALIGN_BYTES>0
            is_dynamic_size_storage
#else
            0
#endif
      )
     ),
    packet_access_bit = packet_traits<Scalar>::Vectorizable && is_aligned ? PacketAccessBit : 0
  };

  public:
    enum { ret = packet_access_bit };
};


template<typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
struct traits<Tensor<Scalar_, NumIndices_, Options_, IndexType_> >
{
  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef IndexType_ Index;
  static const int NumDimensions = NumIndices_;
  static const int Layout = Options_ & RowMajor ? RowMajor : ColMajor;
  enum {
    Options = Options_,
    Flags = compute_tensor_flags<Scalar_, Options_>::ret | (is_const<Scalar_>::value ? 0 : LvalueBit)
  };
};


template<typename Scalar_, typename Dimensions, int Options_, typename IndexType_>
struct traits<TensorFixedSize<Scalar_, Dimensions, Options_, IndexType_> >
{
  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef IndexType_ Index;
  static const int NumDimensions = array_size<Dimensions>::value;
  static const int Layout = Options_ & RowMajor ? RowMajor : ColMajor;
  enum {
    Options = Options_,
    Flags = compute_tensor_flags<Scalar_, Options_>::ret | (is_const<Scalar_>::value ? 0: LvalueBit)
  };
};


template<typename PlainObjectType, int Options_>
struct traits<TensorMap<PlainObjectType, Options_> >
  : public traits<PlainObjectType>
{
  typedef traits<PlainObjectType> BaseTraits;
  typedef typename BaseTraits::Scalar Scalar;
  typedef typename BaseTraits::StorageKind StorageKind;
  typedef typename BaseTraits::Index Index;
  static const int NumDimensions = BaseTraits::NumDimensions;
  static const int Layout = BaseTraits::Layout;
  enum {
    Options = Options_,
    Flags = BaseTraits::Flags
  };
};

template<typename PlainObjectType>
struct traits<TensorRef<PlainObjectType> >
  : public traits<PlainObjectType>
{
  typedef traits<PlainObjectType> BaseTraits;
  typedef typename BaseTraits::Scalar Scalar;
  typedef typename BaseTraits::StorageKind StorageKind;
  typedef typename BaseTraits::Index Index;
  static const int NumDimensions = BaseTraits::NumDimensions;
  static const int Layout = BaseTraits::Layout;
  enum {
    Options = BaseTraits::Options,
    Flags = BaseTraits::Flags
  };
};


template<typename _Scalar, int NumIndices_, int Options, typename IndexType_>
struct eval<Tensor<_Scalar, NumIndices_, Options, IndexType_>, Eigen::Dense>
{
  typedef const Tensor<_Scalar, NumIndices_, Options, IndexType_>& type;
};

template<typename _Scalar, int NumIndices_, int Options, typename IndexType_>
struct eval<const Tensor<_Scalar, NumIndices_, Options, IndexType_>, Eigen::Dense>
{
  typedef const Tensor<_Scalar, NumIndices_, Options, IndexType_>& type;
};

template<typename Scalar_, typename Dimensions, int Options, typename IndexType_>
struct eval<TensorFixedSize<Scalar_, Dimensions, Options, IndexType_>, Eigen::Dense>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options, IndexType_>& type;
};

template<typename Scalar_, typename Dimensions, int Options, typename IndexType_>
struct eval<const TensorFixedSize<Scalar_, Dimensions, Options, IndexType_>, Eigen::Dense>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options, IndexType_>& type;
};

template<typename PlainObjectType, int Options>
struct eval<TensorMap<PlainObjectType, Options>, Eigen::Dense>
{
  typedef const TensorMap<PlainObjectType, Options>& type;