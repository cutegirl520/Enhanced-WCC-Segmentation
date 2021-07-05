// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BINARY_FUNCTORS_H
#define EIGEN_BINARY_FUNCTORS_H

namespace Eigen {

namespace internal {

//---------- associative binary functors ----------

template<typename Arg1, typename Arg2>
struct binary_op_base
{
  typedef Arg1 first_argument_type;
  typedef Arg2 second_argument_type;
};

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, class VectorwiseOp, DenseBase::sum()
  */
template<typename LhsScalar,typename RhsScalar>
struct scalar_sum_op : binary_op_base<LhsScalar,RhsScalar>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar,RhsScalar,scalar_sum_op>::ReturnType result_type;
#ifndef EIGEN_SCALAR_BINARY_OP_PLUGIN
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
#else
  scalar_sum_op() {
    EIGEN_SCALAR_BINARY_OP_PLUGIN
  }
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const { return a + b; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::padd(a,b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type predux(const Packet& a) const
  { return internal::predux(a); }
};
template<typename LhsScalar,typename RhsScalar>
struct functor_traits<scalar_sum_op<LhsScalar,RhsScalar> > {
  enum {
    Cost = (NumTraits<LhsScalar>::AddCost+NumTraits<RhsScalar>::AddCost)/2, // rough estimate!
    PacketAccess = is_same<LhsScalar,RhsScalar>::value && packet_traits<LhsScalar>::HasAdd && packet_traits<RhsScalar>::HasAdd
    // TODO vectorize mixed sum
  };
};

/** \internal
  * \brief Template specialization to deprecate the summation of boolean expressions.
  * This is required to solve Bug 426.
  * \sa DenseBase::count(), DenseBase::any(), ArrayBase::cast(), MatrixBase::cast()
  */
template<> struct scalar_sum_op<bool,bool> : scalar_sum_op<int,int> {
  EIGEN_DEPRECATED
  scalar_sum_op() {}
};


/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, Cwise::operator*(), class VectorwiseOp, MatrixBase::redux()
  */
template<typename L