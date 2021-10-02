
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_CWISE_BINARY_OP_H
#define EIGEN_SPARSE_CWISE_BINARY_OP_H

namespace Eigen { 

// Here we have to handle 3 cases:
//  1 - sparse op dense
//  2 - dense op sparse
//  3 - sparse op sparse
// We also need to implement a 4th iterator for:
//  4 - dense op dense
// Finally, we also need to distinguish between the product and other operations :
//                configuration      returned mode
//  1 - sparse op dense    product      sparse
//                         generic      dense
//  2 - dense op sparse    product      sparse
//                         generic      dense
//  3 - sparse op sparse   product      sparse
//                         generic      sparse
//  4 - dense op dense     product      dense
//                         generic      dense
//
// TODO to ease compiler job, we could specialize product/quotient with a scalar
//      and fallback to cwise-unary evaluator using bind1st_op and bind2nd_op.

template<typename BinaryOp, typename Lhs, typename Rhs>
class CwiseBinaryOpImpl<BinaryOp, Lhs, Rhs, Sparse>
  : public SparseMatrixBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  public:
    typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> Derived;
    typedef SparseMatrixBase<Derived> Base;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)
    CwiseBinaryOpImpl()
    {
      EIGEN_STATIC_ASSERT((
                (!internal::is_same<typename internal::traits<Lhs>::StorageKind,
                                    typename internal::traits<Rhs>::StorageKind>::value)
            ||  ((Lhs::Flags&RowMajorBit) == (Rhs::Flags&RowMajorBit))),
            THE_STORAGE_ORDER_OF_BOTH_SIDES_MUST_MATCH);
    }
};

namespace internal {

  
// Generic "sparse OP sparse"
template<typename XprType> struct binary_sparse_evaluator;

template<typename BinaryOp, typename Lhs, typename Rhs>
struct binary_evaluator<CwiseBinaryOp<BinaryOp, Lhs, Rhs>, IteratorBased, IteratorBased>
  : evaluator_base<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
protected:
  typedef typename evaluator<Lhs>::InnerIterator  LhsIterator;
  typedef typename evaluator<Rhs>::InnerIterator  RhsIterator;
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename XprType::StorageIndex StorageIndex;
public:

  class ReverseInnerIterator;
  class InnerIterator
  {
  public:
    
    EIGEN_STRONG_INLINE InnerIterator(const binary_evaluator& aEval, Index outer)
      : m_lhsIter(aEval.m_lhsImpl,outer), m_rhsIter(aEval.m_rhsImpl,outer), m_functor(aEval.m_functor)
    {
      this->operator++();
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      if (m_lhsIter && m_rhsIter && (m_lhsIter.index() == m_rhsIter.index()))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), m_rhsIter.value());
        ++m_lhsIter;
        ++m_rhsIter;
      }
      else if (m_lhsIter && (!m_rhsIter || (m_lhsIter.index() < m_rhsIter.index())))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), Scalar(0));
        ++m_lhsIter;
      }
      else if (m_rhsIter && (!m_lhsIter || (m_lhsIter.index() > m_rhsIter.index())))
      {
        m_id = m_rhsIter.index();
        m_value = m_functor(Scalar(0), m_rhsIter.value());
        ++m_rhsIter;
      }
      else
      {
        m_value = 0; // this is to avoid a compilation warning
        m_id = -1;
      }
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_value; }

    EIGEN_STRONG_INLINE StorageIndex index() const { return m_id; }
    EIGEN_STRONG_INLINE Index row() const { return Lhs::IsRowMajor ? m_lhsIter.row() : index(); }
    EIGEN_STRONG_INLINE Index col() const { return Lhs::IsRowMajor ? index() : m_lhsIter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_id>=0; }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
    Scalar m_value;
    StorageIndex m_id;
  };
  
  
  enum {
    CoeffReadCost = evaluator<Lhs>::CoeffReadCost + evaluator<Rhs>::CoeffReadCost + functor_traits<BinaryOp>::Cost,
    Flags = XprType::Flags
  };
  
  explicit binary_evaluator(const XprType& xpr)
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  {
    EIGEN_INTERNAL_CHECK_COST_VALUE(functor_traits<BinaryOp>::Cost);
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }
  
  inline Index nonZerosEstimate() const {