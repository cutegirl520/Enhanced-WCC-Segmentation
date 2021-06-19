// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELECT_H
#define EIGEN_SELECT_H

namespace Eigen { 

/** \class Select
  * \ingroup Core_Module
  *
  * \brief Expression of a coefficient wise version of the C++ ternary operator ?:
  *
  * \param ConditionMatrixType the type of the \em condition expression which must be a boolean matrix
  * \param ThenMatrixType the type of the \em then expression
  * \param ElseMatrixType the type of the \em else expression
  *
  * This class represents an expression of a coefficient wise version of the C++ ternary operator ?:.
  * It is the return type of DenseBase::select() and most of the time this is the only way it is used.
  *
  * \sa DenseBase::select(const DenseBase<ThenDerived>&, const DenseBase<ElseDerived>&) const
  */

namespace internal {
template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
struct traits<Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType> >
 : traits<ThenMatrixType>
{
  typedef typename traits<ThenMatrixType>::Scalar Scalar;
  typedef Dense StorageKind;
  typedef typename traits<ThenMatrixType>::XprKind XprKind;
  typedef typename ConditionMatrixType::Nested ConditionMatrixNested;
  typedef typename ThenMatrixType::Nested ThenMatrixNested;
  typedef typename ElseMatrixType::Nested ElseMatrixNested;
  enum {
    RowsAtCompileTime = ConditionMatrixType::RowsAtCompileTime,
    ColsAtCompileTime = ConditionMatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ConditionMatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ConditionMatrixType::MaxColsAtCompileTime,
    Flags = (unsigned int)ThenMatrixType::Flags & ElseMatrixType::Flags & RowMajorBit
  };
};
}

template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
class Select : public internal::dense_xpr_base< Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType> >::type,
               internal::no_assignment_operator
{
  public:

    typedef typename internal::dense_xpr_base<Select>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Select)

    inline EIGEN_DEVICE_FUNC
    Select(const ConditionMatrixType& a_conditionMatrix,
           const ThenMatrixType& a_thenMatrix,
           con