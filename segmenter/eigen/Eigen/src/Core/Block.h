// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_H
#define EIGEN_BLOCK_H

namespace Eigen { 

namespace internal {
template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
struct traits<Block<XprType, BlockRows, BlockCols, InnerPanel> > : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::XprKind XprKind;
  typedef typename ref_selector<XprType>::type XprTypeNested;
  typedef typename remove_reference<XprTypeNested>::type _XprTypeNested;
  enum{
    MatrixRows = traits<XprType>::RowsAtCompileTime,
    MatrixCols = traits<XprType>::ColsAtCompileTime,
    RowsAtCompileTime = MatrixRows == 0 ? 0 : BlockRows,
    ColsAtCompileTime = MatrixCols == 0 ? 0 : BlockCols,
    MaxRowsAtCompileTime = BlockRows==0 ? 0
                         : RowsAtCompileTime != Dynamic ? int(RowsAtCompileTime)
                         : int(traits<XprType>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = BlockCols==0 ? 0
                         : ColsAtCompileTime != Dynamic ? int(ColsAtCompileTime)
                         : int(traits<XprType>::MaxColsAtCompileTime),

    XprTypeIsRowMajor = (int(traits<XprType>::Flags)&RowMajorBit) != 0,
    IsRowMajor = (MaxRowsAtCompileTime==1&&MaxColsAtCompileTime!=1) ? 1
               : (MaxColsAtCompileTime==1&&MaxRowsAtCompileTime!=1) ? 0
               : XprTypeIsRowMajor,
    HasSameStorageOrderAsXprType = (IsRowMajor == XprTypeIsRowMajor),
    InnerSize = IsRowMajor ? int(ColsAtCompileTime) : int(RowsAtCompileTime),
    InnerStrideAtCompileTime = HasSameStorageOrderAsXprType
                             ? int(inner_stride_at_compile_time<XprType>::ret)
                             : int(outer_stride_at_compile_time<XprType>::ret),
    OuterStrideAtCompileTime = HasSameStorageOrderAsXprType
                             ? int(outer_stride_at_compile_time<XprType>::ret)
                             : int(inner_stride_at_compile_time<XprType>::ret),

    // FIXME, this traits is rather specialized for dense object and it needs to be cleaned further
    FlagsLvalueBit = is_lvalue<XprType>::value ? LvalueBit : 0,
    FlagsRowMajorBit = IsRowMajor ? RowMajorBit : 0,
    Flags = (traits<XprType>::Flags & (DirectAccessBit | (InnerPanel?CompressedAccessBit:0))) | FlagsLvalueBit | FlagsRowMajorBit,
    // FIXME DirectAccessBit should not be handled by expressions
    // 
    // Alignment is needed by MapBase's assertions
    // We can sefely set it to false here. Internal alignment errors will be detected by an eigen_internal_assert in the respective evaluator
    Alignment = 0
  };
};

template<typename XprType, int BlockRows=Dynamic, int BlockCols=Dynamic, bool InnerPanel = false,
         bool HasDirectAccess = internal::has_direct_access<XprType>::ret> class BlockImpl_dense;
         
} // end namespace internal

template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename StorageKind> class BlockImpl;

/** \class Block
  * \ingroup Core_Module
  *
  * \brief Expression of a fixed-size or dynamic-size block
  *
  * \tparam XprType the type of the expression in which we are taking a block
  * \tparam BlockRows the number of rows of the block we are taking at compile time (optional)
  * \tparam BlockCols the number of columns of the block we are taking at compile time (optional)
  * \tparam InnerPanel is true, if the block maps to a set of rows of a row major matrix or
  *         to set of columns of a column major matrix (optional). The parameter allows to determine
  *         at compile time whether aligned access is possible on the block expression.
  *
  * This class represents an expression of either a fixed-size or dynamic-size block. It is the return
  * type of DenseBase::block(Index,Index,Index,Index) and DenseBase::block<int,int>(Index,Index) and
  * most of the time this is the only way it is used.
  *
  * However, if you want to directly maniputate block expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating the dynamic case:
  * \include class_Block.cpp
  * Output: \verbinclude class_Block.out
  *
  * \note Even though this expression has dynamic size, in the case where \a XprType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * Here is an example illustrating the fixed-size case:
  * \include class_FixedBlock.cpp
  * Output: \verbinclude class_FixedBlock.out
  *
  * \sa DenseBase::block(Index,Index,Index,Index), DenseBase::block(Index,Index), class VectorBlock
  */
template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel> class Block
  : public BlockImpl<XprType, BlockRows, BlockCols, InnerPanel, typename internal::traits<XprType>::StorageKind>
{
    typedef BlockImpl<XprType, BlockRows, BlockCols, InnerPanel, typename internal::traits<XprType>::StorageKind> Impl;
  public:
    //typedef typename Impl::Base Base;
    typedef Impl Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(Block)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Block)
    
    typedef typename internal::remove_all<XprType>::type NestedExpression;
  
    /** Column or Row constructor
      */
    EIGEN_DEVICE_FUNC
    inline Block(XprType& xpr, Index i) : Impl(xpr,i)
    {
      eigen_assert( (i>=0) && (
          ((BlockRows==1) && (BlockCols==XprType::ColsAtCompileTime) && i<xpr.rows())
        ||((BlockRows==XprType::RowsAtCompileTime) && (BlockCols==1) && i<xpr.cols())));
    }

    /** Fixed-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Block(XprType& xpr, Index startRow, Index startCol)
      : Impl(xpr, startRow, startCol)
    {
      EIGEN_STATIC_ASSERT(RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic,THIS_METHOD_IS_ONLY_FOR_FIXED_SIZE)
      eigen_assert(startRow >= 0 && BlockRows >= 0 && startRow + BlockRows <= xpr.rows()
             && startCol >= 0 && BlockCols >= 0 && startCol + BlockCols <= xpr.cols());
    }

    /** Dynamic-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Block(XprType& xpr,
          Index startRow, Index startCol,
          Index blockRows, Index blockCols)
      : Impl(xpr, startRow, startCol, blockRows