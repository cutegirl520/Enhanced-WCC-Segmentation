// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CONSTANTS_H
#define EIGEN_CONSTANTS_H

namespace Eigen {

/** This value means that a positive quantity (e.g., a size) is not known at compile-time, and that instead the value is
  * stored in some runtime variable.
  *
  * Changing the value of Dynamic breaks the ABI, as Dynamic is often used as a template parameter for Matrix.
  */
const int Dynamic = -1;

/** This value means that a signed quantity (e.g., a signed index) is not known at compile-time, and that instead its value
  * has to be specified at runtime.
  */
const int DynamicIndex = 0xffffff;

/** This value means +Infinity; it is currently used only as the p parameter to MatrixBase::lpNorm<int>().
  * The value Infinity there means the L-infinity norm.
  */
const int Infinity = -1;

/** This value means that the cost to evaluate an expression coefficient is either very expensive or
  * cannot be known at compile time.
  *
  * This value has to be positive to (1) simplify cost computation, and (2) allow to distinguish between a very expensive and very very expensive expressions.
  * It thus must also be large enough to make sure unrolling won't happen and that sub expressions will be evaluated, but not too large to avoid overflow.
  */
const int HugeCost = 10000;

/** \defgroup flags Flags
  * \ingroup Core_Module
  *
  * These are the possible bits which can be OR'ed to constitute the flags of a matrix or
  * expression.
  *
  * It is important to note that these flags are a purely compile-time notion. They are a compile-time property of
  * an expression type, implemented as enum's. They are not stored in memory at runtime, and they do not incur any
  * runtime overhead.
  *
  * \sa MatrixBase::Flags
  */

/** \ingroup flags
  *
  * for a matrix, this means that the storage order is row-major.
  * If this bit is not set, the storage order is column-major.
  * For an expression, this determines the storage order of
  * the matrix created by evaluation of that expression.
  * \sa \blank  \ref TopicStorageOrders */
const unsigned int RowMajorBit = 0x1;

/** \ingroup flags
  * means the expression should be evaluated by the calling expression */
const unsigned int EvalBeforeNestingBit = 0x2;

/** \ingroup flags
  * \deprecated
  * means the expression should be evaluated before any assignment */
EIGEN_DEPRECATED
const unsigned int EvalBeforeAssigningBit = 0x4; // FIXME deprecated

/** \ingroup flags
  *
  * Short version: means the expression might be vectorized
  *
  * Long version: means that the coefficients can be handled by packets
  * and start at a memory location whose alignment meets the requirements
  * of the present CPU architecture for optimized packet access. In the fixed-size
  * case, there is the additional condition that it be possible to access all the
  * coefficients by packets (this implies the requirement that the size be a multiple of 16 bytes,
  * and that any nontrivial strides don't break the alignment). In the dynamic-size case,
  * there is no such condition on the total size and strides, so it might not be possible to access
  * all coeffs by packets.
  *
  * \note This bit can be set regardless of whether vectorization is actually enabled.
  *       To check for actual vectorizability, see \a ActualPacketAccessBit.
  */
const unsigned int PacketAccessBit = 0x8;

#ifdef EIGEN_VECTORIZE
/** \ingroup flags
  *
  * If vectorization is enabled (EIGEN_VECTORIZE is defined) this constant
  * is set to the value \a PacketAccessBit.
  *
  * If vectorization is not enabled (EIGEN_VECTORIZE is not defined) this constant
  * is set to the value 0.
  */
const unsigned int ActualPacketAccessBit = PacketAccessBit;
#else
const unsigned int ActualPacketAccessBit = 0x0;
#endif

/** \ingroup flags
  *
  * Short version: means the expression can be seen as 1D vector.
  *
  * Long version: means that one can access the coefficients
  * of this expression by coeff(int), and coeffRef(int) in the case of a lvalue expression. These
  * index-based access methods are guaranteed
  * to not have to do any runtime computation of a (row, col)-pair from the index, so that it
  * is guaranteed that whenever it is available, index-based access is at least as fast as
  * (row,col)-based access. Expressions for which that isn't possible don't have the LinearAccessBit.
  *
  * If both PacketAccessBit and LinearAccessBit are set, then the
  * packets of this expression can be accessed by packet(int), and writePacket(int) in the case of a
  * lvalue expression.
  *
  * Typically, all vector expressions have the LinearAccessBit, but there is one exception:
  * Product expressions don't have it, because it would be troublesome for vectorization, even when the
  * Product is a vector expression. Thus, vector Product expressions allow index-based coefficient access but
  * not index-based packet access, so they don't have the LinearAccessBit.
  */
const unsigned int LinearAccessBit = 0x10;

/** \ingroup flags
  *
  * Means the expression has a coeffRef() method, i.e. is writable as its individual coefficients are directly addressable.
  * This rules out read-only expressions.
  *
  * Note that DirectAccessBit and LvalueBit are mutually orthogonal, as there are examples of expression having one but note
  * the other:
  *   \li writable expressions that don't have a very simple memory layout as a strided array, have LvalueBit but not DirectAccessBit
  *   \li Map-to-const expressions, for example Map<const Matrix>, have DirectAccessBit but not LvalueBit
  *
  * Expressions having LvalueBit also have their coeff() method returning a const reference instead of returning a new value.
  */
const unsigned int LvalueBit = 0x20;

/** \ingroup flags
  *
  * Means that the underlying array of coefficients can be directly accessed as a plain strided array. The memory layout
  * of the array of coefficients must be exactly the natural one suggested by rows(), cols(),
  * outerStride(), innerStride(), and the RowMajorBit. This rules out expressions such as Diagonal, whose coefficients,
  * though referencable, do not have such a regular memory layout.
  *
  * See the comment on LvalueBit for an explanation of how LvalueBit and DirectAccessBit are mutually orthogonal.
  */
const unsigned int DirectAccessBit = 0x40;

/** \deprecated \ingroup flags
  *
  * means the first coefficient packet is guaranteed to be aligned.
  * An expression cannot has the AlignedBit without the PacketAccessBit flag.
  * In other words, this means we are allow to perform an aligned packet access to the first element regardless
  * of the expression kind:
  * \code
  * expression.packet<Aligned>(0);
  * \endcode
  */
EIGEN_DEPRECATED const unsigned int AlignedBit = 0x80;

const unsigned int NestByRefBit = 0x100;

/** \ingroup flags
  *
  * for an expression, this means that the storage order
  * can be either row-major or column-major.
  * The precise choice will be decided at evaluation time or when
  * combined with other expressions.
  * \sa \blank  \ref RowMajorBit, \ref TopicStorageOrders */
const unsigned int NoPreferredStorageOrderBit = 0x200;

/** \ingroup flags
  *
  * Means that the underlying coefficients can be accessed through pointers to the sparse (un)compressed storage format,
  * that is, the expression provides:
  * \code
    inline const Scalar* valuePtr() const;
    inline const Index* innerIndexPtr() const;
    inline const Index* outerIndexPtr() const;
    inline const Index* innerNonZeroPtr() const;
    \endcode
  */
const unsigned int CompressedAccessBit = 0x400;


// list of flags that are inherited by default
const unsigned int HereditaryBits = RowMajorBit
                                  | EvalBeforeNestingBit;

/** \defgroup enums Enumerations
  * \ingroup Core_Module
  *
  * Various enumerations used in %Eigen. Many of these are used as template parameters.
  */

/** \ingroup enums
  * Enum containing possible values for the \c Mode or \c UpLo parameter of
  * MatrixBase::selfadjointView() and MatrixBase::triangularView(), and selfadjoint solvers. */
enum UpLoType {
  /** View matrix as a lower triangular matrix. */
  Lower=0x1,                      
  /** View matrix as an upper triangular matrix. */
  Upper=0x2,                      
  /** %Matrix has ones on the diagonal; to be used in combination with #Lower or #Upper. */
  UnitDiag=0x4, 
  /** %Matrix has zeros on the diagonal; to be used in combination with #Lower or #Upper. */
  ZeroDiag=0x8,
  /** View matrix as a lower triangular matrix with ones on the diagonal. */
  UnitLower=UnitDiag|Lower, 
  /** View matrix as an upper triangular matrix with ones on the diagonal. */
  UnitUpper=UnitDiag|Upper,
  /** View matrix as a lower triangular matrix with zeros on the diagonal. */
  StrictlyLower=ZeroDiag|Lower, 
  /** View matrix as an upper triangular matrix with zeros on the diagonal. */
  StrictlyUpper=ZeroDiag|Upper,
  /** Used in BandMatrix and SelfAdjointView to indicate that the matrix is self-adjoint. */
  SelfAdjoint=0x10,
  /** Used to support symmetric, non-selfadjoint, complex matrices. */
  Symmetric=0x20
};

/** \ingroup enums
  * Enum for indicating whether a buffer is aligned or not. */
enum AlignmentType {
  Unaligned=0,        /**< Data pointer has no specific alignment. */
  Aligned8=8,         /**< Data pointer is aligned on a 8 bytes boundary. */
  Aligned16=16,       /**< Data pointer is aligned on a 16 bytes boundary. */
  Aligned32=32,       /**< Data pointer is aligned on a 32 bytes boundary. */
  Aligned64=64,       /**< Data pointer is aligned on a 64 bytes boundary. */
  Aligned128=128,     /**< Data pointer is aligned on a 128 bytes boundary. */
  AlignedMask=255,
  Aligned=16,         /**< \deprecated Synonym for Aligned16. */
#if EIGEN_MAX_ALIGN_BYTES==128
  AlignedMax = Aligned128
#elif EIGEN_MAX_ALIGN_BYTES==64
  AlignedMax = Aligned64
#elif EIGEN_MAX_ALIGN_BYTES==32
  AlignedMax = Aligned32
#elif EIGEN_MAX_ALIGN_BYTES==16
  AlignedMax = Aligned16
#elif EIGEN_MAX_ALIGN_BYTES==8
  AlignedMax = Aligned8
#elif EIGEN_MAX_ALIGN_BYTES==0
  AlignedMax = Unaligned
#else
#error Invalid value for EIGEN_MAX_ALIGN_BYTES
#endif
};

/** \ingroup enums
 * Enum used by DenseBase::corner() in Eigen2 compatibility mode. */
// FIXME after the corner() API change, this was not needed anymore, except by AlignedBox
// TODO: find out what to do with that. Adapt the AlignedBox API ?
enum CornerType { TopLeft, TopRight, BottomLeft, BottomRight };

/** \ingroup enums
  * Enum containing possible values for the \p Direction parameter of
  * Reverse, PartialReduxExpr and VectorwiseOp. */
enum DirectionType { 
  /** For Reverse, all columns are reversed; 
    * for PartialReduxExpr and VectorwiseOp, act on columns. */
  