// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_REDUX_H
#define EIGEN_REDUX_H

namespace Eigen { 

namespace internal {

// TODO
//  * implement other kind of vectorization
//  * factorize code

/***************************************************************************
* Part 1 : the logic deciding a strategy for vectorization and unrolling
***************************************************************************/

template<typename Func, typename Derived>
struct redux_traits
{
public:
    typedef typename find_best_packet<typename Derived::Scalar,Derived::SizeAtCompileTime>::type PacketType;
  enum {
    PacketSize = unpacket_traits<PacketType>::size,
    InnerMaxSize = int(Derived::IsRowMajor)
                 ? Derived::MaxColsAtCompileTime
                 : Derived::MaxRowsAtCompileTime
  };

  enum {
    MightVectorize = (int(Derived::Flags)&ActualPacketAccessBit)
                  && (functor_traits<Func>::PacketAccess),
    MayLinearVectorize = bool(MightVectorize) && (int(Derived::Flags)&LinearAccessBit),
    MaySliceVectorize  = bool(MightVectorize) && int(InnerMaxSize)>=3*PacketSize
  };

public:
  enum {
    Traversal = int(MayLinearVectorize) ? int(LinearVectorizedTraversal)
              : int(MaySliceVectorize)  ? int(SliceVectorizedTraversal)
                                        : int(DefaultTraversal)
  };

public:
  enum {
    Cost = Derived::SizeAtCompileTime == Dynamic ? HugeCost
         : Derived::SizeAtCompileTime * Derived::CoeffReadCost + (Derived::SizeAtCompileTime-1) * functor_traits<Func>::Cost,
    UnrollingLimit = EIGEN_UNROLLING_LIMIT * (int(Traversal) == int(DefaultTraversal) ? 1 : int(PacketSize))
  };

public:
  enum {
    Unrolling = Cost <= UnrollingLimit ? CompleteUnrolling : NoUnrolling
  };
  
#ifdef EIGEN_DEBUG_ASSIGN
  static void debug()
  {
    std::cerr << "Xpr: " << typeid(typename Derived::XprType).name() << std::endl;
    std::cerr.setf(std::ios::hex, std::ios::basefield);
    EIGEN_DEBUG_VAR(Derived::Flags)
    std::cerr.unsetf(std::ios::hex);
    EIGEN_DEBUG_VAR(InnerMaxSize)
    EIGEN_DEBUG_VAR(PacketSize)
    EIGEN_DEBUG_VAR(MightVectorize)
    EIGEN_DEBUG_VAR(MayLinearVectorize)
    EIGEN_DEBUG_VAR(MaySliceVectorize)
    EIGEN_DEBUG_VAR(Traversal)
    EIGEN_DEBUG_VAR(UnrollingLimit)
    EIGEN_DEBUG_VAR(Unrolling)
    std::cerr << std::endl;
  }
#endif
};

/***************************************************************************
* Part 2 : unrollers
***************************************************************************/

/*** no vectorization ***/

template<typename Func, typename Derived, int Start, int Length>
struct redux_novec_unroller
{
  enum {
    HalfLength = Length/2
  };

  typedef typename Derived::Scalar Scalar;

  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Derived &mat, const Func& func)
  {
    return func(redux_novec_unroller<Func, Derived, Start, HalfLength>::run(mat,func),
                redux_novec_unroller<Func, Derived, Start+HalfLength, Length-HalfLength>::run(mat,func));
  }
};

template<typename Func, typename Derived, int Start>
struct redux_novec_unroller<Func, Derived, Start, 1>
{
  enum {
    outer = Start / Derived::InnerSizeAtCompileTime,
    inner = Start % Derived::InnerSizeAtCompileTime
  };

  typedef typename Derived::Scalar Scalar;

  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Derived &mat, const Func&)
  {
    return mat.coeffByOuterInner(outer, inner);
  }
};

// This is actually dead code and will never be called. It is required
// to prevent false warnings regarding failed inlining though
// for 0 length run() will never be called at all.
template<typename Func, typename Derived, int Start>
struct redux_novec_unroller<Func, Derived, Start, 0>
{
  typedef typename Derived::Scalar Scalar;
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE Scalar run(const Derived&, const Func&) { return Scalar(); }
};

/*** vectorization ***/

template<typename Func, typename Derived, int Start, int Length>
struct redux_vec_unroller
{
  enum {
    PacketSize = redux_traits<Func, Derived>::PacketSize,
    HalfLength = Length/2
  };

  typedef typename Derived::Scalar Scalar;
  typedef typename redux_traits<Func, Derived>::PacketType PacketScalar;

  static EIGEN_STRONG_INLINE PacketScalar run(const Derived &mat, const Func& func)
  {
    return func.packetOp(
            redux_vec_unroller<Func, Derived, Start, HalfLength>::run(mat,func),
            redux_vec_unroller<Func, Derived, Start+HalfLength, Length-HalfLength>::run(mat,func) );
  }
};

template<typename Func, typename Derived, int Start>
struct redux_vec_unroller<Func, Derived, Start, 1>
{
  enum {
    index = Start * redux_traits<Func, Derived>::PacketSize,
    outer = index / int(Derived::InnerSizeAtCompileTime),
    inner = index % int(Derived::InnerSizeAtCompileTime),
    alignment = Derived::Alignment
  };

  typedef typename Derived::Scalar Scalar;
  typedef typename redux_traits<Func, Derived>::PacketType PacketScalar;

  static EIGEN_STRONG_INLINE PacketScalar run(const Derived &mat, const Func&)
  {
    return mat.template packetByOuterInner<alignment,PacketScalar>(outer, inner);
  }
};

/***************************************************************************
* Part 3 : implementation of all cases
***************************************************************************/

template<typename Func, typename Derived,
         int Traversal = redux_traits<Func, Derived>::Traversal,
         int Unrolling = redux_traits<Func, Derived>::Unrolling
>
struct redux_impl;

template<typename Func, typename Derived>
struct redux_impl<Func, Derived, DefaultTraversal, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Derived &mat, const Func& func)
  {
    eigen_assert(mat.rows()>0 && mat.cols()>0 && "you are using an empty matrix");
    Scalar res;
    res = mat.coeffByOuterInner(0, 0);
    for(Index i = 1; i < mat.innerSize(); ++i)
      res = func(res, mat.coeffByOuterInner(0, i));
    for(Index i = 1; i < mat.outerSize(); ++i)
      for(Index j = 0; j < mat.innerSize(); ++j)
        res = func(res, mat.coeffByOuterInner(i, j));
    return res;
  }
};

template<typename Func, typename Derived>
struct redux_impl<Func,Derived, DefaultTraversal, CompleteUnrolling>
  : public redux_novec_unroller<Func,Derived, 0, Derived::SizeAtCompileTime>
{};

template<typename Func, typename Derived>
struct redux_impl<Func, Derived, LinearVectorizedTraversal, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename redux_traits<Func, Derived>::PacketType PacketScalar;

  static Scalar run(const Derived &mat, const Func& func)
  {
    const Index size = mat.size();
    
    const Index packetSize = redux_traits<Func, Derived>::PacketSize;
    const int packetAlignment = unpacket_traits<PacketScalar>::alignment;
    enum {
      alignment0 = (bool(Derived::Flags & DirectAccessBit) && bool(packet_traits<Scalar>::AlignedOnScalar)) ? int(packetAlignment) : int(Unaligned),
      alignment = EIGEN_PLAIN_ENUM_MAX(alignment0, Derived::Alignment)
    };
    const Index alignedStart = internal::first_default_aligned(mat.nestedExpression());
    const Index alignedSize2 = ((size-alignedStart)/(2*packetSize))*(2*packetSize);
    const Index alignedSize = ((size-alignedStart)/(packetSize))*(packetSize);
    const Index alignedEnd2 = alignedStart + alignedSize2;
    const Index alignedEnd  = alignedStart + alignedSize;
    Scalar res;
    if(alignedSize)
    {
      PacketScalar packet_res0 = mat.template packet<alignment,PacketScalar>(alignedStart);
      if(alignedSize>packetSize) // we have at least two packets to partly unroll the loop
      {
        PacketScalar packet_res1 = mat.template packet<alignment,PacketScalar>(alignedStart+packetSize);
        for(Index index = alignedStart + 2*packetSize; index < alignedEnd2; index += 2*packetSize)
        {
          packet_res0 = func.packetOp(packet_res0, mat.template packet<alignment,PacketScalar>(index));
          packet_res1 = func.packetOp(packet_res1, mat.template packet<alignment,PacketScalar>(index+packetSize));
        }

        packet_res0 = func.packetOp(packet_res0,packet_res1);
        if(alignedEnd>alignedEnd2)
          packet_res0 = func.packetOp(packet_res0, mat.template packet<alignment,PacketScalar>(alignedEnd2));
      }
      res = func.predux(packet_res0);

      for(Index index = 0; index < alignedStart; ++index)
        res = func(res,mat.coeff(index));

      for(Index index = alignedEnd; index < size; ++index)
        res = func(res,mat.coeff(index));
    }
    else // too small to vectorize anything.
         // since this is dynamic-size hence inefficient anyway for such small sizes, don't try to optimize.
    {
      res = mat.coeff(0);
      for(Index index = 1; index < size; ++index)
        res = func(res,mat.coeff(index));
    }

    return res;
  }
};

// NOTE: for SliceVectorizedTraversal we simply bypass unrolling
template<typename Func, typename Derived, int Unrolling>
struct redux_impl<Func, Derived, SliceVectorizedTraversal, Unrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename redux_traits<Func, Derived>::PacketType PacketType;

  EIGEN_DEVICE_FUNC static Scalar run(const Derived &mat, const Func& func)
  {
    eigen_assert(mat.rows()>0 && mat.cols()>0 && "you are using an empty matrix");
    const Index innerSize = mat.innerSize();
    const Index outerSize = mat.outerSize();
    enum {
      packetSize = redux_traits<Func, Derived>::PacketSize
    };
    const Index packetedInnerSize = ((innerSize)/packetSize)*packetSize;
    Scalar res;
    if(packetedInnerSize)
    {
      PacketType packet_res = mat.template packet<Unaligned,PacketType>(0,0);
      for(Index j=0; j<outerSize; ++j)
        for(Index i=(j==0?packetSize:0); i<packetedInnerSize; i+=Index(packetSize))
 