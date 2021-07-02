// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_ZVECTOR_H
#define EIGEN_PACKET_MATH_ZVECTOR_H

#include <stdint.h>

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 4
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#endif

// NOTE Altivec has 32 registers, but Eigen only accepts a value of 8 or 16
#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS  32
#endif

typedef __vector int                 Packet4i;
typedef __vector unsigned int        Packet4ui;
typedef __vector __bool int          Packet4bi;
typedef __vector short int           Packet8i;
typedef __vector unsigned char       Packet16uc;
typedef __vector double              Packet2d;
typedef __vector unsigned long long  Packet2ul;
typedef __vector long long           Packet2l;

typedef union {
  int32_t   i[4];
  uint32_t ui[4];
  int64_t   l[2];
  uint64_t ul[2];
  double    d[2];
  Packet4i  v4i;
  Packet4ui v4ui;
  Packet2l  v2l;
  Packet2ul v2ul;
  Packet2d  v2d;
} Packet;

// We don't want to write the same code all the time, but we need to reuse the constants
// and it doesn't really work to declare them global, so we define macros instead

#define _EIGEN_DECLARE_CONST_FAST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = reinterpret_cast<Packet4i>(vec_splat_s32(X))

#define _EIGEN_DECLARE_CONST_FAST_Packet2d(NAME,X) \
  Packet2d p2d_##NAME = reinterpret_cast<Packet2d>(vec_splat_s64(X))

#define _EIGEN_DECLARE_CONST_FAST_Packet2l(NAME,X) \
  Packet2l p2l_##NAME = reinterpret_cast<Packet2l>(vec_splat_s64(X))

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = pset1<Packet4i>(X)

#define _EIGEN_DECLARE_CONST_Packet2d(NAME,X) \
  Packet2d p2d_##NAME = pset1<Packet2d>(X)

#define _EIGEN_DECLARE_CONST_Packet2l(NAME,X) \
  Packet2l p2l_##NAME = pset1<Packet2l>(X)

// These constants are endian-agnostic
//static _EIGEN_DECLARE_CONST_FAST_Packet4i(ZERO, 0); //{ 0, 0, 0, 0,}
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ONE, 1); //{ 1, 1, 1, 1}

static _EIGEN_DECLARE_CONST_FAST_Packet2d(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet2l(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet2l(ONE, 1);

static Packet2d p2d_ONE = { 1.0, 1.0 }; 
static Packet2d p2d_ZERO_ = { -0.0, -0.0 };

static Packet4i p4i_COUNTDOWN = { 0, 1, 2, 3 };
static Packet2d p2d_COUNTDOWN = reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet16uc>(p2d_ZERO), reinterpret_cast<Packet16uc>(p2d_ONE), 8));

static Packet16uc p16uc_PSET64_HI = { 0,1,2,3, 4,5,6,7, 0,1,2,3, 4,5,6,7 };
static Packet16uc p16uc_DUPLICATE32_HI = { 0,1,2,3, 0,1,2,3, 4,5,6,7, 4,5,6,7 };

// Mask alignment
#define _EIGEN_MASK_ALIGNMENT	0xfffffffffffffff0

#define _EIGEN_ALIGNED_PTR(x)	((ptrdiff_t)(x) & _EIGEN_MASK_ALIGNMENT)

// Handle endianness properly while loading constants
// Define global static constants:

static Packet16uc p16uc_FORWARD =   { 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15 };
static Packet16uc p16uc_REVERSE32 = { 12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3 };
static Packet16uc p16uc_REVERSE64 = { 8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7 };

static Packet16uc p16uc_PSET32_WODD   = vec_sld((Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 0), (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 2), 8);//{ 0,1,2,3, 0,1,2,3, 8,9,10,11, 8,9,10,11 };
static Packet16uc p16uc_PSET32_WEVEN  = vec_sld(p16uc_DUPLICATE32_HI, (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 3), 8);//{ 4,5,6,7, 4,5,6,7, 12,13,14,15, 12,13,14,15 };
/*static Packet16uc p16uc_HALF64_0_16 = vec_sld((Packet16uc)p4i_ZERO, vec_splat((Packet16uc) vec_abs(p4i_MINUS16), 3), 8);      //{ 0,0,0,0, 0,0,0,0, 16,16,16,16, 16,16,16,16};

static Packet16uc p16uc_PSET64_HI = (Packet16uc) vec_mergeh((Packet4ui)p16uc_PSET32_WODD, (Packet4ui)p16uc_PSET32_WEVEN);     //{ 0,1,2,3, 4,5,6,7, 0,1,2,3, 4,5,6,7 };*/
static Packet16uc p16uc_PSET64_LO = (Packet16uc) vec_mergel((Packet4ui)p16uc_PSET32_WODD, (Packet4ui)p16uc_PSET32_WEVEN);     //{ 8,9,10,11, 12,13,14,15, 8,9,10,11, 12,13,14,15 };
/*static Packet16uc p16uc_TRANSPOSE64_HI = vec_add(p16uc_PSET64_HI, p16uc_HALF64_0_16);                                         //{ 0,1,2,3, 4,5,6,7, 16,17,18,19, 20,21,22,23};
static Packet16uc p16uc_TRANSPOSE64_LO = vec_add(p16uc_PSET64_LO, p16uc_HALF64_0_16);                                         //{ 8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31};*/
static Packet16uc p16uc_TRANSPOSE64_HI = { 0,1,2,3, 4,5,6,7, 16,17,18,19, 20,21,22,23};
static Packet16uc p16uc_TRANSPOSE64_LO = { 8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31};

//static Packet16uc p16uc_COMPLEX32_REV = vec_sld(p16uc_REVERSE32, p16uc_REVERSE32, 8);                                         //{ 4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11 };

//static Packet16uc p16uc_COMPLEX32_REV2 = vec_sld(p16uc_FORWARD, p16uc_FORWARD, 8);                                            //{ 8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7 };


#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  #define EIGEN_ZVECTOR_PREFETCH(ADDR) __builtin_prefetch(ADDR);
#else
  #define EIGEN_ZVECTOR_PREFETCH(ADDR) asm( "   pfd [%[addr]]\n" :: [addr] "r" (ADDR) : "cc" );
#endif

template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet4i type;
  typedef Packet4i half;
  enum {
    // FIXME check the Has*
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 0,

    // FIXME check the Has*
    HasAdd  = 1,
    HasSub  = 1,
    HasMul  = 1,
    HasDiv  = 1,
    HasBlend = 1
  };
};

template<> struct packet_traits<double> : default_packet_traits
{
  typedef Packet2d type;
  typedef Packet2d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 1,

    // FIXME check the Has*
    HasAdd  = 1,
    HasSub  = 1,
    HasMul  = 1,
    HasDiv  = 1,
    HasMin  = 1,
    HasMax  = 1,
    HasAbs  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 0,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasRound = 1,
    HasFloor = 1,
    HasCeil = 1,
    HasNegate = 1,
    HasBlend = 1
  };
};

template<> struct unpacket_traits<Packet4i> { typedef int    type; enum {size=4, alignment=Aligned16}; typedef Packet4i half; };
template<> str