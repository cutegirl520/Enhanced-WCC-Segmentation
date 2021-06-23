// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_NEON_H
#define EIGEN_COMPLEX_NEON_H

namespace Eigen {

namespace internal {

inline uint32x4_t p4ui_CONJ_XOR() {
  static const uint32_t conj_XOR_DATA[] = { 0x00000000, 0x80000000, 0x00000000, 0x80000000 };
  return vld1q_u32( conj_XOR_DATA );
}

inline uint32x2_t p2ui_CONJ_XOR() {
  static const uint32_t conj_XOR_DATA[] = { 0x00000000, 0x80000000 };
  return vld1_u32( conj_XOR_DATA );
}

//---------- float ----------
struct Packet2cf
{
  EIGEN_STRONG_INLINE Packet2cf() {}
  EIGEN_STRONG_INLINE explicit Packet2cf(const Packet4f& a) : v(a) {}
  Packet4f  v;
};

template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet2cf type;
  typedef Packet2cf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,
    HasHalfPacket = 0,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cf> { typedef std::complex<float> type; enum {size=2, alignment=Aligned16}; typedef Packet2cf half; };

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  float32x2_t r64;
  r64 = vld1_f32((float *)&from);

  return Packet2cf(vcombine_f32(r64, r64));
}

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(padd<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(psub<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return Packet2cf(pnegate<Packet4f>(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a)
{
  Packet4ui b = vreinterpretq_u32_f32(a.v);
  return Packet2cf(vreinterpretq_f32_u32(veorq_u32(b, p4ui_CONJ_XOR())));
}

template<> EIGEN_STRONG_INLINE Packe