// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010-2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX32_ALTIVEC_H
#define EIGEN_COMPLEX32_ALTIVEC_H

namespace Eigen {

namespace internal {

static Packet4ui  p4ui_CONJ_XOR = vec_mergeh((Packet4ui)p4i_ZERO, (Packet4ui)p4f_ZERO_);//{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 };
#ifdef __VSX__
#if defined(_BIG_ENDIAN)
static Packet2ul  p2ul_CONJ_XOR1 = (Packet2ul) vec_sld((Packet4ui) p2d_ZERO_, (Packet4ui) p2l_ZERO, 8);//{ 0x8000000000000000, 0x0000000000000000 };
static Packet2ul  p2ul_CONJ_XOR2 = (Packet2ul) vec_sld((Packet4ui) p2l_ZERO,  (Packet4ui) p2d_ZERO_, 8);//{ 0x8000000000000000, 0x0000000000000000 };
#else
static Packet2ul  p2ul_CONJ_XOR1 = (Packet2ul) vec_sld((Packet4ui) p2l_ZERO,  (Packet4ui) p2d_ZERO_, 8);//{ 0x8000000000