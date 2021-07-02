// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin, cos, exp, and log functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_SSE_H
#define EIGEN_MATH_FUNCTIONS_SSE_H

namespace Eigen {

namespace internal {

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f plog<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);

  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(inv_mant_mask, ~0x7f800000);

  /* the smallest non denormalized float number */
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(min_norm_pos,  0x00800000);
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(minus_inf,     0xff800000);//-1.f/0.f);
  
  /* natural logarithm computed for 4 simultaneous float
    return NaN for x <= 0
  */
  _EIGEN_DECLARE_CONST_Packet4f(cephes_SQRTHF, 0.707106781186547524f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p0, 7.0376836292E-2f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p1, - 1.1514610310E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p2, 1.1676998740E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p3, - 1.2420140846E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p4, + 1.4249322787E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p5, - 1.6668057665E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p6, + 2.0000714765E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p7, - 2.4999993993E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_p8, + 3.3333331174E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_q1, -2.12194440e-4f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_log_q2, 0.693359375f);


  Packet4i emm0;

  Packet4f invalid_mask = _mm_cmpnge_ps(x, _mm_setzero_ps()); // not greater equal is true if x is NaN
  Packet4f iszero_mask = _mm_cmpeq_ps(x, _mm_setzero_ps());

  x = pmax(x, p4f_min_norm_pos);  /* cut off denormalized stuff */
  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

  /* keep only the fractional part */
  x = _mm_and_ps(x, p4f_inv_mant_mask);
  x = _mm_or_ps(x, p4f_half);

  emm0 = _mm_sub_epi32(emm0, p4i_0x7f);
  Packet4f e = padd(Packet4f(_mm_cvtepi32_ps(emm0)), p4f_1);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  Packet4f mask = _mm_cmplt_ps(x, p4f_cephes_SQRTHF);
  Packet4f tmp = pand(x, mask);
  x = psub(x, p4f_1);
  e = psub(e, pand(p4f_1, mask));
  x = padd(x, tmp);

  Packet4f x2 = pmul(x,x);
  Packet4f x3 = pmul(x2,x);

  Packet4f y, y1, y2;
  y  = pmadd(p4f_cephes_log_p0, x, p4f_cephes_log_p1);
  y1 = pmadd(p4f_cephes_log_p3, x, p4f_cephes_log_p4);
  y2 = pmadd(p4f_cephes_log_p6, x, p4f_cephes_log_p7);
  y  = pmadd(y , x, p4f_cephes_log_p2);
  y1 = pmadd(y1, x, p4f_cephes_log_p5);
  y2 = pmadd(y2, x, p4f_cephes_log_p8);
  y = pmadd(y, x3, y1);
  y = pmadd(y, x3, y2);
  y = pmul(y, x3);

  y1 = pmul(e, p4f_cephes_log_q1);
  tmp = pmul(x2, p4f_half);
  y = padd(y, y1);
  x = psub(x, tmp);
  y2 = pmul(e, p4f_cephes_log_q2);
  x = padd(x, y);
  x = padd(x, y2);
  // negative arg will be NAN, 0 will be -INF
  return _mm_or_ps(_mm_andnot_ps(iszero_mask, _mm_or_ps(x, invalid_mask)),
                   _mm_and_ps(iszero_mask, p4f_minus_inf));
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);


  _EIGEN_DECLARE_CONST_Packet4f(exp_hi,  88.3762626647950f);
  _EIGEN_DECLARE_CONST_Packet4f(exp_lo, -88.3762626647949f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_LOG2EF, 1.44269504088896341f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C1, 0.693359375f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C2, -2.12194440e-4f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p0, 1.9875691500E-4f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p1, 1.3981999507E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p2, 8.3334519073E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p3, 4.1665795894E-2f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p4, 1.6666665459E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p5, 5.0000001201E-1f);

  Packet4f tmp, fx;
  Packet4i emm0;

  // clamp x
  x = pmax(pmin(x, p4f_exp_hi), p4f_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(x, p4f_cephes_LOG2EF, p4f_half);

#ifdef EIGEN_VECTORIZE_SSE4_1
  fx = _mm_floor_ps(fx);
#else
  emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);
  /* if greater, substract 1 */
  Packet4f mask = _mm_cmpgt_ps(tmp, fx);
  mask = _mm_and_ps(mask, p4f_1);
  fx = psub(tmp, mask);
#endif

  tmp = pmul(fx, p4f_cephes_exp_C1);
  Packet4f z = pmul(fx, p4f_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  z = pmul(x,x);

  Packet4f y = p4f_cephes_exp_p0;
  y = pmadd(y, x, p4f_cephes_exp_p1);
  y = pmadd(y, x, p4f_cephes_exp_p2);
  y = pmadd(y, x, p4f_cephes_exp_p3);
  y = pmadd(y, x, p4f_cephes_exp_p4);
  y = pmadd(y, x, p4f_cephes_exp_p5);
  y = pmadd(y, z, x);
  y = padd(y, p4f_1);

  // build 2^n
  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, p4i_0x7f);
  emm0 = _mm_slli_epi32(emm0, 23);
  return pmax(pmul(y, Packet4f(_mm_castsi128_ps(emm0))), _x);
}
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d& _x)
{
  Packet2d x = _x;

  _EIGEN_DECLARE_CONST_Packet2d(1 , 1.0);
  _EIGEN_DECLARE_CONST_Packet2d(2 , 2.0);
  _EIGEN_DECLARE_CONST_Packet2d(half, 0.5);

  _EIGEN_DECLARE_CONST_Packet2d(exp_hi,  709.437);
  _EIGEN_DECLARE_CONST_Packet2d(exp_lo, -709.436139303);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_LOG2EF, 1.4426950408889634073599);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p0, 1.26177193074810590878e-4);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p1, 3.02994407707441961300e-2);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p2, 9.99999999999999999910e-1);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q0, 3.00198505138664455042e-6);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q1, 2.52448340349684104192e-3);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q2, 2.27265548208155028766e-1);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q3, 2.00000000000000000009e0);

  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C1, 0.693145751953125);
  _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C2, 1.42860682030941723212e-6);
  static const __m128i p4i_1023_0 = _mm_setr_epi32(1023, 1023, 0, 0);

  Packet2d tmp, fx;
  Packet4i emm0;

  // clamp x
  x = pmax(pmin(x, p2d_exp_hi), p2d_exp_lo);
  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(p2d_cephes_LOG2EF, x, p2d_half);

#ifdef EIGEN_VECTORIZE_SSE4_1
  fx = _mm_floor_pd(fx);
#else
  emm0 = _mm_cvttpd_ep