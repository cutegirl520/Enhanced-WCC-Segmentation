// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLASUTIL_H
#define EIGEN_BLASUTIL_H

// This file contains many lightweight helper classes used to
// implement and control fast level 2 and level 3 BLAS-like routines.

namespace Eigen {

namespace internal {

// forward declarations
template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs=false, bool ConjugateRhs=false>
struct gebp_kernel;

template<typename Scalar, typename Index, typename DataMapper, int nr, int StorageOrder, bool Conjugate = false, bool PanelMode=false>
struct gemm_pack_rhs;

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, int StorageOrder, bool Conjugate = false, bool PanelMode = false>
struct gemm_pack_lhs;

template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResStorageOrder>
struct general_matrix_matrix_product;

template<typename Index,
         typename LhsScalar, typename LhsMapper, int LhsStorageOrder, bool ConjugateLhs,
         typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version=Specialized>
struct general_matrix_vector_product;


template<bool Conjugate> struct conj_if;

template<> struct conj_if<true> {
  template<typename T>
  inline T operator()(const T& x) { return numext::conj(x); }
  template<typename T>
  inline T pconj(const T& x) { return internal::pconj(x); }
};

template<> struct conj_if<false> {
  template<typename T>
  inline const T& operator()(const T& x) { return x; }
  template<typename T>
  inline const T& pconj(const T& x) { return x; }
};

template<typename Scalar> struct conj_helper<Scalar,Scalar,false,false>
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const { return internal::pmadd(x,y,c); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const { return internal::pmul(x,y); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, false,true>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) + numext::imag(x)*numext::imag(y), numext::imag(x)*numext::real(y) - numext::real(x)*numext::imag(y)); }
};

template<typename RealScalar> struct