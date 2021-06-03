// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

// source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
// TODO this should better be moved to NumTraits
#define EIGEN_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406L


namespace Eigen {

// On WINCE, std::abs is defined for int only, so let's defined our own overloads:
// This issue has been confirmed with MSVC 2008 only, but the issue might exist for more recent versions too.
#if EIGEN_OS_WINCE && EIGEN_COMP_MSVC && EIGEN_COMP_MSVC<=1500
long        abs(long        x) { return (labs(x));  }
double      abs(double      x) { return (fabs(x));  }
float       abs(float       x) { return (fabsf(x)); }
long double abs(long double x) { return (fabsl(x)); }
#endif

namespace internal {

/** \internal \class global_math_functions_filtering_base
  *
  * What it does:
  * Defines a typedef 'type' as follows:
  * - if type T has a member typedef Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl, then
  *   global_math_functions_filtering_base<T>::type is a typedef for it.
  * - otherwise, global_math_functions_filtering_base<T>::type is a typedef for T.
  *
  * How it's used:
  * To allow to defined the global math functions (like sin...) in certain cases, like the Array expressions.
  * When you do sin(array1+array2), the object array1+array2 has a complicated expression type, all what you want to know
  * is that it inherits ArrayBase. So we implement a partial specialization of sin_impl for ArrayBase<Derived>.
  * So we must make sure to use sin_impl<ArrayBase<Derived> > and not sin_impl<Derived>, otherwise our partial specialization
  * won't be used. How does sin know that? That's exactly what global_math_functions_filtering_base tells it.
  *
  * How it's implemented:
  * SFINAE in the style of enable_if. Highly susceptible of breaking compilers. With GCC, it sure does work, but if you replace
  * the typename dummy by an integer template parameter, it doesn't work anymore!
  */

template<typename T, typename dummy = void>
struct global_math_functions_filtering_base
{
  typedef T type;
};

template<typename T> struct always_void { typedef void type; };

template<typename T>
struct global_math_functions_filtering_base
  <T,
   typename always_void<typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl>::type
  >
{
  typedef typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl type;
};

#define EIGEN_MATHFUNC_IMPL(func, scalar) Eigen::internal::func##_impl<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>
#define EIGEN_MATHFUNC_RETVAL(func, scalar) typename Eigen::internal::func##_retval<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>::type

/****************************************************************************
* Implementation of real                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct real_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct real_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::real;
    return real(x);
  }
};

template<typename Scalar> struct real_impl : real_default_impl<Scalar> {};

template<typename Scalar>
struct real_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of imag                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct imag_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar&)
  {
    return RealScalar(0);
  }
};

template<typename Scalar>
struct imag_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::imag;
    return imag(x);
  }
};

template<typename Scalar> struct imag_impl : imag_default_impl<Scalar> {};

template<typename Scalar>
struct imag_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of real_ref                                             *
****************************************************************************/

template<typename Scalar>
struct real_ref_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[0];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<const RealScalar*>(&x)[0];
  }
};

template<typename Scalar>
struct real_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of imag_ref                                             *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct imag_ref_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
};

template<typename Scalar>
struct imag_ref_default_impl<Scalar, false>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(Scalar&)
  {
    return Scalar(0);
  }
  EIGEN_DEVICE_FUNC
  static inline const Scalar run(const Scalar&)
  {
    return Scalar(0);
  }
};

template<typename Scalar>
struct imag_ref_impl : imag_ref_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct imag_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of conj                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct conj_impl
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct conj_impl<Scalar,true>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    using std::conj;
    return conj(x);
  }
};

template<typename Scalar>
struct conj_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of abs2                                                 *
****************************************************************************/

template<typename Scalar,bool IsComplex>
struct abs2_impl_default
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x*x;
  }
};

template<typename Scalar>
struct abs2_impl_default<Scalar, true> // IsComplex
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return real(x)*real(x) + imag(x)*imag(x);
  }
};

template<typename Scalar>
struct abs2_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return abs2_impl_default<Scalar,NumTraits<Scalar>::IsComplex>::run(x);
  }
};

template<typename Scalar>
struct abs2_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of norm1                                                *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct norm1_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    EIGEN_USING_STD_MATH(abs);
    return 