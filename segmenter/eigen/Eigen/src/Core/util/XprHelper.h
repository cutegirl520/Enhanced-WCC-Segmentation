// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_XPRHELPER_H
#define EIGEN_XPRHELPER_H

// just a workaround because GCC seems to not really like empty structs
// FIXME: gcc 4.3 generates bad code when strict-aliasing is enabled
// so currently we simply disable this optimization for gcc 4.3
#if EIGEN_COMP_GNUC && !EIGEN_GNUC_AT(4,3)
  #define EIGEN_EMPTY_STRUCT_CTOR(X) \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X() {} \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X(const X& ) {}
#else
  #define EIGEN_EMPTY_STRUCT_CTOR(X)
#endif

namespace Eigen {

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE DenseIndex;

/**
 * \brief The Index type as used for the API.
 * \details To change this, \c \#define the preprocessor symbol \c EIGEN_DEFAULT_DENSE_INDEX_TYPE.
 * \sa \blank \ref TopicPreprocessorDirectives, StorageIndex.
 */

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE Index;

namespace internal {

template<typename IndexDest, typename IndexSrc>
EIGEN_DEVICE_FUNC
inline IndexDest convert_index(const IndexSrc& idx) {
  // for sizeof(IndexDest)>=sizeof(IndexSrc) compilers should be able to optimize this away:
  eigen_internal_assert(idx <= NumTraits<IndexDest>::highest() && "Index value to big for target type");
  return IndexDest(idx);
}


// promote_scalar_arg is an helper used in operation between an expression and a scalar, like:
//    expression * scalar
// Its role is to determine how the type T of the scalar operand should be promoted given the scalar type ExprScalar of the given expression.
// The IsSupported template parameter must be provided by the caller as: ScalarBinaryOpTraits<ExprScalar,T,op>::Defined using the proper order for ExprScalar and T.
// Then the logic is as follows:
//  - if the operation is natively supported as defined by IsSupported, then the scalar type is not promoted, and T is returned.
//  - otherwise, NumTraits<ExprScalar>::Literal is returned if T is implicitly convertible to NumTraits<ExprScalar>::Literal AND that this does not imply a float to integer conversion.
//  - otherwise, ExprScalar is returned if T is implicitly convertible to ExprScalar AND that this does not imply a float to integer conversion.
//  - In all other cases, the promoted type is not defined, and the respective operation is thus invalid and not available (SFINAE).
template<typename ExprScalar,typename T, bool IsSupported>
struct promote_scalar_arg;

template<typename S,typename T>
struct promote_scalar_arg<S,T,true>
{
  typedef T type;
};

// Recursively check safe conversion to PromotedType, and then ExprScalar if they are different.
template<typename ExprScalar,typename T,typename PromotedType,
  bool ConvertibleToLiteral = internal::is_convertible<T,PromotedType>::value,
  bool IsSafe = NumTraits<T>::IsInteger || !NumTraits<PromotedType>::IsInteger>
struct promote_scalar_arg_unsupported;

// Start recursion with NumTraits<ExprScalar>::Literal
template<typename S,typename T>
struct promote_scalar_arg<S,T,false> : promote_scalar_arg_unsupported<S,T,typename NumTraits<S>::Literal> {};

// We found a match!
template<typename S,typename T, typename PromotedType>
struct promote_scalar_arg_unsupported<S,T,PromotedType,true,true>
{
  typedef PromotedType type;
};

// No match, but no real-to-integer issues, and ExprScalar and current PromotedType are different,
// so let's try to promote to ExprScalar
template<typename ExprScalar,typename T, typename PromotedType>
struct promote_scalar_arg_unsupported<ExprScalar,T,PromotedType,false,true>
   : promote_scalar_arg_unsupported<ExprScalar,T,ExprScalar>
{};

// Unsafe real-to-integer, let's stop.
template<typename S,typename T, typename PromotedType, bool ConvertibleToLiteral>
struct promote_scalar_arg_unsupported<S,T,PromotedType,ConvertibleToLiteral,false> {};

// T is not even convertible to ExprScalar, let's stop.
template<typename S,typename T>
struct promote_scalar_arg_unsupported<S,T,S,false,true> {};

//classes inheriting no_assignment_operator don't generate a default operator=.
class no_assignment_operator
{
  private:
    no_assignment_operator& operator=(const no_assignment_operator&);
};

/** \internal return the index type with the largest number of bits */
template<typename I1, typename I2>
struct promote_index_type
{
  typedef typename conditional<(sizeof(I1)<sizeof(I2)), I2, I1>::type type;
};

/** \internal If the template parameter Value is Dynamic, this class is just a wrapper around a T variable that
  * can be accessed using value() and setValue().
  * Otherwise, this class is an empty structure and value() just returns the template parameter Value.
  */
template<typename T, int Value> class variable_if_dynamic
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamic)
    EIGEN_DEVICE_FUNC E