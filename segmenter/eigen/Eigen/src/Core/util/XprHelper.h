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
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit variable_if_dynamic(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T value() { return T(Value); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setValue(T) {}
};

template<typename T> class variable_if_dynamic<T, Dynamic>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamic() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit variable_if_dynamic(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T value() const { return m_value; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setValue(T value) { m_value = value; }
};

/** \internal like variable_if_dynamic but for DynamicIndex
  */
template<typename T, int Value> class variable_if_dynamicindex
{
  public:
    EIGEN_EMPTY_STRUCT_CTOR(variable_if_dynamicindex)
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit variable_if_dynamicindex(T v) { EIGEN_ONLY_USED_FOR_DEBUG(v); eigen_assert(v == T(Value)); }
    EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T value() { return T(Value); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setValue(T) {}
};

template<typename T> class variable_if_dynamicindex<T, DynamicIndex>
{
    T m_value;
    EIGEN_DEVICE_FUNC variable_if_dynamicindex() { eigen_assert(false); }
  public:
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit variable_if_dynamicindex(T value) : m_value(value) {}
    EIGEN_DEVICE_FUNC T EIGEN_STRONG_INLINE value() const { return m_value; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void setValue(T value) { m_value = value; }
};

template<typename T> struct functor_traits
{
  enum
  {
    Cost = 10,
    PacketAccess = false,
    IsRepeatable = false
  };
};

template<typename T> struct packet_traits;

template<typename T> struct unpacket_traits
{
  typedef T type;
  typedef T half;
  enum
  {
    size = 1,
    alignment = 1
  };
};

template<int Size, typename PacketType,
         bool Stop = Size==Dynamic || (Size%unpacket_traits<PacketType>::size)==0 || is_same<PacketType,typename unpacket_traits<PacketType>::half>::value>
struct find_best_packet_helper;

template< int Size, typename PacketType>
struct find_best_packet_helper<Size,PacketType,true>
{
  typedef PacketType type;
};

template<int Size, typename PacketType>
struct find_best_packet_helper<Size,PacketType,false>
{
  typedef typename find_best_packet_helper<Size,typename unpacket_traits<PacketType>::half>::type type;
};

template<typename T, int Size>
struct find_best_packet
{
  typedef typename find_best_packet_helper<Size,typename packet_traits<T>::type>::type type;
};

#if EIGEN_MAX_STATIC_ALIGN_BYTES>0
template<int ArrayBytes, int AlignmentBytes,
         bool Match     =  bool((ArrayBytes%AlignmentBytes)==0),
         bool TryHalf   =  bool(AlignmentBytes>EIGEN_MIN_ALIGN_BYTES) >
struct compute_default_alignment_helper
{
  enum { value = 0 };
};

template<int ArrayBytes, int AlignmentBytes, bool TryHalf>
struct compute_default_alignment_helper<ArrayBytes, AlignmentBytes, true, TryHalf> // Match
{
  enum { value = AlignmentBytes };
};

template<int ArrayBytes, int AlignmentBytes>
struct compute_default_alignment_helper<ArrayBytes, AlignmentBytes, false, true> // Try-half
{
  // current packet too large, try with an half-packet
  enum { value = compute_default_alignment_helper<ArrayBytes, AlignmentBytes/2>::value };
};
#else
// If static alignment is disabled, no need to bother.
// This also avoids a division by zero in "bool Match =  bool((ArrayBytes%AlignmentBytes)==0)"
template<int ArrayBytes, int AlignmentBytes>
struct compute_default_alignment_helper
{
  enum { value = 0 };
};
#endif

template<typename T, int Size> struct compute_default_alignment {
  enum { value = compute_default_alignment_helper<Size*sizeof(T),EIGEN_MAX_STATIC_ALIGN_BYTES>::value };
};

template<typename T> struct compute_default_alignment<T,Dynamic> {
  enum { value = EIGEN_MAX_ALIGN_BYTES };
};

template<typename _Scalar, int _Rows, int _Cols,
         int _Options = AutoAlign |
                          ( (_Rows==1 && _Cols!=1) ? RowMajor
                          : (_Cols==1 && _Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
         int _MaxRows = _Rows,
         int _MaxCols = _Cols
> class make_proper_matrix_type
{
    enum {
      IsColVector = _Cols==1 && _Rows!=1,
      IsRowVector = _Rows==1 && _Cols!=1,
      Options = IsColVector ? (_Options | ColMajor) & ~RowMajor
              : IsRowVector ? (_Options | RowMajor) & ~ColMajor
              : _Options
    };
  public:
    typedef Matrix<_Scalar, _Rows, _Cols, Options, _MaxRows, _MaxCols> type;
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class compute_matrix_flags
{
    enum { row_major_bit = Options&RowMajor ? RowMajorBit : 0 };
  public:
    // FIXME currently we still have to handle DirectAccessBit at the expression level to handle DenseCoeffsBase<>
    // and then propagate this information to the evaluator's flags.
    // However, I (Gael) think that DirectAccessBit should only matter at the evaluation stage.
    enum { ret = DirectAccessBit | LvalueBit | NestByRefBit | row_major_bit };
};

template<int _Rows, int _Cols> struct size_at_compile_time
{
  enum { ret = (_Rows==Dynamic || _Cols==Dynamic) ? Dynamic : _Rows * _Cols };
};

template<typename XprType> struct size_of_xpr_at_compile_time
{
  enum { ret = size_at_compile_time<traits<XprType>::RowsAtCompileTime,traits<XprType>::ColsAtCompileTime>::ret };
};

/* plain_matrix_type : the difference from eval is that plain_matrix_type is always a plain matrix type,
 * whereas eval is a const reference in the case of a matrix
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct plain_matrix_type;
template<typename T, typename BaseClassType, int Flags> struct plain_matrix_type_dense;
template<typename T> struct plain_matrix_type<T,Dense>
{
  typedef typename plain_matrix_type_dense<T,typename traits<T>::XprKind, traits<T>::Flags>::type type;
};
template<typename T> struct plain_matrix_type<T,DiagonalShape>
{
  typedef typename T::PlainObject type;
};

template<typename T, int Flags> struct plain_matrix_type_dense<T,MatrixXpr,Flags>
{
  typedef Matrix<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

template<typename T, int Flags> struct plain_matrix_type_dense<T,ArrayXpr,Flags>
{
  typedef Array<typename traits<T>::Scalar,
                traits<T>::RowsAtCompileTime,
                traits<T>::ColsAtCompileTime,
                AutoAlign | (Flags&RowMajorBit ? RowMajor : ColMajor),
                traits<T>::MaxRowsAtCompileTime,
                traits<T>::MaxColsAtCompileTime
          > type;
};

/* eval : the return type of eval(). For matrices, this is just a const reference
 * in order to avoid a useless copy
 */

template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct eval;

template<typename T> struct eval<T,Dense>
{
  typedef typename plain_matrix_type<T>::type type;
//   typedef typename T::PlainObject type;
//   typedef T::Matrix<typename traits<T>::Scalar,
//                 traits<T>::RowsAtCompileTime,
//                 traits<T>::ColsAtCompileTime,
//                 AutoAlign | (traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor),
//                 traits<T>::MaxRowsAtCompileTime,
//                 traits<T>::MaxColsAtCompileTime
//           > type;
};

template<typename T> struct eval<T,DiagonalShape>
{
  typedef typename plain_matrix_type<T>::type type;
};

// for matrices, no need to evaluate, just use a const reference to avoid a useless copy
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct eval<Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Dense>
{
  typedef const Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& type;
};


/* similar to plain_matrix_type, but using the evaluator's Flags */
template<typename T, typename StorageKind = typename traits<T>::StorageKind> struct plain_object_eval;

template<typename T>
struct plain_object_eval<T,Dense>
{
  typedef typename plain_matrix_type_dense<T,typename traits<T>::XprKind, evaluator<T>::Flags>::type type;
};


/* plain_matrix_type_column_major : same as plain_matrix_type but guaranteed to be column-major
 */
template<typename T> struct plain_matrix_type_column_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxRows==1&&MaxCols!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

/* plain_matrix_type_row_major : same as plain_matrix_type but guaranteed to be row-major
 */
template<typename T> struct plain_matrix_type_row_major
{
  enum { Rows = traits<T>::RowsAtCompileTime,
         Cols = traits<T>::ColsAtCompileTime,
         MaxRows = traits<T>::MaxRowsAtCompileTime,
         MaxCols = traits<T>::MaxColsAtCompileTime
  };
  typedef Matrix<typename traits<T>::Scalar,
                Rows,
                Cols,
                (MaxCols==1&&MaxRows!=1) ? RowMajor : ColMajor,
                MaxRows,
                MaxCols
          > type;
};

/** \internal The reference selector for template expressions. The idea is that we don't
  * need to use references for expressions since they are light weight proxy
  * objects which should generate no copying overhead. */
template <typename T>
struct ref_selector
{
  typedef typename conditional<
    bool(traits<T>::Flags & NestByRefBit),
    T const&,
    const T
  >::type type;
  
  typedef typename conditional<
    bool(traits<T>::Flags & NestByRefBit),
    T &,
    T
  >::type non_const_type;
};

/** \internal Adds the const qualifier on the value-type of T2 if and only if T1 is a const type */
template<typename T1, typename T2>
struct transfer_constness
{
  typedef typename conditional<
    bool(internal::is_const<T1>::value),
    typename internal::add_const_on_value_type<T2>::type,
    T2
  >::type type;
};


// However, we still need a mechanism to detect whether an expression which is evaluated multiple time
// has to be evaluated into a temporary.
// That's the purpose of this new nested_eval helper:
/** \internal Determines how a given expression should be nested when evaluated multiple times.
  * For example, when you do a * (b+c), Eigen will determine how the expression b+c should be
  * evaluated into the bigger product expression. The choice is between nesting the expression b+c as-is, or
  * evaluating that expression b+c into a temporary variable d, and nest d so that the resulting expression is
  * a*d. Evaluating can be beneficial for example if every coefficient access in the resulting expression causes
  * many coefficient accesses in the nested expressions -- as is the case with matrix product for example.
  *
  * \tparam T the type of the expre