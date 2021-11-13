

typedef CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived> AbsReturnType;
typedef CwiseUnaryOp<internal::scalar_arg_op<Scalar>, const Derived> ArgReturnType;
typedef CwiseUnaryOp<internal::scalar_abs2_op<Scalar>, const Derived> Abs2ReturnType;
typedef CwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived> SqrtReturnType;
typedef CwiseUnaryOp<internal::scalar_rsqrt_op<Scalar>, const Derived> RsqrtReturnType;
typedef CwiseUnaryOp<internal::scalar_sign_op<Scalar>, const Derived> SignReturnType;
typedef CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived> InverseReturnType;
typedef CwiseUnaryOp<internal::scalar_boolean_not_op<Scalar>, const Derived> BooleanNotReturnType;

typedef CwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived> ExpReturnType;
typedef CwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived> LogReturnType;
typedef CwiseUnaryOp<internal::scalar_log1p_op<Scalar>, const Derived> Log1pReturnType;
typedef CwiseUnaryOp<internal::scalar_log10_op<Scalar>, const Derived> Log10ReturnType;
typedef CwiseUnaryOp<internal::scalar_cos_op<Scalar>, const Derived> CosReturnType;
typedef CwiseUnaryOp<internal::scalar_sin_op<Scalar>, const Derived> SinReturnType;
typedef CwiseUnaryOp<internal::scalar_tan_op<Scalar>, const Derived> TanReturnType;
typedef CwiseUnaryOp<internal::scalar_acos_op<Scalar>, const Derived> AcosReturnType;
typedef CwiseUnaryOp<internal::scalar_asin_op<Scalar>, const Derived> AsinReturnType;
typedef CwiseUnaryOp<internal::scalar_atan_op<Scalar>, const Derived> AtanReturnType;
typedef CwiseUnaryOp<internal::scalar_tanh_op<Scalar>, const Derived> TanhReturnType;
typedef CwiseUnaryOp<internal::scalar_sinh_op<Scalar>, const Derived> SinhReturnType;
typedef CwiseUnaryOp<internal::scalar_cosh_op<Scalar>, const Derived> CoshReturnType;
typedef CwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived> SquareReturnType;
typedef CwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived> CubeReturnType;
typedef CwiseUnaryOp<internal::scalar_round_op<Scalar>, const Derived> RoundReturnType;
typedef CwiseUnaryOp<internal::scalar_floor_op<Scalar>, const Derived> FloorReturnType;
typedef CwiseUnaryOp<internal::scalar_ceil_op<Scalar>, const Derived> CeilReturnType;
typedef CwiseUnaryOp<internal::scalar_isnan_op<Scalar>, const Derived> IsNaNReturnType;
typedef CwiseUnaryOp<internal::scalar_isinf_op<Scalar>, const Derived> IsInfReturnType;
typedef CwiseUnaryOp<internal::scalar_isfinite_op<Scalar>, const Derived> IsFiniteReturnType;

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include Cwise_abs.cpp
  * Output: \verbinclude Cwise_abs.out
  *
  * \sa abs2()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const AbsReturnType
abs() const
{
  return AbsReturnType(derived());
}

/** \returns an expression of the coefficient-wise phase angle of \c *this
  *
  * Example: \include Cwise_arg.cpp
  * Output: \verbinclude Cwise_arg.out
  *
  * \sa abs()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const ArgReturnType
arg() const
{
  return ArgReturnType(derived());
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include Cwise_abs2.cpp
  * Output: \verbinclude Cwise_abs2.out
  *
  * \sa abs(), square()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const Abs2ReturnType
abs2() const
{
  return Abs2ReturnType(derived());
}

/** \returns an expression of the coefficient-wise exponential of *this.
  *
  * This function computes the coefficient-wise exponential. The function MatrixBase::exp() in the
  * unsupported module MatrixFunctions computes the matrix exponential.
  *
  * Example: \include Cwise_exp.cpp
  * Output: \verbinclude Cwise_exp.out
  *
  * \sa pow(), log(), sin(), cos()
  */
EIGEN_DEVICE_FUNC
inline const ExpReturnType
exp() const
{
  return ExpReturnType(derived());
}

/** \returns an expression of the coefficient-wise logarithm of *this.
  *
  * This function computes the coefficient-wise logarithm. The function MatrixBase::log() in the
  * unsupported module MatrixFunctions computes the matrix logarithm.
  *
  * Example: \include Cwise_log.cpp
  * Output: \verbinclude Cwise_log.out
  *
  * \sa exp()
  */
EIGEN_DEVICE_FUNC
inline const LogReturnType
log() const
{
  return LogReturnType(derived());
}

/** \returns an expression of the coefficient-wise logarithm of 1 plus \c *this.
  *
  * In exact arithmetic, \c x.log() is equivalent to \c (x+1).log(),
  * however, with finite precision, this function is much more accurate when \c x is close to zero.
  *
  * \sa log()
  */
EIGEN_DEVICE_FUNC
inline const Log1pReturnType
log1p() const
{
  return Log1pReturnType(derived());
}

/** \returns an expression of the coefficient-wise base-10 logarithm of *this.
  *
  * This function computes the coefficient-wise base-10 logarithm.
  *
  * Example: \include Cwise_log10.cpp
  * Output: \verbinclude Cwise_log10.out
  *
  * \sa log()
  */
EIGEN_DEVICE_FUNC
inline const Log10ReturnType
log10() const
{
  return Log10ReturnType(derived());
}

/** \returns an expression of the coefficient-wise square root of *this.
  *
  * This function computes the coefficient-wise square root. The function MatrixBase::sqrt() in the
  * unsupported module MatrixFunctions computes the matrix square root.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const SqrtReturnType
sqrt() const
{
  return SqrtReturnType(derived());
}

/** \returns an expression of the coefficient-wise inverse square root of *this.
  *
  * This function computes the coefficient-wise inverse square root.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const RsqrtReturnType
rsqrt() const
{
  return RsqrtReturnType(derived());
}

/** \returns an expression of the coefficient-wise signum of *this.
  *
  * This function computes the coefficient-wise signum.
  *
  * Example: \include Cwise_sign.cpp
  * Output: \verbinclude Cwise_sign.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const SignReturnType
sign() const
{
  return SignReturnType(derived());
}


/** \returns an expression of the coefficient-wise cosine of *this.
  *
  * This function computes the coefficient-wise cosine. The function MatrixBase::cos() in the
  * unsupported module MatrixFunctions computes the matrix cosine.
  *
  * Example: \include Cwise_cos.cpp
  * Output: \verbinclude Cwise_cos.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const CosReturnType
cos() const
{
  return CosReturnType(derived());
}


/** \returns an expression of the coefficient-wise sine of *this.
  *
  * This function computes the coefficient-wise sine. The function MatrixBase::sin() in the
  * unsupported module MatrixFunctions computes the matrix sine.
  *
  * Example: \include Cwise_sin.cpp
  * Output: \verbinclude Cwise_sin.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const SinReturnType
sin() const
{
  return SinReturnType(derived());
}

/** \returns an expression of the coefficient-wise tan of *this.
  *
  * Example: \include Cwise_tan.cpp
  * Output: \verbinclude Cwise_tan.out
  *
  * \sa cos(), sin()
  */
EIGEN_DEVICE_FUNC
inline const TanReturnType
tan() const
{
  return TanReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc tan of *this.
  *
  * Example: \include Cwise_atan.cpp
  * Output: \verbinclude Cwise_atan.out
  *
  * \sa tan(), asin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const AtanReturnType
atan() const
{
  return AtanReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc cosine of *this.
  *
  * Example: \include Cwise_acos.cpp
  * Output: \verbinclude Cwise_acos.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const AcosReturnType
acos() const
{
  return AcosReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc sine of *this.
  *
  * Example: \include Cwise_asin.cpp
  * Output: \verbinclude Cwise_asin.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const AsinReturnType
asin() const
{
  return AsinReturnType(derived());
}

/** \returns an