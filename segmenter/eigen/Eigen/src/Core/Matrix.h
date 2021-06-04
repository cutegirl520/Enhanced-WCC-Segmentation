// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_H
#define EIGEN_MATRIX_H

namespace Eigen {

namespace internal {
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct traits<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
private:
  enum { size = internal::size_at_compile_time<_Rows,_Cols>::ret };
  typedef typename find_best_packet<_Scalar,size>::type PacketScalar;
  enum {
      row_major_bit = _Options&RowMajor ? RowMajorBit : 0,
      is_dynamic_size_storage = _MaxRows==Dynamic || _MaxCols==Dynamic,
      max_size = is_dynamic_size_storage ? Dynamic : _MaxRows*_MaxCols,
      default_alignment = compute_default_alignment<_Scalar,max_size>::value,
      actual_alignment = ((_Options&DontAlign)==0) ? default_alignment : 0,
      required_alignment = unpacket_traits<PacketScalar>::alignment,
      packet_access_bit = (packet_traits<_Scalar>::Vectorizable && (EIGEN_UNALIGNED_VECTORIZE || (actual_alignment>=required_alignment))) ? PacketAccessBit : 0
    };
    
public:
  typedef _Scalar Scalar;
  typedef Dense StorageKind;
  typedef Eigen::Index StorageIndex;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = _Rows,
    ColsAtCompileTime = _Cols,
    MaxRowsAtCompileTime = _MaxRows,
    MaxColsAtCompileTime = _MaxCols,
    Flags = compute_matrix_flags<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::ret,
    Options = _Options,
    InnerStrideAtCompileTime = 1,
    OuterStrideAtCompileTime = (Options&RowMajor) ? ColsAtCompileTime : RowsAtCompileTime,
    
    // FIXME, the following flag in only used to define NeedsToAlign in PlainObjectBase
    EvaluatorFlags = LinearAccessBit | DirectAccessBit | packet_access_bit | row_major_bit,
    Alignment = actual_alignment
  };
};
}

/** \class Matrix
  * \ingroup Core_Module
  *
  * \brief The matrix class, also used for vectors and row-vectors
  *
  * The %Matrix class is the work-horse for all \em dense (\ref dense "note") matrices and vectors within Eigen.
  * Vectors are matrices with one column, and row-vectors are matrices with one row.
  *
  * The %Matrix class encompasses \em both fixed-size and dynamic-size objects (\ref fixedsize "note").
  *
  * The first three template parameters are required:
  * \tparam _Scalar Numeric type, e.g. float, double, int or std::complex<float>.
  *                 User defined scalar types are supported as well (see \ref user_defined_scalars "here").
  * \tparam _Rows Number of rows, or \b Dynamic
  * \tparam _Cols Number of columns, or \b Dynamic
  *
  * The remaining template parameters are optional -- in most cases you don't have to worry about them.
  * \tparam _Options A combination of either \b #RowMajor or \b #ColMajor, and of either
  *                 \b #AutoAlign or \b #DontAlign.
  *                 The former controls \ref TopicStorageOrders "storage order", and defaults to column-major. The latter controls alignment, which is required
  *                 for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
  * \tparam _MaxRows Maximum number of rows. Defaults to \a _Rows (\ref maxrows "note").
  * \tparam _MaxCols Maximum number of columns. Defaults to \a _Cols (\ref maxrows "note").
  *
  * Eigen provides a number of typedefs covering the usual cases. Here are some examples:
  *
  * \li \c Matrix2d is a 2x2 square matrix of doubles (\c Matrix<double, 2, 2>)
  * \li \c Vector4f is a vector of 4 floats (\c Matrix<float, 4, 1>)
  * \li \c RowVector3i is a row-vector of 3 ints (\c Matrix<int, 1, 3>)
  *
  * \li \c MatrixXf is a dynamic-size matrix of floats (\c Matrix<float, Dynamic, Dynamic>)
  * \li \c VectorXf is a dynamic-size vector of floats (\c Matrix<float, Dynamic, 1>)
  *
  * \li \c Matrix2Xf is a partially fixed-size (dynamic-size) matrix of floats (\c Matrix<float, 2, Dynamic>)
  * \li \c MatrixX3d is a partially dynamic-size (fixed-size) matrix of double (\c Matrix<double, Dynamic, 3>)
  *
  * See \link matrixtypedefs this page \endlink for a complete list of predefined \em %Matrix and \em Vector typedefs.
  *
  * You can access elements of vectors and matrices using normal subscripting:
  *
  * \code
  * Eigen::VectorXd v(10);
  * v[0] = 0.1;
  * v[1] = 0.2;
  * v(0) = 0.3;
  * v(1) = 0.4;
  *
  * Eigen::MatrixXi m(1