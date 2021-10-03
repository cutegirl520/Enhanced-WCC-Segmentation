// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

namespace Eigen { 

/** \ingroup SparseCore_Module
  *
  * \class SparseMatrix
  *
  * \brief A versatible sparse matrix representation
  *
  * This class implements a more versatile variants of the common \em compressed row/column storage format.
  * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
  * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
  * space inbetween the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
  * can be done with limited memory reallocation and copies.
  *
  * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
  * compatible with many library.
  *
  * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
  *
  * \tparam _Scalar the scalar type, i.e. the type of the coefficients
  * \tparam _Options Union of bit flags controlling the storage scheme. Currently the only possibility
  *                 is ColMajor or RowMajor. The default is 0 which means column-major.
  * \tparam _Index the type of the indices. It has to be a \b signed type (e.g., short, int, std::ptrdiff_t). Default is \c int.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
  */

namespace internal {
template<typename _Scalar, int _Options, typename _Index>
struct traits<SparseMatrix<_Scalar, _Options, _Index> >
{
  typedef _Scalar Scalar;
  typedef _Index StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = _Options | NestByRefBit | LvalueBit | CompressedAccessBit,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

template<typename _Scalar, int _Options, typename _Index, int DiagIndex>
struct traits<Diagonal<SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
{
  typedef SparseMatrix<_Scalar, _Options, _Index> MatrixType;
  typedef typename ref_selector<MatrixType>::type MatrixTypeNested;
  typedef typename remove_reference<MatrixTypeNested>::type _MatrixTypeNested;

  typedef _Scalar Scalar;
  typedef Dense StorageKind;
  typedef _Index StorageIndex;
  typedef MatrixXpr XprKind;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = 1,
    Flags = LvalueBit
  };
};

template<typename _Scalar, int _Options, typename _Index, int DiagIndex>
struct traits<Diagonal<const SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
 : public traits<Diagonal<SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
{
  enum {
    Flags = 0
  };
};

} // end namespace internal

template<typename _Scalar, int _Options, typename _Index>
class SparseMatrix
  : public SparseCompressedBase<SparseMatrix<_Scalar, _Options, _Index> >
{
    typedef SparseCompressedBase<SparseMatrix> Base;
    using Base::convert_index;
  public:
    using Base::isCompressed;
    using Base::nonZeros;
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
    using Base::operator+=;
    using Base::operator-=;

    typedef MappedSparseMatrix<Scalar,Flags> Map;
    typedef Diagonal<SparseMatrix> DiagonalReturnType;
    typedef Diagonal<const SparseMatrix> ConstDiagonalReturnType;
    typedef typename Base::InnerIterator InnerIterator;
    typedef typename Base::ReverseInnerIterator ReverseInnerIterator;
    

    using Base::IsRowMajor;
    typedef internal::CompressedStorage<Scalar,StorageIndex> Storage;
    enum {
      Options = _Options
    };

    typedef typename Base::IndexVector IndexVector;
    typedef typename Base::ScalarVector ScalarVector;
  protected:
    typedef SparseMatrix<Scalar,(Flags&~RowMajorBit)|(IsRowMajor?RowMajorBit:0)> TransposedSparseMatrix;

    Index m_outerSize;
    Index m_innerSize;
    StorageIndex* m_outerIndex;
    StorageIndex* m_innerNonZeros;     // optional, if null then the data is compressed
    Storage m_data;

  public:
    
    /** \returns the number of rows of the matrix */
    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    /** \returns the number of columns of the matrix */
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

    /** \returns the number of rows (resp. columns) of the matrix if the storage order column major (resp. row major) */
    inline Index innerSize() const { return m_innerSize; }
    /** \returns the number of columns (resp. rows) of the matrix if the storage order column major (resp. row major) */
    inline Index outerSize() const { return m_outerSize; }
    
    /** \returns a const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline const Scalar* valuePtr() const { return m_data.valuePtr(); }
    /** \returns a non-const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline Scalar* valuePtr() { return m_data.valuePtr(); }

    /** \returns a const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
   