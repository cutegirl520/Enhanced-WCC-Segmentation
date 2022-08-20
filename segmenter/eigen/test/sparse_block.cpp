// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse.h"

template<typename SparseMatrixType> void sparse_block(const SparseMatrixType& ref)
{
  const Index rows = ref.rows();
  const Index cols = ref.cols();
  const Index inner = ref.innerSize();
  const Index outer = ref.outerSize();

  typedef typename SparseMatrixType::Scalar Scalar;
  typedef typename SparseMatrixType::StorageIndex StorageIndex;

  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  typedef Matrix<Scalar,1,Dynamic> RowDenseVector;

  Scalar s1 = internal::random<Scalar>();
  {
    SparseMatrixType m(rows, cols);
    DenseMatrix refMat = DenseMatrix::Zero(rows, cols);
    initSparse<Scalar>(density, refMat, m);

    VERIFY_IS_APPROX(m, refMat);

    // test InnerIterators and Block expressions
    for (int t=0; t<10; ++t)
    {
      Index j = internal::random<Index>(0,cols-2);
      Index i = internal::random<Index>(0,rows-2);
      Index w = internal::random<Index>(1,cols-j);
      Index h = internal::random<Index>(1,rows-i);

      VERIFY_IS_APPROX(m.block(i,j,h,w), refMat.block(i,j,h,w));
      for(Index c=0; c<w; c++)
      {
        VERIFY_IS_APPROX(m.block(i,j,h,w).col(c), refMat.block(i,j,h,w).col(c));
        for(Index r=0; r<h; r++)
        {
          VERIFY_IS_APPROX(m.block(i,j,h,w).col(c).coeff(r), refMat.block(i,j,h,w).col(c).coeff(r));
          VERIFY_IS_APPROX(m.block(i,j,h,w).coeff(r,c), refMat.block(i,j,h,w).coeff(r,c));
        }
      }
      for(Index r=0; r<h; r++)
      {
        VERIFY_IS_APPROX(m.block(i,j,h,w).row(r), refMat.block(i,j,h,w).row(r));
        for(Index c=0; c<w; c++)
        {
          VERIFY_IS_APPROX(m.block(i,j,h,w).row(r).coeff(c), refMat.block(i,j,h,w).row(r).coeff(c));
          VERIFY_IS_APPROX(m.block(i,j,h,w).coeff(r,c), refMat.block(i,j,h,w).coeff(r,c));
        }
      }
      
      VERIFY_IS_APPROX(m.middleCols(j,w), refMat.middleCols(j,w));
      VERIFY_IS_APPROX(m.middleRows(i,h), refMat.middleRows(i,h));
      for(Index r=0; r<h; r++)
      {
        VERIFY_IS_APPROX(m.middleCols(j,w).row(r), refMat.middleCols(j,w).row(r));
        VERIFY_IS_APPROX(m.middleRows(i,h).row(r), refMat.middleRows(i,h).row(r));
        for(Index c=0; c<w; c++)
        {
          VERIFY_IS_APPROX(m.col(c).coeff(r), refMat.col(c).coeff(r));
          VERIFY_IS_APPROX(m.row(r).coeff(c), refMat.row(r).coeff(c));
          
          VERIFY_IS_APPROX(m.middleCols(j,w).coeff(r,c), refMat.middleCols(j,w).coeff(r,c));
          VERIFY_IS_APPROX(m.middleRows(i,h).coeff(r,c), refMat.middleRows(i,h).coeff(r,c));
          if(m.middleCols(j,w).coeff(r,c) != Scalar(0))
          {
            VERIFY_IS_APPROX(m.middleCols(j,w).coeffRef(r,c), refMat.middleCols(j,w).coeff(r,c));
          }
          if(m.middleRows(i,h).coeff(r,c) != Scalar(0))
          {
            VERIFY_IS_APPROX(m.middleRows(i,h).coeff(r,c), refMat.middleRows(i,h).coeff(r,c));
          }
        }
      }
      for(Index c=0; c<w; c++)
      {
        VERIFY_IS_APPROX(m.middleCols(j,w).col(c), refMat.middleCols(j,w).col(c));
        VERIFY_IS_APPROX(m.middleRows(i,h).col(c), refMat.middleRows(i,h).col(c));
      }
    }

    for(Index c=0; c<cols; c++)
    {
      VERIFY_IS_APPROX(m.col(c) + m.col(c), (m + m).col(c));
      VERIFY_IS_APPROX(m.col(c) + m.col(c), refMat.col(c) + refMat.col(c));
    }

    for(Index r=0; r<rows; r++)
    {
      VERIFY_IS_APPROX(m.row(r) + m.row(r), (m + m).row(r));
      VERIFY_IS_APPROX(m.row(r) + m.row(r), refMat.row(r) + refMat.row(r));
    }
  }

  // test innerVector()
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    Index j0 = internal::random<Index>(0,outer-1);
    Index j1 = internal::random<Index>(0,outer-1);
    if(SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m2.innerVector(j0), refMat2.row(j0));
    else
      VERIFY_IS_APPROX(m2.innerVector(j0), refMat2.col(j0));

    if(SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m2.innerVector(j0)+m2.innerVector(j1), refMat2.row(j0)+refMat2.row(j1));
    else
      VERIFY_IS_APPROX(m2.innerVector(j0)+m2.innerVector(j1), refMat2.col(j0)+refMat2.col(j1));

    SparseMatrixType m3(rows,cols);
    m3.reserve(VectorXi::Constant(outer,int(inner/2)));
    for(Index j=0; j<outer; ++j)
      for(Index k=0; k<(std::min)(j,inner); ++k)
        m3.insertByOuterInner(j,k) = internal::convert_index<StorageIndex>(k+1);
    for(Index j=0; j<(std::min)(outer, inner); ++j)
    {
      VERIFY(j==numext::real(m3.innerVector(j).nonZeros()));
      if(j>0)
        VERIFY(j==numext::real(m3.innerVector(j).lastCoeff()));
    }
    m3.makeCompressed();
    for(Index j=0; j<(std::min)(outer, inner); ++j)
    {
      VERIFY(j==numext::real(m3.innerVector(j).nonZeros()));
      if(j>0)
        VERIFY(j==numext::real(m3.innerVector(j).lastCoeff()));
    }

    VERIFY(m3.innerVector(j0).nonZeros() == m3.transpose().innerVector(j0).nonZeros());

//     m2.innerVector(j0) = 2*m2.innerVector(j1);
//     refMat2.col(j0) = 2*refMat2.col(j1);
//     VERIFY_IS_APPROX(m2, refMat2);
  }

  // test innerVectors()
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    if(internal::random<float>(0,1)>0.5f) m2.makeCompressed();
    Index j0 = internal::random<Index>(0,outer-2);
    Index j1 = internal::random<Index>(0,outer-2);
    Index n0 = internal::random<Index>(1,outer-(std::max)(j0,j1));
    if(SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m2.innerVectors(j0,n0), refMat2.block(j0,0,n0,cols));
    else
      VERIFY_IS_APPROX(m2.innerVectors(j0,n0), refMat2.block(0,j0,rows,n0));
    if(SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m2.innerVectors(j0,n0)+m2.innerVectors(j1,n0),
                       refMat2.middleRows(j0,n0)+refMat2.middleRows(j1,n0));
    else
      VERIFY_IS_APPROX(m2.innerVectors(j0,n0)+m2.inne