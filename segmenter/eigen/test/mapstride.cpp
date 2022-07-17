// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<int Alignment,typename VectorType> void map_class_vector(const VectorType& m)
{
  typedef typename VectorType::Index Index;
  typedef typename VectorType::Scalar Scalar;

  Index size = m.size();

  VectorType v = VectorType::Random(size);

  Index arraysize = 3*size;
  
  Scalar* a_array = internal::aligned_new<Scalar>(arraysize+1);
  Scalar* array = a_array;
  if(Alignment!=Aligned)
    array = (Scalar*)(internal::IntPtr(a_array) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));

  {
    Map<VectorType, Alignment, InnerStride<3> > map(array, size);
    map = v;
    for(int i = 0; i < size; ++i)
    {
      VERIFY(array[3*i] == v[i]);
      VERIFY(map[i] == v[i]);
    }
  }

  {
    Map<VectorType, Unaligned, InnerStride<Dynamic> > map(array, size, InnerStride<Dynamic>(2));
    map = v;
    for(int i = 0; i < size; ++i)
    {
      VERIFY(array[2*i] == v[i]);
      VERIFY(map[i] == v[i]);
    }
  }

  internal::aligned_delete(a_array, arraysize+1);
}

template<int Alignment,typename MatrixType> void map_class_matrix(const MatrixType& _m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;

  Index rows = _m.rows(), cols = _m.cols();

  MatrixType m = MatrixType::Random(rows,cols);
  Scalar s1 = internal::random<Scalar>();

  Index arraysize = 2*(rows+4)*(cols+4);

  Scalar* a_array1 = internal::aligned_new<Scalar>(arraysize+1);
  Scalar* array1 = a_array1;
  if(Alignment!=Aligned)
    array1 = (Scalar*)(internal::IntPtr(a_array1) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));

  Scalar a_array2[256];
  Scalar* array2 = a_array2;
  if(Alignment!=Aligned)
    array2 = (Scalar*)(internal::IntPtr(a_array2) + (internal::packet_traits<Scalar>::AlignedOnScalar?sizeof(Scalar):sizeof(typename NumTraits<Scalar>::Real)));
  else
    array2 = (Scalar*)(((internal::UIntPtr(a_array2)+EIGEN_MAX_ALIGN_BYTES-1)/EIGEN_MAX_ALIGN_BYTES)*EIGEN_MAX_ALIGN_BYTES);
  Index maxsize2 = a_array2 - array2 + 256;
  
  // test no inner stride and some dynamic outer stride
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (m.innerSize()+1)*m.outerSize() > maxsize2)
      break;
    Scalar* array = (k==0 ? array1 : array2);
    
    Map<MatrixType, Alignment, OuterStride<Dynamic> > map(array, rows, cols, OuterStride<Dynamic>(m.innerSize()+1));
    map = m;
    VERIFY(map.outerStride() == map.innerSize()+1);
    for(int i = 0; i < m.outerSize(); ++i)
      for(int j = 0; j < m.innerSize(); ++j)
      {
        VERIFY(array[map.outerStride()*i+j] == m.coeffByOuterInner(i,j));
        VERIFY(map.coeffByOuterInner(i,j) == m.coeffByOuterInner(i,j));
      }
    VERIFY_IS_APPROX(s1*map,s1*m);
    map *= s1;
    VERIFY_IS_APPROX(map,s1*m);
  }

  // test no inner stride and an outer stride of +4. This is quite important as for fixed-size matrices,
  // this allows to hit the special case where it's vectorizable.
  for(int k=0; k<2; ++k)
  {
    if(k==1 && (m.innerSize()+4)*m.outerSize() > maxsize2)
      brea