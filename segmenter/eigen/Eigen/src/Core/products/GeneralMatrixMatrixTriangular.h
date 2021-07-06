// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_TRIANGULAR_H
#define EIGEN_GENERAL_MATRIX_MATRIX_TRIANGULAR_H

namespace Eigen { 

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjLhs, bool ConjRhs>
struct selfadjoint_rank1_update;

namespace internal {

/**********************************************************************
* This file implements a general A * B product while
* evaluating only one triangular part of the product.
* This is a more general version of self adjoint product (C += A A^T)
* as the level 3 SYRK Blas routine.
**********************************************************************/

// forward declarations (defined at the end of this file)
template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjLhs, bool ConjRhs, int UpLo>
struct tribb_kernel;
  
/* Optimized matrix-matrix product evaluating only one triangular half */
template <typename Index,
          typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
                              int ResStorageOrder, int  UpLo, int Version = Specialized>
struct general_matrix_matrix_triangular_product;

// as usual if the result is row major => we transpose the product
template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
                          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs, int  UpLo, int Version>
struct general_matrix_matrix_triangular_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,RowMajor,UpLo,Version>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(Index size, Index depth,const LhsScalar* lhs, Index lhsStride,
                                      const RhsScalar* rhs, Index rhsStride, ResScalar* res, Index resStride,
                                      const ResScalar& alpha, level3_blocking<RhsScalar,LhsScalar>& blocking)
  {
    general_matrix_matrix_triangular_product<Index,
        RhsScalar, RhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateRhs,
        LhsScalar, LhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateLhs,
        ColMajor, UpLo==Lower?Upper:Lower>
      ::run(size,depth,rhs,rhsStride,lhs,lhsStride,res,resStride,alpha,blocking);
  }
};

template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
                          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs, int  UpLo, int Version>
struct general_matrix_matrix_triangular_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,ColMajor,UpLo,Version>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(Index size, Index depth,const LhsScalar* _lhs, Index lhsStride,
                                      const RhsScalar* _rhs, Index rhsStride, ResScalar* _res, Index resStride,
                                      const ResScalar& alpha, level3_blocking<LhsScalar,RhsScalar>& blocking)
  {
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;

    typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar, Index, RhsSt