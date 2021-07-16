// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_H

namespace Eigen { 

namespace internal {

// if the rhs is row major, let's transpose the product
template <typename Scalar, typename Index, int Side, int Mode, bool Conjugate, int TriStorageOrder>
struct triangular_solve_matrix<Scalar,Index,Side,Mode,Conjugate,TriStorageOrder,RowMajor>
{
  static void run(
    Index size, Index cols,
    const Scalar*  tri, Index triStride,
    Scalar* _other, Index otherStride,
    level3_blocking<Scalar,Scalar>& blocking)
  {
    triangular_solve_matrix<
      Scalar, Index, Side==OnTheLeft?OnTheRight:OnTheLeft,
      (Mode&UnitDiag) | ((Mode&Upper) ? Lower : Upper),
      NumTraits<Scalar>::IsComplex && Conjugate,
      TriStorageOrder==RowMajor ? ColMajor : RowMajor, ColMajor>
      ::run(size, cols, tri, triStride, _other, otherStride, blocking);
  }
};

/* Optimized triangular solver with multiple right hand side and the triangular matrix on the left
 */
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder>
struct triangular_solve_matrix<Scalar,Index,OnTheLeft,Mode,Conjugate,TriStorageOrder,ColMajor>
{
  static EIGEN_DONT_INLINE void run(
    Index size, Index otherSize,
    const Scalar* _tri, Index triStride,
    Scalar* _other, Index otherStride,
    level3_blocking<Scalar,Scalar>& blocking);
};
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder>
EIGEN_DONT_INLINE void triangular_solve_matrix<Scalar,Index,OnTheLeft,Mode,Conjugate,TriStorageOrder,ColMajor>::run(
    Index size, Index otherSize,
    const Scalar* _tri, Index triStride,
    Scalar* _other, Index otherStride,
    level3_blocking<Scalar,Scalar>& blocking)
  {
    Index cols = otherSize;

    typedef const_blas_data_mapper<Scalar, Index, TriStorageOrder> TriMapper;
    typedef blas_data_mapper<Scalar, Index, ColMajor> OtherMapper;
    TriMapper tri(_tri, triStride);
    OtherMapper other(_other, otherStride);

    typedef gebp_traits<Scalar,Scalar> Traits;

    enum {
      SmallPanelWidth   = EIGEN_PLAIN_ENUM_MAX(Traits::mr,Traits::nr),
      IsLower = (Mode&Lower) == Lower
    };

    Index kc = blocking.kc();                   // cache block size along the K direction
    Index mc = (std::min)(size,blocking.mc());  // cache block size along the M direction

    std::size_t sizeA = kc*mc;
    std::size_t sizeB = kc*cols;

    ei_declare_aligned_stack_constructed_variable(Scalar, blockA, sizeA, blocking.blockA());
    ei_declare_aligned_stack_constructed_variable(Scalar, blockB, sizeB, blocking.blockB());

    conj_if<Conjugate> conj;
    gebp_kernel<Scalar, Scalar, Index, OtherMapper, Traits::mr, Traits::nr, Conjugate, false> gebp_kernel;
    gemm_pack_lhs<Scalar, Index, TriMapper, Traits::mr, Traits::LhsProgress, TriStorageOrder> pack_lhs;
    gemm_pack_rhs<Scalar, Index, OtherMapper, Traits::nr, ColMajor, false, true> pack_rhs;

    // the goal here is to subdivise the Rhs panels such that we keep some cache
    // coherence when accessing the rhs elements
    std::ptrdiff_t l1, l2, l3;
    manage_caching_sizes(GetAction, &l1, &l2, &l3);
    Index subcols = cols>0 ? l2/(4 * sizeof(Scalar) * std::max<Index>(otherStride,size)) : 0;
    subcols = std::max<Index>((subcols/Traits::nr)*Traits::nr, Traits::nr);

    for(Index k2=IsLower ? 0 : size;
        IsLower ? k2<size : k2>0;
        IsLower ? k2+=kc : k2-=kc)
    {
      const Index actual_kc = (std::min)(IsLower ? size-k2 : k2, kc);

      // We have selected and packed a big horizontal panel R1 of rhs. Let B be the packed copy of this panel,
      // and R2 the remaining part of rhs. The corresponding vertical panel of lhs is split into
      // A11 (the triangular part) and A21 the remaining rectangular part.
      // Then the high level algorithm is:
      //  - B = R1                    => general block copy (done during the next step)
      //  - R1 = A11^-1 B             => tricky part
      //  - update B from the new R1  => actually this has to be performed continuously during the above step
      //  - R2 -= A21 * B             => GEPP

      // The tricky part: compute R1 = A11^-1 B while updating B from R1
      // The idea is to split A11 into multiple small vertical panels.
      // Each panel can be split into a small triangular part T1k which is processed without optimization,
      // and the remaining small part T2k which is processed using gebp with appropriate block strides
      for(Index j2=0; j2<cols; j2+=subcols)
      {
        Index actual_cols = (std::min)(cols-j2,subcols);
        // for each small vertical panels [T1k^T, T2k^T]^T of lhs
        for (Index k1=0; k1<actual_kc; k1+=SmallPanelWidth)
        {
          Index actualPanelWidth = std::min<Index>(actual_kc-k1, SmallPanelWidth);
          // tr solve
          for (Index k=0; k<actualPanelWidth; ++k)
          {
            // TODO write a small kernel handling this (can be shared with trsv)
            Index i  = IsLower ? k2+k1+k : k2-k1-k-1;
            Index rs = actualPanelWidth - k - 1; // remaining size
            Index s  = TriStorageOrder==RowMajor ? (IsLower ? k2+k1 : i+1)
                              