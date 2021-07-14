/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to BLAS F77
 *   Triangular matrix-vector product functionality based on ?TRMV.
 ********************************************************************************
*/

#ifndef EIGEN_TRIANGULAR_MATRIX_VECTOR_BLAS_H
#define EIGEN_TRIANGULAR_MATRIX_VECTOR_BLAS_H

namespace Eigen { 

namespace internal {

/**********************************************************************
* This file implements triangular matrix-vector multiplication using BLAS
**********************************************************************/

// trmv/hemv specialization

template<typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, bool ConjRhs, int StorageOrder>
struct triangular_matrix_vector_product_trmv :
  triangular_matrix_vector_product<Index,Mode,LhsScalar,ConjLhs,RhsScalar,ConjRhs,StorageOrder,BuiltIn> {};

#define EIGEN_BLAS_TRMV_SPECIALIZE(Scalar) \
template<typename Index, int Mode, bool ConjLhs, bool ConjRhs> \
struct triangular_matrix_vector_product<Index,Mode,Scalar,ConjLhs,Scalar,ConjRhs,ColMajor,Specialized> { \
 static void run(Index _rows, Index _cols, const Scalar* _lhs, Index lhsStride, \
                                     const Scalar* _rhs, Index rhsIncr, Scalar* _res, Index resIncr, Scalar alpha) { \
      triangular_matrix_vector_product_trmv<Index,Mode,Scalar,ConjLhs,Scalar,ConjRhs,ColMajor>::run( \
        _rows, _cols, _lhs, lhsStride, _rhs, rhsIncr, _res, resIncr, alpha); \
  } \
}; \
template<typename Index, int Mode, bool ConjLhs, bool ConjRhs> \
struct triangular_matrix_vector_product<Index,Mode,Scalar,ConjLhs,Scalar,ConjRhs,RowMajor,Specialized> { \
 static void run(Index _rows, Index _cols, const Scalar* _lhs, Index lhsStride, \
                                     const Scalar* _rhs, Index rhsIncr, Scalar* _res, Index resIncr, Scalar alpha) { \
      triangular_matrix_vector_product_trmv<Index,Mode,Scalar,ConjLhs,Scalar,ConjRhs,RowMajor>::run( \
        _rows, _cols, _lhs, lhsStride, _rhs, rhsIncr, _res, resIncr, alpha); \
  } \
};

EIGEN_BLAS_TRMV_SPECIALIZE(double)
EIGEN_BLAS_TRMV_SPECIALIZE(float)
EIGEN_BLAS_TRMV_SPECIALIZE(dcomplex)
EIGEN_BLAS_TRMV_SPECIALIZE(scomplex)

// implements col-major: res += alpha * op(triangular) * vector
#define EIGEN_BLAS_TRMV_CM(EIGTYPE, B