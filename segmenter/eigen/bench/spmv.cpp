
//g++-4.4 -DNOMTL  -Wl,-rpath /usr/local/lib/oski -L /usr/local/lib/oski/ -l oski -l oski_util -l oski_util_Tid  -DOSKI -I ~/Coding/LinearAlgebra/mtl4/  spmv.cpp  -I .. -O2 -DNDEBUG -lrt  -lm -l oski_mat_CSC_Tid  -loskilt && ./a.out r200000 c200000 n100 t1 p1

#define SCALAR double

#include <iostream>
#include <algorithm>
#include "BenchTimer.h"
#include "BenchSparseUtil.h"

#define SPMV_BENCH(CODE) BENCH(t,tries,repeats,CODE);

// #ifdef MKL
//
// #include "mkl_types.h"
// #include "mkl_spblas.h"
//
// template<typename Lhs,typename Rhs,typename Res>
// void mkl_multiply(const Lhs& lhs, const Rhs& rhs, Res& res)
// {
//   char n = 'N';
//   float alpha = 1;
//   char matdescra[6];
//   matdescra[0] = 'G';
//   matdescra[1] = 0;
//   matdescra[2] = 0;
//   matdescra[3] = 'C';
//   mkl_scscmm(&n, lhs.rows(), rhs.cols(), lhs.cols(), &alpha, matdescra,
//              lhs._valuePtr(), lhs._innerIndexPtr(), lhs.outerIndexPtr(),
//              pntre, b, &ldb, &beta, c, &ldc);
// //   mkl_somatcopy('C', 'T', lhs.rows()