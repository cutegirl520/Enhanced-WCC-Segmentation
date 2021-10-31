// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SUPERLUSUPPORT_H
#define EIGEN_SUPERLUSUPPORT_H

namespace Eigen {

#if defined(SUPERLU_MAJOR_VERSION) && (SUPERLU_MAJOR_VERSION >= 5)
#define DECL_GSSVX(PREFIX,FLOATTYPE,KEYTYPE)		\
    extern "C" {                                                                                          \
      extern void PREFIX##gssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,                  \
                                char *, FLOATTYPE *, FLOATTYPE *, SuperMatrix *, SuperMatrix *,           \
                                void *, int, SuperMatrix *, SuperMatrix *,                                \
                                FLOATTYPE *, FLOATTYPE *, FLOATTYPE *, FLOATTYPE *,                       \
                                GlobalLU_t *, mem_usage_t *, SuperLUStat_t *, int *);                     \
    }                                                                                                     \
    inline float SuperLU_gssvx(superlu_options_t *options, SuperMatrix *A,                                \
         int *perm_c, int *perm_r, int *etree, char *equed,                                               \
         FLOATTYPE *R, FLOATTYPE *C, SuperMatrix *L,                                                      \
         SuperMatrix *U, void *work, int lwork,                                                           \
         SuperMatrix *B, SuperMatrix *X,                                                                  \
         FLOATTYPE *recip_pivot_growth,                                                                   \
         FLOATTYPE *rcond, FLOATTYPE *ferr, FLOATTYPE *berr,                                              \
         SuperLUStat_t *stats, int *info, KEYTYPE) {                                                      \
    mem_usage_t mem_usage;                                                                                \
    GlobalLU_t gLU;                                                                                       \
    PREFIX##gssvx(options, A, perm_c, perm_r, etree, equed, R, C, L,                                      \
         U, work, lwork, B, X, recip_pivot_growth, rcond,                                                 \
         ferr, berr, &gLU, &mem_usage, stats, info);                                                      \
    return mem_usage.for_lu; /* bytes used by the factor storage */                                       \
  }
#else // version < 5.0
#define DECL_GSSVX(PREFIX,FLOATTYPE,KEYTYPE)		\
    extern "C" {                                                                                          \
      extern void PREFIX##gssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,                  \
                                char *, FLOATTYPE *, FLOATTYPE *, SuperMatrix *, SuperMatrix *,           \
                                void *, int, SuperMatrix *, SuperMatrix *,                                \
                                FLOATTYPE *, FLOATTYPE *, FLOATTYPE *, FLOATTYPE *,                       \
                                mem_usage_t *, SuperLUStat_t *, int *);                                   \
    }                                                                                                     \
    inline float SuperLU_gssvx(superlu_options_t *options, SuperMatrix *A,                                \
         int *perm_c, int *perm_r, int *etree, char *equed,                                               \
         FLOATTYPE *R, FLOATTYPE *C, SuperMatrix *L,                                                      \
         SuperMatrix *U, void *work, int lwork,                                                           \
         SuperMatrix *B, SuperMatrix *X,                                                                  \
         FLOATTYPE *recip_pivot_growth,                                                                   \
         FLOATTYPE *rcond, FLOATTYPE *ferr, FLOATTYPE *berr,                                              \
         SuperLUStat_t *stats, int *info, KEYTYPE) {                                                      \
    mem_usage_t mem_usage;                                                                                \
    PREFIX##gssvx(options, A, perm_c, perm_r, etree, equed, R, C, L,                                      \
         U, work, lwork, B, X, recip_pivot_growth, rcond,                                                 \
         ferr, berr, &mem_usage, stats, info);                                                            \
    return mem_usage.for_lu; /* bytes used by the factor storage */                                       \
  }
#endif

DECL_GSSVX(s,float,float)
DECL_GSSVX(c,float,std::complex<float>)
DECL_GSSVX(d,double,double)
DECL_GSSVX(z,double,std::complex<double>)

#ifdef MILU_ALPHA
#define EIGEN_SUPERLU_HAS_ILU
#endif

#ifdef EIGEN_SUPERLU_HAS_ILU

// similarly for the incomplete factorization using gsisx
#define DECL_GSISX(PREFIX,FLOATTYPE,KEYTYPE)                                                    \
    extern "C" {                                                                                \
      extern void PREFIX##gsisx(superlu_options_t *, SuperMatrix *, int *, int *, int *,        \
                         char *, FLOATTYPE *, FLOATTYPE *, SuperMatrix *, SuperMatrix *,        \
                         void *, int, SuperMatrix *, SuperMatrix *, FLOATTYPE *, FLOATTYPE *,   \
                         mem_usage_t *, SuperLUStat_t *, int *);                        \
    }                                                                                           \
    inline float SuperLU_gsisx(superlu_options_t *options, SuperMatrix *A,                      \
         int *perm_c, int *perm_r, int *etree, char *equed,                                     \
         FLOATTYPE *R, FLOATTYPE *C, SuperMatrix *L,                                            \
         SuperMatrix *U, void *work, int lwork,                                                 \
         SuperMatrix *B, SuperMatrix *X,                                                        \
         FLOATTYPE *recip_pivot_growth,                                                         \
         FLOATTYPE *rcond,                                                                      \
         SuperLUStat_t *stats, int *info, KEYTYPE) {                                            \
    mem_usage_t mem_usage;                                                              \
    PREFIX##gsisx(options, A, perm_c, perm_r, etree, equed, R, C, L,                            \
         U, work, lwork, B, X, recip_pivot_growth, rcond,                                       \
         &mem_usage, stats, info);                                                              \
    return mem_usage.for_lu; /* bytes used by the factor storage */                             \
  }

DECL_GSISX(s,float,float)
DECL_GSISX(c,float,std::complex<float>)
DECL_GSISX(d,double,double)
DECL_GSISX(z,double,std::complex<double>)

#endif

template<typename MatrixType>
struct SluMatrixMapHelper;

/** \internal
  *
  * A wrapper class for SuperLU matrices. It supports only compressed sparse matrices
  * and dense matrices. Supernodal and other fancy format are not supported by this wrapper.
  *
  * This wrapper class mainly aims to avoids the need of dynamic allocation of the storage structure.
  */
struct SluMatrix : SuperMatrix
{
  SluMatrix()
  {
    Store = &storage;
  }

  SluMatrix(const SluMatrix& other)
    : SuperMatrix(other)
  {
    Store = &storage;
    storage = other.storage;
  }

  SluMatrix& operator=(const SluMatrix& other)
  {
    SuperMatrix::operator=(static_cast<const SuperMatrix&>(other));
    Store = &storage;
    storage = other.storage;
    return *this;
  }

  struct
  {
    union {int nnz;int lda;};
    void *values;
    int *innerInd;
    int *outerInd;
  } storage;

  void setStorageType(Stype_t t)
  {
    Stype = t;
    if (t==SLU_NC || t==SLU_NR || t==SLU_DN)
      Store = &storage;
    else
    {
      eigen_assert(false && "storage type not supported");
      Store = 0;
    }
  }

  template<typename Scalar>
  void setScalarType()
  {
    if (internal::is_same<Scalar,float>::value)
      Dtype = SLU_S;
    else if (internal::is_same<Scalar,double>::value)
      Dtype = SLU_D;
    else if (internal::is_same<Scalar,std::complex<float> >::value)
      Dtype = SLU_C;
    else if (internal::is_same<Scalar,std::complex<double> >::value)
      Dtype = SLU_Z;
    else
    {
      eigen_assert(false && "Scalar type not supported by SuperLU");
    }
  }

  template<typename MatrixType>
  static SluMatrix Map(MatrixBase<MatrixType>& _mat)
  {
    MatrixType& mat(_mat.derived());
    eigen_assert( ((MatrixType::Flags&RowMajorBit)!=RowMajorBit) && "row-major dense matrices are not supported by SuperLU");
    SluMatrix res;
    res.setStorageType(SLU_DN);
    res.setScalarType<typename MatrixType::Scalar>();
    res.Mtype     = SLU_GE;

    res.nrow      = internal::convert_index<int>(mat.rows());
    res.ncol      = internal::convert_index<int>(mat.cols());

    res.storage.lda       = internal::convert_index<int>(MatrixType::IsVectorAtCompileTime ? mat.size() : mat.outerStride());
    res.storage.values    = (void*)(mat.data());
    return res;
  }

  template<typename MatrixType>
  static SluMatrix Map(SparseMatrixBase<MatrixType>& a_mat)
  {
    MatrixType &mat(a_mat.derived());
    SluMatrix res;
    if ((MatrixType::Flags&RowMajorBit)==RowMajorBit)
    {
      res.setStorageType(SLU_NR);
      res.nrow   