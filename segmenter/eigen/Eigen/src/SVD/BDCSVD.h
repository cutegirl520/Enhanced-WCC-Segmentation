// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
// 
// We used the "A Divide-And-Conquer Algorithm for the Bidiagonal SVD"
// research report written by Ming Gu and Stanley C.Eisenstat
// The code variable names correspond to the names they used in their 
// report
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
// Copyright (C) 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2014-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BDCSVD_H
#define EIGEN_BDCSVD_H
// #define EIGEN_BDCSVD_DEBUG_VERBOSE
// #define EIGEN_BDCSVD_SANITY_CHECKS

namespace Eigen {

#ifdef EIGEN_BDCSVD_DEBUG_VERBOSE
IOFormat bdcsvdfmt(8, 0, ", ", "\n", "  [", "]");
#endif
  
template<typename _MatrixType> class BDCSVD;

namespace internal {

template<typename _MatrixType> 
struct traits<BDCSVD<_MatrixType> >
{
  typedef _MatrixType MatrixType;
};  

} // end namespace internal
  
  
/** \ingroup SVD_Module
 *
 *
 * \class BDCSVD
 *
 * \brief class Bidiagonal Divide and Conquer SVD
 *
 * \tparam _MatrixType the type of the matrix of which we are computing the SVD decomposition
 *
 * This class first reduces the input matrix to bi-diagonal form using class UpperBidiagonalization,
 * and then performs a divide-and-conquer diagonalization. Small blocks are diagonalized using class JacobiSVD.
 * You can control the switching size with the setSwitchSize() method, default is 16.
 * For small matrice (<16), it is thus preferable to directly use JacobiSVD. For larger ones, BDCSVD is highly
 * recommended and can several order of magnitude faster.
 *
 * \warning this algorithm is unlikely to provide accurate result when compiled with unsafe math optimizations.
 * For instance, this concerns Intel's compiler (ICC), which perfroms such optimization by default unless
 * you compile with the \c -fp-model \c precise option. Likewise, the \c -ffast-math option of GCC or clang will
 * significantly degrade the accuracy.
 *
 * \sa class JacobiSVD
 */
template<typename _MatrixType> 
class BDCSVD : public SVDBase<BDCSVD<_MatrixType> >
{
  typedef SVDBase<BDCSVD> Base;
    
public:
  using Base::rows;
  using Base::cols;
  using Base::computeU;
  using Base::computeV;
  
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime, 
    ColsAtCompileTime = MatrixType::ColsAtCompileTime, 
    DiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_DYNAMIC(RowsAtCompileTime, ColsAtCompileTime), 
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime, 
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime, 
    MaxDiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(MaxRowsAtCompileTime, MaxColsAtCompileTime), 
    MatrixOptions = MatrixType::Options
  };

  typedef typename Base::MatrixUType MatrixUType;
  typedef typename Base::MatrixVType MatrixVType;
  typedef typename Base::SingularValuesType SingularValuesType;
  
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> MatrixX;
  typedef Matrix<RealScalar, Dynamic, Dynamic, ColMajor> MatrixXr;
  typedef Matrix<RealScalar, Dynamic, 1> VectorType;
  typedef Array<RealScalar, Dynamic, 1> ArrayXr;
  typedef Array<Index,1,Dynamic> ArrayXi;
  typedef Ref<ArrayXr> ArrayRef;
  typedef Ref<ArrayXi> IndicesRef;

  /** \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via BDCSVD::compute(const MatrixType&).
   */
  BDCSVD() : m_algoswap(16), m_numIters(0)
  {}


  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem size.
   * \sa BDCSVD()
   */
  BDCSVD(Index rows, Index cols, unsigned int computationOptions = 0)
    : m_algoswap(16), m_numIters(0)
  {
    allocate(rows, cols, computationOptions);
  }

  /** \brief Constructor performing the decomposition of given matrix.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you want full or thin U or V unitaries to be computed.
   *                           By default, none is computed. This is a bit - field, the possible bits are #ComputeFullU, #ComputeThinU, 
   *                           #ComputeFullV, #ComputeThinV.
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number of columns (for example MatrixXf). They also are not
   * available with the (non - default) FullPivHouseholderQR preconditioner.
   */
  BDCSVD(const MatrixType& matrix, unsigned int computationOptions = 0)
    : m_algoswap(16), m_numIters(0)
  {
    compute(matrix, computationOptions);
  }

  ~BDCSVD() 
  {
  }
  
  /** \brief Method performing the decomposition of given matrix using custom options.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you want full or thin U or V unitaries to be com