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
   * \param computationOptions optional parameter allowing to specify if you want full or thin U or V unitaries to be computed.
   *                           By default, none is computed. This is a bit - field, the possible bits are #ComputeFullU, #ComputeThinU, 
   *                           #ComputeFullV, #ComputeThinV.
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number of columns (for example MatrixXf). They also are not
   * available with the (non - default) FullPivHouseholderQR preconditioner.
   */
  BDCSVD& compute(const MatrixType& matrix, unsigned int computationOptions);

  /** \brief Method performing the decomposition of given matrix using current options.
   *
   * \param matrix the matrix to decompose
   *
   * This method uses the current \a computationOptions, as already passed to the constructor or to compute(const MatrixType&, unsigned int).
   */
  BDCSVD& compute(const MatrixType& matrix)
  {
    return compute(matrix, this->m_computationOptions);
  }

  void setSwitchSize(int s) 
  {
    eigen_assert(s>3 && "BDCSVD the size of the algo switch has to be greater than 3");
    m_algoswap = s;
  }
 
private:
  void allocate(Index rows, Index cols, unsigned int computationOptions);
  void divide(Index firstCol, Index lastCol, Index firstRowW, Index firstColW, Index shift);
  void computeSVDofM(Index firstCol, Index n, MatrixXr& U, VectorType& singVals, MatrixXr& V);
  void computeSingVals(const ArrayRef& col0, const ArrayRef& diag, const IndicesRef& perm, VectorType& singVals, ArrayRef shifts, ArrayRef mus);
  void perturbCol0(const ArrayRef& col0, const ArrayRef& diag, const IndicesRef& perm, const VectorType& singVals, const ArrayRef& shifts, const ArrayRef& mus, ArrayRef zhat);
  void computeSingVecs(const ArrayRef& zhat, const ArrayRef& diag, const IndicesRef& perm, const VectorType& singVals, const ArrayRef& shifts, const ArrayRef& mus, MatrixXr& U, MatrixXr& V);
  void deflation43(Index firstCol, Index shift, Index i, Index size);
  void deflation44(Index firstColu , Index firstColm, Index firstRowW, Index firstColW, Index i, Index j, Index size);
  void deflation(Index firstCol, Index lastCol, Index k, Index firstRowW, Index firstColW, Index shift);
  template<typename HouseholderU, typename HouseholderV, typename NaiveU, typename NaiveV>
  void copyUV(const HouseholderU &householderU, const HouseholderV &householderV, const NaiveU &naiveU, const NaiveV &naivev);
  void structured_update(Block<MatrixXr,Dynamic,Dynamic> A, const MatrixXr &B, Index n1);
  static RealScalar secularEq(RealScalar x, const ArrayRef& col0, const ArrayRef& diag, const IndicesRef &perm, const ArrayRef& diagShifted, RealScalar shift);

protected:
  MatrixXr m_naiveU, m_naiveV;
  MatrixXr m_computed;
  Index m_nRec;
  ArrayXr m_workspace;
  ArrayXi m_workspaceI;
  int m_algoswap;
  bool m_isTranspose, m_compU, m_compV;
  
  using Base::m_singularValues;
  using Base::m_diagSize;
  using Base::m_computeFullU;
  using Base::m_computeFullV;
  using Base::m_computeThinU;
  using Base::m_computeThinV;
  using Base::m_matrixU;
  using Base::m_matrixV;
  using Base::m_isInitialized;
  using Base::m_nonzeroSingularValues;

public:  
  int m_numIters;
}; //end class BDCSVD


// Method to allocate and initialize matrix and attributes
template<typename MatrixType>
void BDCSVD<MatrixType>::allocate(Index rows, Index cols, unsigned int computationOptions)
{
  m_isTranspose = (cols > rows);

  if (Base::allocate(rows, cols, computationOptions))
    return;
  
  m_computed = MatrixXr::Zero(m_diagSize + 1, m_diagSize );
  m_compU = computeV();
  m_compV = computeU();
  if (m_isTranspose)
    std::swap(m_compU, m_compV);
  
  if (m_compU) m_naiveU = MatrixXr::Zero(m_diagSize + 1, m_diagSize + 1 );
  else         m_naiveU = MatrixXr::Zero(2, m_diagSize + 1 );
  
  if (m_compV) m_naiveV = MatrixXr::Zero(m_diagSize, m_diagSize);
  
  m_workspace.resize((m_diagSize+1)*(m_diagSize+1)*3);
  m_workspaceI.resize(3*m_diagSize);
}// end allocate

template<typename MatrixType>
BDCSVD<MatrixType>& BDCSVD<MatrixType>::compute(const MatrixType& matrix, unsigned int computationOptions) 
{
#ifdef EIGEN_BDCSVD_DEBUG_VERBOSE
  std::cout << "\n\n\n======================================================================================================================\n\n\n";
#endif
  allocate(matrix.rows(), matrix.cols(), computationOptions);
  using std::abs;

  const RealScalar considerZero = (std::numeric_limits<RealScalar>::min)();
  
  //**** step -1 - If the problem is too small, directly falls back to JacobiSVD and return
  if(matrix.cols() < m_algoswap)
  {
    // FIXME this line involves temporaries
    JacobiSVD<MatrixType> jsvd(matrix,computationOptions);
    if(computeU()) m_matrixU = jsvd.matrixU();
    if(computeV()) m_matrixV = jsvd.matrixV();
    m_singularValues = jsvd.singularValues();
    m_nonzeroSingularValues = jsvd.nonzeroSingularValues();
    m_isInitialized = true;
    return *this;
  }
  
  //**** step 0 - Copy the input matrix and apply scaling to reduce over/under-flows
  RealScalar scale = matrix.cwiseAbs().maxCoeff();
  if(scale==RealScalar(0)) scale = RealScalar(1);
  MatrixX copy;
  if (m_isTranspose) copy = matrix.adjoint()/scale;
  else               copy = matrix/scale;
  
  //**** step 1 - Bidiagonalization
  // FIXME this line involves temporaries
  internal::UpperBidiagonalization<MatrixX> bid(copy);

  //**** step 2 - Divide & Conquer
  m_naiveU.setZero();
  m_naiveV.setZero();
  // FIXME this line involves a temporary matrix
  m_computed.topRows(m_diagSize) = bid.bidiagonal().toDenseMatrix().transpose();
  m_computed.template bottomRows<1>().setZero();
  divide(0, m_diagSize - 1, 0, 0, 0);

  //**** step 3 - Copy singular values and vectors
  for (int i=0; i<m_diagSize; i++)
  {
    RealScalar a = abs(m_computed.coeff(i, i));
    m_singularValues.coeffRef(i) = a * scale;
    if (a<considerZero)
    {
      m_nonzeroSingularValues = i;
      m_singularValues.tail(m_diagSize - i - 1).setZero();
      break;
    }
    else if (i == m_diagSize - 1)
    {
      m_nonzeroSingularValues = i + 1;
      break;
    }
  }

#ifdef EIGEN_BDCSVD_DEBUG_VERBOSE
//   std::cout << "m_naiveU\n" << m_naiveU << "\n\n";
//   std::cout << "m_naiveV\n" << m_naiveV << "\n\n";
#endif
  if(m_isTranspose) copyUV(bid.householderV(), bid.householderU(), m_naiveV, m_naiveU);
  else              copyUV(bid.householderU(), bid.householderV(), m_naiveU, m_naiveV);

  m_isInitialized = true;
  return *this;
}// end compute


template<typename MatrixType>
template<typename HouseholderU, typename HouseholderV, typename NaiveU, typename NaiveV>
void BDCSVD<MatrixType>::copyUV(const HouseholderU &householderU, const HouseholderV &householderV, const NaiveU &naiveU, const NaiveV &naiveV)
{
  // Note exchange of U and V: m_matrixU is set from m_naiveV and vice versa
  if (computeU())
  {
    Index Ucols = m_computeThinU ? m_diagSize : householderU.cols();
    m_matrixU = MatrixX::Identity(householderU.cols(), Ucols);
    m_matrixU.topLeftCorner(m_diagSize, m_diagSize) = naiveV.template cast<Scalar>().topLeftCorner(m_diagSize, m_diagSize);
    householderU.applyThisOnTheLeft(m_matrixU); // FIXME this line involves a temporary buffer
  }
  if (computeV())
  {
    Index Vcols = m_computeThinV ? m_diagSize : householderV.cols();
    m_matrixV = MatrixX::Identity(householderV.cols(), Vcols);
    m_matrixV.topLeftCorner(m_diagSize, m_diagSize) = naiveU.template cast<Scalar>().topLeftCorner(m_diagSize, m_diagSize);
    householderV.applyThisOnTheLeft(m_matrixV); // FIXME this line involves a temporary buffer
  }
}

/** \internal
  * Performs A = A * B exploiting the special structure of the matrix A. Splitting A as:
  *  A = [A1]
  *      [A2]
  * such that A1.rows()==n1, then we assume that at least half of the columns of A1 and A2 are zeros.
  * We can thus pack them prior to the the matrix product. However, this is only worth the effort if the matrix is large
  * enough.
  */
template<typename MatrixType>
void BDCSVD<MatrixType>::structured_update(Block<MatrixXr,Dynamic,Dynamic> A, const MatrixXr &B, Index n1)
{
  Index n = A.rows();
  if(n>100)
  {
    // If the matrices are large enough, let's exploit the sparse structure of A by
    // splitting it in half (wrt n1), and packing the non-zero columns.
    Index n2 = n - n1;
    Map<MatrixXr> A1(m_workspace.data()      , n1, n);
    Map<MatrixXr> A2(m_workspace.data()+ n1*n, n2, n);
    Map<MatrixXr> B1(m_workspace.data()+  n*n, n,  n);
    Map<MatrixXr> B2(m_workspace.data()+2*n*n, n,  n);
    Index k1=0, k2=0;
    for(Index j=0; j<n; ++j)
    {
      if( (A.col(j).head(n1).array()!=0).any() )
      {
        A1.col(k1) = A.col(j).head(n1);
        B1.row(k1) = B.row(j);
        ++k1;
      }
      if( (A.col(j).tail(n2).array()!=0).any() )
      {
        A2.col(k2) = A.col(j).tail(n2);
        B2.row(k2) = B.row(j);
        ++k2;
      }
    }
  
    A.topRows(n1).noalias()    = A1.leftCols(k1) * B1.topRows(k1);
    A.bottomRows(n2).noalias() = A2.leftCols(k2) * B2.topRows(k2);
  }
  else
  {
    Map<MatrixXr,Aligned> tmp(m_workspace.data(),n,n);
    tmp.noalias() = A*B;
    A = tmp;
  }
}

// The divide algorithm is done "in place", we are always working on subsets of the same matrix. The divide methods takes as argument the 
// place of the submatrix we are currently working on.

//@param firstCol : The Index of the first column of the submatrix of m_computed and for m_naiveU;
//@param lastCol : The Index of the last column of the submatrix of m_computed and for m_naiveU; 
// lastCol + 1 - firstCol is the size of the submatrix.
//@param firstRowW : The Index of the first row of the matrix W that we are to change. (see the reference paper section 1 for more information on W)
//@param firstRowW : Same as firstRowW with the column.
//@param shift : Each time one takes the left submatrix, one must add 1 to the shift. Why? Because! We actually want the last column of the U submatrix 
// to become the first column (*coeff) and to shift all the other columns to the right. There are more details on the reference paper.
template<typename MatrixType>
void BDCSVD<MatrixType>::divide (Index firstCol, Index lastCol, Index firstRowW, Index firstColW, Index shift)
{
  // requires rows = cols + 1;
  using std::pow;
  using std::sqrt;
  using std::abs;
  const Index n = lastCol - firstCol + 1;
  const Index k = n/2;
  const RealScalar considerZero = (std::numeric_limits<RealScalar>::min)();
  RealScalar alphaK;
  RealScalar betaK; 
  RealScalar r0; 
  RealScalar lambda, phi, c0, s0;
  VectorType l, f;
  // We use the other algorithm which is more efficient for small 
  // matrices.
  if (n < m_algoswap)
  {
    // FIXME this line involves temporaries
    JacobiSVD<MatrixXr> b(m_computed.block(firstCol, firstCol, n + 1, n), ComputeFullU | (m_compV ? ComputeFullV : 0));
    if (m_compU)
      m_naiveU.block(firstCol, firstCol, n + 1, n + 1).real() = b.matrixU();
    else 
    {
      m_naiveU.row(0).segment(firstCol, n + 1).real() = b.matrixU().row(0);
      m_naiveU.row(1).segment(firstCol, n + 1).real() = b.matrixU().row(n);
    }
    if (m_compV) m_naiveV.block(firstRowW, firstColW, n, n).real() = b.matrixV();
    m_computed.block(firstCol + shift, firstCol + shift, n + 1, n).setZero();
    m_computed.diagonal().segment(firstCol + shift, n) = b.singularValues().head(n);
    return;
  }
  // We use the divide and conquer algorithm
  alphaK =  m_computed(firstCol + k, firstCol + k);
  betaK = m_computed(firstCol + k + 1, firstCol + k);
  // The divide must be done in that order in order to have good results. Divide change the data inside the submatrices
  // and the divide of