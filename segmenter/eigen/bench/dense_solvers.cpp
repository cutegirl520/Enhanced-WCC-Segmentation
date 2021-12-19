#include <iostream>
#include "BenchTimer.h"
#include <Eigen/Dense>
#include <map>
#include <string>
using namespace Eigen;

std::map<std::string,Array<float,1,4> > results;

template<typename Scalar,int Size>
void bench(int id, int size = Size)
{
  typedef Matrix<Scalar,Size,Size> Mat;
  Mat A(size,size);
  A.setRandom();
  A = A*A.adjoint();
  BenchTimer t_llt, t_ldlt, t_lu, t_fplu, t_qr, t_cpqr, t_cod, t_fpqr, t_jsvd, t_bdcsvd;
  
  int tries = 3;
  int rep = 1000/size;
  if(rep==0) rep = 1;
//   rep = rep*rep;
  
  LLT<Mat> llt(A);
  LDLT<Mat> ldlt(A);
  PartialPivLU<Mat> lu(A);
  FullPivLU<Mat> fplu(A);
  HouseholderQR<Mat> qr(A);
  ColPivHouseholderQR<Mat> cpqr(A);
  CompleteOrthogonalDecomposition<Mat> cod(A);
  FullPivHouseholderQR<Mat> fpqr(A);
  JacobiSVD<Mat> jsvd(A.rows(),A.cols());
  BDCSVD<Mat> bdcsvd(A.rows(),A.cols());
  
  BENCH(t_llt, tries, rep, llt.compute(A));
  BENCH(t_ldlt, tries, rep, ldlt.compute(A));
