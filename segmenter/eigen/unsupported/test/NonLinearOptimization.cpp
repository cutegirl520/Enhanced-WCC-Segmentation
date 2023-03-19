// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>

#include <stdio.h>

#include "main.h"
#include <unsupported/Eigen/NonLinearOptimization>

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

using std::sqrt;

// tolerance for chekcing number of iterations
#define LM_EVAL_COUNT_TOL 4/3

int fcn_chkder(const VectorXd &x, VectorXd &fvec, MatrixXd &fjac, int iflag)
{
    /*      subroutine fcn for chkder example. */

    int i;
    assert(15 ==  fvec.size());
    assert(3 ==  x.size());
    double tmp1, tmp2, tmp3, tmp4;
    static const double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
        3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};


    if (iflag == 0)
        return 0;

    if (iflag != 2)
        for (i=0; i<15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;
            tmp3 = tmp1;
            if (i >= 8) tmp3 = tmp2;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
    else {
        for (i = 0; i < 15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;

            /* error introduced into next statement for illustration. */
            /* corrected statement should read    tmp3 = tmp1 . */

            tmp3 = tmp2;
            if (i >= 8) tmp3 = tmp2;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4=tmp4*tmp4;
            fjac(i,0) = -1.;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
    }
    return 0;
}


void testChkder()
{
  const int m=15, n=3;
  VectorXd x(n), fvec(m), xp, fvecp(m), err;
  MatrixXd fjac(m,n);
  VectorXi ipvt;

  /*      the following values should be suitable for */
  /*      checking the jacobian matrix. */
  x << 9.2e-1, 1.3e-1, 5.4e-1;

  internal::chkder(x, fvec, fjac, xp, fvecp, 1, err);
  fcn_chkder(x, fvec, fjac, 1);
  fcn_chkder(x, fvec, fjac, 2);
  fcn_chkder(xp, fvecp, fjac, 1);
  internal::chkder(x, fvec, fjac, xp, fvecp, 2, err);

  fvecp -= fvec;

  // check those
  VectorXd fvec_ref(m), fvecp_ref(m), err_ref(m);
  fvec_ref <<
      -1.181606, -1.429655, -1.606344,
      -1.745269, -1.840654, -1.921586,
      -1.984141, -2.022537, -2.468977,
      -2.827562, -3.473582, -4.437612,
      -6.047662, -9.267761, -18.91806;
  fvecp_ref <<
      -7.724666e-09, -3.432406e-09, -2.034843e-10,
      2.313685e-09,  4.331078e-09,  5.984096e-09,
      7.363281e-09,   8.53147e-09,  1.488591e-08,
      2.33585e-08,  3.522012e-08,  5.301255e-08,
      8.26666e-08,  1.419747e-07,   3.19899e-07;
  err_ref <<
      0.1141397,  0.09943516,  0.09674474,
      0.09980447,  0.1073116, 0.1220445,
      0.1526814, 1, 1,
      1, 1, 1,
      1, 1, 1;

  VERIFY_IS_APPROX(fvec, fvec_ref);
  VERIFY_IS_APPROX(fvecp, fvecp_ref);
  VERIFY_IS_APPROX(err, err_ref);
}

// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct lmder_functor : Functor<double>
{
    lmder_functor(void): Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }

    int df(const VectorXd &x, MatrixXd &fjac) const
    {
        double tmp1, tmp2, tmp3, tmp4;
        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
            fjac(i,0) = -1;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
        return 0;
    }
};

void testLmder1()
{
  int n=3, info;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmder()
{
  const int m=15, n=3;
  int info;
  double fnorm, covfac;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.minimize(x);

  // check return values
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  fnorm = lm.fvec.blueNorm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covfac = fnorm*fnorm/(m-n);
  internal::covar(lm.fjac, lm.permutation.indices()); // TODO : move this as a function of lm

  MatrixXd cov_ref(n,n);
  cov_ref <<
      0.0001531202,   0.002869941,  -0.002656662,
      0.002869941,    0.09480935,   -0.09098995,
      -0.002656662,   -0.09098995,    0.08778727;

//  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov =  covfac*lm.fjac.topLeftCorner<n,n>();
  VERIFY_IS_APPROX( cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

struct hybrj_functor : Functor<double>
{
    hybrj_functor(void) : Functor<double>(9,9) {}

    int operator()(const VectorXd &x, VectorXd &fvec)
    {
        double temp, temp1, temp2;
        const VectorXd::Index n = x.size();
        assert(fvec.size()==n);
        for (VectorXd::Index k = 0; k < n; k++)
        {
            temp = (3. - 2.*x[k])*x[k];
            temp1 = 0.;
            if (k) temp1 = x[k-1];
            temp2 = 0.;
            if (k != n-1) temp2 = x[k+1];
            fvec[k] = temp - temp1 - 2.*temp2 + 1.;
        }
        return 0;
    }
    int df(const VectorXd &x, MatrixXd &fjac)
    {
        const VectorXd::Index n = x.size();
        assert(fjac.rows()==n);
        assert(fjac.cols()==n);
        for (VectorXd::Index k = 0; k < n; k++)
        {
            for (VectorXd::Index j = 0; j < n; j++)
                fjac(k,j) = 0.;
            f