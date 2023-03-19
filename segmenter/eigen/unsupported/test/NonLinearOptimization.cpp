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
            fjac(k,k) = 3.- 4.*x[k];
            if (k) fjac(k,k-1) = -1.;
            if (k != n-1) fjac(k,k+1) = -2.;
        }
        return 0;
    }
};


void testHybrj1()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  info = solver.hybrj1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrj()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);


  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solve(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);

}

struct hybrd_functor : Functor<double>
{
    hybrd_functor(void) : Functor<double>(9,9) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double temp, temp1, temp2;
        const VectorXd::Index n = x.size();

        assert(fvec.size()==n);
        for (VectorXd::Index k=0; k < n; k++)
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
};

void testHybrd1()
{
  int n=9, info;
  VectorXd x(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  info = solver.hybrd1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 20);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrd()
{
  const int n=9;
  int info;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  solver.parameters.nb_of_subdiagonals = 1;
  solver.parameters.nb_of_superdiagonals = 1;
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solveNumericalDiff(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 14);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref <<
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

struct lmstr_functor : Functor<double>
{
    lmstr_functor(void) : Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec)
    {
        /*  subroutine fcn for lmstr1 example. */
        double tmp1, tmp2, tmp3;
        static const double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        assert(15==fvec.size());
        assert(3==x.size());

        for (int i=0; i<15; i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }
    int df(const VectorXd &x, VectorXd &jac_row, VectorXd::Index rownb)
    {
        assert(x.size()==3);
        assert(jac_row.size()==x.size());
        double tmp1, tmp2, tmp3, tmp4;

        VectorXd::Index i = rownb-2;
        tmp1 = i+1;
        tmp2 = 16 - i - 1;
        tmp3 = (i>=8)? tmp2 : tmp1;
        tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
        jac_row[0] = -1;
        jac_row[1] = tmp1*tmp2/tmp4;
        jac_row[2] = tmp1*tmp3/tmp4;
        return 0;
    }
};

void testLmstr1()
{
  const int n=3;
  int info;

  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.lmstr1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 6);
  VERIFY_IS_EQUAL(lm.njev, 5);

  // check norm
  VERIFY_IS_APPROX(lm.fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695 ;
  VERIFY_IS_APPROX(x, x_ref);
}

void testLmstr()
{
  const int n=3;
  int info;
  double fnorm;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmstr_functor functor;
  LevenbergMarquardt<lmstr_functor> lm(functor);
  info = lm.minimizeOptimumStorage(x);

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

}

struct lmdif_functor : Functor<double>
{
    lmdif_functor(void) : Functor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        int i;
        double tmp1,tmp2,tmp3;
        static const double y[15]={1.4e-1,1.8e-1,2.2e-1,2.5e-1,2.9e-1,3.2e-1,3.5e-1,3.9e-1,
            3.7e-1,5.8e-1,7.3e-1,9.6e-1,1.34e0,2.1e0,4.39e0};

        assert(x.size()==3);
        assert(fvec.size()==15);
        for (i=0; i<15; i++)
        {
            tmp1 = i+1;
            tmp2 = 15 - i;
            tmp3 = tmp1;

            if (i >= 8) tmp3 = tmp2;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }
};

void testLmdif1()
{
  const int n=3;
  int info;

  VectorXd x(n), fvec(15);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  DenseIndex nfev;
  info = LevenbergMarquardt<lmdif_functor>::lmdif1(functor, x, &nfev);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(nfev, 26);

  // check norm
  functor(x, fvec);
  VERIFY_IS_APPROX(fvec.blueNorm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.0824106, 1.1330366, 2.3436947;
  VERIFY_IS_APPROX(x, x_ref);

}

void testLmdif()
{
  const int m=15, n=3;
  int info;
  double fnorm, covfac;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmdif_functor functor;
  NumericalDiff<lmdif_functor> numDiff(functor);
  LevenbergMarquardt<NumericalDiff<lmdif_functor> > lm(numDiff);
  info = lm.minimize(x);

  // check return values
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 26);

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
      0.0001531202,   0.002869942,  -0.002656662,
      0.002869942,    0.09480937,   -0.09098997,
      -0.002656662,   -0.09098997,    0.08778729;

//  std::cout << fjac*covfac << std::endl;

  MatrixXd cov;
  cov =  covfac*lm.fjac.topLeftCorner<n,n>();
  VERIFY_IS_APPROX( cov, cov_ref);
  // TODO: why isn't this allowed ? :
  // VERIFY_IS_APPROX( covfac*fjac.topLeftCorner<n,n>() , cov_ref);
}

struct chwirut2_functor : Functor<double>
{
    chwirut2_functor(void) : Functor<double>(3,54) {}
    static const double m_x[54];
    static const double m_y[54];
    int operator()(const VectorXd &b, VectorXd &fvec)
    {
        int i;

        assert(b.size()==3);
        assert(fvec.size()==54);
        for(i=0; i<54; i++) {
            double x = m_x[i];
            fvec[i] = exp(-b[0]*x)/(b[1]+b[2]*x) - m_y[i];
        }
        return 0;
    }
    int df(const VectorXd &b, MatrixXd &fjac)
    {
        assert(b.size()==3);
        assert(fjac.rows()==54);
        assert(fjac.cols()==3);
        for(int i=0; i<54; i++) {
            double x = m_x[i];
            double factor = 1./(b[1]+b[2]*x);
            double e = exp(-b[0]*x);
            fjac(i,0) = -x*e*factor;
            fjac(i,1) = -e*factor*factor;
            fjac(i,2) = -x*e*factor*factor;
        }
        return 0;
    }
};
const double chwirut2_functor::m_x[54] = { 0.500E0, 1.000E0, 1.750E0, 3.750E0, 5.750E0, 0.875E0, 2.250E0, 3.250E0, 5.250E0, 0.750E0, 1.750E0, 2.750E0, 4.750E0, 0.625E0, 1.250E0, 2.250E0, 4.250E0, .500E0, 3.000E0, .750E0, 3.000E0, 1.500E0, 6.000E0, 3.000E0, 6.000E0, 1.500E0, 3.000E0, .500E0, 2.000E0, 4.000E0, .750E0, 2.000E0, 5.000E0, .750E0, 2.250E0, 3.750E0, 5.750E0, 3.000E0, .750E0, 2.500E0, 4.000E0, .750E0, 2.500E0, 4.000E0, .750E0, 2.500E0, 4.000E0, .500E0, 6.000E0, 3.000E0, .500E0, 2.750E0, .500E0, 1.750E0};
const double chwirut2_functor::m_y[54] = { 92.9000E0 ,57.1000E0 ,31.0500E0 ,11.5875E0 ,8.0250E0 ,63.6000E0 ,21.4000E0 ,14.2500E0 ,8.4750E0 ,63.8000E0 ,26.8000E0 ,16.4625E0 ,7.1250E0 ,67.3000E0 ,41.0000E0 ,21.1500E0 ,8.1750E0 ,81.5000E0 ,13.1200E0 ,59.9000E0 ,14.6200E0 ,32.9000E0 ,5.4400E0 ,12.5600E0 ,5.4400E0 ,32.0000E0 ,13.9500E0 ,75.8000E0 ,20.0000E0 ,10.4200E0 ,59.5000E0 ,21.6700E0 ,8.5500E0 ,62.0000E0 ,20.2000E0 ,7.7600E0 ,3.7500E0 ,11.8100E0 ,54.7000E0 ,23.7000E0 ,11.5500E0 ,61.3000E0 ,17.7000E0 ,8.7400E0 ,59.2000E0 ,16.3000E0 ,8.6200E0 ,81.0000E0 ,4.8700E0 ,14.6200E0 ,81.7000E0 ,17.1700E0 ,81.3000E0 ,28.9000E0  };

// http://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml
void testNistChwirut2(void)
{
  const int n=3;
  int info;

  VectorXd x(n);

  /*
   * First try
   */
  x<< 0.1, 0.01, 0.02;
  // do the computation
  chwirut2_functor functor;
  LevenbergMarquardt<chwirut2_functor> lm(functor);
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 10);
  VERIFY_IS_EQUAL(lm.njev, 8);
  // check norm^2
  VERIFY_IS_APPROX(lm.fvec.squaredNorm(), 5.1304802941E+02);
  // check x
  VERIFY_IS_APPROX(x[0], 1.6657666537E-01);
  VERIFY_IS_APPROX(x[1], 5.1653291286E-03);
  VERIFY_IS_APPROX(x[2], 1.2150007096E-02);

  /*
   * Second try
   */
  x<< 0.15, 0.008, 0.010;
  // do the computation
  lm.resetParameters();
  lm.parameters.ftol = 1.E6*NumTraits<double>::epsilon();
  lm.parameters.xtol = 1.E6*NumTraits<double>::epsilon();
  info = lm.minimize(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(lm.nfev, 7);
  VERIFY_IS_EQUA