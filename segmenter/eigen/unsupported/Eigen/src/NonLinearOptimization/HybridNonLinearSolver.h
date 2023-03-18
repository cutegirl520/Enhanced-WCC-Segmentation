
// -*- coding: utf-8
// vim: set fileencoding=utf-8

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HYBRIDNONLINEARSOLVER_H
#define EIGEN_HYBRIDNONLINEARSOLVER_H

namespace Eigen { 

namespace HybridNonLinearSolverSpace { 
    enum Status {
        Running = -1,
        ImproperInputParameters = 0,
        RelativeErrorTooSmall = 1,
        TooManyFunctionEvaluation = 2,
        TolTooSmall = 3,
        NotMakingProgressJacobian = 4,
        NotMakingProgressIterations = 5,
        UserAsked = 6
    };
}

/**
  * \ingroup NonLinearOptimization_Module
  * \brief Finds a zero of a system of n
  * nonlinear functions in n variables by a modification of the Powell
  * hybrid method ("dogleg").
  *
  * The user must provide a subroutine which calculates the
  * functions. The Jacobian is either provided by the user, or approximated
  * using a forward-difference method.
  *
  */
template<typename FunctorType, typename Scalar=double>
class HybridNonLinearSolver
{
public:
    typedef DenseIndex Index;

    HybridNonLinearSolver(FunctorType &_functor)
        : functor(_functor) { nfev=njev=iter = 0;  fnorm= 0.; useExternalScaling=false;}

    struct Parameters {
        Parameters()
            : factor(Scalar(100.))
            , maxfev(1000)
            , xtol(std::sqrt(NumTraits<Scalar>::epsilon()))
            , nb_of_subdiagonals(-1)
            , nb_of_superdiagonals(-1)
            , epsfcn(Scalar(0.)) {}
        Scalar factor;
        Index maxfev;   // maximum number of function evaluation
        Scalar xtol;
        Index nb_of_subdiagonals;
        Index nb_of_superdiagonals;
        Scalar epsfcn;
    };
    typedef Matrix< Scalar, Dynamic, 1 > FVectorType;
    typedef Matrix< Scalar, Dynamic, Dynamic > JacobianType;
    /* TODO: if eigen provides a triangular storage, use it here */
    typedef Matrix< Scalar, Dynamic, Dynamic > UpperTriangularType;

    HybridNonLinearSolverSpace::Status hybrj1(
            FVectorType  &x,
            const Scalar tol = std::sqrt(NumTraits<Scalar>::epsilon())
            );

    HybridNonLinearSolverSpace::Status solveInit(FVectorType  &x);
    HybridNonLinearSolverSpace::Status solveOneStep(FVectorType  &x);
    HybridNonLinearSolverSpace::Status solve(FVectorType  &x);

    HybridNonLinearSolverSpace::Status hybrd1(
            FVectorType  &x,
            const Scalar tol = std::sqrt(NumTraits<Scalar>::epsilon())
            );

    HybridNonLinearSolverSpace::Status solveNumericalDiffInit(FVectorType  &x);
    HybridNonLinearSolverSpace::Status solveNumericalDiffOneStep(FVectorType  &x);
    HybridNonLinearSolverSpace::Status solveNumericalDiff(FVectorType  &x);

    void resetParameters(void) { parameters = Parameters(); }
    Parameters parameters;
    FVectorType  fvec, qtf, diag;
    JacobianType fjac;
    UpperTriangularType R;
    Index nfev;
    Index njev;
    Index iter;
    Scalar fnorm;
    bool useExternalScaling; 
private:
    FunctorType &functor;
    Index n;
    Scalar sum;