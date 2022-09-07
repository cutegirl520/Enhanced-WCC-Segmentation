// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse.h"
#include <Eigen/SparseCore>
#include <sstream>

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(IterativeSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess& g, Result &x) {
  if(internal::random<bool>())
  {
    // With a temporary through evaluator<SolveWithGuess>
    x = solver.derived().solveWithGuess(b,g) + Result::Zero(x.rows(), x.cols());
  }
  else
  {
    // direct evaluation within x through Assignment<Result,SolveWithGuess>
    x = solver.derived().solveWithGuess(b.derived(),g);
  }
}

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess& , Result& x) {
  if(internal::random<bool>())
    x = solver.derived().solve(b) + Result::Zero(x.rows(), x.cols());
  else
    x = solver.derived().solve(b);
}

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const SparseMatrixBase<Rhs>& b, const Guess& , Result& x) {
  x = solver.derived().solve(b);
}

template<typename Solver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const DenseMat& dA, const DenseRhs& db)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::StorageIndex StorageIndex;

  DenseRhs refX = dA.householderQr().solve(db);
  {
    Rhs x(A.cols(), b.cols());
    Rhs oldb = b;

    solver.compute(A);
    if (solver.info() != Success)
    {
      std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
      VERIFY(solver.info() == Success);
    }
    x = solver.solve(b);
    if (solver.info() != Success)
    {
      std::cerr << "WARNING | sparse solver testing: solving failed (" << typeid(Solver).name() << ")\n";
      return;
    }
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));

    x.setZero();
    solve_with_guess(solver, b, x, x);
    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    
    x.setZero();
    // test the analyze/factorize API
    solver.analyzePattern(A);
    solver.factorize(A);
    VERIFY(solver.info() == Success && "factorization failed