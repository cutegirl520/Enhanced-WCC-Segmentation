// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_1
#define EIGEN_UNALIGNED_VECTORIZE 1
#endif

#ifdef EIGEN_TEST_PART_2
#define EIGEN_UNALIGNED_VECTORIZE 0
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#undef EIGEN_DEFAULT_TO_ROW_MAJOR
#endif
#define EIGEN_DEBUG_ASSIGN
#include "main.h"
#include <typeinfo>

using internal::demangle_flags;
using internal::demangle_traversal;
using internal::demangle_unrolling;

template<typename Dst, typename Src>
bool test_assign(const Dst&, const Src&, int traversal, int unrolling)
{
  typedef internal::copy_using_evaluator_traits<internal::evaluator<Dst>,internal::evaluator<Src>, internal::assign_op<typename Dst::Scalar,typename Src::Scalar> > traits;
  bool res = traits::Traversal==traversal;
  if(unrolling==InnerUnrolling+CompleteUnrolling)
    res = res && (int(traits::Unrolling)==InnerUnrolling || int(traits::Unrolling)==CompleteUnrolling);
  else
    res = res && int(traits::Unrolling)==unrolling;
  if(!res)
  {
    std::cerr << "Src: " << demangle_flags(Src::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Src>::Flags) << std::endl;
    std::cerr << "Dst: " << demangle_flags(Dst::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Dst>::Flags) << std::endl;
    traits::debug();
    std::cerr << " Expected Traversal == " << demangle_traversal(traversal)
              << " got " << demangle_traversal(traits::Traversal) << "\n";
    std::cerr << " Expected Unrolling == " << demangle_unrolling(unrolling)
              << " got " << demangle_unrolling(traits::Unrolling) << "\n";
  }
  return res;
}

template<typename Dst, typename Src>
bool test_assign(int traversal, int unrolling)
{
  typedef internal::copy_using_evaluator_traits<internal::evaluator<Dst>,internal::evaluator<Src>, internal::assign_op<typename Dst::Scalar,typename Src::Scalar> > traits;
  bool res = traits::Traversal==traversal && traits::Unrolling==unrolling;
  if(!res)
  {
    std::cerr << "Src: " << demangle_flags(Src::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Src>::Flags) << std::endl;
    std::cerr << "Dst: " << demangle_flags(Dst::Flags) << std::endl;
    std::cerr << "     " << demangle_flags(internal::evaluator<Dst>::Flags) << std::endl;
    traits::debug();
    std::cerr << " Expected Traversal == " << demangle_traversal(traversal)
              << " got " << demangle_traversal(traits::Traversal) << "\n";
    std::cerr << " Expected Unrolling == " << demangle_unrolling(unrolling)
              << " got " << demangle_unrolling(traits::Unrolling) << "\n";
  }
  return res;
}

template<typename Xpr>
bool test_redux(const Xpr&, int traversal, int unrolling)
{
  typedef typename Xpr::Scalar Scalar;
  typedef internal::redux_traits<internal::scalar_sum_op<Scalar,Scalar>,internal::redux_evaluator<Xpr> > traits;
  
  bool res = tra