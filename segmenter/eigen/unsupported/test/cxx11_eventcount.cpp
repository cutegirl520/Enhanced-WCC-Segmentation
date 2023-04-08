// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS
#include "main.h"
#include <Eigen/CXX11/ThreadPool>

// Visual studio doesn't implement a rand_r() function since its
// implementation of rand() is already thread safe
int rand_reentrant(unsigned int* s) {
#ifdef EIGEN_COMP_MSVC_STRICT
  EIGEN_UNUSED_VARIABLE(s);
  return rand();
#else
  return rand_r(s);
#endif
}

static void test_basic_eventcount()
{
  std::vector<EventCount::Waiter> waiters(1);
  EventCount ec(waiters);
  EventCount::Waiter& w = waiters[0];
  ec.Notify(false);
  ec.Prewait(&w);
  ec.Notify(true);
  ec.CommitWait(&w);
  ec.Prewait(&w);
  ec.CancelWait(