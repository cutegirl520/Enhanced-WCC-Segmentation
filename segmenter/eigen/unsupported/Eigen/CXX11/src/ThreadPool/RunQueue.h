// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_RUNQUEUE_H_
#define EIGEN_CXX11_THREADPOOL_RUNQUEUE_H_


namespace Eigen {

// RunQueue is a fixed-size, partially non-blocking deque or Work items.
// Operations on front of the queue must be done by a single thread (owner),
// operations on back of the queue can be done by multiple threads concurrently.
//
// Algorithm outline:
// All remote threads operating on the queue back are serialized by a mutex.
// This ensures that at most two threads access state: owner and one remote
// thread (Size aside). The algorithm ensures that the occupied region of the
// underlying array is logically continuous (can wraparound, but no stray
// occupied elements). Owner operates on one end of this region, remote thread
// operates on the other end. Synchronization between these threads
// (potential consumption of the last element and take up of the last empty
// element) happens by means of state variable in each element. States are:
// empty, busy (in process of insertion of removal) and ready. Threads claim
// elements (empty->busy and ready->busy transitions) by means of a CAS
// operation. The finishing transition (busy->empty and busy->ready) are done
// with plain store as the element is exclusively owned by the current thread.
//
// Note: we could permit only pointers as elements, then we would not need
// separate state variable as null/non-null pointer value would serve as state,
// but that would require malloc/free per operation for large, complex values
// (and this is designed to store std::function<()>)