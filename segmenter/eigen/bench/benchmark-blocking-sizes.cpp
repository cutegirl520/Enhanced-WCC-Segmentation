// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Jacob <benoitjacob@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <memory>
#include <cstdio>

bool eigen_use_specific_block_size;
int eigen_block_size_k, eigen_block_size_m, eigen_block_size_n;
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZES eigen_use_specific_block_size
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_K eigen_block_size_k
#define EIGEN_TEST_