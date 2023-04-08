// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <array>
#include <Eigen/CXX11/src/util/CXX11Meta.h>

using Eigen::internal::is_same;
using Eigen::internal::type_list;
using Eigen::internal::numeric_list;
using Eigen::internal::gen_numeric_list;
using Eigen::internal::gen_numeric_list_reversed;
using Eigen::internal::gen_numeric_list_swapped_pair;
using Eigen::internal::gen_numeric_list_repeated;
using Eigen::internal::concat;
using Eigen::internal::mconcat;
using Eigen::internal::take;
using Eigen::internal::skip;
using Eigen::internal::slice;
using Eigen::internal::get;
using Eigen::internal::id_numeric;
using Eigen::internal::id_type;
using Eigen::internal::is_same_gf;
using Eigen::internal::apply_op_from_left;
using Eigen::internal::apply_op_from_right;
using Eigen::internal::contained_in_list;
using Eigen::internal::contained_in_list_gf;
using Eigen::internal::arg_prod;
using Eigen::internal::arg_sum;
using Eigen::internal::sum_op;
using Eigen::internal::product_op;
using Eigen::internal::array_reverse;
using Eigen::internal::array_sum;
using Eigen::internal::array_prod;
using 