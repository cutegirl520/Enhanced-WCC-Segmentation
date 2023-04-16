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
using Eigen::internal::array_reduce;
using Eigen::internal::array_zip;
using Eigen::internal::array_zip_and_reduce;
using Eigen::internal::array_apply;
using Eigen::internal::array_apply_and_reduce;
using Eigen::internal::repeat;
using Eigen::internal::instantiate_by_c_array;

struct dummy_a {};
struct dummy_b {};
struct dummy_c {};
struct dummy_d {};
struct dummy_e {};

// dummy operation for testing apply
template<typename A, typename B> struct dummy_op;
template<> struct dummy_op<dummy_a, dummy_b> { typedef dummy_c type; };
template<> struct dummy_op<dummy_b, dummy_a> { typedef dummy_d type; };
template<> struct dummy_op<dummy_b, dummy_c> { typedef dummy_a type; };
template<> struct dummy_op<dummy_c, dummy_b> { typedef dummy_d type; };
template<> struct dummy_op<dummy_c, dummy_a> { typedef dummy_b type; };
template<> struct dummy_op<dummy_a, dummy_c> { typedef dummy_d type; };
template<> struct dummy_op<dummy_a, dummy_a> { typedef dummy_e type; };
template<> struct dummy_op<dummy_b, dummy_b> { typedef dummy_e type; };
template<> struct dummy_op<dummy_c, dummy_c> { typedef dummy_e type; };

template<typename A, typename B> struct dummy_test { constexpr static bool value = false; constexpr static int global_flags = 0; };
template<> struct dummy_test<dummy_a, dummy_a>     { constexpr static bool value = true;  constexpr static int global_flags = 1; };
template<> struct dummy_test<dummy_b, dummy_b>     { constexpr static bool value = true;  constexpr static int global_flags = 2; };
template<> struct dummy_test<dummy_c, dummy_c>     { constexpr static bool value = true;  constexpr static int global_flags = 4; };

struct times2_op { template<typename A> static A run(A v) { return v * 2; } };

struct dummy_inst
{
  int c;

  dummy_inst() : c(0) {}
  explicit dummy_inst(int) : c(1) {}
  dummy_inst(int, int) : c(2) {}
  dummy_inst(int, int, int) : c(3) {}
  dummy_inst(int, int, int, int) : c(4) {}
  dummy_inst(int, int, int, int, int) : c(5) {}
};

static void test_gen_numeric_list()
{
  VERIFY((is_same<typename gen_numeric_list<int, 0>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 1>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 2>::type, numeric_list<int, 0, 1>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 5>::type, numeric_list<int, 0, 1, 2, 3, 4>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 10>::type, numeric_list<int, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9>>::value));

  VERIFY((is_same<typename gen_numeric_list<int, 0, 42>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 1, 42>::type, numeric_list<int, 42>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 2, 42>::type, numeric_list<int, 42, 43>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 5, 42>::type, numeric_list<int, 42, 43, 44, 45, 46>>::value));
  VERIFY((is_same<typename gen_numeric_list<int, 10, 42>::type, numeric_list<int, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51>>::value));

  VERIFY((is_same<typename gen_numeric_list_reversed<int, 0>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 1>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 2>::type, numeric_list<int, 1, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 5>::type, numeric_list<int, 4, 3, 2, 1, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 10>::type, numeric_list<int, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0>>::value));

  VERIFY((is_same<typename gen_numeric_list_reversed<int, 0, 42>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 1, 42>::type, numeric_list<int, 42>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 2, 42>::type, numeric_list<int, 43, 42>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 5, 42>::type, numeric_list<int, 46, 45, 44, 43, 42>>::value));
  VERIFY((is_same<typename gen_numeric_list_reversed<int, 10, 42>::type, numeric_list<int, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42>>::value));

  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 0, 2, 3>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 1, 2, 3>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 2, 2, 3>::type, numeric_list<int, 0, 1>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 5, 2, 3>::type, numeric_list<int, 0, 1, 3, 2, 4>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 10, 2, 3>::type, numeric_list<int, 0, 1, 3, 2, 4, 5, 6, 7, 8, 9>>::value));

  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 0, 44, 45, 42>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 1, 44, 45, 42>::type, numeric_list<int, 42>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 2, 44, 45, 42>::type, numeric_list<int, 42, 43>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 5, 44, 45, 42>::type, numeric_list<int, 42, 43, 45, 44, 46>>::value));
  VERIFY((is_same<typename gen_numeric_list_swapped_pair<int, 10, 44, 45, 42>::type, numeric_list<int, 42, 43, 45, 44, 46, 47, 48, 49, 50, 51>>::value));

  VERIFY((is_same<typename gen_numeric_list_repeated<int, 0, 0>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename gen_numeric_list_repeated<int, 1, 0>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_repeated<int, 2, 0>::type, numeric_list<int, 0, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_repeated<int, 5, 0>::type, numeric_list<int, 0, 0, 0, 0, 0>>::value));
  VERIFY((is_same<typename gen_numeric_list_repeated<int, 10, 0>::type, numeric_list<int, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>>::value));
}

static void test_concat()
{
  VERIFY((is_same<typename concat<type_list<dummy_a, dummy_a>, type_list<>>::type, type_list<dummy_a, dummy_a>>::value));
  VERIFY((is_same<typename concat<type_list<>, type_list<dummy_a, dummy_a>>::type, type_list<dummy_a, dummy_a>>::value));
  VERIFY((is_same<typename concat<type_list<dummy_a, dummy_a>, type_list<dummy_a, dummy_a>>::type, type_list<dummy_a, dummy_a, dummy_a, dummy_a>>::value));
  VERIFY((is_same<typename concat<type_list<dummy_a, dummy_a>, type_list<dummy_b, dummy_c>>::type, type_list<dummy_a, dummy_a, dummy_b, dummy_c>>::value));
  VERIFY((is_same<typename concat<type_list<dummy_a>, type_list<dummy_b, dummy_c>>::type, type_list<dummy_a, dummy_b, dummy_c>>::value));

  VERIFY((is_same<typename concat<numeric_list<int, 0, 0>, numeric_list<int>>::type, numeric_list<int, 0, 0>>::value));
  VERIFY((is_same<typename concat<numeric_list<int>, numeric_list<int, 0, 0>>::type, numeric_list<int, 0, 0>>::value));
  VERIFY((is_same<typename concat<numeric_list<int, 0, 0>, numeric_list<int, 0, 0>>::type, numeric_list<int, 0, 0, 0, 0>>::value));
  VERIFY((is_same<typename concat<numeric_list<int, 0, 0>, numeric_list<int, 1, 2>>::type, numeric_list<int, 0, 0, 1, 2>>::value));
  VERIFY((is_same<typename concat<numeric_list<int, 0>, numeric_list<int, 1, 2>>::type, numeric_list<int, 0, 1, 2>>::value));

  VERIFY((is_same<typename mconcat<type_list<dummy_a>>::type, type_list<dummy_a>>::value));
  VERIFY((is_same<typename mconcat<type_list<dummy_a>, type_list<dummy_b>>::type, type_list<dummy_a, dummy_b>>::value));
  VERIFY((is_same<typename mconcat<type_list<dummy_a>, type_list<dummy_b>, type_list<dummy_c>>::type, type_list<dummy_a, dummy_b, dummy_c>>::value));
  VERIFY((is_same<typename mconcat<type_list<dummy_a>, type_list<dummy_b, dummy_c>>::type, type_list<dummy_a, dummy_b, dummy_c>>::value));
  VERIFY((is_same<typename mconcat<type_list<dummy_a, dummy_b>, type_list<dummy_c>>::type, type_list<dummy_a, dummy_b, dummy_c>>::value));

  VERIFY((is_same<typename mconcat<numeric_list<int, 0>>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename mconcat<numeric_list<int, 0>, numeric_list<int, 1>>::type, numeric_list<int, 0, 1>>::value));
  VERIFY((is_same<typename mconcat<numeric_list<int, 0>, numeric_list<int, 1>, numeric_list<int, 2>>::type, numeric_list<int, 0, 1, 2>>::value));
  VERIFY((is_same<typename mconcat<numeric_list<int, 0>, numeric_list<int, 1, 2>>::type, numeric_list<int, 0, 1, 2>>::value));
  VERIFY((is_same<typename mconcat<numeric_list<int, 0, 1>, numeric_list<int, 2>>::type, numeric_list<int, 0, 1, 2>>::value));
}

static void test_slice()
{
  typedef type_list<dummy_a, dummy_a, dummy_b, dummy_b, dummy_c, dummy_c> tl;
  typedef numeric_list<int, 0, 1, 2, 3, 4, 5> il;

  VERIFY((is_same<typename take<0, tl>::type, type_list<>>::value));
  VERIFY((is_same<typename take<1, tl>::type, type_list<dummy_a>>::value));
  VERIFY((is_same<typename take<2, tl>::type, type_list<dummy_a, dummy_a>>::value));
  VERIFY((is_same<typename take<3, tl>::type, type_list<dummy_a, dummy_a, dummy_b>>::value));
  VERIFY((is_same<typename take<4, tl>::type, type_list<dummy_a, dummy_a, dummy_b, dummy_b>>::value));
  VERIFY((is_same<typename take<5, tl>::type, type_list<dummy_a, dummy_a, dummy_b, dummy_b, dummy_c>>::value));
  VERIFY((is_same<typename take<6, tl>::type, type_list<dummy_a, dummy_a, dummy_b, dummy_b, dummy_c, dummy_c>>::value));

  VERIFY((is_same<typename take<0, il>::type, numeric_list<int>>::value));
  VERIFY((is_same<typename take<1, il>::type, numeric_list<int, 0>>::value));
  VERIFY((is_same<typename take<2, il>::type, numeric_list<int, 0, 1>>::value));
  VERIFY((is_same<typename take<3, il>::type, numeric_list<int, 0, 1, 2>>::value));
  VERIFY((is_same<typename take<4, il>::type, numeric_list<int, 0, 1, 2, 3>>::value));
  VERIFY((is_same<typename take<5, il>::type, numeric_list<int, 0, 1, 2, 3, 4>>::value));
  VERIFY((is_same<typename take<6, il>::type, numeric_list<int, 0, 1, 2, 3, 4, 5>>::value));
  
  VERIFY((is_same<typename skip<0, tl>::type, type_list<dummy_a, dummy_a, dummy_b, dummy_b, dummy_c, dummy_c>>::value));
  VERIFY((is_same<typename skip<1, tl>::type, type_list<dummy_a, dummy_b, dummy_b, dummy_c, dummy_c>>::value));
  VERIFY((is_same<typename skip<2, tl>::type, type_list<dummy_b, dummy_b, dummy_c, dummy_c>>::value));
  VERIFY((is_same<typename skip<3, tl>::type, type_list<dummy_b, dummy_c, dummy_c>>::value));
  VERIFY((is_same<typename skip<4, tl>::type, type_list<dummy_c, dummy_c>>::value));
  VERIFY((is_same<typename skip<5, tl>::type, type_list<dummy_c>>::value));
  VERIFY((is_same<typename skip<6, tl>::type, type_list<>>::value));

  VERIFY((is_same<typename skip<0, il>::type, numeric_list<int, 0, 1, 2, 3, 4, 5>>::value));
  VERIFY((is_same<typename skip<1, il>::type, numeric_list<int, 1, 2, 3, 4, 5>>::value));
  VERIFY((is_same<typename skip<2, il>::type, numeric_list<int, 2, 3, 4, 5>>::value));
  VERIFY((is_same<typename skip<3, il>::type, numeric_list<int, 3, 4, 5>>::value));
  VERIFY((is_same<typename skip<4, il>::type, numeric_list<int, 4, 5>>::value));
  VERIFY((is_same<typename skip<5, il>::type, numeric_list<int, 5>>::value));
  VERIFY((is_same<typename skip<6, il>::type, numeric_list<int>>::value));

  VERIFY((is_same<typename slice<0, 3, tl>::type, typename take<3, tl>::type>::value));
  VERIFY((is_same<typename slice<0, 3, il>::type, typename 