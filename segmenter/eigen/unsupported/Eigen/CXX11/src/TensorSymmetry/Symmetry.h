// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSORSYMMETRY_SYMMETRY_H
#define EIGEN_CXX11_TENSORSYMMETRY_SYMMETRY_H

namespace Eigen {

enum {
  NegationFlag           = 0x01,
  ConjugationFlag        = 0x02
};

enum {
  GlobalRealFlag         = 0x01,
  GlobalImagFlag         = 0x02,
  GlobalZeroFlag         = 0x03
};

namespace internal {

template<std::size_t NumIndices, typename... Sym>                   struct tensor_symmetry_pre_analysis;
template<std::size_t NumIndices, typename... Sym>                   struct tensor_static_symgroup;
template<bool instantiate, std::size_t NumIndices, typename... Sym> struct tensor_static_symgroup_if;
template<typename Tensor_> struct tensor_symmetry_calculate_flags;
template<typename Tensor_> struct tensor_symmetry_assign_value;
template<typename... Sym> struct tensor_symmetry_num_indices;

} // end namespace internal

template<int One_, int Two_>
struct Symmetry
{
  static_assert(One_ != Two_, "Symmetries must cover distinct indices.");
  constexpr static int One = One_;
  constexpr static int Two = Two_;
  constexpr static int Flags = 0;
};

template<int One_, int Two_>
struct AntiSymmetry
{
  static_assert(One_ != Two_, "Symmetries must cover distinct indices.");
  constexpr static int One = One_;
  constexpr static int Two = Two_;
  constexpr static int Flags = NegationFlag;
};

template<int One_, int Two_>
struct