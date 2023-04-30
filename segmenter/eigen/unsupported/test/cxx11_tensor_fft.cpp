// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Jianwei Cui <thucjw@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_fft_2D_golden() {
  Tensor<float, 2, DataLayout> input(2, 3);
  input(0, 0) = 1;
  input(0, 1) = 2;
  input(0, 2) = 3;
  input(1, 0) = 4;
  input(1, 1) = 5;
  input(1, 2) = 6;

  array<ptrdiff_t, 2> fft;
  fft[0] = 0;
  fft[1] = 1;

  Tensor<std::complex<float>, 2, DataLayout> output = input.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);

  std::complex<float> output_golden[6]; // in ColMajor order
  output_golden[0] = std::complex<float>(21, 0);
  output_golden[1] = std::complex<float>(-9, 0);
  output_golden[2] = std::complex<float>(-3, 1.73205);
  output_golden[3] = std::complex<float>( 0, 0);
  output_golden[4] = std::complex<float>(-3, -1.73205);
  output_golden[5] = std::complex<float>(0 ,0);

  std::complex<float> c_offset = std::complex<float>(1.0, 1.0);

  if (DataLayout == ColMajor) {
    VERIFY_IS_APPROX(output(0) + c_offset, output_golden[0] + c_offset);
    VERIFY_IS_APPROX(output(1) + c_offset, output_golden[1] + c_offset);
    VERIFY_IS_APPROX(output(2) + c_offset, output_golden[2] + c_offset);
    VERIFY_IS_APPROX(output(3) + c_offset, output_golden[3] + c_offset);
    VERIFY_IS_APPROX(output(4) + c_offset, output_golden[4] + c_offset);
    VERIFY_IS_APPROX(output(5) + c_offset, output_golden[5] + c_offset);
  }
  else {
    VERIFY_IS_APPROX(output(0)+ c_offset, output_golden[0]+ c_offset);
    VERIFY_IS_APPROX(output(1)+ c_offset, output_golden[2]+ c_offset);
    VERIFY_IS_APPROX(output(2)+ c_offset, output_golden[4]+ c_offset);
    VERIFY_IS_APPROX(output(3)+ c_offset, output_golden[1]+ c_offset);
    VERIFY_IS_APPROX(output(4)+ c_offset, output_golden[3]+ c_offset);
    VERIFY_IS_APPROX(output(5)+ c_offset, output_golden[5]+ c_offset);
  }
}

static void test_fft_complex_input_golden() {
  Tensor<std::complex<float>, 1, ColMajor> input(5);
  input(0) = std::complex<float>(1, 1);
  input(1) = std::complex<float>(2, 2);
  input(2) = std::complex<float>(3, 3);
  input(3) = std::complex<float>(4, 4);
  input(4) = std::complex<float>(5, 5);

  array<ptrdiff_t, 1> fft;
  fft[0] = 0;

  Tensor<std::complex<float>, 1, ColMajor> forward_output_both_parts = input.fft<BothParts, FFT_FORWARD>(fft);
  Tensor<std::complex<float>, 1, ColMajor> reverse_output_both_parts = input.fft<BothParts, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor> forward_output_real_part = input.fft<RealPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor> reverse_output_real_part = input.fft<RealPart, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor> forward_output_imag_part = input.fft<ImagPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor> reverse_output_imag_part = input.fft<ImagPart, FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(forward_output_both_parts.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_both_parts.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_real_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_real_part.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_imag_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_imag_part.dimension(0), input.dimension(0));

  std::complex<float> forward_golden_result[5];
  std::complex<float> reverse_golden_result[5];

  forward_golden_result[0] = std::complex<float>(15.000000000000000,+15.000000000000000);
  forward_golden_result[1] = std::complex<float>(-5.940954801177935, +0.940954801177934);
  forward_golden_result[2] = std::complex<float>(-3.312299240582266, -1.687700759417735);
  forward_golden_result[3] = std::complex<float>(-1.687700759417735, -3.312299240582266);
  forward_golden_result[4] = std::complex<float>( 0.940954801177934, -5.940954801177935);

  reverse_golden_result[0] = std::complex<float>( 3.000000000000000, + 3.000000000000000);
  reverse_golden_result[1] = std::complex<float>( 0.188190960235587, - 1.188190960235587);
  reverse_golden_result[2] = std::complex<float>(-0.337540151883547, - 0.662459848116453);
  reverse_golden_result[3] = std::complex<float>(-0.662459848116453, - 0.337540151883547);
  reverse_golden_result[4] = std::complex<float>(-1.1881909602