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
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>
#include <memory>

#include <Eigen/Core>

using namespace std;

const int default_precision = 4;

// see --only-cubic-sizes
bool only_cubic_sizes = false;

// see --dump-tables
bool dump_tables = false;

uint8_t log2_pot(size_t x) {
  size_t l = 0;
  while (x >>= 1) l++;
  return l;
}

uint16_t compact_size_triple(size_t k, size_t m, size_t n)
{
  return (log2_pot(k) << 8) | (log2_pot(m) << 4) | log2_pot(n);
}

// just a helper to store a triple of K,M,N sizes for matrix product
struct size_triple_t
{
  uint16_t k, m, n;
  size_triple_t() : k(0), m(0), n(0) {}
  size_triple_t(size_t _k, size_t _m, size_t _n) : k(_k), m(_m), n(_n) {}
  size_triple_t(const size_triple_t& o) : k(o.k), m(o.m), n(o.n) {}
  size_triple_t(uint16_t compact)
  {
    k = 1 << ((compact & 0xf00) >> 8);
    m = 1 << ((compact & 0x0f0) >> 4);
    n = 1 << ((compact & 0x00f) >> 0);
  }
  bool is_cubic() const { return k == m && m == n; }
};

ostream& operator<<(ostream& s, const size_triple_t& t)
{
  return s << "(" << t.k << ", " << t.m << ", " << t.n << ")";
}

struct inputfile_entry_t
{
  uint16_t product_size;
  uint16_t pot_block_size;
  size_triple_t nonpot_block_size;
  float gflops;
};

struct inputfile_t
{
  enum class type_t {
    unknown,
    all_pot_sizes,
    default_sizes
  };

  string filename;
  vector<inputfile_entry_t> entries;
  type_t type;

  inputfile_t(const string& fname)
    : filename(fname)
    , type(type_t::unknown)
  {
    ifstream stream(filename);
    if (!stream.is_open()) {
      cerr << "couldn't open input file: " << filename << endl;
      exit(1);
    }
    string line;
    while (getline(stream, line)) {
      if (line.empty()) continue;
      if (line.find("BEGIN MEASUREMENTS ALL POT SIZES") == 0) {
        if (type != type_t::unknown) {
          cerr << "Input file " << filename << " contains redundant BEGIN MEASUREMENTS lines";
          exit(1);
        }
        type = type_t::all_pot_sizes;
        continue;
      }
      if (line.find("BEGIN MEASUREMENTS DEFAULT SIZES") == 0) {
        if (type != type_t::unknown) {
          cerr << "Input file " << filename << " contains redundant BEGIN MEASUREMENTS lines";
          exit(1);
        }
        type = type_t::default_sizes;
        continue;
      }
      

      if (type == type_t::unknown) {
        continue;
      }
      switch(type) {
        case type_t::all_pot_sizes: {
          unsigned int product_size, block_size;
          float gflops;
          int sscanf_result =
            sscanf(line.c_str(), "%x %x %f",
                   &product_size,
                   &block_size,
                   &gflops);
          if (3 != sscanf_result ||
              !product_size ||
              product_size > 0xfff ||
              !block_size ||
              block_size > 0xfff ||
              !isfinite(gflops))
          {
            cerr << "ill-formed input file: " << filename << endl;
            cerr << "offending line:" << endl << line << endl;
            exit(1);
          }
          if (only_cubic_sizes && !size_triple_t(product_size).is_cubic()) {
            continue;
          }
          inputfile_entry_t entry;
          entry.product_size = uint16_t(product_size);
          entry.pot_block_size = uint16_t(block_size);
          entry.gflops = gflops;
          entries.push_back(entry);
          break;
        }
        case type_t::default_sizes: {
          unsigned int product_size;
          float gflops;
          int bk, bm, bn;
          int sscanf_result =
            sscanf(line.c_str(), "%x default(%d, %d, %d) %f",
                   &product_size,
                   &bk, &bm, &bn,
                   &gflops);
          if (5 != sscanf_result ||
              !product_size ||
              product_size > 0xfff ||
              !isfinite(gflops))
          {
            cerr << "ill-formed input file: " << filename << endl;
            cerr << "offending line:" << endl << line << endl;
            exit(1);
          }
          if (only_cubic_sizes && !size_triple_t(product_size).is_cubic()) {
            continue;
          }
          inputfile_entry_t entry;
          entry.product_size = uint16_t(product_size);
          entry.pot_block_size = 0;
          entry.nonpot_block_size = size_triple_t(bk, bm, bn);
          entry.gflops = gflops;
          entries.push_back(entry);
          break;
        }
        
        default:
          break;
      }
    }
    stream.close();
    if (type == type_t::unknown) {
      cerr << "Unrecognized input file " << filename << endl;
      exit(1);
    }
    if (entries.empty()) {
      cerr << "didn't find any measurements in input file: " << filename << endl;
      exit(1);
    }
  }
};

struct preprocessed_inputfile_entry_t
{
  uint16_t product_size;
  uint16_t block_size;

  float efficiency;
};

bool lower_efficiency(const preprocessed_inputfile_entry_t& e1, const preprocessed_inputfile_entry_t& e2)
{
  return e1.efficiency < e2.efficiency;
}

struct preprocessed_inputfile_t
{
  string filename;
  vector<preprocessed_inputfile_entry_t> entries;

  preprocessed_inputfile_t(const inputfile_t& inputfile)
    : filename(inputfile.filename)
  {
    if (inputfile.type != inputfile_t::type_t::all_pot_sizes) {
      abort();
    }
    auto it = inputfile.entries.begin();
    auto it_first_with_given_product_size = it;
    while (it != inputfile.entries.end()) {
      ++it;
      if (it == inputfile.entries.end() ||
        it->product_size != it_first_with_given