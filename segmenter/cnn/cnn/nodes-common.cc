#include "cnn/nodes.h"

#include <limits>
#include <cmath>
#include <sstream>

using namespace std;

namespace cnn {

inline bool LooksLikeVector(const Dim& d) {
  if (d.ndims() == 1) return true;
  if (d.ndims() > 1) {
    for (unsigned i = 1; i < d.ndims(); ++i)
      if (d[i] != 1) return false;
  }
  return true;
}

string SparsemaxLoss::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparsemax(" << arg_names[0] << ", q)";
  return s.str();
}

Dim SparsemaxLoss::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1 || !LooksLikeVector(xs[0])) {
    ostringstream s; s << "Bad input dimensions in SparsemaxLoss: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({1});
}

string Sparsemax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sparsemax(" << arg_names[0] << ")";
  return s.str();
}

Dim Sparsemax::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1 || !LooksLikeVector(xs[0])) {
    ostringstream s; s << "Bad input dimensions in Sparsemax: " << xs;
    throw std::invalid_argument(s.str());
  }
  return xs[0];
}

string MatrixInverse::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inverse(" << arg_names[0] << ")";
  return s.str();
}

Dim MatrixInverse::dim_forward(const vector<Dim>& xs) const {
  return xs[0];
}

Dim LogDet::dim_forward(const vector<Dim>& xs) const {
    if (xs[0].ndims() > 2 || (xs[0].rows() != xs[0].cols())) {
        cerr << "Bad arguments in LogDet: " << xs << endl;
        throw std::invalid_argument("invalid arguments to LogDet");
    }
    return Dim({1});
}

string LogDet::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "logdet(" << arg_names[0] << ")";
  return s.str();
}

string AddMv::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "addmv(" << arg_names[0] << ", " << arg_names[1] << ")";
  return s.str();
}

Dim AddMv::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 ||
      xs[0].ndims() > 2 ||
      xs[0].rows() != xs[1].rows() ||
      xs[1].ndims() != 1) {
    cerr << "Bad arguments in AddMv: " << xs << endl;
    throw std::invalid_argument("invalid arguments to AddMv");
  }
  return xs[0];
}

string SelectRows::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "select_rows(" << arg_names[0] << ", {rsize=" << prows->size() << "})";
  return s.str();
}

Dim SelectRows::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1 || xs[0].ndims() > 2) {
    cerr << "Bad arguments in SelectRows: " << xs << endl;
    throw std::invalid_argument("invalid arguments to SelectRows");
  }
  unsigned nrows = prows->size();
  if (xs[0].ndims() == 1) return Dim({nrows});
  return Dim({nrows, xs[0].cols()});
}

string SelectCols::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "select_cols(" << arg_names[0] << ", {csize=" << pcols->size() << "})";
  return s.str();
}

Dim SelectCols::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1 || xs[0].ndims() != 2) {
    cerr << "Bad arguments in SelectCols: " << xs << endl;
    throw std::invalid_argument("invalid arguments to SelectCols");
  }
  unsigned ncols = pcols->size();
  return Dim({xs[0].rows(), ncols});
}

string Min::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "min{" << arg_names[0] << ", " << arg_names[1] << "}";
  return s.str();
}

Dim Min::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 || xs[0] != xs[1]) {
    cerr << "Bad arguments in Min: " << xs << endl;
    throw std::invalid_argument("invalid arguments to Min");
  }
  return xs[0].bd >= xs[1].bd ? xs[0] : xs[1];
}

string Max::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "max{" << arg_names[0] << ", " << arg_names[1] << "}";
  return s.str();
}

Dim Max::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 || xs[0] != xs[1]) {
    cerr << "Bad arguments in Max: " << xs << endl;
    throw std::invalid_argument("invalid arguments to Max");
  }
  return xs[0].bd >= xs[1].bd ? xs[0] : xs[1];
}

string TraceOfProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "Tr(" << arg_names[0] << " * " << arg_names[1] << "^T)";
  return s.str();
}

Dim TraceOfProduct::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 || xs[0] != xs[1]) {
    cerr << "Bad arguments in TraceOfProduct: " << xs << endl;
    throw std::invalid_argument("invalid arguments to TraceOfProduct");
  }
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

string ConstScalarMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << alpha;
  return s.str();
}

Dim ConstScalarMultiply::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1) {
    cerr << "ConstScalarMultiply expects one argument: " << xs << endl;
    throw std::invalid_argument("ConstScalarMultiply expects one argument");
  }
  return xs[0];
}

string DotProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "^T . " << arg_names[1];
  return s.str();
}

Dim DotProduct::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 ||
      !LooksLikeVector(xs[0]) ||
      !LooksLikeVector(xs[1]) ||
      xs[0].rows() != xs[1].rows()) {
    cerr << "Bad arguments to DotProduct: " << xs << endl;
    throw std::invalid_argument("Bad arguments to DotProduct");
  }
  return Dim({1}, max(xs[0].bd, xs[1].bd));
}

string Transpose::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "^T";
  return s.str();
}

Dim Transpose::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1) {
    cerr << "Bad arguments to Transpose: " << xs << endl;
    throw std::invalid_argument("Bad arguments to Transpose");
  }
  return xs[0].transpose();
}

string Reshape::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "reshape(" << arg_names[0] << " --> " << to << ')';
  return s.str();
}

Dim Reshape::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  assert(xs[0].size() == to.size());
  return to;
}

string SumColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_cols(matrix=" << arg_names[0];
  if (arg_names.size() == 2) s << ", col_weighting=" << arg_names[1];
  s << ')';
  return s.str();
}

Dim SumColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1 || xs.size() == 2);
  int bd = (xs.size() == 1 ? xs[0].bd : max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows()}, bd);
}

string KMHNGram::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "kmh-ngram(" << arg_names[0] << ')';
  return s.str();
}

Dim KMHNGram::dim_forward(const vector<Dim>& xs) const {
  assert(xs[0].ndims() == 2);
  const unsigned new_cols = xs[0].cols() - n + 1;
  if (new_cols < 1) {
    ostringstream s; s << "Bad input dimensions in KMHNGram: " << xs;
    throw std::invalid_argument(s.str());
  }
  return Dim({xs[0][0], new_cols});
}

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dot(" << arg_names[0] << "," << arg_names[1] << ')';
  if (arg_names.size() == 3) s << " + " << arg_names[2];
  return s.str();
}

Dim InnerProduct3D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3)
    throw std::invalid_argument("Expected two or three arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      xs[1].ndims() != 1 ||
      xs[0].size(2) != xs[1].size(0)) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0), xs[0].size(1)}, max(xs[0].bd, xs[1].bd));
  if(xs.size() == 3) d.bd = max(d.bd, xs[2].bd);
  if (xs.size() == 3 && xs[2] != d) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

string InnerProduct3D_1D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dotdot(" << arg_names[0] << "," << arg_names[1] << "," << arg_names[2] << ')';
  if (arg_names.size() == 4) s << " + " << arg_names[3];
  return s.str();
}

Dim InnerProduct3D_1D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 3 && xs.size() != 4)
    throw std::invalid_argument("Expected three or four arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      xs[1].ndims() != 1 ||
      xs[2].ndims() != 1) {
    // TODO fix add check
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0)}, max(max(xs[0].bd, xs[1].bd), xs[2].bd));
  if(xs.size() == 4) d.bd = max(d.bd, xs[3].bd);
  if (xs.size() == 4 && xs[3] != d) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

string GaussianNoise::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + N(0," << stddev << ')';
  return s.str();
}

Dim GaussianNoise::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Dropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim Dropout::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string BlockDropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "block_dropout(" << arg_names[0] << ",dropout_probability=" << dropout_probability << ')';
  return s.str();
}

Dim BlockDropout::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string ConstantPlusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " + " << arg_names[0];
  return s.str();
}

Dim ConstantPlusX::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string ConstantMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " - " << arg_names[0];
  return s.str();
}

Dim ConstantMinusX::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string LogSumExp::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log(exp " << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + exp " << arg_names[i];
  s << ")";
  return s.str();
}

Dim LogSumExp::dim_forward(const vector<Dim>& xs) const {
  Dim d = xs[0].truncate();
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (d.single_batch() != xs[i].truncate().single_batch()) {
      ostringstream s; s << "Mismatched input dimensions in LogSumExp: " << xs;
      throw std::invalid_argument(s.str());
    }
    d.bd = max(xs[i].bd, d.bd);
  }
  return d;
}
string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

Dim Sum::dim_forward(const vector<Dim>& xs) const {
  Dim d = xs[0].truncate();
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (d.single_batch() != xs[i].truncate().single_batch()) {
      ostringstream s; s << "Mismatched input dimensions in Sum: " << xs;
      throw std::invalid_argument(s.str());
    }
    d.bd = max(xs[i].bd, d.bd);
  }
  return d;
}

string SumBatches::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_batches( " << arg_names[0] << " )";
  return s.str();
}

Dim SumBatches::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0].single_batch();
}

string Average::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "average(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << ", " << arg_names[i];
  s << ")";
  return s.str();
}

Dim Average::dim_forward(const vector<Dim>& xs) const {
  Dim d(xs[0]);
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (xs[0].single_batch() != xs[1].single_batch()) {
      ostringstream s; s << "Mismatched input dimensions in Average: " << xs;
      throw std::invalid_argument(s.str());
    }
    d.bd = max(xs[i].bd, d.bd);
  }
  return d;
}

string Sqrt::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sqrt(" << arg_names[0] << ')';
  return s.str();
}

Dim Sqrt::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Erf::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "erf(" << arg_names[0] << ')';
  return s.str();
}

Dim Erf::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Tanh::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Square::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "square(" << arg_names[0] << ')';
  return s.str();
}

Dim Square::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}
