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
      xs[0].nd