#include "cnn/expr.h"

#include <initializer_list>

#include "cnn/nodes.h"
#include "cnn/conv.h"

namespace cnn { namespace expr {

using std::vector;

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>& data) { return Expression(&g, g.add_input(d, data)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression const_parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_const_parameters(p)); }
Expression parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>* pindices) { return Expression