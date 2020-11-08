#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <boost/serialization/strong_typedef.hpp>

#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/tensor.h"
#include "cnn/model.h"
#include "cnn/devices.h"

// Computation graph where nodes represent forward and backward intermediate
// values, and edges represent functions of multiple values. To represent the
// fact that a function may have multiple arguments, edges have a single head
// and 0, 1, 2, or more tails. (Constants, inputs, and parameters are
// represented as functions of 0 parameters.)
// Example: given the function z = f(x, y), z, x, and y are nodes, and there
// is an edge representing f with which points to the z node (i.e., its head),
// and x and y are the tails of the edge.

namespace cnn {

extern AlignedMemoryPool* fxs;
extern AlignedMemoryPool* dEdfs;
extern AlignedMemoryPool* ps;
extern float* kSCALAR_MINUSONE;
extern float* kSCALAR_ONE;
extern float* kSCALAR_ZERO;

// devices provide information about GPUs and CPUs
// these include any API information that is required to make calls
// to the GPU as well as the memory pools for the device
// Device is not copyable, so you can use the pointer to uniquely
// identify the device
//extern std::vector<Device*> devices; // [0] is always the CPU
extern Device* default_device; // where parameters go by default

class ExecutionEngine;
struct ParameterNodeBase;
struct Node;
namespace expr { struct Expression; }

BOOST_STRONG_TYPEDEF(unsigned, VariableIndex)
inline void swap(VariableIndex& i1, VariableIndex& i2) {
  VariableIndex t = i1;
  i1 = i2;
  i2 = t;
}

struct ComputationGraph {
  ComputationGraph();
  ~ComputationGraph();

  // INPUTS
  // the computational network will pull inputs in from the user's data
  // structures and make them available to the computation
  VariableIndex add_input(real s);  // add scalar
  VariableIndex add_input(const real* ps);  // add pointer to scalar
  VariableIndex add_input(const Dim& d, cons