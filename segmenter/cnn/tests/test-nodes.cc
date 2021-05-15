#include <cnn/cnn.h>
#include <cnn/expr.h>
#include <cnn/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace cnn;
using namespace cnn::expr;
using namespace std;


struct NodeTest {
  NodeTest() {
    // set up some dummy arguments to cnn
    for (auto x : {"NodeTest", "--cnn-mem", "10"}) {
      av.push_back(strdup(x));
    }
    char **argv = &av[0];
    int argc = av.size();
    cnn::Initialize(argc, argv);
    ones3_vals = {1.f,1.f,1.f};
    first_one_vals = {1.f,0.f,0.f};
    ones2_vals = {1.f,1.f};
    batch_vals = {1.f,2.f,3.f,4.f,5.f,6.f};
    // Create parameters
    std::vector<float> param1_vals = {1.1f,-2.2f,3.3f};
    std::vector<float> param2_vals = {2.2f,3.4f,-1.2f};
    std::vector<float> param3_vals = {1.1f,2.2f,3.3f};
    std::vector<float> param_scalar1_vals = {2.2f};
    std::vector<float> param_scalar2_vals = {1.1f};
    param1 = mod.add_parameters({3});
    TensorTools::SetElements(param1->values,param1_vals);
    param2 = mod.add_parameters({3});
    TensorTools::SetElements(param2->values,param2_vals);
    param3 = mod.add_parameters({3});
    TensorTools::SetElements(param3->values,param3_vals);
    param_scalar1 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar1->values,param_scalar1_vals);
    param_scalar2 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar2->values,param_scalar2_vals);
  }
  ~NodeTest() {
    for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if(vec.size()) oss << vec[0];
    for(size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  std::vector<float> ones3_vals, ones2_vals, first_one_vals, batch_vals;
  std::vector<char*> av;
  cnn::Model mod;
  cnn::Parameters *param1, *param2, *param3, *param_scalar1, *param_scalar2;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(node_test, NodeTest);


// Expression operator-(const Expression& x);
BOOST_AUTO_TEST_CASE( negate_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = -x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( add_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( addscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1+2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalaradd_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0+x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( subtract_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalarsubtract_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0-x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( subtractscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1-2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1*transpose(x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  ones3 * y * transpose(ones3);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_batch_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = x1*transpose(x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  sum_batches(ones3 * y * transpose(ones3));
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = affine_transform({x1, x2, scalar});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  ones3 * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = affine_transform({x1, x2, scalar});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  sum_batches(ones3 * y);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch2_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = input(cg, Dim({1,3},2), batch_vals);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = affine_transform({x1, scalar, transpose(x2) });
  Expression ones3 = input(cg, {3,1}, ones3_vals);
  sum_batches(y * ones3);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, float y);
BOOST_AUTO_TEST_CASE( multiplyscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1*2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// inline Expression operator*(float y, const Expression& x) { return x * y; }
BOOST_AUTO_TEST_CASE( scalarmultiply_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0*x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }
BOOST_AUTO_TEST_CASE( dividescalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1/2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cdiv(x1, x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = colwise_add(x1 * transpose(x2), x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  ones3 * y * transpose(ones3);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y);
// TODO

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
// TODO

// Expression sqrt(const Expression& x);
BOOST_AUTO_TEST_CASE( sqrt_gradient ) {
  cnn::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = sqrt(x3);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression erf(const Expression& x);
BOOST_AUTO_TEST_CASE( erf_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = erf(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression tanh(const Expression& x);
BOOST_AUTO_TEST_CASE( tanh_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = tanh(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression exp(const Expression& x);
BOOST_AUTO_TEST_CASE( exp_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = exp(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression square(const Expression& x);
BOOST_AUTO_TEST_CASE( square_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = square(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression cube(const Expression& x);
BOOST_AUTO_TEST_CASE( cube_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cube(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression lgamma(const Expression& x);
BOOST_AUTO_TEST_CASE( lgamma_gradient ) {
  cnn::ComputationGraph cg;
  Expression x2 = parameter(cg, param2);
  Expression y = lgamma(x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression log(const Expression& x);
BOOST_AUTO_TEST_CASE( log_gradient ) {
  cnn::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = log(x3);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression logistic(const Expression& x);
BOOST_AUTO_TEST_CASE( logistic_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = logistic(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression rectify(const Expression& x);
BOOST_AU