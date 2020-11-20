#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "CNNEdges"
#include <boost/test/unit_test.hpp>

#include <vector>

#include "cnn/tests/test_utils.h"
#include "cnn/tensor.h"
#include "cnn/edges.h"
#include "cnn/c2w.h"

using namespace std;
using namespace cnn;

BOOST_GLOBAL_FIXTURE(TestTensorSetup)

Dim size(const Tensor& t) {
  if (t.cols() > 1)
    return Dim(t.rows(), t.cols());
  return Dim(t.rows());
}

BOOST_AUTO_TEST_CASE(ESqrL2)
{
  auto U = Ccm({2}, {4,5});
  auto V = Ccm({2}, {1,1});
  cerr << str(U) << endl;
  SquaredEuclideanDistance e;
  vector<const Tensor*> xs = {&U, &V};
  Tensor W = e.forward(xs); 
  cerr << "Norm^2:" << str(W) << endl;
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(W,0),25., eps);
  Tensor dEdf = Ccm({1}, {1});
  Tensor d1 = e.backward(xs, W, dEdf, 0);
  Tensor d2 = e.backward(xs, W, dEdf, 1);
  cerr << d1 << endl;
  cerr << d2 << endl;
  BOOST_CHECK_CLOSE(t(d1,0), 6., eps);
  BOOST_CHECK_CLOSE(t(d1,1), 8., eps);
  BOOST_CHECK_CLOSE(t(d2,0), -6., eps);
  BOOST_CHECK_CLOSE(t(d2,1), -8., eps);
}

BOOST_AUTO_TEST_CASE(EMatrixMultiply) {
  Tensor U = Ccm({2,3}, {1,2,3,4,5,6});
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tensor*> xs = {&U, &V};
  Tensor W = mm.forward(xs);
  BOOST_REQUIRE_EQUAL(Dim({2,2}),size(W));
  double eps = 1e-5;
  BOOST_CHECK_CLOSE(t(W,0,0), 76., eps);
  BOOST_CHECK_CLOSE(t(W,1,0), 100., eps);
  BOOST_CHECK_CLOSE(t(W,0,1), 103., eps);
  BOOST_CHECK_CLOSE(t(W,1,1), 136., eps);
  Tensor dEdf = Ccm({2,2}, {-1,0.5,1,2});
  Tensor dEdx0 = mm.backward(xs, W, dEdf, 0);
  cerr << str(dEdx0) << endl;
  BOOST_CHECK_CLOSE(t(dEdx0,0,0),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,0,1),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,0,2),3.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,0),23.5,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,1),26.,eps);
  BOOST_CHECK_CLOSE(t(dEdx0,1,2),28.5,eps);
  Tensor dEdx1 = mm.backward(xs, W, dEdf, 1);
  cerr << str(dEdx1) << endl;
  BOOST_CHECK_CLOSE(t(dEdx1,0,0),0.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,1,0),-1.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,2,0),-2.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,0,1),5.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,1,1),11.,eps);
  BOOST_CHECK_CLOSE(t(dEdx1,2,1),17.,eps);
}

BOOST_AUTO_TEST_CASE(EColumnConcat)
{
  Tensor u1 = Ccm({2}, {1, 2});
  Tensor u2 = Ccm({2}, {3, 4});
  Tensor u3 = Ccm({2}, {5, 6});
  cerr << u1 << endl;
  cerr << u2 << endl;
  cerr << u3 << endl;
  vector<const Tensor*> xs = {&u1, &u2, &u3};
  ConcatenateColumns cc;
  Tensor U = cc.forward(xs);
  cerr << U << endl;
  Tensor V = Ccm({3,2}, {7,8,9,10,11,12});
  MatrixMultiply mm;
  vector<const Tens