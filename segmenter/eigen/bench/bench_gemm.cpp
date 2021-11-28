
// g++-4.4 bench_gemm.cpp -I .. -O2 -DNDEBUG -lrt -fopenmp && OMP_NUM_THREADS=2  ./a.out
// icpc bench_gemm.cpp -I .. -O3 -DNDEBUG -lrt -openmp  && OMP_NUM_THREADS=2  ./a.out

// Compilation options:
// 
// -DSCALAR=std::complex<double>
// -DSCALARA=double or -DSCALARB=double
// -DHAVE_BLAS
// -DDECOUPLED
//

#include <iostream>
#include <Eigen/Core>
#include <bench/BenchTimer.h>

using namespace std;
using namespace Eigen;

#ifndef SCALAR
// #define SCALAR std::complex<float>
#define SCALAR float
#endif

#ifndef SCALARA
#define SCALARA SCALAR
#endif

#ifndef SCALARB
#define SCALARB SCALAR
#endif

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef Matr