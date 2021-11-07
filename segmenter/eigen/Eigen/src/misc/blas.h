#ifndef BLAS_H
#define BLAS_H

#ifdef __cplusplus
extern "C"
{
#endif

#define BLASFUNC(FUNC) FUNC##_

#ifdef __WIN64__
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

int    BLASFUNC(xerbla)(const char *, int *info, int);

float  BLASFUNC(sdot)  (int *, float  *, int *, float  *, int *);
float  BLASFUNC(sdsdot)(int *, float  *,        float  *, int *, float  *, int *);

double BLASFUNC(dsdot) (int *, float  *, int *, float  *, int *);
double BLASFUNC(ddot)  (int *, double *, int *, double *, int *);
double BLASFUNC(qdot)  (int *, double *, int *, double *, int *);

int  BLASFUNC(cdotuw)  (int *, float  *, int *, float  *, int *, float*);
int  BLASFUNC(cdotcw)  (int *, float  *, int *, float  *, int *, float*);
int  BLASFUNC(zdotuw)  (int *, double  *, int *, double  *, int *, double*);
int  BLASFUNC(zdotcw)  (int *, double  *, int *, double  *, int *, double*);

int    BLASFUNC(saxpy) (const int *, const float  *, const float  *, const int *, float  *, const int *);
int    BLASFUNC(daxpy) (const int *, const double *, const double *, const int *, double *, const int *);
int    BLASFUNC(qaxpy) (const int *, const double *, const double *, const int *, double *, const int *);
int    BLASFUNC(caxpy) (const int *, const float  *, const float  *, const int *, float  *, const int *);
int    BLASFUNC(zaxpy) (const int *, const double *, const double *, const int *, double *, const int *);
int    BLASFUNC(xaxpy) (const int *, const double *, const double *, const int *, double *, const int *);
int    BLASFUNC(caxpyc)(const int *, const float  *, const float  *, const int *, float  *, const int *);
int    BLASFUNC(zaxpyc)(const int *, const double *, const double *, const int *, double *, const int *);
int    BLASFUNC(xaxpyc)(const int *, const double *, const double *, const int *, double *, const int *);

int    BLASFUNC(scopy) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(dcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(qcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(ccopy) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(zcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(xcopy) (int *, double *, int *, double *, int *);

int    BLASFUNC(sswap) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(dswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(qswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(cswap) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(zswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(xswap) (int *, double *, int *, double *, int *);

float  BLASFUNC(sasum) (int *, float  *, int *);
float  BLASFUNC(scasum)(int *, float  *, int *);
double BLASFUNC(dasum) (int *, double *, int *);
double BLASFUNC(qasum) (int *, double *, int *);
double BLASFUNC(dzasum)(int *, double *, int *);
double BLASFUNC(qxasum)(int *, double *, int *);

int    BLASFUNC(isamax)(int *, float  *, int *);
int    BLASFUNC(idamax)(int *, double *, int *);
int    BLASFUNC(iqamax)(int *, double *, int *);
int    BLASFUNC(icamax)(int *, float  *, int *);
int    BLASFUNC(izamax)(int *, double *, int *);
int    BLASFUNC(ixamax)(int *, double *, int *);

int    BLASFUNC(ismax) (int *, float  *, int *);
int    BLASFUNC(idmax) (int *, double *, int *);
int    BLASFUNC(iqmax) (int *, double *, int *);
int    BLASFUNC(icmax) (int *, float  *, int *);
int    BLASFUNC(izmax) (int *, double *, int *);
int    BLASFUNC(ixmax) (int *, double *, int *);

int    BLASFUNC(isamin)(int *, float  *, int *);
int    BLASFUNC(idamin)(int *, double *, int *);
int    BLASFUNC(iqamin)(int *, double *, int *);
int    BLASFUNC(icamin)(int *, float  *, int *);
int    BLASFUNC(izamin)(int *, double *, int *);
int    BLASFUNC(ixamin)(int *, double *, int *);

int    BLASFUNC(ismin)(int *, float  *, int *);
int    BLASFUNC(idmin)(int *, double *, int *);
int    BLASFUNC(iqmin)(int *, double *, int *);
int    BLASFUNC(icmin)(int *, float  *, int *);
int    BLASFUNC(izmin)(int *, double *, int *);
int    BLASFUNC(ixmin)(int *, double *, int *);

float  BLASFUNC(samax) (int *, float  *, int *);
double BLASFUNC(damax) (int *, double *, int *);
double BLASFUNC(qamax) (int *, double *, int *);
float  BLASFUNC(scamax)(int *, float  *, int *);
double BLASFUNC(dzamax)(int *, double *, int *);
double BLASFUNC(qxamax)(int *, double *, int *);

float  BLASFUNC(samin) (int *, float  *, int *);
double BLASFUNC(damin) (int *, double *, int *);
double BLASFUNC(qamin) (int *, double *, int *);
float  BLASFUNC(scamin)(int *, float  *, int *);
double BLASFUNC(dz