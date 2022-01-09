
project(EigenBlas CXX)

include("../cmake/language_support.cmake")

workaround_9220(Fortran EIGEN_Fortran_COMPILER_WORKS)

if(EIGEN_Fortran_COMPILER_WORKS)
  enable_language(Fortran OPTIONAL)
  if(NOT CMAKE_Fortran_COMPILER)
    set(EIGEN_Fortran_COMPILER_WORKS OFF)
  endif()
endif()

add_custom_target(blas)

set(EigenBlas_SRCS  single.cpp double.cpp complex_single.cpp complex_double.cpp xerbla.cpp
                    f2c/srotm.c   f2c/srotmg.c  f2c/drotm.c f2c/drotmg.c
                    f2c/lsame.c   f2c/dspmv.c   f2c/ssbmv.c f2c/chbmv.c
                    f2c/sspmv.c   f2c/zhbmv.c   f2c/chpmv.c f2c/dsbmv.c
                    f2c/zhpmv.c   f2c/dtbmv.c   f2c/stbmv.c f2c/ctbmv.c
                    f2c/ztbmv.c   f2c/d_cnjg.c  f2c/r_cnjg.c
   )

if (EIGEN_Fortran_COMPILER_WORKS)
  set(EigenBlas_SRCS ${EigenBlas_SRCS} fortran/complexdots.f)
else()
  set(EigenBlas_SRCS ${EigenBlas_SRCS} f2c/complexdots.c)
