
# Find LAPACK library
#
# This module finds an installed library that implements the LAPACK
# linear-algebra interface (see http://www.netlib.org/lapack/).
# The approach follows mostly that taken for the autoconf macro file, acx_lapack.m4
# (distributed at http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).
#
# This module sets the following variables:
#  LAPACK_FOUND - set to true if a library implementing the LAPACK interface
#    is found
#  LAPACK_INCLUDE_DIR - Directories containing the LAPACK header files
#  LAPACK_DEFINITIONS - Compilation options to use LAPACK
#  LAPACK_LINKER_FLAGS - Linker flags to use LAPACK (excluding -l
#    and -L).
#  LAPACK_LIBRARIES_DIR - Directories containing the LAPACK libraries.
#     May be null if LAPACK_LIBRARIES contains libraries name using full path.
#  LAPACK_LIBRARIES - List of libraries to link against LAPACK interface.
#     May be null if the compiler supports auto-link (e.g. VC++).
#  LAPACK_USE_FILE - The name of the cmake module to include to compile
#     applications or libraries using LAPACK.
#
# This module was modified by CGAL team:
# - find libraries for a C++ compiler, instead of Fortran
# - added LAPACK_INCLUDE_DIR, LAPACK_DEFINITIONS and LAPACK_LIBRARIES_DIR
# - removed LAPACK95_LIBRARIES


include(CheckFunctionExists)

# This macro checks for the existence of the combination of fortran libraries
# given by _list.  If the combination is found, this macro checks (using the