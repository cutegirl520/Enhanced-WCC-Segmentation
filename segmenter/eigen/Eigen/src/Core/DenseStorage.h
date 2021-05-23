// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2010-2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

#ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
  #define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN EIGEN_DENSE_STORAGE_CTOR_PLUGIN;
#else
  #define EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
#endif

namespace Eigen {

namespace internal {

struct constructor_without_unaligned_array_assert {};

template<typename T, int Size>
EIGEN_DEVICE_FUNC
void check_static_allocation_size()
{
  // if EIGEN_STACK_ALLOCATION_LIMIT is defined to 0, then no limit
  #if EIGEN_STACK_ALLOCATION_LIMIT
  EIGEN_STATIC_ASSERT(Size * sizeof(T) <= EIGEN_STACK_ALLOCATION_LIMIT, OBJECT_ALLOCATED_ON_STACK_IS_TOO_BIG);
  #endif
}

/** \internal
  * Static array. If the MatrixOrArrayOptions require auto-alignment, the array will be automatically aligned:
  * to 16 bytes boundary if the total size is a multiple of 16 bytes.
  */
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions&DontAlign) ? 0
                        : compute_default_alignment<T,Size>::value >
struct plain_array
{
  T array[Size];

  EIGEN_DEVICE_FUNC
  plain_array()
  { 
    check_static_allocation_size<T,Size>();
  }

  EIGEN_DEVICE_FUNC
  plain_array(constructor_without_unaligned_array_assert)
  { 
    check_static_allocation_size<T,Size>();
  }
};

#if defined(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask)
#elif EIGEN_GNUC_AT_LEAST(4,7) 
  // GCC 4.7 is too aggressive in its optimizations and remove the alignement test based on the fact the array is declared to be aligned.
  // See this bug report: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=53900
  // Hiding the origin of the array pointer behind a function argument seems to do the trick even if the function is inlined:
  template<typename PtrType>
  EIGEN_ALWAYS_INLINE PtrType eigen_unaligned_array_assert_workaround_gcc47(PtrType array) { return array; }
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask) \
    eigen_assert((internal::UIntPtr(eigen_unaligned_array_assert_workaround_gcc47(array)) & (sizemask)) == 0 \
              && "this assertion is explained here: " \
              "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html" \
              " **** READ THIS WEB PAGE !!! ****");
#else
  #define EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(sizemask) \
    eigen_assert((internal::UIntPtr(array) & (sizemask)) == 0 \
              && "this assertion is explained here: " \
              "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html" \
              " **** READ THIS WEB PAGE !!! ****");
#endif

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 8>
{
  EIGEN_ALIGN_TO_BOUNDARY(8) T array[Size];

  EIGEN_DEVICE_FUNC
  plain_array() 
  {
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(7);
    check_static_allocation_size<T,Size>();
  }

  EIGEN_DEVICE_FUNC
  plain_array(constructor_without_unaligned_array_assert) 
  { 
    check_static_allocation_size<T,Size>();
  }
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 16>
{
  EIGEN_ALIGN_TO_BOUNDARY(16) T array[Size];

  EIGEN_DEVICE_FUNC
  plain_array() 
  { 
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(15);
    check_static_allocation_size<T,Size>();
  }

  EIGEN_DEVICE_FUNC
  plain_array(constructor_without_unaligned_array_assert) 
  { 
    check_static_allocation_size<T,Size>();
  }
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 32>
{
  EIGEN_ALIGN_TO_BOUNDARY(32) T array[Size];

  EIGEN_DEVICE_FUNC
  plain_array() 
  {
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(31);
    check_static_allocation_size<T,Size>();
  }

  EIGEN_DEVICE_FUNC
  plain_array(constructor_without_unaligned_array_assert) 
  { 
    check_static_allocation_size<T,Size>();
  }
};

template <typename T, int Size, int MatrixOrArrayOptions>
struct plain_array<T, Size, MatrixOrArrayOptions, 64>
{
  EIGEN_ALIGN_TO_BOUNDARY(64) T array[Size];

  EIGEN_DEVICE_FUNC
  plain_array() 
  { 
    EIGEN_MAKE_UNALIGNED_ARRAY_ASSERT(63);
    check_static_allocation_size<T,Size>();
  }

  EIGEN_DEVICE_FUNC
  plain_array(constructor_without_unaligned_array_assert) 
  { 
    check_static_allocation_size<T,Size>();
  }
};

template <typename T, int MatrixOrArrayOptions, int Alignment>
struct plain_array<T, 0, MatrixOrArrayOptions, Alignment>
{
  T array[1];
  EIGEN_DEVICE_FUNC plain_array() {}
  EIGEN_DEVICE_FUNC plain_array(constructor_without_unaligned_array_assert) {}
};

} // end namespace internal

/** \internal
  *
  * \class DenseStorage
  * \ingroup Core_Module
  *
  * \brief Stores the data of a matrix
  *
  * This class stores the data of fixed-size, dynamic-size or mixed matrices
  * in a way as compact as possible.
  *
  * \sa Matrix
  */
template<typename T, int Size, int _Rows, int _Cols, int _Options> class DenseStorage;

// purely fixed-size matrix
template<typename T, int Size, int _Rows, int _Cols, int _Options> class DenseStorage
{
    internal::plain_array<T,Size,_Options> m_data;
  public:
    EIGEN_DEVICE_FUNC DenseStorage() {}
    EIGEN_DEVICE_FUNC
    explicit DenseStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(internal::constructor_without_unaligned_array_assert()) {}
    EIGEN_DEVICE_FUNC 
    DenseStorage(const DenseStorage& other) : m_data(other.m_data) {}
    EIGEN_DEVICE_FUNC 
    DenseStorage& operator=(const DenseStorage& other)
    { 
      if (this != &other) m_data = other.m_data;
      return *this; 
    }
    EIGEN_DEVICE_FUNC DenseStorage(Index size, Index rows, Index cols) {
      EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      eigen_internal_assert(size==rows*cols && rows==_Rows && cols==_Cols);
      EIGEN_UNUSED_VARIABLE(size);
      EIGEN_UNUSED_VARIABLE(rows);
      EIGEN_UNUSED_VARIABLE(cols);
    }
    EIGEN_DEVICE_FUNC void swap(DenseStorage& other) { std::swap(m_data,other.m_data); }
    EIGEN_DEVICE_FUNC static Index rows(void) {return _Rows;}
    EIGEN_DEVICE_FUNC static Index cols(void) {return _Cols;}
    EIGEN_DEVICE_FUNC void conservativeResize(Index,Index,Index) {}
    EIGEN_DEVICE_FUNC void resize(Index,Index,Index) {}
    EIGEN_DEVICE_FUNC const T *data() const { return m_data.array; }
    EIGEN_DEVICE_FUNC T *data() { return m_data.array; }
};

// null matrix
template<typename T, int _Rows, int _Cols, int _Option