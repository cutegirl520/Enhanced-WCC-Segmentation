// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

namespace Eigen { 

/** \ingroup SparseCore_Module
  *
  * \class SparseMatrix
  *
  * \brief A versatible sparse matrix representation
  *
  * This class implements a more versatile variants of the common \em compressed row/column storage format.
  * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
  * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
  * space inbetween the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
  * can be done with limited memory reallocation and copies.
  *
  * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
  * compatible with many library.
  *
  * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
  *
  * \tparam _Scalar the scalar type, i.e. the type of the coefficients
  * \tparam _Options Union of bit flags controlling the storage scheme. Currently the only possibility
  *                 is ColMajor or RowMajor. The default is 0 which means column-major.
  * \tparam _Index the type of the indices. It has to be a \b signed type (e.g., short, int, std::ptrdiff_t). Default is \c int.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
  */

namespace internal {
template<typename _Scalar, int _Options, typename _Index>
struct traits<SparseMatrix<_Scalar, _Options, _Index> >
{
  typedef _Scalar Scalar;
  typedef _Index StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = _Options | NestByRefBit | LvalueBit | CompressedAccessBit,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

template<typename _Scalar, int _Options, typename _Index, int DiagIndex>
struct traits<Diagonal<SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
{
  typedef SparseMatrix<_Scalar, _Options, _Index> MatrixType;
  typedef typename ref_selector<MatrixType>::type MatrixTypeNested;
  typedef typename remove_reference<MatrixTypeNested>::type _MatrixTypeNested;

  typedef _Scalar Scalar;
  typedef Dense StorageKind;
  typedef _Index StorageIndex;
  typedef MatrixXpr XprKind;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = 1,
    Flags = LvalueBit
  };
};

template<typename _Scalar, int _Options, typename _Index, int DiagIndex>
struct traits<Diagonal<const SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
 : public traits<Diagonal<SparseMatrix<_Scalar, _Options, _Index>, DiagIndex> >
{
  enum {
    Flags = 0
  };
};

} // end namespace internal

template<typename _Scalar, int _Options, typename _Index>
class SparseMatrix
  : public SparseCompressedBase<SparseMatrix<_Scalar, _Options, _Index> >
{
    typedef SparseCompressedBase<SparseMatrix> Base;
    using Base::convert_index;
  public:
    using Base::isCompressed;
    using Base::nonZeros;
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
    using Base::operator+=;
    using Base::operator-=;

    typedef MappedSparseMatrix<Scalar,Flags> Map;
    typedef Diagonal<SparseMatrix> DiagonalReturnType;
    typedef Diagonal<const SparseMatrix> ConstDiagonalReturnType;
    typedef typename Base::InnerIterator InnerIterator;
    typedef typename Base::ReverseInnerIterator ReverseInnerIterator;
    

    using Base::IsRowMajor;
    typedef internal::CompressedStorage<Scalar,StorageIndex> Storage;
    enum {
      Options = _Options
    };

    typedef typename Base::IndexVector IndexVector;
    typedef typename Base::ScalarVector ScalarVector;
  protected:
    typedef SparseMatrix<Scalar,(Flags&~RowMajorBit)|(IsRowMajor?RowMajorBit:0)> TransposedSparseMatrix;

    Index m_outerSize;
    Index m_innerSize;
    StorageIndex* m_outerIndex;
    StorageIndex* m_innerNonZeros;     // optional, if null then the data is compressed
    Storage m_data;

  public:
    
    /** \returns the number of rows of the matrix */
    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    /** \returns the number of columns of the matrix */
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

    /** \returns the number of rows (resp. columns) of the matrix if the storage order column major (resp. row major) */
    inline Index innerSize() const { return m_innerSize; }
    /** \returns the number of columns (resp. rows) of the matrix if the storage order column major (resp. row major) */
    inline Index outerSize() const { return m_outerSize; }
    
    /** \returns a const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline const Scalar* valuePtr() const { return m_data.valuePtr(); }
    /** \returns a non-const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline Scalar* valuePtr() { return m_data.valuePtr(); }

    /** \returns a const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline const StorageIndex* innerIndexPtr() const { return m_data.indexPtr(); }
    /** \returns a non-const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline StorageIndex* innerIndexPtr() { return m_data.indexPtr(); }

    /** \returns a const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline const StorageIndex* outerIndexPtr() const { return m_outerIndex; }
    /** \returns a non-const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline StorageIndex* outerIndexPtr() { return m_outerIndex; }

    /** \returns a const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline const StorageIndex* innerNonZeroPtr() const { return m_innerNonZeros; }
    /** \returns a non-const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline StorageIndex* innerNonZeroPtr() { return m_innerNonZeros; }

    /** \internal */
    inline Storage& data() { return m_data; }
    /** \internal */
    inline const Storage& data() const { return m_data; }

    /** \returns the value of the matrix at position \a i, \a j
      * This function returns Scalar(0) if the element is an explicit \em zero */
    inline Scalar coeff(Index row, Index col) const
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      return m_data.atInRange(m_outerIndex[outer], end, StorageIndex(inner));
    }

    /** \returns a non-const reference to the value of the matrix at position \a i, \a j
      *
      * If the element does not exist then it is inserted via the insert(Index,Index) function
      * which itself turns the matrix into a non compressed form if that was not the case.
      *
      * This is a O(log(nnz_j)) operation (binary search) plus the cost of insert(Index,Index)
      * function if the element does not already exist.
      */
    inline Scalar& coeffRef(Index row, Index col)
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      eigen_assert(end>=start && "you probably called coeffRef on a non finalized matrix");
      if(end<=start)
        return insert(row,col);
      const Index p = m_data.searchLowerIndex(start,end-1,StorageIndex(inner));
      if((p<end) && (m_data.index(p)==inner))
        return m_data.value(p);
      else
        return insert(row,col);
    }

    /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.
      * The non zero coefficient must \b not already exist.
      *
      * If the matrix \c *this is in compressed mode, then \c *this is turned into uncompressed
      * mode while reserving room for 2 x this->innerSize() non zeros if reserve(Index) has not been called earlier.
      * In this case, the insertion procedure is optimized for a \e sequential insertion mode where elements are assumed to be
      * inserted by increasing outer-indices.
      * 
      * If that's not the case, then it is strongly recommended to either use a triplet-list to assemble the matrix, or to first
      * call reserve(const SizesType &) to reserve the appropriate number of non-zero elements per inner vector.
      *
      * Assuming memory has been appropriately reserved, this function performs a sorted insertion in O(1)
      * if the elements of each inner vector are inserted in increasing inner index order, and in O(nnz_j) for a random insertion.
      *
      */
    Scalar& insert(Index row, Index col);

  public:

    /** Removes all non zeros but keep allocated memory
      *
      * This function does not free the currently allocated memory. To release as much as memory as possible,
      * call \code mat.data().squeeze(); \endcode after resizing it.
      * 
      * \sa resize(Index,Index), data()
      */
    inline void setZero()
    {
      m_data.clear();
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(StorageIndex));
      if(m_innerNonZeros)
        memset(m_innerNonZeros, 0, (m_outerSize)*sizeof(StorageIndex));
    }

    /** Preallocates \a reserveSize non zeros.
      *
      * Precondition: the matrix must be in compressed mode. */
    inline void reserve(Index reserveSize)
    {
      eigen_assert(isCompressed() && "This function does not make sense in non compressed mode.");
      m_data.reserve(reserveSize);
    }
    
    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** Preallocates \a reserveSize[\c j] non zeros for each column (resp. row) \c j.
      *
      * This function turns the matrix in non-compressed mode.
      * 
      * The type \c SizesType must expose the following interface:
        \code
        typedef value_type;
        const value_type& operator[](i) const;
        \endcode
      * for \c i in the [0,this->outerSize()[ range.
      * Typical choices include std::vector<int>, Eigen::VectorXi, Eigen::VectorXi::Constant, etc.
      */
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes);
    #else
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::value_type& enableif =
    #if (!EIGEN_COMP_MSVC) || (EIGEN_COMP_MSVC>=1500) // MSVC 2005 fails to compile with this typename
        typename
    #endif
        SizesType::value_type())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    #endif // EIGEN_PARSED_BY_DOXYGEN
  protected:
    template<class SizesType>
    inline void reserveInnerVectors(const SizesType& reserveSizes)
    {
      if(isCompressed())
      {
        Index totalReserveSize = 0;
        // turn the matrix into non-compressed mode
        m_innerNonZeros = static_cast<StorageIndex*>(std::malloc(m_outerSize * sizeof(StorageIndex)));
        if (!m_innerNonZeros) internal::throw_std_bad_alloc();
        
        // temporarily use m_innerSizes to hold the new starting points.
        StorageIndex* newOuterIndex = m_innerNonZeros;
        
        StorageIndex count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          count += reserveSizes[j] + (m_outerIndex[j+1]-m_outerIndex[j]);
          totalReserveSize += reserveSizes[j];
        }
        m_data.reserve(totalReserveSize);
        StorageIndex previousOuterIndex = m_outerIndex[m_outerSize];
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          StorageIndex innerNNZ = previousOuterIndex - m_outerIndex[j];
          for(Index i=innerNNZ-1; i>=0; --i)
          {
            m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
            m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
          }
          previousOuterIndex = m_outerIndex[j];
          m_outerIndex[j] = newOuterIndex[j];
          m_innerNonZeros[j] = innerNNZ;
        }
        m_outerIndex[m_outerSize] = m_outerIndex[m_outerSize-1] + m_innerNonZeros[m_outerSize-1] + reserveSizes[m_outerSize-1];
        
        m_data.resize(m_outerIndex[m_outerSize]);
      }
      else
      {
        StorageIndex* newOuterIndex = static_cast<StorageIndex*>(std::malloc((m_outerSize+1)*sizeof(StorageIndex)));
        if (!newOuterIndex) internal::throw_std_bad_alloc();
        
        StorageIndex count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          StorageIndex alreadyReserved = (m_outerIndex[j+1]-m_outerIndex[j]) - m_innerNonZeros[j];
          StorageIndex toReserve = std::max<StorageIndex>(reserveSizes[j], alreadyReserved);
          count += toReserve + m_innerNonZeros[j];
        }
        newOuterIndex[m_outerSize] = count;
        
        m_data.resize(count);
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          Index offset = newOuterIndex[j] - m_outerIndex[j];
          if(offset>0)
          {
            StorageIndex innerNNZ = m_innerNonZeros[j];
            for(Index i=innerNNZ-1; i>=0; --i)
            {
              m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
              m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
            }
          }
        }
        
        std::swap(m_outerIndex, newOuterIndex);
        std::free(newOuterIndex);
      }
      
    }
  public:

    //--- low level purely coherent filling ---

    /** \internal
      * \returns a reference to the non zero coefficient at position \a row, \a col assuming that:
      * - the nonzero does not already exist
      * - the new coefficient is the last one according to the storage order
      *
      * Before filling a given inner vector you must call the statVec(Index) function.
      *
      * After an insertion session, you should call the finalize() function.
      *
      * \sa insert, insertBackByOuterInner, startVec */
    inline Scalar& insertBack(Index row, Index col)
    {
      return insertBackByOuterInner(IsRowMajor?row:col, IsRowMajor?col:row);
    }

    /** \internal
      * \sa insertBack, startVec */
    inline Scalar& insertBackByOuterInner(Index outer, Index inner)
    {
      eigen_assert(Index(m_outerIndex[outer+1]) == m_data.size() && "Invalid ordered insertion (invalid outer index)");
      eigen_assert( (m_outerIndex[outer+1]-m_outerIndex[outer]==0 || m_data.index(m_data.size()-1)<inner) && "Invalid ordered insertion (invalid inner index)");
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \warning use it only if you know what you are doing */
    inline Scalar& insertBackByOuterInnerUnordered(Index outer, Index inner)
    {
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \sa insertBack, insertBackByOuterInner */
    inline void startVec(Index outer)
    {
      eigen_assert(m_outerIndex[outer]==Index(m_data.size()) && "You must call startVec for each inner vector sequentially");
      eigen_assert(m_outerIndex[outer+1]==0 && "You must call startVec for each inner vector sequentially");
      m_outerIndex[outer+1] = m_outerIndex[outer];
    }

    /** \internal
      * Must be called after inserting a set of non zero entries using the low level compressed API.
      */
    inline void finalize()
    {
      if(isCompressed())
      {
        StorageIndex size = internal::convert_index<StorageIndex>(m_data.size());
        Index i = m_outerSize;
        // find the last filled column
        while (i>=0 && m_outerIndex[i]==0)
          --i;
        ++i;
        while (i<=m_outerSize)
        {
          m_outerIndex[i] = size;
          ++i;
        }
      }
    }

    //---

    template<typename InputIterators>
    void setFromTriplets(const InputIterators& begin, const InputIterators& end);

    template<typename InputIterators,typename DupFunctor>
    void setFromTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

    void sumupDuplicates() { collapseDuplicates(internal::scalar_sum_op<Scalar,Scalar>()); }

    template<typename DupFunctor>
    void collapseDuplicates(DupFunctor dup_func = DupFunctor());

    //---
    
    /** \internal
      * same as insert(Index,Index) except that the indices are given relative to the storage order */
    Scalar& insertByOuterInner(Index j, Index i)
    {
      return insert(IsRowMajor ? j : i, IsRowMajor ? i : j);
    }

    /** Turns the matrix into the \em compressed format.
      */
    void makeCompressed()
    {
      if(isCompressed())
        return;
      
      eigen_internal_assert(m_outerIndex!=0 && m_outerSize>0);
      
      Index oldStart = m_outerIndex[1];
      m_outerIndex[1] = m_innerNonZeros[0];
      for(Index j=1; j<m_outerSize; ++j)
      {
        Index nextOldStart = m_outerIndex[j+1];
        Index offset = oldStart - m_outerIndex[j];
        if(offset>0)
        {
          for(Index k=0; k<m_innerNonZeros[j]; ++k)
          {
            m_data.index(m_outerIndex[j]+k) = m_data.index(oldStart+k);
            m_data.value(m_outerIndex[j]+k) = m_data.value(oldStart+k);
          }
        }
        m_outerIndex[j+1] = m_outerIndex[j] + m_innerNonZeros[j];
        oldStart = nextOldStart;
      }
      std::free(m_innerNonZeros);
      m_innerNonZeros = 0;
      m_data.resize(m_outerIndex[m_outerSize]);
      m_data.squeeze();
    }

    /** Turns the matrix into the uncompressed mode */
    void uncompress()
    {
      if(m_innerNonZeros != 0)
        return; 
      m_innerNonZeros = static_cast<StorageIndex*>(std::malloc(m_outerSize * sizeof(StorageIndex)));
      for (Index i = 0; i < m_outerSize; i++)
      {
        m_innerNonZeros[i] = m_outerIndex[i+1] - m_outerIndex[i]; 
      }
    }
    
    /** Suppresses all nonzeros which are \b much \b smaller \b than \a reference under the tolerence \a epsilon */
    void prune(const Scalar& reference, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision())
    {
      prune(default_prunning_func(reference,epsilon));
    }
    
    /** Turns the matrix into compressed format, and suppresses all nonzeros which do not satisfy the predicate \a keep.
      * The functor type \a KeepFunc must implement the following function:
      * \code
      * bool operator() (const Index& row, const Index& col, const Scalar& value) const;
      * \endcode
      * \sa prune(Scalar,RealScalar)
      */
    template<typename KeepFunc>
    void prune(const KeepFunc& keep = KeepFunc())
    {
      // TODO optimize the uncompressed mode to avoid moving and allocating the data twice
      makeCompressed();

      StorageIndex k = 0;
      for(Index j=0; j<m_outerSize; ++j)
      {
        Index previousStart = m_outerIndex[j];
        m_outerIndex[j] = k;
        Index end = m_outerIndex[j+1];
        for(Index i=previousStart; i<end; ++i)
        {
          if(keep(IsRowMajor?j:m_data.index(i), IsRowMajor?m_data.index(i):j, m_data.value(i)))
          {
            m_data.value(k) = m_data.value(i);
            m_data.index(k) = m_data.index(i);
            ++k;
          }
        }
      }
      m_outerIndex[m_outerSize] = k;
      m_data.resize(k,0);
    }

    /** Resizes the matrix to a \a rows x \a cols matrix leaving old values untouched.
      *
      * If the sizes of the matrix are decreased, then the matrix is turned to \b uncompressed-mode
      * and the storage of the out of bounds coefficients is kept and reserved.
      * Call makeCompressed() to pack the entries and squeeze extra memory.
      *
      * \sa reserve(), setZero(), makeCompressed()
      */
    void conservativeResize(Index rows, Index cols) 
    {
      // No change
      if (this->rows() == rows && this->cols() == cols) return;
      
      // If one dimension is null, then there is nothing to be preserved
      if(rows==0 || cols==0) return resize(rows,cols);

      Index innerChange = IsRowMajor ? cols - this->cols() : rows - this->rows();
      Index outerChange = IsRowMajor ? rows - this->rows() : cols - this->cols();
      StorageIndex newInnerSize = convert_index(IsRowMajor ? cols : rows);

      // Deals with inner non zeros
      if (m_innerNonZeros)
      {
        // Resize m_innerNonZeros
        StorageIndex *newInnerNonZeros = static_cast<StorageIndex*>(std::realloc(m_innerNonZeros, (m_outerSize + outerChange) * sizeof(StorageIndex)));
        if (!newInnerNonZeros) internal::throw_std_bad_alloc();
        m_innerNonZeros = newInnerNonZeros;
        
        for(Index i=m_outerSize; i<m_outerSize+outerChange; i++)          
          m_innerNonZeros[i] = 0;
      } 
      else if (innerChange < 0) 
      {
        // Inner size decreased: allocate a new m_innerNonZeros
        m_innerNonZeros = static_cast<StorageIndex*>(std::malloc((m_outerSize+outerChange+1) * sizeof(StorageIndex)));
        if (!m_innerNonZeros) internal::throw_std_bad_alloc();
        for(Index i = 0; i < m_outerSize; i++)
          m_innerNonZeros[i] = m_outerIndex[i+1] - m_outerIndex[i];
      }
      
      // Change the m_innerNonZeros in case of a decrease of inner size
      if (m_innerNonZeros && innerChange < 0)
      {
        for(Index i = 0; i < m_outerSize + (std::min)(outerChange, Index(0)); i++)
        {
          StorageIndex &n = m_innerNonZeros[i];
          StorageIndex start = m_outerIndex[i];
          while (n > 0 && m_data.index(start+n-1) >= newInnerSize) --n; 
        }
      }
      
      m_innerSize = newInnerSize;

      // Re-allocate outer index structure if necessary
      if (outerChange == 0)
        return;
          
      StorageIndex *newOuterIndex = static_cast<StorageIndex*>(std::realloc(m_outerIndex, (m_outerSize + outerChange + 1) * sizeof(StorageIndex)));
      if (!newOuterIndex) internal::throw_std_bad_alloc();
      m_outerIndex = newOuterIndex;
      if (outerChange > 0)
      {
        StorageIndex last = m_outerSize == 0 ? 0 : m_outerIndex[m_outerSize];
        for(Index i=m_outerSize; i<m_outerSize+outerChange+1; i++)          
          m_outerIndex[i] = last; 
      }
      m_outerSize += outerChange;
    }
    
    /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero.
      * 
      * This function does not free the currently allocated memory. To release as much as memory as possible,
      * call \code mat.data().squeeze(); \endcode after resizing it.
      * 
      * \sa reserve(), setZero()
      */
    void resize(Index rows, Index cols)
    {
      const Index outerSize = IsRowMajor ? rows : cols;
      m_innerSize = IsRowMajor ? cols : rows;
      m_data.clear();
      if (m_outerSize != outerSize || m_outerSize==0)
      {
        std::free(m_outerIndex);
        m_outerIndex = static_cast<StorageIndex*>(std::malloc((outerSize + 1) * sizeof(StorageIndex)));
        if (!m_outerIndex) internal::throw_std_bad_alloc();
        
        m_outerSize = outerSize;
      }
      if(m_innerNonZeros)
      {
        std::free(m_innerNonZeros);
        m_innerNonZeros = 0;
      }
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(StorageIndex));
    }

    /** \internal
      * Resize the nonzero vector to \a size */
    void resizeNonZeros(Index size)
    {
      m_data.resize(size);
    }

    /** \returns a const expression of the diagonal coefficients. */
    const ConstDiagonalReturnType diagonal() const { return ConstDiagonalReturnType(*this); }
    
    /** \returns a read-write expression of the diagonal coefficients.
      * \warning If the diagonal entries are written, then all diagonal
      * entries \b must already exist, otherwise an assertion will be raised.
      */
    DiagonalReturnType diagonal() { return DiagonalReturnType(*this); }

    /** Default constructor yielding an empty \c 0 \c x \c 0 matrix */
    inline SparseMatrix()
      : m_outerSize(-1), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      resize(0, 0);
    }

    /** Constructs a \a rows \c x \a cols empty matrix */
    inline SparseMatrix(Index rows, Index cols)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      resize(rows, cols);
    }

    /** Constructs a sparse matrix from the sparse expression \a other */
    template<typename OtherDerived>
    inline SparseMatrix(const SparseMatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      check_template_parameters();
      const bool needToTranspose = (Flags & RowMajorBit) != (internal::evaluator<OtherDerived>::Flags & RowMajorBit);
      if (needToTranspose)
        *this = other.derived();
      else
      {
        #ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
          EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
        #endif
        internal::call_assignment_no_alias(*this, other.derived());
      }
    }
    
    /** Constructs a sparse matrix from the sparse selfadjoint view \a other */
    template<typename OtherDerived, unsigned int UpLo>
    inline SparseMatrix(const SparseSelfAdjointView<OtherDerived, UpLo>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      Base::operator=(other);
    }

    /** Copy constructor (it performs a deep copy) */
    inline SparseMatrix(const SparseMatrix& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      *this = other.derived();
    }

    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    SparseMatrix(const ReturnByValue<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      initAssignment(other);
      other.evalTo(*this);
    }
    
    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    explicit SparseMatrix(const DiagonalBase<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      check_template_parameters();
      *this = other.derived();
    }

    /** Swaps the content of two sparse matrices of the same type.
      * This is a fast operation that simply swaps the underlying pointers and parameters. */
    inline void swap(SparseMatrix& other)
    {
      //EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
      std::swap(m_outerIndex, other.m_outerIndex);
      std::swap(m_innerSize, other.m_innerSize);
      std::swap(m_outerSize, other.m_outerSize);
      std::swap(m_innerNonZeros, other.m_innerNonZeros);
      m_data