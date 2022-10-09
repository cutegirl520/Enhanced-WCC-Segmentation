// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_H

namespace Eigen {

/** \class Tensor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor class.
  *
  * The %Tensor class is the work-horse for all \em dense tensors within Eigen.
  *
  * The %Tensor class encompasses only dynamic-size objects so far.
  *
  * The first two template parameters are required:
  * \tparam Scalar_ \anchor tensor_tparam_scalar Numeric type, e.g. float, double, int or std::complex<float>.
  *                 User defined scalar types are supported as well (see \ref user_defined_scalars "here").
  * \tparam NumIndices_ Number of indices (i.e. rank of the tensor)
  *
  * The remaining template parameters are optional -- in most cases you don't have to worry about them.
  * \tparam Options_ \anchor tensor_tparam_options A combination of either \b #RowMajor or \b #ColMajor, and of either
  *                 \b #AutoAlign or \b #DontAlign.
  *                 The former controls \ref TopicStorageOrders "storage order", and defaults to column-major. The latter controls alignment, which is required
  *                 for vectorization. It defaults to aligning tensors. Note that tensors currently do not support any operations that profit from vectorization.
  *                 Support for such operations (i.e. adding two tensors etc.) is planned.
  *
  * You can access elements of tensors using normal subscripting:
  *
  * \code
  * Eigen::Tensor<double, 4> t(10, 10, 10, 10);
  * t(0, 1, 2, 3) = 42.0;
  * \endcode
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_TENSOR_PLUGIN.
  *
  * <i><b>Some notes:</b></i>
  *
  * <dl>
  * <dt><b>Relation to other parts of Eigen:</b></dt>
  * <dd>The midterm developement goal for this class is to have a similar hierarchy as Eigen uses for matrices, so that
  * taking blocks or using tensors in expressions is easily possible, including an interface with the vector/matrix code
  * by providing .asMatrix() and .asVector() (or similar) methods for rank 2 and 1 tensors. However, currently, the %Tensor
  * class does not provide any of these features and is only available as a stand-alone class that just allows for
  * coefficient access. Also, when fixed-size tensors are implemented, the number of template arguments is likely to
  * change dramatically.</dd>
  * </dl>
  *
  * \ref TopicStorageOrders
  */

template<typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
class Tensor : public TensorBase<Tensor<Scalar_, NumIndices_, Options_, IndexType_> >
{
  public:
    typedef Tensor<Scalar_, NumIndices_, Options_, IndexType_> Self;
    typedef TensorBase<Tensor<Scalar_, NumIndices_, Options_, IndexType_> > Base;
    typedef typename Eigen::internal::nested<Self>::type Nested;
    typedef typename internal::traits<Self>::StorageKind StorageKind;
    typedef typename internal::traits<Self>::Index Index;
    typedef Scalar_ Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename Base::CoeffReturnType CoeffReturnType;

    enum {
      IsAligned = bool(EIGEN_MAX_ALIGN_BYTES>0) & !(Options_&DontAlign),
      Layout = Options_ & RowMajor ? RowMajor : ColMajor,
      CoordAccess = true,
      RawAccess = true
    };

    static const int Options = Options_;
    static const int NumIndices = NumIndices_;
    typedef DSizes<Index, NumIndices_> Dimensions;

  protected:
    TensorStorage<