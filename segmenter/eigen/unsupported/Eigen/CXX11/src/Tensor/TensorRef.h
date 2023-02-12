// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REF_H
#define EIGEN_CXX11_TENSOR_TENSOR_REF_H

namespace Eigen {

namespace internal {

template <typename Dimensions, typename Scalar>
class TensorLazyBaseEvaluator {
 public:
  TensorLazyBaseEvaluator() : m_refcount(0) { }
  virtual ~TensorLazyBaseEvaluator() { }

  EIGEN_DEVICE_FUNC virtual const Dimensions& dimensions() const = 0;
  EIGEN_DEVICE_FUNC virtual const Scalar* data() const = 0;

  EIGEN_DEVICE_FUNC virtual const Scalar coeff(DenseIndex index) const = 0;
  EIGEN_DEVICE_FUNC virtual Scalar& coeffRef(DenseIndex index) = 0;

  void incrRefCount() { ++m_refcount; }
  void decrRefCount() { --m_refcount; }
  int refCount() const { return m_refcount; }

 private:
  // No copy, no assigment;
  TensorLazyBaseEvaluator(const TensorLazyBaseEvaluator& other);
  TensorLazyBaseEvaluator& operator = (const TensorLazyBaseEvaluator& other);

  int m_refcount;
};


template <typename Dimensions, typename Expr, typename Device>
class TensorLazyEvaluatorReadOnly : public TensorLazyBaseEvaluator<Dimensions, typename TensorEvaluator<Expr, Device>::Scalar> {
 public:
  //  typedef typename TensorEvaluator<Expr, Device>::Dimensions Dimensions;
  typedef typename TensorEvaluator<Expr, Device>::Scalar Scalar;

  TensorLazyEvaluatorReadOnly(const Expr& expr, const Device& device) : m_impl(expr, device), m_dummy(Scalar(0)) {
    m_dims = m_impl.dimensions();
    m_impl.evalSubExprsIfNeeded(NULL);
  }
  virtual ~TensorLazyEvaluatorReadOnly() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC virtual const Dimensions& dimensions() const {
    return m_dims;
  }
  EIGEN_DEVICE_FUNC virtual const Scalar* data() const {
    return m_impl.data();
  }

  EIGEN_DEVICE_FUNC virtual const Scalar coeff(DenseIndex index) const {
    return m_impl.coeff(index);
  }
  EIGEN_DEVICE_FUNC virtual Scalar& coeffRef(DenseIndex /*index*/) {
    eigen_assert(false && "can't reference the coefficient of a rvalue");
    return m_dummy;
  };

 protected:
  TensorEvaluator<Expr, Device> m_impl;
  Dimensions m_dims;
  Scalar m_dummy;
};

template <typename Dimensions, typename Expr, typename Device>
class TensorLazyEvaluatorWritable : public TensorLazyEvaluatorReadOnly<Dimensions, Expr, Device> {
 public:
  typedef TensorLazyEvaluatorReadOnly<Dimensions, Expr, Device> Base;
  typedef typename Base::Scalar Scalar;

  TensorLazyEvaluatorWritable(const Expr& expr, const Device& device) : Base(expr, device) {
  }
  virtual ~TensorLazyEvaluatorWritable() {
  }

  EIGEN_DEVICE_FUNC virtual Scalar& coeffRef(DenseIndex index) {
    return this->m_impl.coeffRef(index);
  }
};

template <typename Dimensions, typename Expr, typename Device>
class TensorLazyEvaluator : public internal::conditional<bool(internal::is_lvalue<Expr>::value),
                            TensorLazyEvaluatorWritable<Dimensions, Expr, Device>,
                            TensorLazyEvaluatorReadOnly<Dimensions, const Expr, Device> >::type {
 public:
  typedef typename internal::conditional<bool(internal::is_lvalue<Expr>::value),
                  