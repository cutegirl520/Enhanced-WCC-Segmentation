
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSORSYMMETRY_TEMPLATEGROUPTHEORY_H
#define EIGEN_CXX11_TENSORSYMMETRY_TEMPLATEGROUPTHEORY_H

namespace Eigen {

namespace internal {

namespace group_theory {

/** \internal
  * \file CXX11/Tensor/util/TemplateGroupTheory.h
  * This file contains C++ templates that implement group theory algorithms.
  *
  * The algorithms allow for a compile-time analysis of finite groups.
  *
  * Currently only Dimino's algorithm is implemented, which returns a list
  * of all elements in a group given a set of (possibly redundant) generators.
  * (One could also do that with the so-called orbital algorithm, but that
  * is much more expensive and usually has no advantages.)
  */

/**********************************************************************
 *                "Ok kid, here is where it gets complicated."
 *                         - Amelia Pond in the "Doctor Who" episode
 *                           "The Big Bang"
 *
 * Dimino's algorithm
 * ==================
 *
 * The following is Dimino's algorithm in sequential form:
 *
 * Input: identity element, list of generators, equality check,
 *        multiplication operation
 * Output: list of group elements
 *
 * 1. add identity element
 * 2. remove identities from list of generators
 * 3. add all powers of first generator that aren't the
 *    identity element
 * 4. go through all remaining generators:
 *        a. if generator is already in the list of elements
 *                -> do nothing
 *        b. otherwise
 *                i.   remember current # of elements
 *                     (i.e. the size of the current subgroup)
 *                ii.  add all current elements (which includes
 *                     the identity) each multiplied from right
 *                     with the current generator to the group
 *                iii. add all remaining cosets that are generated
 *                     by products of the new generator with itself
 *                     and all other generators seen so far
 *
 * In functional form, this is implemented as a long set of recursive
 * templates that have a complicated relationship.
 *
 * The main interface for Dimino's algorithm is the template
 * enumerate_group_elements. All lists are implemented as variadic