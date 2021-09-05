// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INCOMPLETE_LUT_H
#define EIGEN_INCOMPLETE_LUT_H


namespace Eigen { 

namespace internal {
    
/** \internal
  * Compute a quick-sort split of a vector 
  * On output, the vector row is permuted such that its elements satisfy
  * abs(row(i)) >= abs(row(ncut)) if i<ncut
  * abs(row(i)) <= abs(row(ncut)) if i>ncut 
  * \param row The vector of values
  * \param ind The array of index for the elements in @p row
  * \param ncut  The number of largest elements to keep
  **/ 
template <typename VectorV, t