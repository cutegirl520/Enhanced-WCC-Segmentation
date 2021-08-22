// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HOMOGENEOUS_H
#define EIGEN_HOMOGENEOUS_H

namespace Eigen { 

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Homogeneous
  *
  * \brief Expression of one (or a set of) homogeneous vector(s)
  *
  * \param MatrixType the type of the object in which we are making homogeneous
  *
  * This class represents an ex