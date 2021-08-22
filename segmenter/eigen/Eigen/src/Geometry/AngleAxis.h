// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ANGLEAXIS_H
#define EIGEN_ANGLEAXIS_H

namespace Eigen { 

/** \geometry_module \ingroup Geometry_Module
  *
  * \class AngleAxis
  *
  * \brief Represents a 3D rotation as a rotation angle around an arbitrary 3D axis
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  *
  * \warning When setting up an AngleAxis object, the axis vector \b must \b be \b normalized.
  *
  * The following two typedefs are provided for convenience:
  * \li \c AngleAxisf for \c float
  * \li \c AngleAxisd for \c double
  *
  * Combined with Matrix