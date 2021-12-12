//=============================================================================
// File      : utilities.h
// Created   : mar jun 19 13:18:14 CEST 2001
// Author    : Antoine YESSAYAN, Paul RASCLE, EDF
// Project   : SALOME
// Copyright : EDF 2001
// $Header$
//=============================================================================

/* ---  Definition macros file to print information if _DEBUG_ is defined --- */

# ifndef UTILITIES_H
# define UTILITIES_H

# include <stdlib.h>
//# include <iostream> ok for gcc3.01
# include <iostream>

/* ---  INFOS is always defined (without _DEBUG_): to be used for warnings, with release version --- */

# define HEREWEARE cout<<flush ; cerr << __FILE__ << " ["