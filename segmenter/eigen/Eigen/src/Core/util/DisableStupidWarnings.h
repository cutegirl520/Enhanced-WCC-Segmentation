#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

#ifdef _MSC_VER
  // 4100 - unreferenced formal parameter (occurred e.g. in aligned_allocator::destroy(pointer p))
  // 4101 - unreferenced local variable
  // 4127 - conditional expression is constant
  // 4181 - qualifier applied to reference type ignored
  // 4211 - nonstandard extension used : redefined extern to static
  // 4244 - 'argument' : conversion from 'type1' to 'type2', possible loss of data
  // 4273 - QtAlignedMalloc, inconsistent DLL linkage
  // 4324 - structure was padded due to declspec(align())
  // 4503 - decorated name length exceeded, name was truncated
  // 4512 - assignment operator could not be generated
  // 4522 - 'class' : multiple assignment operators specified
  // 4700 - uninitialized local variable 'xyz' used
  // 4717 - 'function' : recursive on all control paths, function will cause runtime stack overflow
  // 4800 - 'type' : forcing value to bool 'true' or 'false' (performance warning)
  #ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    #pragma warning( push )
  #