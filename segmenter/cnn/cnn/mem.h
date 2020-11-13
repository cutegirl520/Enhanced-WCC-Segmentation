#ifndef CNN_MEM_H
#define CNN_MEM_H

#include <vector>

namespace cnn {

// allocates memory from the device (CPU, GPU)
// only used to create the memory pools
// creates alignment appropriate for that device
struct MemAllocator {
  explic