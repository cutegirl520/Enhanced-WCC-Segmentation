#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <vector>
#include "cnn/model.h"
#include "cnn/shadow-params.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m, real lam, real e0) :
    eta0(e0), eta(e0), eta_decay(), epoch(), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates(), model(m) {}
  virtual ~Trainer();

  virtual void update(real scale = 1.0) = 0;
  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  // if clipping is enabled and the gradient is too big, return the amo