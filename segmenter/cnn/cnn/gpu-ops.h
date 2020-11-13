#ifndef CNN_GPU_OPS_H
#define CNN_GPU_OPS_H

namespace cnn {
namespace gpu {

void vpairwise_rank_loss(int n, float margin, const float* xgood, const float* xbad, float* y);
void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const float* fx, const float* dEdf, float* dEdx);
void vcwise_product(int n, const float* x0, const float* x1, float* y);
void vcwise_product_backward(int n, const float* dEdy, const float* x_other, float* dEdx);
void vconstant_minusx(int n, float c, const float* x, float* y);
void vnegate(int n, const float* x, float* y);
void vnegate_backward(int n, const float* dEdf, float* dEdx);
void vrelu(int n, const float* x, float* y);
void vrelu_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vtanh(int n, const float* x, float* y);
void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vlog(int n, const float* x, float* y);
void vlog_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void vlogistic(int n, const float* x, float* y);
void vlogistic_backward(int n, const float* fx, const float* dEdf, float* dEdx);
void l2_norm_reduc