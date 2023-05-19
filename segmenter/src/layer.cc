#include "layer.h"

StaticInputLayer::StaticInputLayer(cnn::Model* model,
  unsigned size_word, unsigned dim_word,
  unsigned size_postag, unsigned dim_postag,
  unsigned size_pretrained_word, unsigned dim_pretrained_word,
  unsigned dim_output,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained) :
  p_w(nullptr), p_p(nullptr), p_t(nullptr),
  p_ib(nullptr), p_w2l(nullptr), p_p2l(nullptr), p_t2l(nullptr),
  use_word(true), use_postag(true), use_pretrained_word(true) {

  p_ib = model->add_parameters({ dim_output, 1 });
  if (dim_word == 0) {
    std::cerr << "Word dim should be greater than 0." << std::endl;
    std::cerr << "Fine-tuned word embedding is inactivated." << std::endl;
    use_word = false;
  } else {
    p_w = model->add_lookup_parameters(size_word, { dim_word, 1 });
    p_w2l = model->add_parameters({ dim_output, dim_word });
  }

  if (dim_postag == 0) {
    std::cerr << "Postag dim should be greater than 0." << std::endl;
    std::cerr << "Fine-tuned postag embedding is inactivated." << std::endl;
    use_postag = false;
  } else {
    p_p = model->add_lookup_parameters(size_postag, { dim_postag, 1 });
    p_p2l = model->add_parameters({ dim_output, dim_postag });
  }

  if (dim_pretrained_word == 0) {
    std::cerr << "Pretrained word embedding dim should be greater than 0." << std::endl;
    std::cerr << "Pretrained word embedding is inactivated." << std::endl;
    use_pretrained_word = false;
  } else {
    p_t = model->add_lookup_parameters(size_pretrained_word, { dim_pretrained_word, 1 });
    for (auto it : pretrained) { p_t->Initialize(it.first, it.second); }
    p_t2l = model->add_parameters({ dim_output, dim_pretrained_word });
  }
}


cnn::expr::Expression StaticInputLayer::add_input(cnn::ComputationGraph* hg,
  unsigned wid, unsigned pid, unsigned pre_wid) {
  cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
  if (use_word && wid > 0) {
    cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
    cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
    expr = cnn::expr::affine_transform({ expr, w2l, w });
  }
  if (use_postag && pid > 0) {
    cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
    cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
    expr = cnn::expr::affine_transform({ expr, p2l, p });
  }
  if (use_pretrained_word && pre_wid > 0) {
    cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
    cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
    expr = cnn::expr::affine_transform({ expr, t2l, t });
  }
  return cnn::expr::rectify(expr);
}


DynamicInputLayer::DynamicInputLayer(cnn::Model* model,
  unsigned size_word, unsigned dim_word,
  unsigned size_postag, unsigned dim_postag,
  unsigned size_pretrained_word, unsigned dim_pretrained_word,
  unsigned size_label, unsigned dim_label,
  unsigned dim_output,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained):
  StaticInputLayer(model, size_word, dim_word, size_postag, dim_postag,
  size_pretrained_word, dim_pretrained_word, dim_output, pretrained),
  p_l(nullptr), p_l2l(nullptr), use_label(true) {
  if (dim_label == 0) {
    std::cerr << "Label embedding dim should be greater than 0." << std::endl;
    std::cerr << "Label embedding is inactivated." << std::endl;
    use_label = false;
  } else {
    p_l = model->add_lookup_parameters(size_label, { dim_label, 1 });
    p_l2l = model->add_parameters({ dim_output, dim_label });
  }
}


cnn::expr::Expression DynamicInputLayer::add_input2(cnn::ComputationGraph* hg,
  unsigned wid, unsigned pid, unsigned pre_wid, unsigned lid) {
  cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
  if (use_word && wid > 0) {
    cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
    cnn::