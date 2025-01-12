
#include "crf.h"
#include "cnn/expr.h"


CRFBuilder::CRFBuilder(cnn::Model *model, boost::program_options::variables_map conf, Corpus corpus,
                             const std::unordered_map<unsigned int, std::vector<float>> &unigram_pretrained,
                             const std::unordered_map<unsigned int, std::vector<float>> &bigram_pretrained,
                             const std::unordered_map<unsigned int, std::vector<float>> &ounigram_pretrained,
                             const std::unordered_map<unsigned int, std::vector<float>> &obigram_pretrained) :
  unigram_pretrained(unigram_pretrained), bigram_pretrained(bigram_pretrained),
  ounigram_pretrained(ounigram_pretrained), obigram_pretrained(obigram_pretrained),
  for_lstm(conf["layers"].as<unsigned>(), conf["lstm_input_dim"].as<unsigned>(), conf["lstm_hidden_dim"].as<unsigned>(), model),
  rev_lstm(conf["layers"].as<unsigned>(), conf["lstm_input_dim"].as<unsigned>(), conf["lstm_hidden_dim"].as<unsigned>(), model) {

  dropout_rate = conf["dropout"].as<float>();
  use_train = (conf["use_train"].as<unsigned>() == 1);
  is_finetune = (conf["finetune"].as<unsigned>() == 1);
  is_ofinetune = (conf["ofinetune"].as<unsigned>() == 1);
  UNK_IDX = 0;

  BOOST_ASSERT_MSG(use_train or unigram_pretrained.size() > 0 or ounigram_pretrained.size() > 0, "No input features");

  unsigned LAYERS = conf["layers"].as<unsigned>();
  unsigned LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  unsigned LSTM_HIDDEN_DIM = conf["lstm_hidden_dim"].as<unsigned>();
  unsigned UNIGRAM_DIM = conf["unigram_dim"].as<unsigned>();
  unsigned BIGRAM_DIM = conf["bigram_dim"].as<unsigned>();
  unsigned N_UNIGRAMS = corpus.max_unigram;
  unsigned N_BIGRAMS = corpus.max_bigram;
  unsigned N_LABELS = static_cast<int>(corpus.id2label.size());
  unsigned LABEL_DIM = conf["label_dim"].as<unsigned>();
  unsigned HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();

  n_labels = N_LABELS;
  p_u = model->add_lookup_parameters(N_UNIGRAMS, {UNIGRAM_DIM});
  p_b = model->add_lookup_parameters(N_BIGRAMS, {BIGRAM_DIM});

  // initialize pretrained in domain embedding
  if (unigram_pretrained.size() > 0) {
    p_pu = model->add_lookup_parameters(N_UNIGRAMS, {UNIGRAM_DIM});
    for(auto &it : unigram_pretrained){ p_pu->Initialize(it.first, it.second); }
    p_pu2l = model->add_parameters({LSTM_INPUT_DIM, UNIGRAM_DIM});
  }
  else {
    p_pu = nullptr;
    p_pu2l = nullptr;
  }
  if (bigram_pretrained.size() > 0) {
    p_pb = model->add_lookup_parameters(N_BIGRAMS, {BIGRAM_DIM});
    for(auto &it : bigram_pretrained){ p_pb->Initialize(it.first, it.second); }
    p_pb2l = model->add_parameters({LSTM_INPUT_DIM, BIGRAM_DIM});
  }
  else {
    p_pb = nullptr;
    p_pb2l = nullptr;
  }

  // initialize pretrained out-of-domain embedding
  if (ounigram_pretrained.size() > 0) {
    p_pou = model->add_lookup_parameters(N_UNIGRAMS, {UNIGRAM_DIM});
    for(auto &it : ounigram_pretrained){ p_pou->Initialize(it.first, it.second); }
    p_pou2l = model->add_parameters({LSTM_INPUT_DIM, UNIGRAM_DIM});
  }
  else {
    p_pou = nullptr;
    p_pou2l = nullptr;
  }
  if (obigram_pretrained.size() > 0) {
    p_pob = model->add_lookup_parameters(N_BIGRAMS, {BIGRAM_DIM});
    for(auto &it : obigram_pretrained){ p_pob->Initialize(it.first, it.second); }
    p_pob2l = model->add_parameters({LSTM_INPUT_DIM, BIGRAM_DIM});
  }
  else {
    p_pob = nullptr;
    p_pob2l = nullptr;
  }

  p_u2l = model->add_parameters({LSTM_INPUT_DIM, UNIGRAM_DIM});
  p_b2l = model->add_parameters({LSTM_INPUT_DIM, BIGRAM_DIM});
  p_lb = model->add_parameters({LSTM_INPUT_DIM});

  //bi_hidden to hidden
  p_bi2h = model->add_parameters({HIDDEN_DIM, 2*LSTM_HIDDEN_DIM});
  p_bif2h = model->add_parameters({HIDDEN_DIM, LSTM_HIDDEN_DIM});
  p_bir2h = model->add_parameters({HIDDEN_DIM, LSTM_HIDDEN_DIM});
  p_label2h = model->add_parameters({HIDDEN_DIM, LABEL_DIM});
  p_hb = model->add_parameters({HIDDEN_DIM});

  //hidden to output
  p_h2o = model->add_parameters({1, HIDDEN_DIM});
  p_ob = model->add_parameters({1});

  p_labels = model->add_lookup_parameters(N_LABELS, {LABEL_DIM});
  p_trans = model->add_lookup_parameters(N_LABELS * N_LABELS, {1});

  p_for_guard = model->add_parameters({LSTM_INPUT_DIM});
  p_rev_guard = model->add_parameters({LSTM_INPUT_DIM});

  std::string label;
  id2label = corpus.id2label;
  for (int i=0; i<corpus.id2label.size(); i++){
    label = corpus.id2label[i];
    if (label == "B") B = i;
    else if(label == "M") M = i;
    else if(label == "E") E = i;
    else if(label == "S") S = i;
  }
  set_valid_trans(corpus.id2label);
}

Expression CRFBuilder::supervised_loss(cnn::ComputationGraph *cg, const std::vector<unsigned int> &raw_unigrams,
                                       const std::vector<unsigned int> &raw_bigrams, const std::vector<unsigned int> &unigrams,
                                       const std::vector<unsigned int> &bigrams, const std::vector<unsigned int> &labels) {
  unsigned len = static_cast<int>(unigrams.size());
  BOOST_ASSERT_MSG( len == labels.size(), "label and sentence size not match" );

  std::vector<ExpressionRow> emit_matrix(len, ExpressionRow(n_labels));
  std::vector<ExpressionRow> tran_matrix(n_labels, ExpressionRow(n_labels));
  std::vector<Expression> uni_labels(n_labels);
  for (unsigned t = 0; t < n_labels; ++t) {
    uni_labels[t] = cnn::expr::lookup(*cg, p_labels, t);
    for (unsigned pt = 0; pt < n_labels; ++pt) {
      tran_matrix[pt][t] = cnn::expr::lookup(*cg, p_trans, pt * n_labels + t);
    }
  }

  Expression lb = parameter(*cg, p_lb);
  Expression u2l = parameter(*cg, p_u2l);
  Expression b2l = parameter(*cg, p_b2l);
  Expression pu2l, pb2l, pou2l, pob2l;
  if (p_pu2l) { pu2l = parameter(*cg, p_pu2l); }
  if (p_pb2l) { pb2l = parameter(*cg, p_pb2l); }
  if (p_pou2l) {pou2l = parameter(*cg, p_pou2l); }
  if (p_pob2l) {pob2l = parameter(*cg, p_pob2l); }

  // inputs
  std::vector<Expression> inputs(len);
  Expression uni, bi, pu, pb, pou, pob;
  unsigned int idx;
  for (unsigned i = 0; i < len; ++i) {
    if(use_train) {
      uni = lookup(*cg, p_u, unigrams[i]);
      bi = lookup(*cg, p_b, bigrams[i]);
      inputs[i] = affine_transform({lb, u2l, uni, b2l, bi});
      if(p_pu2l) {
        if (unigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        if(is_finetune) pu = lookup(*cg, p_pu, idx);
        else pu = const_lookup(*cg, p_pu, idx);
        inputs[i] = affine_transform({inputs[i], pu2l, pu});
      }
      if(p_pb2l){
        if (bigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        if (is_finetune) pb = lookup(*cg, p_pb, idx);
        else pb = const_lookup(*cg, p_pb, idx);
        inputs[i] = affine_transform({inputs[i], pb2l, pb});
      }
      if(p_pou2l) {
        if (ounigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        if(is_finetune) pou = lookup(*cg, p_pou, idx);
        else pou = const_lookup(*cg, p_pou, idx);
        inputs[i] = affine_transform({inputs[i], pou2l, pou});
      }
      if(p_pob2l){
        if (obigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        if (is_finetune) pob = lookup(*cg, p_pob, idx);
        else pob = const_lookup(*cg, p_pob, idx);
        inputs[i] = affine_transform({inputs[i], pob2l, pob});
      }

    }
    else{
      BOOST_ASSERT(p_pu2l != nullptr);
      if(p_pu2l) {
        if (unigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        if(is_finetune) pu = lookup(*cg, p_pu, idx);
        else pu = const_lookup(*cg, p_pu, idx);
        inputs[i] = affine_transform({lb, pu2l, pu});
      }
      if(p_pb2l){
        if (bigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        if (is_finetune) pb = lookup(*cg, p_pb, idx);
        else pb = const_lookup(*cg, p_pb, idx);
        inputs[i] = affine_transform({inputs[i], pb2l, pb});
      }
      if(p_pou2l) {
        if (ounigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        if(is_ofinetune) pou = lookup(*cg, p_pou, idx);
        else pou = const_lookup(*cg, p_pou, idx);
        inputs[i] = affine_transform({inputs[i], pou2l, pou});
      }
      if(p_pob2l){
        if (obigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        if (is_ofinetune) pob = lookup(*cg, p_pob, idx);
        else pob = const_lookup(*cg, p_pob, idx);
        inputs[i] = affine_transform({inputs[i], pob2l, pob});
      }
    }
    inputs[i] = tanh(inputs[i]);
  }

  // bilstm
  Expression f_guard = parameter(*cg, p_for_guard);
  Expression r_guard = parameter(*cg, p_rev_guard);

  for_lstm.new_graph(*cg);
  rev_lstm.new_graph(*cg);
  for_lstm.start_new_sequence();
  rev_lstm.start_new_sequence();

  if(dropout_rate > 0){
    for_lstm.set_dropout(dropout_rate);
    rev_lstm.set_dropout(dropout_rate);
  }
  else{
    for_lstm.disable_dropout();
    rev_lstm.disable_dropout();
  }

  for_lstm.add_input(f_guard);
  rev_lstm.add_input(r_guard);
  for (int i=0; i<len; i++){
    for_lstm.add_input(inputs[i]);
    rev_lstm.add_input(inputs[len-i-1]);
  }

  Expression hb = parameter(*cg, p_hb);
  Expression label2h = parameter(*cg, p_label2h);
  Expression bif2h = parameter(*cg, p_bif2h);
  Expression bir2h = parameter(*cg, p_bir2h);
  Expression ob = parameter(*cg, p_ob);
  Expression h2o = parameter(*cg, p_h2o);
  for (unsigned i = 0; i < len; ++i) {
    Expression for_i = for_lstm.get_h(RNNPointer(i+1)).back();
    Expression rev_i = rev_lstm.get_h(RNNPointer(len-i)).back();
    for (unsigned t = 0; t < n_labels; ++t) {
      emit_matrix[i][t] = tanh(dropout(affine_transform({hb, bif2h, for_i, bir2h, rev_i, label2h, uni_labels[t]}), dropout_rate));
      emit_matrix[i][t] = affine_transform({ob, h2o, emit_matrix[i][t]});
    }
  }

  std::vector<ExpressionRow> alpha(len, ExpressionRow(n_labels));
  std::vector<Expression> path(len);

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      std::vector<Expression> f;
      if (i == 0) {
        f.push_back(emit_matrix[i][t]);
        if (t == labels[i]) {
          path[i] = emit_matrix[i][t];
        }
      } else {
        for (unsigned pt = 0; pt < n_labels; ++pt) {
          f.push_back(alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t]);
          if (pt == labels[i - 1] && t == labels[i]) {
            path[i] = path[i - 1] + emit_matrix[i][t] + tran_matrix[pt][t];
          }
        }
      }
      alpha[i][t] = cnn::expr::logsumexp(f);
      //if (fr.size() > 0) { path[i] = cnn::expr::logsumexp(fr); }
    }
  }

  std::vector<Expression> f;
  for (unsigned t = 0; t < n_labels; ++t) {
    f.push_back(alpha[len - 1][t]);
  }
  return cnn::expr::logsumexp(f) - path.back();
}


void CRFBuilder::decode(cnn::ComputationGraph *cg, const std::vector<unsigned int> &raw_unigrams,
                        const std::vector<unsigned int> &raw_bigrams, const std::vector<unsigned int> &unigrams,
                        const std::vector<unsigned int> &bigrams, std::vector<unsigned int> &pred_labels) {
  unsigned len = static_cast<int>(unigrams.size());
  std::vector<std::vector<double>> emit_matrix(len, std::vector<double>(n_labels));
  std::vector<std::vector<double>> tran_matrix(n_labels, std::vector<double>(n_labels));

  std::vector<Expression> uni_labels(n_labels);
  for (unsigned t = 0; t < n_labels; ++t) {
    uni_labels[t] = cnn::expr::lookup(*cg, p_labels, t);
    for (unsigned pt = 0; pt < n_labels; ++pt) {
      tran_matrix[pt][t] = as_scalar(cg->get_value(lookup(*cg, p_trans, pt * n_labels + t)));
    }
  }

  Expression lb = parameter(*cg, p_lb);
  Expression u2l = parameter(*cg, p_u2l);
  Expression b2l = parameter(*cg, p_b2l);
  Expression pu2l, pb2l, pou2l, pob2l;
  if (p_pu2l) { pu2l = parameter(*cg, p_pu2l); }
  if (p_pb2l) { pb2l = parameter(*cg, p_pb2l); }
  if (p_pou2l) {pou2l = parameter(*cg, p_pou2l); }
  if (p_pob2l) {pob2l = parameter(*cg, p_pob2l); }

  // inputs
  std::vector<Expression> inputs(len);
  Expression uni, bi, pu, pb, pou, pob;
  unsigned int idx;
  for (unsigned i = 0; i < len; ++i) {
    if(use_train) {
      uni = lookup(*cg, p_u, unigrams[i]);
      bi = lookup(*cg, p_b, bigrams[i]);
      inputs[i] = affine_transform({lb, u2l, uni, b2l, bi});
      if(p_pu2l) {
        if (unigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        pu = const_lookup(*cg, p_pu, idx);
        inputs[i] = affine_transform({inputs[i], pu2l, pu});
      }
      if(p_pb2l){
        if (bigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        pb = const_lookup(*cg, p_pb, idx);
        inputs[i] = affine_transform({inputs[i], pb2l, pb});
      }
      if(p_pou2l) {
        if (ounigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        pou = const_lookup(*cg, p_pou, idx);
        inputs[i] = affine_transform({inputs[i], pou2l, pou});
      }
      if(p_pob2l){
        if (obigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        pob = const_lookup(*cg, p_pob, idx);
        inputs[i] = affine_transform({inputs[i], pob2l, pob});
      }

    }
    else{
      BOOST_ASSERT(p_pu2l != nullptr);
      if(p_pu2l) {
        if (unigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        pu = const_lookup(*cg, p_pu, idx);
        inputs[i] = affine_transform({lb, pu2l, pu});
      }
      if(p_pb2l){
        if (bigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        pb = const_lookup(*cg, p_pb, idx);
        inputs[i] = affine_transform({inputs[i], pb2l, pb});
      }
      if(p_pou2l) {
        if (ounigram_pretrained.count(raw_unigrams[i])) idx = raw_unigrams[i];
        else idx = UNK_IDX;
        pou = const_lookup(*cg, p_pou, idx);
        inputs[i] = affine_transform({inputs[i], pou2l, pou});
      }
      if(p_pob2l){
        if (obigram_pretrained.count(raw_bigrams[i])) idx = raw_bigrams[i];
        else idx = UNK_IDX;
        pob = const_lookup(*cg, p_pob, idx);
        inputs[i] = affine_transform({inputs[i], pob2l, pob});
      }
    }
    inputs[i] = tanh(inputs[i]);
  }

  // bilstm
  Expression f_guard = parameter(*cg, p_for_guard);
  Expression r_guard = parameter(*cg, p_rev_guard);

  for_lstm.new_graph(*cg);
  rev_lstm.new_graph(*cg);
  for_lstm.start_new_sequence();
  rev_lstm.start_new_sequence();

  for_lstm.disable_dropout();
  rev_lstm.disable_dropout();

  for_lstm.add_input(f_guard);
  rev_lstm.add_input(r_guard);
  for (int i=0; i<len; i++){
    for_lstm.add_input(inputs[i]);
    rev_lstm.add_input(inputs[len-i-1]);
  }
  Expression hb = parameter(*cg, p_hb);
  Expression label2h = parameter(*cg, p_label2h);
  Expression bif2h = parameter(*cg, p_bif2h);
  Expression bir2h = parameter(*cg, p_bir2h);
  Expression ob = parameter(*cg, p_ob);
  Expression h2o = parameter(*cg, p_h2o);
  for (unsigned i = 0; i < len; ++i) {
    Expression for_i = for_lstm.get_h(RNNPointer(i+1)).back();
    Expression rev_i = rev_lstm.get_h(RNNPointer(len-i)).back();
    for (unsigned t = 0; t < n_labels; ++t) {
      Expression tmp = tanh(affine_transform({hb, bif2h, for_i, bir2h, rev_i, label2h, uni_labels[t]}));
      emit_matrix[i][t] = as_scalar(cg->get_value(affine_transform({ob, h2o, tmp})));
    }
  }
  std::vector<std::vector<double>> alpha(len, std::vector<double>(n_labels));
  std::vector<std::vector<unsigned>> path(len, std::vector<unsigned>(n_labels));

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      if (i == 0) {
        alpha[i][t] = emit_matrix[i][t];
        path[i][t] = n_labels;
        continue;
      }

      for (unsigned pt = 0; pt < n_labels; ++pt) {
        if (pt == 0) {
          alpha[i][t] = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          path[i][t] = pt;
        } else {
          double score = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          if (score > alpha[i][t]) {
            alpha[i][t] = score;
            path[i][t] = pt;
          }
        }
      }
    }
  }

  unsigned best = 0; double best_score = alpha[len - 1][0];
  for (unsigned t = 1; t < n_labels; ++t) {
    if (best_score < alpha[len - 1][t]) { best = t; best_score = alpha[len - 1][t]; }
  }
  pred_labels.clear(); pred_labels.push_back(best);
  for (unsigned i = len - 1; i > 0; -- i) {
    best = path[i][best];
    pred_labels.push_back(best);
  }
  std::reverse(pred_labels.begin(), pred_labels.end());
  BOOST_ASSERT(pred_labels.size() == unigrams.size());
}

void CRFBuilder::set_valid_trans(const std::vector<std::string> &id2labels) {
  unsigned len = static_cast<unsigned >(id2labels.size());
  BOOST_ASSERT(len == 4);
  for (unsigned i=0; i<len; i++){
    if (id2labels[i] == "B") { Bid = i; }
    else if (id2labels[i] == "M") { Mid = i; }
    else if (id2labels[i] == "E") { Eid = i; }
    else if (id2labels[i] == "S") { Sid = i ;}
    else { BOOST_ASSERT_MSG(false, "unknown label"); }
  }

  valid_trans.insert(Bid*4 + Mid);
  valid_trans.insert(Bid*4 + Eid);
  valid_trans.insert(Mid*4 + Eid);
  valid_trans.insert(Mid*4 + Mid);
  valid_trans.insert(Eid*4 + Sid);
  valid_trans.insert(Eid*4 + Bid);
  valid_trans.insert(Sid*4 + Sid);
  valid_trans.insert(Sid*4 + Bid);
}

void CRFBuilder::get_valid_labels(std::vector<unsigned int> &cur_valid_labels, unsigned int len,
                                  unsigned int cur_position,
                                  std::vector<unsigned int> &pred_labels) {
  if(cur_position == 0){
    if(cur_position == len-1){
      cur_valid_labels.push_back(S);
    }
    else{
      cur_valid_labels.push_back(S);
      cur_valid_labels.push_back(B);
    }
  }
  else if (cur_position == len-1){
    int prev_label = pred_labels.back();
    if (prev_label == B || prev_label == M){
      cur_valid_labels.push_back(E);
    }
    else if (prev_label == E || prev_label == S){
      cur_valid_labels.push_back(S);
    }
  }
  else{
    int prev_label = pred_labels.back();
    if (prev_label == B || prev_label == M){
      cur_valid_labels.push_back(M);
      cur_valid_labels.push_back(E);
    }
    else if(prev_label == E || prev_label == S){
      cur_valid_labels.push_back(B);
      cur_valid_labels.push_back(S);
    }
  }
}