#include <iostream>
#include <map>
#include <chrono>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cassert>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "logging.h"
#include "training_utils.h"
#include "crf.h"
#include "../corpus.h"
#include "glog/logging.h"
#include "eval.h"

namespace po = boost::program_options;

Corpus corpus;

void print_conf(const boost::program_options::variables_map &conf);

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description opts("LSTM-GREEDY");
  opts.add_options()
    ("is_train,i", po::value<unsigned>(), "train or test")
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
    ("model_dir", po::value<std::string>()->default_value("model/"), "Model dir")
    ("train,T", po::value<std::string>(), "The path to the train data.")
    ("dev,d", po::value<std::string>(), "The path to the dev data")
    ("test,t", po::value<std::string>(), "The path to the test data")
    ("model,m", po::value<std::string>(), "The path to the model, used for test time.")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy.")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probabilty with which to replace singletons.")
    ("layers", po::value<unsigned>()->default_value(1), "number of LSTM layers")
    ("unigram, c", po::value<std::string>(), "The path to the in domian unigram embedding.")
    ("bigram, b", po::value<std::string>(), "The path to the in domain bigram embedding")
    ("ounigram, oc", po::value<std::string>(), "The path to the out of domian unigram embedding.")
    ("obigram, ob", po::value<std::string>(), "The path to the out of domain bigram embedding")
    ("unigram_dim", po::value<unsigned>()->default_value(50), "unigram embedding dim")
    ("bigram_dim", po::value<unsigned>()->default_value(50), "bigram embedding dim")
    ("label_dim", po::value<unsigned>()->default_value(32), "label embedding dim")
    ("lstm_hidden_dim", po::value<unsigned>()->default_value(100), "LSTM hidden dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(100), "LSTM input dimension")
    ("hidden_dim", po::value<unsigned>()->default_value(100), "hidden dim")
    ("maxiter", po::value<unsigned>()->default_value(30), "Max number of iterations.")
    ("dropout", po::value<float>()->default_value(0.3), "the dropout rate")
    ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
    ("outfile", po::value<std::strin