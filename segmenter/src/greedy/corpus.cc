#include "utils.h"
#include "logging.h"
#include "corpus.h"
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <cassert>


Corpus::Corpus()
  : max_unigram(0), max_bigram(0){
}


// TODO add radical and strokes
void Corpus::load_training_data(const std::string &filename, const std::string &mode) {
  std::ifstream train_file(filename);
  BOOST_ASSERT_MSG(train_file, "Failed to open training file.");
  std::string line, prev_unigram;

  // add UNK
  if (mode == "train"){
    BOOST_ASSERT_MSG(max_bigram==0 && max_unigram==0, "max unigram and bigram should be 0");
    unigram2id[Corpus::UNK] = 0, id2unigram[0] = Corpus::UNK;
    bigram2id[Corpus::UNK] = 0, id2bigram[0] = Corpus::UNK;
    radical2id[Corpus::UNK] = 0, id2radical[0] = Corpus::UNK;
    max_unigram = 1, max_bigram = 1, max_radical = 1;
  }
  else{
    BOOST_ASSERT(max_unigram >= 1);
    BOOST_ASSERT(max_bigram >= 1);
    BOOST_ASSERT(max_radical >= 1);
    BOOST_ASSERT(id2label.size() == 4);
  }

  std::vector<unsigned> cur_unigrams;
  std::vector<unsigned> cur_bigrams;
  std::vector<unsigned> cur_radicals;
  std::vector<unsigned> cur_labels;

  unsigned sid = 0;
  while (std::getline(train_file, line)){
    if (line.empty()){
      if (cur_unigrams.size() == 0){
        continue;
      }
      if (mode == "train"){
        train_unigram_sentences[sid] = cur_unigrams;
        train_bigram_sentences[sid] = cur_bigrams;
        train_radical_sentences[sid] = cur_radicals;
        train_labels[sid] = cur_labels;
        sid ++;
        n_train = sid;
      }
      else if(mode == "dev"){
        dev_unigram_sentences[sid] = cur_unigrams;
        dev_bigram_sentences[sid] = cur_bigrams;
        dev_radical_sentences[sid] = cur_radicals;
        dev_labels[sid] = cur_labels;
        sid ++;
        n_dev = sid;
      }
      else if(mode == "test"){
        test_unigram_sentences[sid] = cur_unigrams;
        test_bigram_sentences[sid] = cur_bigrams;
        test_radical_sentences[sid] = cur_radicals;
        test_labels[sid] = cur_labels;
        sid ++;
        n_test = sid;
      }
      else{
        BOOST_ASSERT_MSG(false, "wrong mode of training data");
      }

      cur_unigrams.clear();
      cur_bigrams.clear();
      cur_labels.clear();
      cur_radicals.clear();
    }
    else{
      std::vector<std::string> items;

      boost::algorithm::trim(line);
      boost::algorithm::split(items, line, boost::is_any_of("\t"), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() >= 2, "Ill format of training data");
      std::string& unigram = items[0];
      std::string& radical = items[1];
      std::string& label = items.back();

      std::string bigram;
      if (cur_unigrams.size() == 0) {
        bigram = Corpus::BOS + unigram;
      }
      else{
        bigram = prev_unigram + unigram;
      }
      prev_unigram = unigram;

      if (mode == "train"){
        add(unigram, max_unigram, unigram2id, id2unigram);
        add(bi