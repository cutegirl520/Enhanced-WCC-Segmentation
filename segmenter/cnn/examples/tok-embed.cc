#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 1;
unsigned CODE_DIM = 64;
unsigned CHAR_DIM = 64;
unsigned EMBED_DIM = 64;
unsigned SEG_DIM = 32;
unsigned H1DIM = 48;
unsigned H2DIM = 36;
unsigned TAG_DIM = 16;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned DURATION_DIM = 8;

bool eval = false;
cnn::Dict d;
int kNONE;
int kSOW;
int kEOW;

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else abort();
}

struct PrefixNode {
  PrefixNode() :
      terminal(false),
      bias(nullptr), pred(nullptr), zero_cond(nullptr), zero_child(nullptr),
      one_cond(nullptr), one_child(nullptr) {}

  ~PrefixNode() {
    delete zero_child;
    delete one_child;
  }

  bool terminal;
  Parameters* bias;
  Parameters* pred;
  Parameters* zero_cond;
  PrefixNode* zero_child;
  Parameters* one_cond;
  PrefixNode* one_child;
};

// instructions for use
//   1) add all codes using add("0000") and the like
//   2) then call AllocateParameters(m, dim)
struct PrefixCode {
  PrefixCode() : params_allocated(false) {}

  PrefixNode* add(const string& pfc) {
    assert(!params_allocated);
    PrefixNode* cur = &root;
    for (unsigned i = 0; i < pfc.size(); ++i) {
      if (cur->terminal) {
        cerr << "Prefix property violated at position " << i << " of " << pfc << endl;
        abort();
      }
      assert(pfc[i] == '0' || pfc[i] == '1');
      PrefixNode*& next = pfc[i] == '0' ? cur->zero_child : cur->one_child;
      if (!next) next = new PrefixNode;
      cur = next;
    }
    cur->terminal = true;
    return cur;
  }

  void AllocateParameters_rec(Model& m, unsigned dim, PrefixNode* n) {
    if (!n->terminal) {
      if (!n->zero_child || !n->one_child) {
        cerr << "Non-binary production in prefix code\n";
        abort();
      }
      n->bias = m.add_parameters({1});
      n->pred = m.add_parameters({dim});
   