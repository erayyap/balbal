// Exact re-implementation of flashlight-text 0.0.7 LexiconFreeDecoder (CTC, no silence token, no ASG) that returns the
// same best hypothesis with less work. Three changes, none of which alters which candidates survive:
//  1. Hypotheses of the previous frame are visited best first, so the running best score (and with it the beam threshold) rises early.
//  2. For each hypothesis the tokens are visited in descending emission order (as before). A candidate can score at most
//     prev + emission + lm_weight * lm_upper (lm_upper = the largest score the LM can return, i.e. the token bonus for KenLM log10 <= 0).
//     If that bound is below best - threshold, candidatesAdd would drop this and every later (lower-emission) token of the hypothesis,
//     so the loop stops without querying the LM.
//  3. Only the best final hypothesis is built and returned (flashlight returns all of them as Python lists).
// Ties in score are the only case where order could matter; flashlight already orders candidates by LM-state address there.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace py = pybind11;
using fl::lib::text::LMPtr;
using fl::lib::text::LMStatePtr;

namespace fastdec {

struct State {
  double score; LMStatePtr lmState; const State* parent; int token; bool prevBlank; double amScore, lmScore;
  State(double s, const LMStatePtr& l, const State* p, int t, bool b, double am, double lm)
      : score(s), lmState(l), parent(p), token(t), prevBlank(b), amScore(am), lmScore(lm) {}
  int compareNoScoreStates(const State* o) const {   // same order as LexiconFreeDecoderState
    int c = lmState->compare(o->lmState);
    if (c != 0) return c > 0 ? 1 : -1;
    if (token != o->token) return token > o->token ? 1 : -1;
    if (prevBlank != o->prevBlank) return prevBlank > o->prevBlank ? 1 : -1;
    return 0;
  }
};

class FastCTCDecoder {
 public:
  FastCTCDecoder(int beam, int beamToken, double threshold, double lmWeight, LMPtr lm, int blank, double lmUpper, bool logAdd)
      : beam_(beam), beamToken_(beamToken), thr_(threshold), lmw_(lmWeight), lm_(std::move(lm)), blank_(blank), lmUpper_(lmUpper), logAdd_(logAdd) {}

  std::vector<int> decode(uintptr_t ptr, int T, int N) {
    const float* em = reinterpret_cast<const float*>(ptr);
    hyp_.assign(T + 2, {});
    hyp_[0].emplace_back(0.0, lm_->start(false), nullptr, -1, false, 0.0, 0.0);
    const int K = std::min(beamToken_, N);
    std::vector<int> idx(N);
    const double lmBonusMax = std::max(0.0, lmw_ * lmUpper_);
    for (int t = 0; t < T; t++) {
      const float* e = em + (size_t)t * N;
      std::iota(idx.begin(), idx.end(), 0);
      if (N > beamToken_) std::partial_sort(idx.begin(), idx.begin() + beamToken_, idx.end(), [e](int l, int r) { return e[l] > e[r]; });
      reset();
      order_.clear();
      for (auto& h : hyp_[t]) order_.push_back(&h);
      std::sort(order_.begin(), order_.end(), [](const State* a, const State* b) { return a->score > b->score; });
      for (const State* prev : order_) {
        for (int r = 0; r < K; ++r) {
          const int n = idx[r];
          const double am = e[n];
          const double score = prev->score + am;
          if (score + lmBonusMax < best_ - thr_) break;   // this and all lower-emission tokens would be dropped
          if (n != blank_ && (n != prev->token || prev->prevBlank)) {
            auto ls = lm_->score(prev->lmState, n);
            add(score + lmw_ * ls.second, ls.first, prev, n, false, prev->amScore + am, prev->lmScore + ls.second);
            if (ls.second > maxLm_) maxLm_ = ls.second;
          } else if (n == blank_) {
            add(score, prev->lmState, prev, n, true, prev->amScore + am, prev->lmScore);
          } else {
            add(score, prev->lmState, prev, n, false, prev->amScore + am, prev->lmScore);
          }
        }
      }
      store(hyp_[t + 1], false);
    }
    reset();   // decodeEnd: LM finish, sil_ = -1 token
    for (auto& prev : hyp_[T]) {
      auto ls = lm_->finish(prev.lmState);
      add(prev.score + lmw_ * ls.second, ls.first, &prev, -1, false, prev.amScore, prev.lmScore + ls.second);
    }
    store(hyp_[T + 1], true);
    std::vector<int> tokens;
    if (hyp_[T + 1].empty()) return tokens;
    std::vector<int> rev;
    for (const State* s = &hyp_[T + 1][0]; s; s = s->parent) rev.push_back(s->token);
    tokens.assign(rev.rbegin(), rev.rend());   // frame-aligned like DecodeResult.tokens (index 0 = start state)
    return tokens;
  }
  double maxLm_ = -std::numeric_limits<double>::infinity();

 private:
  void reset() { best_ = -std::numeric_limits<double>::infinity(); cand_.clear(); ptrs_.clear(); }
  void add(double score, const LMStatePtr& l, const State* p, int t, bool b, double am, double lm) {
    if (score >= best_) best_ = score;
    if (score >= best_ - thr_) cand_.emplace_back(score, l, p, t, b, am, lm);
  }
  void store(std::vector<State>& out, bool sorted) {   // candidatesStore
    out.clear();
    if (cand_.empty()) return;
    const double threshold = best_ - thr_;
    for (auto& c : cand_) if (c.score >= threshold) ptrs_.push_back(&c);
    std::sort(ptrs_.begin(), ptrs_.end(), [](const State* a, const State* b) {
      int c = a->compareNoScoreStates(b); return c == 0 ? a->score > b->score : c > 0; });
    int m = 1;
    for (size_t i = 1; i < ptrs_.size(); i++) {
      if (ptrs_[i]->compareNoScoreStates(ptrs_[m - 1]) != 0) ptrs_[m++] = ptrs_[i];
      else {
        double mx = std::max(ptrs_[m - 1]->score, ptrs_[i]->score);
        if (logAdd_) { double mn = std::min(ptrs_[m - 1]->score, ptrs_[i]->score); ptrs_[m - 1]->score = mx + std::log1p(std::exp(mn - mx)); }
        else ptrs_[m - 1]->score = mx;
      }
    }
    ptrs_.resize(m);
    auto byScore = [](const State* a, const State* b) { return a->score > b->score; };
    int nValid = ptrs_.size(), fin = std::min(nValid, beam_);
    if (!sorted && nValid > beam_) std::nth_element(ptrs_.begin(), ptrs_.begin() + fin, ptrs_.begin() + nValid, byScore);
    else if (sorted) std::partial_sort(ptrs_.begin(), ptrs_.begin() + fin, ptrs_.begin() + nValid, byScore);
    out.reserve(fin);
    for (int i = 0; i < fin; i++) out.emplace_back(std::move(*ptrs_[i]));
  }
  int beam_, beamToken_; double thr_, lmw_; LMPtr lm_; int blank_; double lmUpper_; bool logAdd_;
  std::vector<std::vector<State>> hyp_;
  std::vector<State> cand_; std::vector<State*> ptrs_; std::vector<const State*> order_;
  double best_ = 0;
};

}  // namespace fastdec

PYBIND11_MODULE(ctc_fast_decoder, m) {
  py::module_::import("flashlight.lib.text.decoder");
  py::class_<fastdec::FastCTCDecoder>(m, "FastCTCDecoder")
      .def(py::init<int, int, double, double, LMPtr, int, double, bool>(), py::arg("beam"), py::arg("beam_token"), py::arg("threshold"),
           py::arg("lm_weight"), py::arg("lm"), py::arg("blank"), py::arg("lm_upper"), py::arg("log_add") = true)
      .def("decode", &fastdec::FastCTCDecoder::decode, py::call_guard<py::gil_scoped_release>())
      .def_readonly("max_lm", &fastdec::FastCTCDecoder::maxLm_);
}
