// Incremental version of ctc_fast_decoder.cpp (exact flashlight-text 0.0.7 LexiconFreeDecoder
// best path, CTC, no silence token) for streaming. Same search per frame (best-first previous hypotheses, early break on the emission
// bound, candidatesStore merge); the beam is kept between calls so every committed frame is searched once.
//   step(frames)  : extend the beam by T committed frames
//   peek(frames)  : best collapsed tokens if T more frames were appended, without committing them (provisional text for the lookahead)
//   best()        : collapsed tokens of the current best hypothesis (no LM end-of-sentence)
//   finish()      : LM end-of-sentence + best collapsed tokens (= FastCTCDecoder.decode on all frames of the segment); then reset()
//   rebase()      : keep only the best hypothesis (its LM state and CTC state) as the new start; returns its collapsed tokens and frees
//                   the history (long speech without pauses: bounded memory, the LM context carries over)
// Collapsed tokens = frame path with repeats merged and blank / start dropped (itertools.groupby on the frame path, blanks dropped).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <vector>
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace py = pybind11;
using fl::lib::text::LMPtr;
using fl::lib::text::LMStatePtr;

namespace streamdec {

struct State {
  double score; LMStatePtr lmState; const State* parent; int token; bool prevBlank;
  State(double s, const LMStatePtr& l, const State* p, int t, bool b) : score(s), lmState(l), parent(p), token(t), prevBlank(b) {}
  int compareNoScoreStates(const State* o) const {   // same order as LexiconFreeDecoderState
    int c = lmState->compare(o->lmState);
    if (c != 0) return c > 0 ? 1 : -1;
    if (token != o->token) return token > o->token ? 1 : -1;
    if (prevBlank != o->prevBlank) return prevBlank > o->prevBlank ? 1 : -1;
    return 0;
  }
};

class StreamCTCDecoder {
 public:
  StreamCTCDecoder(int beam, int beamToken, double threshold, double lmWeight, LMPtr lm, int blank, double lmUpper, bool logAdd)
      : beam_(beam), beamToken_(beamToken), thr_(threshold), lmw_(lmWeight), lm_(std::move(lm)), blank_(blank), lmUpper_(lmUpper), logAdd_(logAdd) {
    reset();
  }

  void reset() {
    hyp_.clear(); hyp_.emplace_back();
    hyp_[0].emplace_back(0.0, lm_->start(false), nullptr, -1, false);
  }

  void step(uintptr_t ptr, int T, int N) {
    const float* em = reinterpret_cast<const float*>(ptr);
    for (int t = 0; t < T; t++) frame(em + (size_t)t * N, N);
  }

  std::vector<int> peek(uintptr_t ptr, int T, int N) {
    const float* em = reinterpret_cast<const float*>(ptr);
    const size_t keep = hyp_.size();
    for (int t = 0; t < T; t++) frame(em + (size_t)t * N, N);
    std::vector<int> out = collapse(bestState());
    while (hyp_.size() > keep) hyp_.pop_back();
    return out;
  }

  std::vector<int> best() const { return collapse(bestState()); }

  std::vector<int> finish() {
    reset_cand();
    for (auto& prev : hyp_.back()) {
      auto ls = lm_->finish(prev.lmState);
      add(prev.score + lmw_ * ls.second, ls.first, &prev, -1, false);
    }
    std::vector<State> fin; store(fin, true);
    std::vector<int> out = fin.empty() ? std::vector<int>() : collapse(&fin[0]);
    reset();
    return out;
  }

  std::vector<int> rebase() {
    const State* b = bestState();
    std::vector<int> out = collapse(b);
    State s(0.0, b->lmState, nullptr, b->token, b->prevBlank);
    hyp_.clear(); hyp_.emplace_back(); hyp_[0].push_back(s);   // carried CTC state: a repeat of its token stays merged
    return out;
  }

  int frames() const { return (int)hyp_.size() - 1; }

 private:
  void frame(const float* e, int N) {
    const int K = std::min(beamToken_, N);
    idx_.resize(N); std::iota(idx_.begin(), idx_.end(), 0);
    if (N > beamToken_) std::partial_sort(idx_.begin(), idx_.begin() + beamToken_, idx_.end(), [e](int l, int r) { return e[l] > e[r]; });
    const double lmBonusMax = std::max(0.0, lmw_ * lmUpper_);
    reset_cand();
    order_.clear();
    for (auto& h : hyp_.back()) order_.push_back(&h);
    std::sort(order_.begin(), order_.end(), [](const State* a, const State* b) { return a->score > b->score; });
    for (const State* prev : order_) {
      for (int r = 0; r < K; ++r) {
        const int n = idx_[r];
        const double score = prev->score + e[n];
        if (score + lmBonusMax < best_ - thr_) break;
        if (n != blank_ && (n != prev->token || prev->prevBlank)) {
          auto ls = lm_->score(prev->lmState, n);
          add(score + lmw_ * ls.second, ls.first, prev, n, false);
        } else if (n == blank_) {
          add(score, prev->lmState, prev, n, true);
        } else {
          add(score, prev->lmState, prev, n, false);
        }
      }
    }
    hyp_.emplace_back(); store(hyp_.back(), false);
  }

  const State* bestState() const {
    const auto& v = hyp_.back(); const State* b = &v[0];
    for (auto& s : v) if (s.score > b->score) b = &s;
    return b;
  }

  std::vector<int> collapse(const State* s) const {
    std::vector<int> rev;
    for (; s; s = s->parent) rev.push_back(s->token);
    std::vector<int> out;
    int prev = -2; bool first = true;   // the first path element is the start state or, after rebase(), the carried state (already returned)
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
      int t = *it;
      if (first) { first = false; prev = t; continue; }   // start / carried state
      if (t != prev && t != blank_ && t >= 0) out.push_back(t);
      prev = t;
    }
    return out;
  }

  void reset_cand() { best_ = -std::numeric_limits<double>::infinity(); cand_.clear(); ptrs_.clear(); }
  void add(double score, const LMStatePtr& l, const State* p, int t, bool b) {
    if (score >= best_) best_ = score;
    if (score >= best_ - thr_) cand_.emplace_back(score, l, p, t, b);
  }
  void store(std::vector<State>& out, bool sorted) {   // candidatesStore
    out.clear(); ptrs_.clear();
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
  std::deque<std::vector<State>> hyp_;   // deque: frames never move, so parent pointers stay valid
  std::vector<State> cand_; std::vector<State*> ptrs_; std::vector<const State*> order_; std::vector<int> idx_;
  double best_ = 0;
};

}  // namespace streamdec

PYBIND11_MODULE(ctc_stream_decoder, m) {
  py::module_::import("flashlight.lib.text.decoder");
  py::class_<streamdec::StreamCTCDecoder>(m, "StreamCTCDecoder")
      .def(py::init<int, int, double, double, LMPtr, int, double, bool>(), py::arg("beam"), py::arg("beam_token"), py::arg("threshold"),
           py::arg("lm_weight"), py::arg("lm"), py::arg("blank"), py::arg("lm_upper"), py::arg("log_add") = true)
      .def("reset", &streamdec::StreamCTCDecoder::reset)
      .def("step", &streamdec::StreamCTCDecoder::step, py::call_guard<py::gil_scoped_release>())
      .def("peek", &streamdec::StreamCTCDecoder::peek, py::call_guard<py::gil_scoped_release>())
      .def("best", &streamdec::StreamCTCDecoder::best)
      .def("finish", &streamdec::StreamCTCDecoder::finish)
      .def("rebase", &streamdec::StreamCTCDecoder::rebase)
      .def("frames", &streamdec::StreamCTCDecoder::frames);
}
