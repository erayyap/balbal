// Native token-length reward for Flashlight's lexicon-free CTC decoder.
// The decoder calls score() only when a new collapsed token is emitted:
// repeated frames, blanks, BOS and EOS receive no length reward.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <climits>
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace py = pybind11;
using namespace fl::lib::text;

class TokenBonusLM : public LM {
 public:
  TokenBonusLM(LMPtr base, float bonus) : base_(std::move(base)), bonus_(bonus) {}
  LMStatePtr start(bool empty) override { return base_->start(empty); }
  std::pair<LMStatePtr, float> score(const LMStatePtr& state, int token) override {
    auto result = base_->score(state, token);
    result.second += bonus_;
    return result;
  }
  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override {
    return base_->finish(state);
  }
  void updateCache(std::vector<LMStatePtr> states) override {
    base_->updateCache(std::move(states));
  }
 private:
  LMPtr base_;
  float bonus_;
};

// Per-request hotword (keyword) biasing, 2026-10-15. Wraps any LM with a trie over token sequences of the
// requested words. Each token that extends a match earns that edge's bonus; when the match breaks, or the
// hypothesis ends inside an unfinished word, the bonus earned so far is taken back, so only completed hotwords
// keep their reward. States mirror the base LM's state tree (one cached child per token), so hypotheses that
// the base LM would merge are only merged here when they are also at the same hotword position.
struct HotwordState : LMState {
  LMStatePtr base;
  int node = 0;
  float earned = 0;  // bonus of the current unfinished match (refunded on mismatch / finish)
  float delta = 0;   // hotword part of the score that led to this state
};

class HotwordLM : public LM {
 public:
  HotwordLM(LMPtr base, const std::vector<std::vector<int>>& seqs, const std::vector<std::vector<float>>& bonuses)
      : base_(std::move(base)) {
    if (seqs.size() != bonuses.size()) throw std::invalid_argument("one bonus list per sequence");
    nodes_.emplace_back();
    for (size_t s = 0; s < seqs.size(); ++s) {
      if (seqs[s].empty() || seqs[s].size() != bonuses[s].size())
        throw std::invalid_argument("each sequence needs one bonus per token");
      int node = 0;
      for (size_t i = 0; i < seqs[s].size(); ++i) {
        auto it = nodes_[node].next.find(seqs[s][i]);
        if (it == nodes_[node].next.end()) {
          nodes_.emplace_back();
          nodes_.back().bonus = bonuses[s][i];
          it = nodes_[node].next.emplace(seqs[s][i], int(nodes_.size()) - 1).first;
        }
        node = it->second;
      }
      nodes_[node].terminal = true;
    }
  }
  LMStatePtr start(bool empty) override {
    auto state = std::make_shared<HotwordState>();
    state->base = base_->start(empty);
    return state;
  }
  std::pair<LMStatePtr, float> score(const LMStatePtr& state, int token) override {
    auto in = std::static_pointer_cast<HotwordState>(state);
    auto result = base_->score(in->base, token);
    auto out = in->child<HotwordState>(token);
    if (!out->base) {
      out->base = result.first;
      int node = in->node;
      float earned = in->earned, delta = 0;
      int next = step(node, token);
      if (next < 0 && node != 0) {  // match broke: take back the partial bonus, retry as a new word start
        delta -= earned; earned = 0; node = 0;
        next = step(0, token);
      }
      if (next >= 0) {
        delta += nodes_[next].bonus; earned += nodes_[next].bonus; node = next;
        if (nodes_[next].terminal) {  // completed hotword keeps its bonus
          earned = 0;
          if (nodes_[next].next.empty()) node = 0;
        }
      } else {
        node = 0;
      }
      out->node = node; out->earned = earned; out->delta = delta;
    }
    return {out, result.second + out->delta};
  }
  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override {
    auto in = std::static_pointer_cast<HotwordState>(state);
    auto result = base_->finish(in->base);
    auto out = in->child<HotwordState>(INT_MIN);
    out->base = result.first;
    return {out, result.second - in->earned};
  }
  void updateCache(std::vector<LMStatePtr> states) override {
    for (auto& s : states) s = std::static_pointer_cast<HotwordState>(s)->base;
    base_->updateCache(std::move(states));
  }
  int nodes() const { return int(nodes_.size()); }
 private:
  struct Node { std::unordered_map<int, int> next; float bonus = 0; bool terminal = false; };
  int step(int node, int token) const {
    auto it = nodes_[node].next.find(token);
    return it == nodes_[node].next.end() ? -1 : it->second;
  }
  LMPtr base_;
  std::vector<Node> nodes_;
};

PYBIND11_MODULE(ctc_lm_bonus, m) {
  py::module_::import("flashlight.lib.text.decoder");
  py::class_<TokenBonusLM, LM, std::shared_ptr<TokenBonusLM>>(m, "TokenBonusLM")
      .def(py::init<LMPtr, float>());
  py::class_<HotwordLM, LM, std::shared_ptr<HotwordLM>>(m, "HotwordLM")
      .def(py::init<LMPtr, const std::vector<std::vector<int>>&, const std::vector<std::vector<float>>&>())
      .def_property_readonly("nodes", &HotwordLM::nodes);
}
