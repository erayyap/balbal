// Native token-length reward for Flashlight's lexicon-free CTC decoder.
// The decoder calls score() only when a new collapsed token is emitted:
// repeated frames, blanks, BOS and EOS receive no length reward.
#include <pybind11/pybind11.h>
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

PYBIND11_MODULE(ctc_lm_bonus, m) {
  py::module_::import("flashlight.lib.text.decoder");
  py::class_<TokenBonusLM, LM, std::shared_ptr<TokenBonusLM>>(m, "TokenBonusLM")
      .def(py::init<LMPtr, float>());
}
