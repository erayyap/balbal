// Conformer rel-pos attention score as one onnxruntime CPU custom op (domain "balbal", op "RelPosScore").
//   out[0,h,i,j] = (ac[0,h,i,j] + bd[0,h,i, j - i + T - 1]) / div        ac [1,H,T,T], bd [1,H,T,2T-1], div scalar
// It replaces the exported rel-shift chain (Range/Sub/Expand of a [1,H,T,T] int64 index, GatherElements, Add, Div). The shifted row of bd
// is a contiguous slice, so no index tensor is built. Same IEEE float add and divide as ORT's Add/Div: build WITHOUT -ffast-math
// (that would turn the divide into a multiply by 1/div) so the output is bit-identical.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#include <vector>

namespace {

struct RelJob { const float *ac, *bd; float div, *out; int64_t T; };

void rows(void* p, size_t r) {   // one task = one (h, i) row
  auto& j = *static_cast<RelJob*>(p);
  const int64_t T = j.T, P = 2 * T - 1, h = int64_t(r) / T, i = int64_t(r) % T;
  const float* a = j.ac + (h * T + i) * T;
  const float* b = j.bd + (h * T + i) * P + (T - 1 - i);
  float* o = j.out + (h * T + i) * T;
  const float d = j.div;
  for (int64_t k = 0; k < T; ++k) o[k] = (a[k] + b[k]) / d;
}

struct RelKernel {
  RelKernel(const OrtApi&, const OrtKernelInfo*) {}
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto ac = ctx.GetInput(0), bd = ctx.GetInput(1), dv = ctx.GetInput(2);
    auto as = ac.GetTensorTypeAndShapeInfo().GetShape(), bs = bd.GetTensorTypeAndShapeInfo().GetShape();
    if (as.size() != 4 || bs.size() != 4 || as[0] != 1 || bs[0] != 1 || as[1] != bs[1] || as[2] != as[3] || bs[2] != as[2] || bs[3] != 2 * as[2] - 1
        || dv.GetTensorTypeAndShapeInfo().GetElementCount() != 1)
      ORT_CXX_API_THROW("RelPosScore: want ac [1,H,T,T], bd [1,H,T,2T-1], scalar div", ORT_INVALID_ARGUMENT);
    RelJob j{ac.GetTensorData<float>(), bd.GetTensorData<float>(), dv.GetTensorData<float>()[0], nullptr, as[2]};
    j.out = ctx.GetOutput(0, as.data(), as.size()).GetTensorMutableData<float>();
    const size_t n = size_t(as[1] * as[2]);
    if (n) ctx.ParallelFor(rows, n, 0, &j);
  }
};

struct RelOp : Ort::CustomOpBase<RelOp, RelKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new RelKernel(api, info); }
  const char* GetName() const { return "RelPosScore"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

}  // namespace

void AddRelPosOps(Ort::CustomOpDomain& domain) { static RelOp op; domain.Add(&op); }
