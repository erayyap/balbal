// Fused product-key memory read as an onnxruntime CPU custom op (domain "balbal", op "PkmRead").
// Same math as balbal/model/conformer_acoustic_memory_multihead.py retrieve() in eval mode (export_onnx.py PkmReadFn):
//   per frame t, head h: left/right sub-key scores = q[t,h,side,:] . keys[h,side,s,:]  (s < S), top-K of each side,
//   K*K candidate sums, top-K of those, softmax(score * inv_temp[h]), out[t,:] += sum_k w_k * values[li*S + ri, :]  (summed over heads).
// Inputs: query [1,T,H,2,D] (already L2-normalized), keys [H,2,S,D] (normalized), values [N,V], inv_temp [H]; attribute top_k.
// PkmReadQ8 (same attribute): values int8 [N,V] + per-row scale [N] (values ~= q * scale; 4x less RAM), inputs
//   query, keys, values_q, scales, inv_temp.
// Output: [1,T,V]. Frames run in parallel on the session's intra-op pool (KernelContext::ParallelFor). Selection is unsorted
// (nth_element): order inside the top-K does not change softmax or the weighted sum; ties / float summation order may differ.
// Build: ./build.sh (pkm_read.cc with -ffast-math, relpos.cc without, one libbalbal_ops.so)
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#include <algorithm>
#include <functional>
#include <mutex>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace {

constexpr int64_t TB = 8;   // frames per parallel task (score block [TB, S] stays in L1/L2, each key row reused TB times)
constexpr int64_t SB = 32;  // keys per register tile: acc[TB][SB] = 16 zmm accumulators, one pass over D per tile

struct Job {
  const float *q, *kt, *values, *inv_temp;   // kt = keys transposed to [H, 2, D, SP], SP = S rounded up to SB (zero padded)
  const int8_t* values_q; const float* scales;   // PkmReadQ8 (values == nullptr)
  float* out;
  int64_t T, H, S, D, V, K, SP;
};

// index of the FIRST minimum of v[0..K) (same tie rule as a forward scalar scan)
inline int64_t argmin_first(const float* __restrict v, int64_t K) {
#ifdef __AVX512F__
  if (K % 16 == 0) {
    __m512 m = _mm512_loadu_ps(v);
    for (int64_t k = 16; k < K; k += 16) m = _mm512_min_ps(m, _mm512_loadu_ps(v + k));
    const float mn = _mm512_reduce_min_ps(m);
    const __m512 b = _mm512_set1_ps(mn);
    for (int64_t k = 0; k < K; k += 16) {
      __mmask16 e = _mm512_cmp_ps_mask(_mm512_loadu_ps(v + k), b, _CMP_EQ_OQ);
      if (e) return k + __builtin_ctz(e);
    }
  }
#endif
  int64_t mi = 0;
  for (int64_t k = 1; k < K; ++k) if (v[k] < v[mi]) mi = k;
  return mi;
}

// top-K of x[0..n) (unsorted): running K-best buffer + its current minimum; only values above the minimum touch the buffer
// (~K ln(n/K) replacements). Ties at the boundary keep the earliest index. AVX-512: 16-wide filter against the minimum + vector argmin;
// visits and replaces exactly the same elements in the same order as the scalar loop.
inline void topk(const float* __restrict x, int64_t n, int64_t K, int32_t* __restrict ids, float* __restrict vals) {
  for (int64_t k = 0; k < K; ++k) { vals[k] = x[k]; ids[k] = int32_t(k); }
  int64_t mi = argmin_first(vals, K);
  float mn = vals[mi];
  int64_t i = K;
#ifdef __AVX512F__
  for (; i + 16 <= n; i += 16) {
    const __m512 xv = _mm512_loadu_ps(x + i);
    __mmask16 m = _mm512_cmp_ps_mask(xv, _mm512_set1_ps(mn), _CMP_GT_OQ);
    while (m) {
      const int l = __builtin_ctz(m);
      vals[mi] = x[i + l]; ids[mi] = int32_t(i + l);
      mi = argmin_first(vals, K); mn = vals[mi];
      m = _mm512_cmp_ps_mask(xv, _mm512_set1_ps(mn), _CMP_GT_OQ) & __mmask16(0xFFFFu << (l + 1));
    }
  }
#endif
  for (; i < n; ++i) {
    if (x[i] <= mn) continue;
    vals[mi] = x[i]; ids[mi] = int32_t(i);
    mi = argmin_first(vals, K); mn = vals[mi];
  }
}

// Fast top-K: (1) 16-lane vertical maxima over blocks of 16 vectors give G >= K maxima of disjoint element groups; their K-th largest tau
// is a lower bound for the K-th largest element, so (2) only elements >= tau (kept in index order, ~4% of S) can be in the top-K;
// (3) the exact running-min topk runs on those survivors. Same top-K set whenever scores are distinct; falls back to topk() otherwise
// Returns false when the result could depend on tie order (another survivor equals the K-th value); the caller then reruns topk().
inline bool topk_fast(const float* __restrict x, int64_t n, int64_t K, int32_t* __restrict ids, float* __restrict vals) {
#ifdef __AVX512F__
  const int64_t nv = n / 16;                              // full vectors
  thread_local std::vector<float> gm, sv; thread_local std::vector<int32_t> si;
  gm.resize((nv + 15) / 16 * 16); sv.resize(n); si.resize(n);
  int64_t G = 0;
  for (int64_t v0 = 0; v0 < nv; v0 += 16) {
    __m512 m = _mm512_loadu_ps(x + v0 * 16);
    for (int64_t v = v0 + 1; v < std::min(nv, v0 + 16); ++v) m = _mm512_max_ps(m, _mm512_loadu_ps(x + v * 16));
    _mm512_storeu_ps(gm.data() + G, m); G += 16;
  }
  if (G < K) return false;
  std::nth_element(gm.begin(), gm.begin() + (K - 1), gm.begin() + G, std::greater<float>());
  const float tau = gm[K - 1];
  const __m512 tv = _mm512_set1_ps(tau);
  __m512i iv = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i step = _mm512_set1_epi32(16);
  int64_t c = 0;
  for (int64_t v = 0; v < nv; ++v, iv = _mm512_add_epi32(iv, step)) {
    const __m512 xv = _mm512_loadu_ps(x + v * 16);
    const __mmask16 k = _mm512_cmp_ps_mask(xv, tv, _CMP_GE_OQ);
    if (k) { _mm512_mask_compressstoreu_ps(sv.data() + c, k, xv); _mm512_mask_compressstoreu_epi32(si.data() + c, k, iv); c += __builtin_popcount(k); }
  }
  for (int64_t i = nv * 16; i < n; ++i) if (x[i] >= tau) { sv[c] = x[i]; si[c] = int32_t(i); ++c; }
  if (c < K) return false;                              // cannot happen (>= K group maxima are >= tau); kept as a guard
  topk(sv.data(), c, K, ids, vals);
  float mn = vals[0];
  for (int64_t k = 1; k < K; ++k) mn = std::min(mn, vals[k]);
  int64_t ge = 0;
  for (int64_t i = 0; i < c; ++i) ge += sv[i] >= mn;
  if (ge != K) return false;                            // tie at the boundary
  for (int64_t k = 0; k < K; ++k) ids[k] = si[ids[k]];
  return true;
#else
  return false;
#endif
}

// sort a K list descending (values + ids together); K is small (16 / 32)
inline void sort_desc(float* v, int32_t* id, int64_t K) {
  for (int64_t i = 1; i < K; ++i) {
    float x = v[i]; int32_t y = id[i]; int64_t j = i - 1;
    while (j >= 0 && v[j] < x) { v[j + 1] = v[j]; id[j + 1] = id[j]; --j; }
    v[j + 1] = x; id[j + 1] = y;
  }
}

template <bool Q8>
void block(void* data, size_t b) {
  const Job& j = *static_cast<const Job*>(data);
  const int64_t H = j.H, S = j.S, D = j.D, V = j.V, K = j.K;
  const int64_t t0 = int64_t(b) * TB, tn = std::min<int64_t>(TB, j.T - t0);
  const int64_t SP = j.SP;
  thread_local std::vector<float> sc, topv, w, cv, ql;
  thread_local std::vector<int32_t> topi, ci;
  sc.resize(2 * TB * SP); topv.resize(2 * TB * K); topi.resize(2 * TB * K); w.resize(K); cv.resize(K); ci.resize(K); ql.resize(TB * D);
  std::memset(j.out + t0 * V, 0, sizeof(float) * tn * V);
  for (int64_t h = 0; h < H; ++h) {
    for (int side = 0; side < 2; ++side) {               // scores [TB, SP] = Q[TB, D] . KT[D, SP], register-tiled
      float* __restrict scs = sc.data() + side * TB * SP;
      float* __restrict q = ql.data();                   // this block's query rows (zero rows past tn)
      for (int64_t t = 0; t < TB; ++t)
        for (int64_t d = 0; d < D; ++d) q[t * D + d] = t < tn ? j.q[((t0 + t) * H + h) * 2 * D + side * D + d] : 0.f;
      const float* __restrict kt = j.kt + (h * 2 + side) * D * SP;
      for (int64_t s0 = 0; s0 < SP; s0 += SB) {
        float acc[TB][SB] = {};
        for (int64_t d = 0; d < D; ++d) {
          const float* __restrict kr = kt + d * SP + s0;
          for (int64_t t = 0; t < TB; ++t) {
            const float qd = q[t * D + d];
            for (int64_t s = 0; s < SB; ++s) acc[t][s] += qd * kr[s];
          }
        }
        for (int64_t t = 0; t < TB; ++t) std::memcpy(scs + t * SP + s0, acc[t], sizeof(float) * SB);
      }
      for (int64_t t = 0; t < tn; ++t)
      {
        float* v = topv.data() + (side * TB + t) * K; int32_t* id = topi.data() + (side * TB + t) * K;
        // exact: fall back to the scalar-order topk whenever ties could make the choice or the order differ
        bool ok = topk_fast(scs + t * SP, S, K, id, v);
        if (ok) { sort_desc(v, id, K); for (int64_t k = 1; k < K && ok; ++k) ok = v[k] != v[k - 1]; }
        if (!ok) { topk(scs + t * SP, S, K, id, v); sort_desc(v, id, K); }
      }
    }
    const float it = j.inv_temp[h];
    for (int64_t t = 0; t < tn; ++t) {
      const float *lv = topv.data() + t * K, *rv = topv.data() + (TB + t) * K;
      const int32_t *li = topi.data() + t * K, *ri = topi.data() + (TB + t) * K;
      // top-K of lv[a] + rv[c] over K*K pairs; lists sorted descending -> row a can stop once lv[a] + rv[c] <= current K-th best
      int64_t m = 0, mi = 0; float mn = -INFINITY;
      for (int64_t a = 0; a < K; ++a) {
        if (m == K && lv[a] + rv[0] <= mn) break;
        for (int64_t c = 0; c < K; ++c) {
          const float x = lv[a] + rv[c];
          if (m < K) {
            cv[m] = x; ci[m] = int32_t(a * K + c); ++m;
            if (m == K) { mi = argmin_first(cv.data(), K); mn = cv[mi]; }
            continue;
          }
          if (x <= mn) break;
          cv[mi] = x; ci[mi] = int32_t(a * K + c);
          mi = argmin_first(cv.data(), K); mn = cv[mi];
        }
      }
      float mx = -INFINITY, sum = 0.f;
      for (int64_t k = 0; k < K; ++k) mx = std::max(mx, cv[k]);
      for (int64_t k = 0; k < K; ++k) { w[k] = std::exp((cv[k] - mx) * it); sum += w[k]; }
      const float inv = 1.f / sum;
      float* __restrict o = j.out + (t0 + t) * V;
      for (int64_t k = 0; k < K; ++k) {                  // issue all K row loads first (random rows of a large table)
        const int64_t row = int64_t(li[ci[k] / K]) * S + ri[ci[k] % K];
        const char* p = Q8 ? reinterpret_cast<const char*>(j.values_q + row * V) : reinterpret_cast<const char*>(j.values + row * V);
        for (int64_t off = 0; off < V * int64_t(Q8 ? 1 : 4); off += 64) __builtin_prefetch(p + off);
      }
      for (int64_t k = 0; k < K; ++k) {
        const int64_t row = int64_t(li[ci[k] / K]) * S + ri[ci[k] % K];
        if constexpr (Q8) {
          const int8_t* __restrict vr = j.values_q + row * V;
          const float wk = w[k] * inv * j.scales[row];
          for (int64_t v = 0; v < V; ++v) o[v] += wk * float(vr[v]);
        } else {
          const float* __restrict vr = j.values + row * V;
          const float wk = w[k] * inv;
          for (int64_t v = 0; v < V; ++v) o[v] += wk * vr[v];
        }
      }
    }
  }
}

template <bool Q8>
struct PkmKernel {
  int64_t top_k;
  const float* kt_src = nullptr;   // keys pointer the transposed copy was made from (constant initializer -> made once)
  std::vector<float> kt;
  std::mutex mu;
  PkmKernel(const OrtApi&, const OrtKernelInfo* info) { top_k = Ort::ConstKernelInfo(info).GetAttribute<int64_t>("top_k"); }
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto q = ctx.GetInput(0), keys = ctx.GetInput(1), values = ctx.GetInput(2), temp = ctx.GetInput(Q8 ? 4 : 3);
    auto qs = q.GetTensorTypeAndShapeInfo().GetShape(), ks = keys.GetTensorTypeAndShapeInfo().GetShape(), vs = values.GetTensorTypeAndShapeInfo().GetShape();
    if (qs.size() != 5 || qs[0] != 1 || qs[3] != 2 || ks.size() != 4 || ks[0] != qs[2] || ks[1] != 2 || ks[3] != qs[4] || vs.size() != 2)
      ORT_CXX_API_THROW("PkmRead: bad shapes", ORT_INVALID_ARGUMENT);
    const float* kp = keys.GetTensorData<float>();
    const int64_t H = qs[2], S = ks[2], D = qs[4];
    std::unique_lock<std::mutex> lock(mu);
    const int64_t SP = (S + SB - 1) / SB * SB;
    if (kp != kt_src) {                                  // [H, 2, S, D] -> [H, 2, D, SP] (zero padded)
      kt.assign(H * 2 * SP * D, 0.f);
      for (int64_t hs = 0; hs < H * 2; ++hs)
        for (int64_t s = 0; s < S; ++s)
          for (int64_t d = 0; d < D; ++d) kt[(hs * D + d) * SP + s] = kp[(hs * S + s) * D + d];
      kt_src = kp;
    }
    lock.unlock();
    Job j{q.GetTensorData<float>(), kt.data(), nullptr, temp.GetTensorData<float>(), nullptr, nullptr, nullptr, qs[1], H, S, D, vs[1], top_k, SP};
    if constexpr (Q8) {
      auto sc = ctx.GetInput(3);
      if (sc.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(vs[0])) ORT_CXX_API_THROW("PkmReadQ8: scales must be [N]", ORT_INVALID_ARGUMENT);
      j.values_q = values.GetTensorData<int8_t>(); j.scales = sc.GetTensorData<float>();
    } else j.values = values.GetTensorData<float>();
    if (vs[0] < S * S) ORT_CXX_API_THROW("PkmRead: values must have >= S*S rows", ORT_INVALID_ARGUMENT);
    if (j.K < 1 || j.K > j.S || temp.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(j.H))
      ORT_CXX_API_THROW("PkmRead: bad top_k / inv_temp", ORT_INVALID_ARGUMENT);
    const int64_t T = qs[1];
    std::vector<int64_t> os{1, T, j.V};
    j.out = ctx.GetOutput(0, os.data(), os.size()).GetTensorMutableData<float>();
    if (T) ctx.ParallelFor(block<Q8>, size_t((T + TB - 1) / TB), 0, &j);
  }
};

template <bool Q8>
struct PkmOp : Ort::CustomOpBase<PkmOp<Q8>, PkmKernel<Q8>> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new PkmKernel<Q8>(api, info); }
  const char* GetName() const { return Q8 ? "PkmReadQ8" : "PkmRead"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return Q8 ? 5 : 4; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return Q8 && i == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

}  // namespace

void AddRelPosOps(Ort::CustomOpDomain& domain);   // relpos.cc (built without -ffast-math)

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  static Ort::CustomOpDomain domain{"balbal"};
  static PkmOp<false> op;
  static PkmOp<true> op_q8;
  try {
    domain.Add(&op);
    domain.Add(&op_q8);
    AddRelPosOps(domain);
    Ort::UnownedSessionOptions(options).Add(domain);
  } catch (const Ort::Exception& e) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
  }
  return nullptr;
}
