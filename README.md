# balbal-v1 — Türkçe konuşma tanıma (kod)

Bu depo **balbal-v1** Türkçe konuşma tanıma modelinin çıkarım (inference), metin normalizasyonu ve eğitim kodunu içerir.

> **Ağırlıklar şu an yayımlanmıyor.** Model ağırlıkları, dil modeli (KenLM) ve eğitim verisi bu depoda yoktur.
> Kod, ağırlıkları sizin vereceğiniz bir klasörden (`MODEL_DIR`) okur.
> Sentetik veri üretim kodu (TTS ile veri üretimi, etiket üretimi) bu depoya dahil değildir.

## Mimari

| Parça | Ayrıntı |
|---|---|
| Akustik model | Conformer-CTC (NeMo `EncDecCTCModelBPE`), 16 katman, d_model 176, 80 mel, 4x alt örnekleme (40 ms kare) |
| Döngülü düzeltici | Kodlayıcı çıktısının üstünde 2 katmanlı Transformer düzeltici, **3 geçiş** (her geçiş bir önceki CTC dağılımını girdi alır) |
| Akustik bellek | Ürün-anahtar (product-key) bellek: 500×500 alt anahtar, 4 kafa, top-16, değer boyutu 120, 3 karelik bağlam |
| Sözlük | SentencePiece 1024 parça + CTC boşluk (1025 sınıf) |
| Dil modeli | 6-gram **token** KenLM (SentencePiece parçaları üzerinde) |
| Kod çözücü | flashlight-text sözlüksüz ışın araması: alpha 0.75, beta 1.5, sıcaklık 1.2, ışın 64, top-12 token, eşik 8 |
| Parametre | **45.4M saklanan / 15.4M aktif** (bellek tablosunun çoğu her karede okunmaz) |

Uzun ses 30 saniyelik sabit parçalara bölünür (VAD yok); her parça ayrı çözülür.

## Sonuçlar

| Değerlendirme | WER |
|---|---|
| FLEURS tr (test), v2 puanlama, KenLM | **6.54** |
| Mu-Bench tr-TR, resmi puanlama | **11.93** |

v2 puanlama: `normalizer/metrics_v2.py` (aşağıya bakın).

## Depo yapısı

```
balbal_model/   model tanımı: döngülü CTC düzeltici + akustik bellek (NeMo Conformer'a eklenir)
configs/        model_config.yaml (NeMo), refiner.json (düzeltici/bellek), decoder.json (kod çözücü ayarları)
inference/      PyTorch/NeMo çıkarımı, ONNX CPU çalışma zamanı, KenLM kod çözücü, ONNX dışa aktarma
normalizer/     Türkçe normalizasyon, v2 puanlama, rakam→kelime (eğitim etiketleri), ITN (kelime→rakam)
training/       eğitim döngüsü, Muon+AdamW, veri/augmentasyon, ağırlık çorbası, KenLM kurulumu, kalibrasyon
```

## MODEL_DIR düzeni

```
MODEL_DIR/
  conformer.pt                     PyTorch state dict (balbal-v1)
  tokenizer/tokenizer.model        SentencePiece
  tokenizer/tokenizer.vocab
  tokenizer/vocab.txt
  lm/model.binary                  6-gram token KenLM (trie)
  lm/tokenizer.model               aynı SentencePiece
  model-ext.onnx, model-ext.data   (yalnızca ONNX çalışma zamanı için; export_onnx.py üretir)
```

`--model-dir MODEL_DIR` ya da `BALBAL_MODEL_DIR` ortam değişkeni ile verilir.

## Çıkarım

Linux x86-64, Python 3.12.

**PyTorch/NeMo (CPU veya GPU):**

```bash
python -m venv .venv && . .venv/bin/activate
python inference/setup_env.py --device cpu --model-dir MODEL_DIR   # torch 2.10 + sabitlenmiş NeMo, kod çözücüyü derler
python inference/transcribe.py --model-dir MODEL_DIR ses.wav --device cpu
```

Çıktı dosya başına bir JSON satırı (`text`, parçalar, süreler); `--text-only` yalnızca metni yazar.
WAV/FLAC/OGG doğrudan, diğer biçimler FFmpeg ile okunur; 16 kHz'e eğitimdeki ile aynı yeniden örnekleyici (scipy polyphase) kullanılır.

**ONNX (yalnızca CPU, torch/NeMo gerekmez):**

```bash
python inference/export_onnx.py --model-dir MODEL_DIR ornek1.wav ornek2.wav   # bir kez, PyTorch ortamında
pip install -r inference/requirements-onnx.txt && python inference/build_native.py
python inference/runtime_onnx.py --model-dir MODEL_DIR ses.wav                  # ASR_THREADS ile iş parçacığı
```

`export_onnx.py` mel ön ucunu DFT konvolüsyonu olarak, düzeltici dikkatini ve bellek okumayı şekilden bağımsız işlemler olarak
yeniden yazar; verdiğiniz kliplerde NeMo modeliyle farkı (`export.json`) ölçer ve ağırlıkları 64 KiB hizalı ayrı bir dosyaya
taşır (onnxruntime bellek eşlemesiyle okur).

Kod çözücü testi (yapay girdiyle): `python inference/self_test.py --model-dir MODEL_DIR`.

Python içinden:

```python
import sys; sys.path.insert(0, 'inference')
from transcribe import Transcriber
asr = Transcriber('MODEL_DIR', device='cpu')
print(asr.transcribe('ses.wav')['text'])
```

## Normalizer

- `normalizer/turkish.py` — `normalize_turkish()`: küçük harf (Türkçe İ/ı), noktalama, sayılar/semboller/e-posta sözcükleri; v1 puanlama
  ve dil modelinden tutulan (held-out) cümleleri dışlamak için tam metin anahtarı.
- `normalizer/metrics_v2.py` — v2 puanlama: v1 + dolgu sesleri (ee, hı), "ile" kaynaşması, şapka, sıra sayıları, birimler,
  kesme işareti, bitişik/ayrı yazım farkları, e-postada "at/et", harf-rakam kodları ve **sayıların rakam rakam karşılaştırılması**
  (`doksan` = `9 0` = `dokuz sıfır`). `corpus_metrics_v2(refs, hyps)` WER/CER döndürür.
- `normalizer/numbers.py` — `verbalize()`: eğitim etiketlerindeki rakamları kelimeye çevirir (`180'e` → `yüz seksene`);
  belirsiz biçimlerde `None` döner ve satır atılır. balbal-v1 **yalnızca kelime** etiketleriyle eğitildi, rakam yazmaz.
- `normalizer/label_clean.py` — gerçek konuşma etiketlerini temizler (sahne notları, kekeleme, sembol/URL, saniyede >25 karakter,
  5'ten fazla tekrar eden metin).
- `normalizer/itn.py` — ters normalizasyon: `itn_text('sıfır beş yüz otuz iki')` → `0 532`; `itn_digits()` puanlama için
  rakam rakam biçim.

## Eğitim

balbal-v1, önceki aşamanın ağırlıklarından başlayan bir devam eğitimidir (bu aşama: 9000 güncelleme, yayımlanan = adım 6000
ham ağırlıkları):

```bash
TEL_P=0.5 EDGE_P=0.5 LOADER_WORKERS=8 python training/train.py runs/balbal-v1 \
    --init ONCEKI_ASAMA.pt --tokenizer-dir MODEL_DIR/tokenizer \
    --train train.jsonl --dev dev.jsonl \
    --updates 9000 --lr 1e-4 --eval-every 3000 --soup 0.5 --micro 16
```

- **Veri** (`training/data.py`): JSONL satırları (`audio_path`, `pcm_sha256`, `duration_seconds`, `text`, `source`) veya aynı kaydın
  ardışık parçalarını anında birleştiren sanal satırlar (`parts`). CTC'ye sığmayan klipler atılır.
- **Augmentasyon**: telefon bandı (`TEL_P`: 8 kHz, 300 Hz yüksek geçiren, mu-law), kenar biçimi (`EDGE_P`: sessiz kenarları kırp,
  0–0.6 s dijital sıfır ekle), NeMo SpecAugment.
- **Hedef**: 0.3 × CTC(kodlayıcı) + 0.7 × ortalama CTC(3 düzeltici geçişi), bf16.
- **Optimizasyon**: Muon (gizli doğrusal katmanlar, kafa başına) + AdamW (geri kalanı; bellek değer tablosu 5× öğrenme oranı),
  10 adım ısınma + kosinüs (1 → 0.1), gradyan kırpma 1.0, güncelleme başına 32 klip.
- **Değerlendirme**: her `--eval-every` adımda `--dev` üzerinde greedy v2 WER (ham ve `--soup` çorbası); en iyi 5 kontrol noktası
  saklanır. (Orijinal koşu kontrol noktalarını ayrıca KenLM ile iç DEV kümesinde sıraladı; o kısım bu depoda yok.)
- **Dil modeli**: `training/build_lm.py` — modelin gördüğü tüm eğitim metinleri üzerinde 6-gram token KenLM; held-out metinler tam
  eşleşmeyle dışlanır.
- **Kalibrasyon**: `training/calibrate.py` — DEV'de alpha × sıcaklık 3×3 ızgara; varsayılan ayar ancak eşleştirilmiş bootstrap
  %95 güven aralığı tamamen 0'ın altındaysa değişir (balbal-v1'de değişmedi).
- **Çorba**: `training/soup.py OUT.pt 0.5:a.pt 0.5:b.pt`.

ROCm'da eğitim adımlarında MIOpen kapalıdır (yaklaşık %41 daha hızlı); değerlendirmede açıktır.

## Lisans

Kod [Apache License 2.0](LICENSE) ile lisanslanmıştır (Copyright 2026 Eray Yapağcı). Lisans yalnızca bu depodaki kodu kapsar;
model ağırlıkları bu lisansa dahil değildir ve henüz yayımlanmadı. Üçüncü taraf lisansları: `inference/licenses/`
(flashlight-text, KenLM).

---

## English summary

Code for **balbal-v1**, a Turkish speech recognizer: Conformer-CTC (16 layers, d_model 176, SentencePiece 1024) with a 2-layer
refiner run for 3 passes and a product-key acoustic memory (45.4M stored / 15.4M active parameters), decoded with a 6-gram token
KenLM (flashlight-text, alpha 0.75, beta 1.5, T 1.2, beam 64). Results: FLEURS tr test **6.54 WER** (v2 scoring),
Mu-Bench tr-TR **11.93 WER** (official scoring).

**Weights are not released yet**; the code loads them from a `MODEL_DIR` you provide (layout above). Synthetic-data generation
code is not included.

- `inference/` — PyTorch/NeMo transcriber (`transcribe.py`), torch-free ONNX CPU runtime (`runtime_onnx.py`), ONNX export
  (`export_onnx.py`), KenLM beam decoder with a small native extension (`build_native.py`).
- `normalizer/` — Turkish normalization, v2 scoring (`corpus_metrics_v2`), digit verbalization for words-only labels, label
  cleaning, inverse text normalization (number words → digits).
- `training/` — the training loop of the final balbal-v1 stage (Muon + AdamW, telephone/edge augmentation, 0.3/0.7 base/refiner CTC
  objective), weight soups, KenLM build, decoder calibration. The original run also ranked checkpoints with KenLM on an internal
  DEV set; here checkpoint ranking uses greedy v2 WER on your `--dev`.

License: the code is under the [Apache License 2.0](LICENSE) (Copyright 2026 Eray Yapağcı). It covers the code in this
repository only; the model weights are not covered and are not released yet. Third-party licenses: `inference/licenses/`.
