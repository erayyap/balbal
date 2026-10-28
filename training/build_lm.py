"""6-gram token KenLM for the beam decoder (MODEL_DIR/lm), built on the transcripts of the rows the acoustic model was trained on
(+ optional plain text, one sentence per line).

Tokens are the acoustic model's SentencePiece pieces (piece id i written as the character chr(i + 100)), so the LM
scores exactly the decoder's tokens. Recipe: whitespace-normalised text, each (id, pcm_sha256) once; rows whose
normalize_turkish(text) equals any held-out text (dev/test/control manifests) are excluded; lmplz -o 6 -S 4G
--discount_fallback --prune 0 0 1 1 2 2; build_binary trie; ARPA vs binary parity probe (< 1e-4).
Needs KenLM's lmplz and build_binary (https://github.com/kpu/kenlm; set KENLM_BIN to their directory) and the kenlm module.
  python training/build_lm.py --tokenizer TOKENIZER_DIR/tokenizer.model --train a.jsonl b.jsonl --heldout dev.jsonl test.jsonl \
      --out MODEL_DIR/lm [--scratch DIR]
Writes OUT/model.binary, OUT/tokenizer.model, OUT/complete.json.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(l) for l in open(path)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--train', type=Path, nargs='+', required=True)
    p.add_argument('--text', type=Path, nargs='*', default=[], help='plain text files, one sentence per line')
    p.add_argument('--heldout', type=Path, nargs='*', default=[])
    p.add_argument('--quantize', action='store_true', help='8-bit quantized trie (-q 8 -b 8 -a 22)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scratch', type=Path, default=None, help='temporary dir for the corpus/ARPA (needs several GB)')
    p.add_argument('--memory', default='4G', help='lmplz -S')
    args = p.parse_args()
    import kenlm
    import sentencepiece as spm
    from balbal.normalizer.turkish import normalize_turkish
    tools = Path(os.getenv('KENLM_BIN', ''))
    tool = lambda name: str(tools / name) if str(tools) else (shutil.which(name) or name)
    args.out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    blocked = {normalize_turkish(r['text']) for path in args.heldout for r in rows(path)}
    scratch = Path(tempfile.mkdtemp(prefix='kenlm-', dir=args.scratch)); corpus = scratch / 'corpus.txt'
    counts, excluded, seen, probes, token_count = Counter(), Counter(), set(), [], 0
    try:
        with corpus.open('w') as target:
            for manifest in args.train:
                for row in rows(manifest):
                    ident = (row['id'], row.get('pcm_sha256'))
                    if ident in seen: continue
                    seen.add(ident); text = ' '.join(row['text'].split()); source = row.get('source', '?')
                    if not text or normalize_turkish(text) in blocked: excluded[source] += 1; continue
                    ids = sp.encode(text)
                    if not ids or sp.unk_id() in ids: raise ValueError('Unrepresentable LM target: ' + row['id'])
                    encoded = ' '.join(chr(i+100) for i in ids); target.write(encoded + '\n')
                    counts[source] += 1; token_count += len(ids)
                    if len(probes) < 48: probes.append(encoded)
            for path in args.text:
                for line in open(path, encoding='utf-8'):
                    text = ' '.join(line.split())
                    if not text or normalize_turkish(text) in blocked: excluded[path.name] += 1; continue
                    ids = sp.encode(text)
                    if not ids or sp.unk_id() in ids: excluded[path.name] += 1; continue
                    encoded = ' '.join(chr(i+100) for i in ids); target.write(encoded + '\n')
                    counts[path.name] += 1; token_count += len(ids)
        arpa, binary = scratch / 'model.arpa', scratch / 'model.binary'
        with corpus.open('rb') as source, (args.out / 'build.log').open('a') as log:
            subprocess.run([tool('lmplz'), '-o', '6', '-S', args.memory, '-T', str(scratch) + '/', '--arpa', str(arpa),
                            '--discount_fallback', '--prune', '0', '0', '1', '1', '2', '2'],
                           stdin=source, stdout=log, stderr=log, check=True)
            subprocess.run([tool('build_binary'), *(['-q', '8', '-b', '8', '-a', '22'] if args.quantize else []), 'trie', str(arpa), str(binary)], stdout=log, stderr=log, check=True)
        a, b = kenlm.Model(str(arpa)), kenlm.Model(str(binary))
        delta = max(abs(a.score(q) - b.score(q)) for q in probes); assert delta < (.05 if args.quantize else 1e-4), delta; del a, b
        shutil.copy2(args.tokenizer, args.out / 'tokenizer.model'); shutil.copy2(binary, args.out / 'model.binary')
        result = dict(status='complete', binary_sha256=sha(args.out / 'model.binary'), tokenizer_sha256=sha(args.tokenizer),
                      train_hashes={p.name: sha(p) for p in args.train + args.text}, quantized=args.quantize, heldout_hashes={p.name: sha(p) for p in args.heldout},
                      source_counts=dict(counts), tokens=token_count, excluded_exact_heldout_or_empty=dict(excluded),
                      corpus_sha256=sha(corpus), arpa_binary_max_error=delta, prune=[0, 0, 1, 1, 2, 2])
        (args.out / 'complete.json').write_text(json.dumps(result, indent=1) + '\n')
        print(json.dumps(result, indent=1))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
