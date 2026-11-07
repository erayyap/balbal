"""The first N clips (default 8) of FLEURS Turkish test (google/fleurs, CC BY 4.0) for the CI model tests, read from the start of the
archive without downloading all of it.
  python .github/scripts/fleurs_clips.py OUT_DIR [N]"""
from pathlib import Path
import sys
import tarfile
import urllib.request

URL = 'https://huggingface.co/datasets/google/fleurs/resolve/main/data/tr_tr/audio/test.tar.gz'


def main(out, n=8):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); count = 0
    with urllib.request.urlopen(URL, timeout=120) as response, tarfile.open(fileobj=response, mode='r|gz') as archive:
        for member in archive:
            if member.isfile() and member.name.endswith('.wav'):
                (out / Path(member.name).name).write_bytes(archive.extractfile(member).read()); count += 1
                if count == n: break
    print(f'{count} clips in {out}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
