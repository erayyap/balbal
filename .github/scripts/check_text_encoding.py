"""Text files opened without an encoding: open() / Path.open() in text mode, Path.read_text() / write_text(), and subprocess text
output, with no encoding= argument, use the locale's code page on Windows (cp1252, cp1254...) and break on UTF-8 manifests,
transcripts and pages. Lists every such call in the repository's Python code; exit status 1 when there is any.
  python .github/scripts/check_text_encoding.py [paths...]      (default: the whole repository except balbal-v1/)"""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SKIP = {'balbal-v1', '.git', 'third_party', 'build', 'dist', '.venv', 'venv'}
BINARY_MODULES = {'gzip', 'bz2', 'lzma', 'tarfile', 'zipfile', 'wave', 'sf', 'soundfile', 'Image', 'webbrowser', 'os', 'h5py', 'tf'}


def mode_of(call, position):
    """The mode argument (keyword or the given position) if it is a string literal; 'r' when absent; None when not literal."""
    for k in call.keywords:
        if k.arg == 'mode': return k.value.value if isinstance(k.value, ast.Constant) and isinstance(k.value.value, str) else None
    if len(call.args) > position:
        a = call.args[position]
        return a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
    return 'r'


def problems(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call): continue
        keywords = {k.arg for k in node.keywords}
        if 'encoding' in keywords or None in keywords: continue   # encoding given, or **kwargs
        f = node.func
        if isinstance(f, ast.Name) and f.id == 'open':
            mode = mode_of(node, 1)
        elif isinstance(f, ast.Attribute) and f.attr == 'open' and not (isinstance(f.value, ast.Name) and f.value.id in BINARY_MODULES):
            if isinstance(f.value, ast.Name) and f.value.id == 'io': mode = mode_of(node, 1)
            else: mode = mode_of(node, 0)
        elif isinstance(f, ast.Attribute) and f.attr in ('read_text', 'write_text'):
            mode = 'r'
        elif isinstance(f, ast.Attribute) and f.attr in ('run', 'check_output', 'Popen') and (
                any(k.arg in ('text', 'universal_newlines') and not (isinstance(k.value, ast.Constant) and k.value.value is False)
                    for k in node.keywords)):
            mode = 'r'
        else:
            continue
        if mode is not None and 'b' in mode: continue
        yield node.lineno, ast.unparse(node)[:110]


def main(paths):
    files = []
    for p in paths or [ROOT]:
        p = Path(p).resolve()
        files += [p] if p.is_file() else [f for f in sorted(p.rglob('*.py')) if not SKIP & set(f.relative_to(p).parts)]
    found = 0
    for f in files:
        for line, text in problems(f):
            print(f'{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}:{line}: {text}'); found += 1
    print(f'{found} text open(s) without an encoding in {len(files)} files')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
