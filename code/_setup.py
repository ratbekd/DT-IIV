"""Path setup shared by all scripts: adds code/lib to the import path and
returns the output directory for the calling script's folder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))


def outdir(sub):
    d = os.path.join(HERE, '..', 'output', sub)
    os.makedirs(d, exist_ok=True)
    return d


class Tee:
    """Print to the console and to a results file."""
    def __init__(self, path):
        self.f = open(path, 'w')
    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()
