"""Path setup for the application and simulation scripts: puts code/lib on the
import path and sets the output folder (output/<folder of the script>, or the
folder in the environment variable IIV_OUT)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
OUT = os.environ.get('IIV_OUT', os.path.join(HERE, '..', '..', 'output', os.path.basename(HERE)))
os.makedirs(OUT, exist_ok=True)


def out(name):
    return os.path.join(OUT, name)
