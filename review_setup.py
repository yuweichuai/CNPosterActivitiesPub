"""Portable Stata setup; no API access or credentials are required."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_state = {}

def initialize_stata():
    if _state:
        return _state['stata']
    import stata_setup
    configured = os.environ.get('STATA_PATH')
    candidates = [configured] if configured else [
        '/Applications/StataNow', '/Applications/Stata',
        '/usr/local/stata19', '/usr/local/stata18', '/usr/local/stata',
        'C:/Program Files/StataNow19', 'C:/Program Files/Stata19', 'C:/Program Files/Stata18',
    ]
    installation = next((p for p in candidates if p and Path(p).is_dir()), None)
    if installation is None:
        raise RuntimeError('Set STATA_PATH to a licensed Stata installation containing PyStata; see README.md.')
    edition = os.environ.get('STATA_EDITION', 'mp')
    stata_setup.config(installation, edition, splash=False)
    from pystata import stata as engine
    from sfi import Scalar as scalar, Matrix as matrix
    _state.update(stata=engine, Scalar=scalar, Matrix=matrix)
    return engine

class _LazyStataObject:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, name):
        initialize_stata()
        return getattr(_state[self.name], name)

stata = _LazyStataObject('stata')
Scalar = _LazyStataObject('Scalar')
Matrix = _LazyStataObject('Matrix')

def prepare_workspace():
    # All inputs and generated outputs are confined to this review package.
    os.chdir(ROOT)
    for folder in ('figures', 'data', 'notebooks', 'logs'):
        (ROOT / 'outputs' / folder).mkdir(parents=True, exist_ok=True)
