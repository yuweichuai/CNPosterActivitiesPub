"""Run review notebooks in fresh kernels; never overwrite the supplied notebooks."""
import argparse
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', help='Notebook filenames; default: all in numeric order.')
    parser.add_argument('--kernel', default='python3', help='Jupyter kernel with the required Python packages.')
    parser.add_argument('--timeout', type=int, default=-1, help='Seconds per cell; -1 means no time limit.')
    args = parser.parse_args()
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted(
        p for p in ROOT.glob('*.ipynb') if re.match(r'^(?:1_[12]|2_[1-4]|3_[1-5]|4_[1-3]|5)_', p.name)
    )
    for path in paths:
        if path.parent.resolve() != ROOT or not path.is_file() or path.suffix != '.ipynb':
            parser.error('Supply notebook filenames from this package.')
        notebook = nbformat.read(path, as_version=4)
        destination = ROOT / 'outputs/notebooks' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f'Running {path.name}', flush=True)
        client = NotebookClient(notebook, kernel_name=args.kernel, timeout=args.timeout,
                                resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, destination)
        print(f'Saved outputs/notebooks/{path.name}', flush=True)


if __name__ == '__main__':
    main()
