"""Execute notebooks 01-04 in order, in place, so their outputs are always fresh --
this is the "run the pipeline" entrypoint: no manual per-notebook clicking.

Usage:
    python run_notebooks.py                  # run everything, 01 through 04
    python run_notebooks.py --from 03_features  # skip earlier stages (e.g. raw data is already fresh)
    python run_notebooks.py --timeout 3600    # raise the per-notebook timeout (01's yfinance pull is slow)
"""
import argparse
import pathlib
import subprocess
import sys

NOTEBOOKS = ['01_data_ingestion', '02_eda', '03_features', '04_models']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', default=NOTEBOOKS[0], choices=NOTEBOOKS,
                     help='first notebook to run (skip earlier ones)')
    ap.add_argument('--timeout', type=int, default=1800, help='per-notebook execution timeout, seconds')
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    start_idx = NOTEBOOKS.index(args.start)

    for name in NOTEBOOKS[start_idx:]:
        path = root / 'notebooks' / f'{name}.ipynb'
        print(f'--- running {path.name} ---', flush=True)
        result = subprocess.run([
            sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
            f'--ExecutePreprocessor.timeout={args.timeout}', str(path),
        ])
        if result.returncode != 0:
            print(f'FAILED: {path.name} (exit {result.returncode}) -- stopping.', file=sys.stderr)
            sys.exit(result.returncode)

    print('all notebooks executed successfully.')


if __name__ == '__main__':
    main()
