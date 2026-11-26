"""Check source/config paths without loading weights or running a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=str(ROOT / 'configs/sa2va_8b.json'))
    parser.add_argument('--check-files', action='store_true')
    parser.add_argument('--runtime-imports', action='store_true', help='Check installed model dependencies on CPU')
    args = parser.parse_args()
    from egoafford_sa2va.config import configure
    x = configure(args.config)
    if args.check_files:
        for key in ('upstream', 'backbone', 'sam2', 'train', 'val'):
            if not Path(x[key]).exists():
                parser.error(f'{key} not found: {x[key]}')
    if args.runtime_imports:
        from egoafford_sa2va.model import EgoAffordSa2VA
        from egoafford_sa2va.dataset import EgoAffordDataset
        from egoafford_sa2va.evaluate_internvl import infer_batch
        from egoafford_sa2va.config import build_config
        build_config()
    print(json.dumps(dict(initialization=x['initialization'], prompt=x['prompt'],
          policy=x['policy'], epochs=x['epochs'], effective_batch=x['effective_batch'],
          micro_batch=x['micro_batch'], upstream=x['upstream']), indent=2))


if __name__ == '__main__':
    main()
