import argparse
import json
from pathlib import Path
from .metrics import Metrics
from .data import read_jsonl, load_masks, validate_manifest


def evaluate(gt_file, pred_file, data_root, prediction_root):
    gt, pred = read_jsonl(gt_file), read_jsonl(pred_file)
    if gt.keys() != pred.keys():
        raise ValueError(f'ID mismatch: {len(gt.keys()-pred.keys())} missing, {len(pred.keys()-gt.keys())} unexpected predictions')
    scores = Metrics()
    for ident, row in gt.items():
        g = load_masks(row, data_root)
        p = load_masks(pred[ident], prediction_root, g.shape[1:], allow_absent=True)
        scores.update(p, g)
    return scores.compute()


def main():
    parser = argparse.ArgumentParser(description='EgoAfford fixed O/I/D evaluation. Scores are fractions.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--ground-truth', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--prediction-root', required=True)
    p.add_argument('--output')
    p = sub.add_parser('evaluate-counts', help='Recompute metrics from recorded intersections/unions and presence flags')
    p.add_argument('records', nargs='+')
    p.add_argument('--output')
    p = sub.add_parser('validate')
    p.add_argument('manifest')
    p.add_argument('--root', required=True)
    p.add_argument('--check-images', action='store_true')
    p = sub.add_parser('visualize')
    p.add_argument('manifest')
    p.add_argument('--root', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            result = validate_manifest(args.manifest, args.root, args.check_images)
        elif args.command == 'evaluate':
            result = evaluate(args.ground_truth, args.predictions, args.data_root, args.prediction_root)
        elif args.command == 'visualize':
            from .visualize import save_overlay
            save_overlay(read_jsonl(args.manifest)[args.id], args.root, args.output)
            result = dict(output=args.output)
        else:
            scores, seen = Metrics(), set()
            for name in args.records:
                for ident, row in read_jsonl(name).items():
                    if ident in seen:
                        raise ValueError(f'Duplicate id across record files: {ident}')
                    seen.add(ident)
                    scores.update_counts(row['intersection'], row['union'], row['gt_present'], row['pred_present'])
            result = scores.compute()
    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
    text = json.dumps(result, indent=2, allow_nan=False)
    if getattr(args, 'output', None) and args.command != 'visualize':
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
