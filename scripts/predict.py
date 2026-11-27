"""Single-image or manifest inference through the paper's native Sa2VA path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/sa2va_8b.json'))
    parser.add_argument('--checkpoint', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image')
    source.add_argument('--manifest')
    parser.add_argument('--action', help='Required with --image')
    parser.add_argument('--output', required=True, help='Directory for masks and predictions.jsonl')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true', help='Also score manifest GT')
    parser.add_argument('--trusted-checkpoint', action='store_true',
                        help='Allow pickle loading of a trusted original MMEngine checkpoint')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.image and not args.action):
        parser.error('Use a positive batch size and provide --action with --image')
    if args.image and args.evaluate:
        parser.error('--evaluate requires a manifest with GT masks')
    output = Path(args.output).resolve()
    if (output / 'predictions.jsonl').exists():
        parser.error('predictions.jsonl already exists; choose a new output directory')
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error('Checkpoint not found: ' + str(checkpoint))

    from egoafford_sa2va.config import configure
    from egoafford_sa2va.common import rows, ROLES, statistics, ground_truth, aggregate, atomic
    x = configure(args.config)
    items = rows(args.manifest) if args.manifest else [dict(
        id=Path(args.image).stem, image=str(Path(args.image).resolve()), action=args.action)]
    if not items or len({r['id'] for r in items}) != len(items):
        parser.error('Manifest must have unique IDs and at least one sample')
    for row in items:
        if not isinstance(row['action'], str) or not row['action'].strip():
            parser.error('Every sample must have a non-empty action')
        if not row.get('image') or not Path(row['image']).is_file():
            parser.error('Image unavailable for ' + row['id'] + '; real images must be obtained and aligned locally')
        if args.evaluate and not ('masks' in row or 'mask_path' in row or 'mask' in row):
            parser.error('--evaluate requires GT masks for every sample')

    import torch
    from PIL import Image
    from egoafford_sa2va.evaluate_internvl import load_native, infer_batch
    if not torch.cuda.is_available():
        parser.error('Native Sa2VA inference requires a CUDA GPU')
    torch.cuda.set_device(0)
    model, processor, _ = load_native(None)
    # Original trainer checkpoints may include non-tensor metadata.
    if checkpoint.suffix == '.safetensors':
        from safetensors.torch import load_file
        payload = load_file(str(checkpoint), device='cpu')
    else:
        payload = torch.load(checkpoint, map_location='cpu', mmap=True,
                             weights_only=not args.trusted_checkpoint)
    state = payload.get('state_dict', payload)
    result = model.load_state_dict(state, strict=False)
    missing = model._ego_trainable_keys.intersection(result.missing_keys)
    if missing or result.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch: {len(missing)} missing trained keys; '
                         f'{len(result.unexpected_keys)} unexpected keys')
    del state, payload
    (output / 'masks').mkdir(parents=True, exist_ok=True)
    records = []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream, \
            (output / 'records.jsonl').open('w', encoding='utf-8') as details:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            results = infer_batch(model, processor, batch, x['prompt'], x['policy'])
            for row, (masks, info) in zip(batch, results):
                key = hashlib.sha256(row['id'].encode('utf-8')).hexdigest()
                paths = {}
                for role, mask in zip(ROLES, masks):
                    name = f'masks/{key}_{role}.png'
                    Image.fromarray(mask.astype('uint8') * 255).save(output / name)
                    paths[role] = name
                stream.write(json.dumps(dict(id=row['id'], masks=paths), ensure_ascii=False) + '\n')
                record = dict(id=row['id'], action=row['action'], **info)
                if args.evaluate:
                    record.update(statistics(masks, ground_truth(row)))
                details.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
            print(f'{min(start + len(batch), len(items))}/{len(items)}', flush=True)
    if args.evaluate:
        metrics = aggregate(records)
        if any(row.get('schema_version') == 'functional_role_annotation/1.0' for row in items):
            # Public IDs are opaque hashes, not the original scene/image hierarchy.
            metrics.pop('num_scenes', None)
        atomic(output / 'metrics.json', metrics)
    atomic(output / 'inference.json', dict(config=Path(args.config).name,
           checkpoint=checkpoint.name, samples=len(items), prompt=x['prompt'], policy=x['policy']))


if __name__ == '__main__':
    main()
