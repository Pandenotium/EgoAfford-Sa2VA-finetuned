"""Portable JSONL manifests with explicit binary mask files."""
import json
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from . import ROLES


def read_jsonl(path):
    records = {}
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = row.get('id')
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError(f'{path}:{n}: missing non-empty id')
        if ident in records:
            raise ValueError(f'{path}:{n}: duplicate id {ident}')
        records[ident] = row
    if not records:
        raise ValueError(f'{path}: no samples')
    return records


def relative_file(root, name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Asset paths must be relative POSIX paths.')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or ':' in p.parts[0]:
        raise ValueError(f'Non-portable asset path: {name}')
    base = Path(root).resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f'Asset resolves outside root: {name}')
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def load_masks(row, root, shape=None, allow_absent=False):
    mapping = row.get('masks')
    if not isinstance(mapping, dict) or set(mapping) != set(ROLES):
        raise ValueError(f'{row.get("id")}: masks must explicitly name O/I/D.')
    masks = []
    for role in ROLES:
        name = mapping[role]
        if name is None and allow_absent:
            if shape is None:
                raise ValueError('Explicit empty predictions need a known image shape.')
            a = np.zeros(shape, dtype=bool)
        else:
            path = relative_file(root, name)
            with Image.open(path) as image:
                a = np.asarray(image)
            if a.ndim != 2 or not np.isin(a, [0, 1, 255]).all():
                raise ValueError(f'{path}: expected a single-channel binary mask, not an RGB overlay.')
            a = a != 0
        if shape is None:
            shape = a.shape
        if a.shape != shape:
            raise ValueError(f'{row.get("id")}/{role}: mask size mismatch; resizing is not implicit.')
        masks.append(a)
    return np.stack(masks)


def validate_manifest(path, root, check_images=False):
    records = read_jsonl(path)
    groups = set()
    for row in records.values():
        if not isinstance(row.get('action'), str) or not row['action'].strip():
            raise ValueError(f'{row["id"]}: missing action')
        if row.get('source') not in ('real', 'synthetic'):
            raise ValueError(f'{row["id"]}: source must be real or synthetic')
        if not isinstance(row.get('group_id'), str) or not row['group_id']:
            raise ValueError(f'{row["id"]}: missing episode/task group_id')
        groups.add((row['source'], row['group_id']))
        masks = load_masks(row, root)
        if check_images:
            with Image.open(relative_file(root, row.get('image'))) as image:
                if image.size != masks.shape[:0:-1]:
                    raise ValueError(f'{row["id"]}: image and masks have different dimensions')
    return dict(num_samples=len(records), num_groups=len(groups))
