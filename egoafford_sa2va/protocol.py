"""Explicit Sa2VA protocol; data audit and scoring are shared snapshots."""
import re,os
import numpy as np
from .common import ROLES, SYSTEM, ground_truth, repair, box, present

SEG = '[SEG]'
SPECIAL_TOKENS = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

def validate_options(mode, policy):
    if mode not in ('minimal_user', 'full_rule') or policy not in ('emit_empty', 'omit_empty'):
        raise ValueError((mode, policy))

def prompt(action, mode, policy):
    validate_options(mode, policy)
    text = f"Action: '{action}'"
    if mode == 'full_rule':
        rule = ('Append one native segmentation token to each role line, including absent roles.'
                if policy == 'emit_empty' else
                'Append one native segmentation token only for present roles. For None roles output the name and zero box without a segmentation token.')
        token = '<segmentation token>' if policy == 'emit_empty' else '<segmentation token only if present>'
        text += ('\nSegment three functional components: direct_object (the manipulated object), instrument (the tool; None for bare hands), destination (the receiving target; None without transfer).'
                 '\nIMPORTANT RULES:\n- Segment the functional part, not the whole object.'
                 '\n- Output each role name and bounding box in original-image pixels; use None and [0, 0, 0, 0] if absent.'
                 '\n- ' + rule + '\nOUTPUT FORMAT:\nComponents:\n' +
                 '\n'.join(f'{r}: <name>; box: [x1, y1, x2, y2] {token}' for r in ROLES))
    if os.environ.get('EGOAFFORD_SUPERVISION','full')=='no_name':
        if policy!='emit_empty':raise ValueError('No-name ablation is emit-only')
        text=text.replace('Output each role name and bounding box in original-image pixels; use None and [0, 0, 0, 0] if absent.', 'Output each role bounding box in original-image pixels; use [0, 0, 0, 0] if absent.')
        text=text.replace(': <name>; box:', ': box:')
    if SEG in text:
        raise ValueError('Input may not contain native segmentation tokens')
    return text

def role_indices(answer, policy, allow_zero=False):
    """Strict training/format parser. Indices follow SEG occurrence order, not GT."""
    validate_options('minimal_user', policy)
    lines = answer.strip().splitlines()
    if len(lines) != 4 or lines[0] != 'Components:':
        raise ValueError('Expected Components header and exactly three role lines')
    if os.environ.get('EGOAFFORD_SUPERVISION','full')=='no_name':
        if policy!='emit_empty':raise ValueError('No-name ablation is emit-only')
        for role,line in zip(ROLES,lines[1:]):
            if not re.fullmatch(re.escape(role)+r': box: \[(\d+), (\d+), (\d+), (\d+)\] \[SEG\]',line):
                raise ValueError('Malformed no-name role line: '+line)
        if answer.count(SEG)!=3:raise ValueError('Expected exactly three SEG tokens')
        return [0,1,2]
    active = []
    for i, (role, line) in enumerate(zip(ROLES, lines[1:])):
        pattern = re.escape(role) + r': ([^\n;]+); box: \[(\d+), (\d+), (\d+), (\d+)\]( \[SEG\])?'
        match = re.fullmatch(pattern, line)
        if not match:
            raise ValueError(f'Malformed or reordered role line: {line}')
        name = match[1]
        if SEG in name:
            raise ValueError('SEG in role name')
        absent = name == 'None'
        coords = list(map(int, match.group(2, 3, 4, 5)))
        if absent and coords != [0, 0, 0, 0]:
            raise ValueError('None must have zero box')
        emit = bool(match[6])
        if emit != (policy == 'emit_empty' or not absent):
            raise ValueError('SEG presence does not match declared role/policy')
        if emit:
            active.append(i)
    if answer.count(SEG) != len(active) or (not active and not allow_zero):
        raise ValueError('Invalid SEG count')
    return active

def target(row, policy):
    masks = ground_truth(row).copy()
    names = dict(row['names'])
    if row.get('training_policy') == 'alt':
        names, masks = repair(names, masks)
    text = 'Components:\n' + '\n'.join(
        (f'{r}: box: {[int(x) for x in box(m)]}' if os.environ.get('EGOAFFORD_SUPERVISION','full')=='no_name' else f'{r}: {names[r]}; box: {[int(x) for x in box(m)]}') +
        (' ' + SEG if policy == 'emit_empty' or present(names[r]) else '')
        for r, m in zip(ROLES, masks))
    indices = list(range(3)) if policy == 'emit_empty' else [i for i,r in enumerate(ROLES) if present(names[r])]
    return text, masks, indices

def map_prediction(answer, decoded_masks, shape, policy):
    """GT-independent mapping, retain identifiable masks even if bbox text is malformed."""
    validate_options('minimal_user', policy)
    pred = np.zeros((3, *shape), dtype=bool)
    valid = [False] * 3
    spans = list(re.finditer(re.escape(SEG), answer))
    if len(spans) != len(decoded_masks):
        return pred, valid, False
    used = []
    for i, role in enumerate(ROLES):
        lines = list(re.finditer(r'(?m)^' + role + r':([^\n]*)', answer))
        if len(lines) != 1:
            continue
        line = lines[0]
        ids = [j for j, s in enumerate(spans) if line.start() <= s.start() < line.end()]
        absent = bool(re.match(re.escape(role) + r':\s*None\s*;\s*box:', line.group()))
        if len(ids) == 1:
            mask = np.asarray(decoded_masks[ids[0]])
            if mask.shape != tuple(shape):
                raise ValueError('Decoded mask shape mismatch')
            pred[i] = mask.astype(bool)
            used.extend(ids)
            valid[i] = True
        elif not ids and absent and policy == 'omit_empty':
            valid[i] = True
    try:
        role_indices(answer, policy, allow_zero=True)
        fmt = all(valid) and len(used) == len(spans)
    except ValueError:
        fmt = False
    return pred, valid, fmt
