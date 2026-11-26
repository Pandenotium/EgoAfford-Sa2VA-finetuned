import json
import os
import re
from functools import lru_cache
from pathlib import Path
import numpy as np

ROLES = ('direct_object', 'instrument', 'destination')
LABELS = ('direct object', 'instrument', 'destination')
ALLOWED = {(1,0,0),(1,1,0),(1,0,1),(0,1,1),(1,1,1)}
SYSTEM = 'You are a helpful assistant that can see images and perform reasoning segmentation.'
QUESTION = """Action: '{action}'
Segment the following three functional components by bounding boxes for the action step:
- The Direct Object (the object being manipulated).
- The Instrument (the tool used to perform the action, absent if performed by hands).
- The Destination (the target location or container of transfer, absent if the action means no transfer).
A component could be absent (by putting 'None') if not used.

IMPORTANT RULES:
- segment the functional part, not the whole object.
- Output each component's name and bounding box in original-image pixels. Use None and [0, 0, 0, 0] if absent.

OUTPUT FORMAT:
Components:
direct object: <name or None>, [x1, y1, x2, y2]
instrument: <name or None>, [x1, y1, x2, y2]
destination: <name or None>, [x1, y1, x2, y2]
"""

def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def rows(path):
    out=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    root=Path(os.environ.get('EGOAFFORD_DATA_ROOT', Path(path).resolve().parent)).resolve()
    for row in out:
        for key in ('image', 'mask_path'):
            if row.get(key):
                row[key]=str((root/row[key]).resolve())
        if 'masks' in row:
            row['masks']={k: str((root/v).resolve()) if v is not None else None for k,v in row['masks'].items()}
        if 'source' in row and 'data_source' not in row:
            row['data_source']=row['source']
    return out

@lru_cache(maxsize=1)
def sparse(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(z['shape']);m=np.zeros(shape,dtype=bool)
        idx=z['nonempty_idx']
        if idx.size:
            ns=tuple(z['nonempty_shape']);v=np.unpackbits(z['packed'])[:int(np.prod(ns))].reshape(ns)
            m[idx[:,0],idx[:,1]]=v
    m.flags.writeable=False
    return m

def present(name):
    if name is None:return False
    if not isinstance(name,str):raise ValueError('Malformed role name')
    return name.strip().casefold() not in ('','none','null')

def repair(names,masks):
    # No mutation of file-backed/cached annotations or action text.
    n=dict(names);m=masks.copy()
    for j,r in enumerate(ROLES):
        if not present(n[r]):m[j]=False;n[r]='None'
        elif not m[j].any():n[r]='None'
    return n,m

def decisions(names,masks):
    t=tuple(present(names[r]) for r in ROLES);v=tuple(bool(x.any()) for x in masks)
    return dict(strict=t==v and v in ALLOWED,alt=tuple(a and b for a,b in zip(t,v)) in ALLOWED,
                text=list(t),mask=list(v))

def box(mask):
    y,x=np.where(mask)
    return [float(x.min()),float(y.min()),float(x.max()),float(y.max())] if len(x) else [0.,0.,0.,0.]

def answer(names,masks):
    return 'Components:\n'+''.join(f'{label}: {names[r]}, {box(m)}\n' for label,r,m in zip(LABELS,ROLES,masks))

def messages(action):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTION.format(action=action)}]}]

def format_valid(reply):
    pattern=r'^Components:\s*\n'+r'\s*\n'.join(re.escape(k)+r':\s*[^\n]+,\s*\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3}\s*\]' for k in LABELS)+r'\s*$'
    return bool(re.fullmatch(pattern,reply.strip()))

def ground_truth(row):
    if 'masks' in row:
        return _png_masks(row)
    return sparse(row['mask_path'])[row['mask_index']]

def _png_masks(row):
    from PIL import Image
    masks=[]
    for role in ROLES:
        with Image.open(row['masks'][role]) as image:
            mask=np.asarray(image)
        if mask.ndim!=2 or not np.isin(mask, [0,1,255]).all():
            raise ValueError('GT masks must be single-channel binary PNGs')
        masks.append(mask.astype(bool))
    return np.stack(masks)

def statistics(pred,gt):
    inter=(pred&gt).sum((1,2));union=(pred|gt).sum((1,2))
    iou=np.divide(inter,union,out=np.ones(3),where=union!=0)
    return dict(intersection=inter.tolist(),union=union.tolist(),iou=iou.tolist(),
                gt_present=gt.any((1,2)).tolist(),pred_present=pred.any((1,2)).tolist())

def aggregate(records):
    inter=np.array([r['intersection'] for r in records]);union=np.array([r['union'] for r in records]);iou=np.array([r['iou'] for r in records])
    gp=np.array([r['gt_present'] for r in records]);pp=np.array([r['pred_present'] for r in records])
    component={};counts=[]
    for j,r in enumerate(ROLES):
        tp=int((gp[:,j]&pp[:,j]).sum());fp=int((~gp[:,j]&pp[:,j]).sum());fn=int((gp[:,j]&~pp[:,j]).sum());den=2*tp+fp+fn
        component[r]=dict(NE_mIoU=float(iou[gp[:,j],j].mean()) if gp[:,j].any() else None,gt_present=int(gp[:,j].sum()),TP=tp,FP=fp,FN=fn,
                          presence_F1=2*tp/den if den else None)
        counts.append((tp,fp,fn))
    tp,fp,fn=np.array(counts).sum(0);f1=[v['presence_F1'] for v in component.values() if v['presence_F1'] is not None]
    return dict(protocol='given-action-canonical-OID-v1',num_samples=len(records),num_scenes=len({r['id'].rsplit('/',1)[0] for r in records}),
                gIoU=float(iou.mean()),cIoU=float(inter.sum()/union.sum()) if union.sum() else 1.,component_wise=component,
                presence_macro_F1=float(np.mean(f1)) if f1 else None,presence_micro_F1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None,
                format_valid_rate=(None if all(r['format_valid'] is None for r in records) else float(np.mean([r['format_valid'] for r in records]))),planning=False)
