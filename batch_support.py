# Modified for the EgoAfford release (2026): supervised-row CE memory optimization.
# SPDX-License-Identifier: Apache-2.0
# See THIRD_PARTY_NOTICES.md and docs/SOURCE_PROVENANCE.md.
"""Cluster micro-batches, fixed effective batch and sample-normalized CE."""
import json,os
from pathlib import Path

PROFILES={
 'iterative':{'legacy':1,'h100':4,'h200':4},
 'unipixel':{'legacy':1,'h100':4,'h200':4},
 'sa2va':{'legacy':1,'h100':4,'h200':4},
 'samtok':{'legacy':1,'h100':8,'h200':32},
 'egolens':{'legacy':16,'h100':16,'h200':16},
}

def resolve(family,effective,world=8):
 profile=os.environ.get('EGO_BATCH_PROFILE','legacy').lower()
 if profile=='auto':
  import torch
  name=torch.cuda.get_device_name().lower()
  profile='h200' if 'h200' in name else 'h100' if 'h100' in name else 'legacy'
 if family not in PROFILES or profile not in PROFILES[family]:raise ValueError('Unknown model/batch profile')
 micro=int(os.environ.get('EGO_MICRO_BATCH',PROFILES[family][profile]))
 if micro<1 or effective%(world*micro):raise ValueError(f'Cannot preserve effective batch {effective}: world={world}, micro={micro}')
 return dict(profile=profile,micro=micro,accumulation=effective//(world*micro),world=world,effective_batch=effective)

def sequence_ce(logits,labels):
 """Match the arithmetic mean of the original single-sample CE losses."""
 import torch
 import torch.nn.functional as F
 values=[]
 for scores,target in zip(logits,labels):
  target=target[1:];valid=target!=-100
  if not valid.any():raise ValueError('No shifted supervised tokens')
  # Ignored prompt/image positions contribute neither loss nor gradient.
  # Select supervised rows before FP32 conversion to avoid a full T×V copy.
  values.append(F.cross_entropy(scores[:-1][valid].float(),target[valid]))
 return torch.stack(values)

def install_sequence_mean(model):
 """Native HF forward/architecture unchanged; replace only loss reduction."""
 if getattr(model,'_ego_sequence_mean_installed',False):return
 def before(module,args,kwargs):
  module._ego_sequence_labels=None
  labels=kwargs.get('labels')
  if module.training and labels is not None and labels.shape[0]>1:
   module._ego_sequence_labels=labels
   kwargs=dict(kwargs,labels=None)
  return args,kwargs
 def after(module,args,kwargs,output):
  labels=module._ego_sequence_labels;module._ego_sequence_labels=None
  if labels is not None:output['loss']=sequence_ce(output.logits,labels).mean()
  return output
 model.register_forward_pre_hook(before,with_kwargs=True)
 model.register_forward_hook(after,with_kwargs=True)
 model._ego_sequence_mean_installed=True

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--effective',type=int);p.add_argument('--world',type=int,default=8);p.add_argument('--shell',action='store_true');p.add_argument('--config');p.add_argument('--out');a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text()) if a.config else None
 result=resolve(a.family,a.effective if cfg is None else cfg['effective_batch'],a.world)
 if cfg is not None:
  cfg.update(micro_batch=result['micro'],batch_execution=result)
  Path(a.out).write_text(json.dumps(cfg,indent=2));print(json.dumps(result));raise SystemExit(0)
 print(f"{result['micro']} {result['accumulation']}" if a.shell else json.dumps(result))
