"""Batched native InternVL generation; current-image SAM2 masks and O/I/D mapping."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .protocol import SEG,map_prediction
from .dataset_internvl import ImageProcessor
from .internvl_generation import generate

def load_native(group):
 from xtuner.registry import BUILDER
 from .config import build_config
 torch.manual_seed(42);np.random.seed(42)
 model=BUILDER.build(build_config().model)
 model.mllm.gradient_checkpointing_disable()
 model.mllm.model.language_model.config.use_cache=True
 expected=None
 model._ego_trainable_keys={name for name,param in model.named_parameters() if param.requires_grad}
 model.eval();model.requires_grad_(False)
 processor=ImageProcessor(os.environ['SA2VA_QWEN'])
 return model.to('cuda'),processor,expected

def seg_hidden(hidden,sequences,seg_id):
 # Official InternVL get_seg_hidden_states: generated IDs exclude the prompt;
 # generation states include prefill, so use the final len(generated[:-1]) states.
 tokens=sequences[:-1];n=len(tokens)
 if n==0:return hidden[:0]
 selected=hidden[-n:][tokens==seg_id]
 if len(selected)!=int((sequences==seg_id).sum()):raise ValueError('Unprocessed final SEG')
 return selected

@torch.inference_mode()
def infer_batch(model,processor,items,mode='full_rule',policy='emit_empty',max_new_tokens=1536):
 prepared=[];sizes=[]
 for row in items:
  with Image.open(row['image']) as im:
   image=im.convert('RGB');sizes.append((image.height,image.width));prepared.append(processor.prepare(image,row['action'],mode,policy))
 length=max(len(x['input_ids']) for x in prepared)
 if length+max_new_tokens>8192:raise ValueError('Prompt exceeds generation context')
 tokenizer=processor.tokenizer;pad=tokenizer.pad_token_id or tokenizer.eos_token_id
 ids=torch.full((len(items),length),pad,dtype=torch.long,device='cuda');attention=torch.zeros_like(ids,dtype=torch.bool)
 for i,x in enumerate(prepared):
  n=len(x['input_ids']);ids[i,-n:]=x['input_ids'].to('cuda');attention[i,-n:]=True
 pixels=torch.cat([x['pixel_values'] for x in prepared]).to('cuda',dtype=model.mllm.model.vision_model.dtype)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  generated=generate(model.mllm.model,pixel_values=pixels,input_ids=ids,attention_mask=attention,
   do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,eos_token_id=tokenizer.eos_token_id,pad_token_id=pad,
   output_hidden_states=True,return_dict_in_generate=True,use_cache=True)
  sequences=generated.sequences
  hidden=torch.cat([step[-1] for step in generated.hidden_states],dim=1)
  includes_prompt=sequences.shape[1]>=length and torch.equal(sequences[:,:length],ids)
  result=[]
  for i,(row,shape,data) in enumerate(zip(items,sizes,prepared)):
   tokens=sequences[i,length:] if includes_prompt else sequences[i]
   # Remove batch padding after the first EOS, retaining the EOS itself.
   ends=(tokens==tokenizer.eos_token_id).nonzero()
   trimmed=tokens[:int(ends[0])+1] if len(ends) else tokens
   response=tokenizer.decode(trimmed,skip_special_tokens=False)
   answer=response.removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
   try:
    if includes_prompt:
     full=sequences[i,:-1];h=hidden[i][full==model.seg_token_idx]
    else:h=seg_hidden(hidden[i],tokens,model.seg_token_idx)
    n=int((tokens==model.seg_token_idx).sum())
    if n>16:raise ValueError('Too many segmentation tokens')
    if n:
     lang=model.text_hidden_fcs(h)[:,None]
     raw=data['g_pixel_values'].to('cuda');grounding=model.grounding_encoder.preprocess_image(raw)[None]
     sam=model.grounding_encoder.get_sam2_embeddings(grounding,expand_size=n)
     logits=model.grounding_encoder.inject_language_embd(sam,lang,nf_nobj=(1,n))
     masks=F.interpolate(logits[0].unsqueeze(1),size=shape,mode='bilinear',align_corners=False)
     decoded=(masks[:,0].sigmoid()>.5).cpu().numpy()
    else:decoded=np.zeros((0,*shape),dtype=bool)
    pred,valid,fmt=map_prediction(answer,decoded,shape,policy);error=None
   except ValueError as ex:
    pred=np.zeros((3,*shape),dtype=bool);valid=[False]*3;fmt=False;error=repr(ex);n=0
   result.append((pred,dict(response=response,answer=answer,valid=valid,format_valid=fmt,seg_count=n,error=error)))
 return result

def infer(model,processor,row,mode,policy):return infer_batch(model,processor,[row],mode,policy)[0]
