"""SC07 native image mixin, direct quarter members; syn alt only, real off."""
import torch
from PIL import Image
from transformers import AutoTokenizer
from projects.sa2va.datasets.base import Sa2VADatasetMixin
from projects.sa2va.models import DirectResize
from .common import rows,SYSTEM
from .protocol import SPECIAL_TOKENS,SEG,prompt,target


class ImageProcessor(Sa2VADatasetMixin):
    def __init__(self,path):
        self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True,padding_side='right')
        self.tokenizer.add_tokens(SPECIAL_TOKENS,special_tokens=True)
        self.seg_id=self.tokenizer.convert_tokens_to_ids(SEG)
        self._init_architecture_config('intern_vl')
        self._init_image_processing_config(min_dynamic_patch=1,max_dynamic_patch=12,image_size=448,
            use_thumbnail=True,downsample_ratio=.5,patch_size=14)
        self._init_image_processor()
        self._init_extra_image_processor(dict(type=DirectResize,target_length=1024))

    def prepare(self,image,action,mode,policy):
        data=self._process_single_image(image,single_image_mode=False)
        visual='<img>'+'<IMG_CONTEXT>'*data['num_image_tokens']+'</img>'
        text=('<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'+visual+'\n'
              +prompt(action,mode,policy)+'<|im_end|>\n<|im_start|>assistant\n')
        data.update(input_ids=torch.tensor(self.tokenizer.encode(text,add_special_tokens=False),dtype=torch.long),prefix=text)
        return data


class EgoAffordDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,processor_path,mode='minimal_user',policy='emit_empty',max_length=8192):
        self.rows=[dict(row,training_policy='alt' if row.get('data_source')=='synthetic' else 'off') for row in rows(manifest)]
        self.processor=ImageProcessor(processor_path);self.tokenizer=self.processor.tokenizer
        self.mode,self.policy,self.max_length=mode,policy,max_length

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];answer,masks,active=target(row,self.policy)
        with Image.open(row['image']) as image:data=self.processor.prepare(image.convert('RGB'),row['action'],self.mode,self.policy)
        prefix=data['input_ids']
        answer_ids=torch.tensor(self.tokenizer.encode(answer+'<|im_end|>',add_special_tokens=False),dtype=torch.long)
        sep=torch.tensor(self.tokenizer.encode('\n',add_special_tokens=False),dtype=torch.long)
        ids=torch.cat((prefix,answer_ids,sep))
        if len(ids)>self.max_length:raise ValueError(f'Context limit exceeded: {row["id"]}')
        labels=torch.cat((torch.full_like(prefix,-100),answer_ids,torch.full_like(sep,-100)))
        return dict(input_ids=ids,labels=labels,pixel_values=data['pixel_values'],g_pixel_values=data['g_pixel_values'],
            masks=torch.from_numpy(masks[active].copy()).float(),role_indices=active,
            seg_id=self.processor.seg_id,policy=self.policy,sample_id=row['id'],answer=answer)


def collate(items):
    if len(items)>1:return dict(data={'ego_micro_items':[collate([item])['data'] for item in items]},data_samples=None)
    item=items[0];ids=item['input_ids'][None]
    data=dict(input_ids=ids,labels=item['labels'][None],attention_mask=torch.ones_like(ids,dtype=torch.bool),
        position_ids=torch.arange(ids.shape[1])[None],pixel_values=[item['pixel_values']],
        g_pixel_values=[item['g_pixel_values']],masks=[item['masks']],frames_per_batch=[1],
        ego_role_indices=item['role_indices'],ego_policy=item['policy'],ego_seg_id=item['seg_id'])
    return dict(data=data,data_samples=None)
