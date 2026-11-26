"""Native Sa2VA batching without role replication/cropping or sample reweighting."""
class RoleAlignmentMixin:
    def check_obj_number(self, embeddings, masks, fix_number=None):
        return embeddings,masks

    def checked(self,data):
        data=dict(data);active=data.pop('ego_role_indices');data.pop('ego_policy');data.pop('ego_seg_id')
        return data,len(active)

    def forward(self,data,data_samples=None,mode='loss'):
        if 'ego_micro_items' not in data:
            checked,_=self.checked(data)
            return super().forward(checked,data_samples,mode)
        import torch
        from torch.nn.utils.rnn import pad_sequence
        items=data['ego_micro_items'];groups={};total=len(items);result={}
        for item in items:
            checked,n=self.checked(item);groups.setdefault(n,[]).append(checked)
        for objects,group in groups.items():
            ids=pad_sequence([x['input_ids'][0] for x in group],batch_first=True,padding_value=self.mllm.model.config.pad_token_id or 151643)
            labels=pad_sequence([x['labels'][0] for x in group],batch_first=True,padding_value=-100)
            attention=pad_sequence([x['attention_mask'][0] for x in group],batch_first=True,padding_value=0)
            batch=dict(input_ids=ids,labels=labels,attention_mask=attention,position_ids=torch.arange(ids.shape[1],device=ids.device)[None].expand(len(group),-1))
            for key in ('pixel_values','image_grid_thw','g_pixel_values','masks','frames_per_batch'):
                if key in group[0]:batch[key]=[v for x in group for v in x[key]]
            losses=super().forward(batch,data_samples,mode)
            # Keep the native epsilon contribution per original single image.
            if self.loss_sample_points:
                n=len(group)*objects;epsilon=1e-4
                losses['loss_dice']*= (n+epsilon)/(len(group)*(objects+epsilon))
                points=self.num_points
                losses['loss_mask']*= (n*points+epsilon)/(len(group)*(objects*points+epsilon))
            for key,value in losses.items():result[key]=result.get(key,0)+value*(len(group)/total)
        return result
