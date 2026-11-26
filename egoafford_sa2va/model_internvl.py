"""Official InternVL/Sa2VA classes; only protocol checks and batch loss reduction."""
from projects.sa2va.models.sa2va import Sa2VAModel
from projects.sa2va.models.mllm.internvl import InternVLMLLM
from .alignment import RoleAlignmentMixin

class SampleMeanInternVL(InternVLMLLM):
 def __init__(self,*args,llm_lora=None,**kwargs):
  super().__init__(*args,llm_lora=llm_lora,**kwargs)
  # Upstream delays adapter creation but its parent reset this flag to False.
  # Restore the requested setting before Sa2VAModel loads weights then adds LoRA.
  self.use_llm_lora=llm_lora is not None
 def _compute_loss(self,logits,labels):
  from batch_support import sequence_ce
  return sequence_ce(logits,labels).mean()

class EgoAffordSa2VA(RoleAlignmentMixin,Sa2VAModel):
 def __init__(self,*args,**kwargs):
  self._loading_initial_checkpoint=kwargs.get('pretrained_pth') is not None
  super().__init__(*args,**kwargs)
  self._loading_initial_checkpoint=False
  self.mllm.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 def load_state_dict(self,state_dict,strict=True,assign=False):
  if self._loading_initial_checkpoint:strict=True
  return super().load_state_dict(state_dict,strict=strict,assign=assign)
