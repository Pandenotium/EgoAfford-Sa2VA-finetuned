"""Paper recipe with portable paths and an explicit effective batch size."""
import json,os,sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('upstream', 'backbone', 'sam2', 'official_checkpoint', 'train', 'val', 'run')

def resolve_path(value):
    if value is None:
        return None
    value = os.path.expandvars(os.path.expanduser(value))
    if '$' in value:
        raise ValueError('Unresolved environment variable in path: ' + value)
    path = Path(value)
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def load(path):
    path=Path(path).resolve();value=json.loads(path.read_text())
    base=load(path.parent/value.pop('base')) if 'base' in value else {}
    base.update(value);return base


def configure(path):
    x=load(path)
    x.update({k: resolve_path(x[k]) for k in PATH_KEYS if k in x})
    x['tests'] = {k: dict(v, path=resolve_path(v['path'])) for k,v in x.get('tests', {}).items()}
    sys.path.insert(0, x['upstream'])
    os.environ.update(EGO_QUARTER_CONFIG=str(Path(path).resolve()),SA2VA_QWEN=x['backbone'],
        SA2VA_UPSTREAM=x['upstream'],EGOAFFORD_PROMPT=x['prompt'],EGOAFFORD_SEG_POLICY=x['policy'],
        EGOAFFORD_SUPERVISION='full')
    return x


def build_config():
    from mmengine.config import Config
    from mmengine.config.lazy import LazyObject
    x=configure(os.environ['EGO_QUARTER_CONFIG'])
    micro=int(os.environ.get('EGO_MICRO_BATCH',x['micro_batch']))
    world=int(os.environ.get('WORLD_SIZE', '8'))
    if micro<1 or world<1:
        raise ValueError('micro batch and world size must be positive')
    accumulation=x['effective_batch']//(world*micro)
    if micro<1 or accumulation<1 or accumulation*world*micro!=x['effective_batch']:
        raise ValueError('world size × micro batch must divide the effective batch')
    cfg=Config.fromfile(str(Path(x['upstream'])/'projects/sa2va/configs/sa2va_finetune.py'))
    cfg.model.type=LazyObject('egoafford_sa2va.model','EgoAffordSa2VA')
    cfg.model.arch_type='intern_vl'
    cfg.model.mllm.type=LazyObject('egoafford_sa2va.model','SampleMeanInternVL')
    cfg.model.mllm.model_path=x['backbone']
    cfg.model.tokenizer.pretrained_model_name_or_path=x['backbone']
    cfg.model.pretrained_pth=x['official_checkpoint'] if x['initialization']=='sa2va' else None
    cfg.model.grounding_encoder.ckpt_path=x['sam2']
    cfg.model.training_bs=micro
    cfg.train_dataloader=dict(batch_size=micro,num_workers=8,persistent_workers=True,pin_memory=True,
        dataset=dict(type=LazyObject('egoafford_sa2va.dataset','EgoAffordDataset'),manifest=x['train'],
            processor_path=x['backbone'],mode=x['prompt'],policy=x['policy']),
        sampler=dict(type='DefaultSampler',shuffle=True),
        collate_fn=dict(type=LazyObject('egoafford_sa2va.dataset','collate')))
    cfg.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=x['epochs'])
    cfg.optim_wrapper.optimizer.lr=4e-5
    cfg.optim_wrapper.accumulative_counts=accumulation
    cfg.optim_wrapper.loss_scale=dict(enabled=False)
    warmup=.05*x['epochs']
    cfg.param_scheduler[0].end=warmup;cfg.param_scheduler[1].begin=warmup;cfg.param_scheduler[1].end=x['epochs']
    cfg.default_hooks.checkpoint.by_epoch=True;cfg.default_hooks.checkpoint.interval=1
    cfg.default_hooks.checkpoint.max_keep_ckpts=-1
    cfg.randomness=dict(seed=42,deterministic=False)
    cfg.model_wrapper_cfg=dict(type='MMDistributedDataParallel',find_unused_parameters=True)
    cfg.work_dir=str(Path(x['run'])/'checkpoints')
    cfg.custom_hooks=[]
    return cfg
