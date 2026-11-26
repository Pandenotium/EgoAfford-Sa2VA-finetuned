"""Native MMEngine entry; seed before construction, no startup probe."""
import argparse,random
import numpy as np
import torch
from .config import configure,build_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();configure(args.config)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    from mmengine.runner import Runner
    cfg=build_config();cfg.launcher='pytorch';cfg.resume=args.resume;cfg.load_from=None
    Runner.from_cfg(cfg).train()


if __name__=='__main__':main()
