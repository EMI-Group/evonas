import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import argparse
import numpy as np
from tqdm import tqdm
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.amp import autocast as mautocast
from mmengine.evaluator import Evaluator
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from DetectionNAS.networks import model

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils import convert_arg_line_to_args, get_root_logger, unwrap_model, str2list

# Register modules
init_default_scope('mmdet')
register_all_modules(init_default_scope=True)

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDetection Evaluation', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ckpt_path', type=str, help='path to load checkpoints', required=True)
    
    parser.add_argument('--devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--log_directory', type=str, default='work_dirs/eval', help='directory to save logs')
    
    # Architecture params
    parser.add_argument('--mlp_ratio', type=str2list, default=[4.0], help='mlp ratio list')
    parser.add_argument('--d_state', type=str2list, default=[64], help='d_state list')
    parser.add_argument('--ssd_expand', type=str2list, default=[4], help='ssd_expand list')
    parser.add_argument('--depth', type=str2list, default=[[1]*2, [1]*4, [1]*8, [1]*4], help='depth list (bitmasks)')

    # Eval options
    parser.add_argument('--amp', action='store_true', help='enable mixed precision (AMP)')

    # Multi-gpu training
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
    
    return parser.parse_args()

@torch.no_grad()
def evaluate(model, dataloader, evaluator, logger, sample_config=None, amp=False):
    model.eval()
    model_module = unwrap_model(model)
    
    if sample_config:
        # Assuming the structure is model.backbone.backbone.set_sample_config based on retrain.py
        if hasattr(model_module, 'backbone') and hasattr(model_module.backbone, 'backbone') and hasattr(model_module.backbone.backbone, 'set_sample_config'):
             logger.info(f"Setting sample config: {sample_config}")
             model_module.backbone.backbone.set_sample_config(sample_config=sample_config)
        else:
             logger.warning("Could not find set_sample_config method on model.backbone.backbone. Architecture configuration might be ignored.")

    prog_bar = tqdm(dataloader) if (not dist.is_initialized() or dist.get_rank() == 0) else dataloader
    for i, data_batch in enumerate(prog_bar):
        with mautocast(enabled=amp):
            preds = model_module.val_step(data_batch)
        evaluator.process(data_samples=preds, data_batch=data_batch)

    metrics = evaluator.evaluate(len(dataloader.dataset))
    return metrics

def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu
    
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Init logger
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        if not os.path.exists(args.log_directory):
            os.makedirs(args.log_directory, exist_ok=True)
        logger = get_root_logger(args.log_directory)
        logger.info(f"Args: {args}")
    else:
        logger = None

    # Build Dataloader
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    
    # Build Evaluator
    evaluator = Evaluator(metrics=cfg.val_evaluator)
    if hasattr(dataloader.dataset, 'metainfo'):
        evaluator.dataset_meta = dataloader.dataset.metainfo

    # Build Model
    if logger: logger.info(f"Building model from config...")
    model = MODELS.build(cfg.model)
    model.init_weights()
    
    # Load Checkpoint
    if args.ckpt_path:
        if logger: logger.info(f"Loading checkpoint from {args.ckpt_path}")
        try:
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Handle possible DDP prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            incompatible = model.load_state_dict(new_state_dict, strict=False)
            if logger: logger.info(f"Checkpoint loaded. Incompatible keys: {incompatible}")
        except Exception as e:
            if logger: logger.error(f"Failed to load checkpoint: {e}")
            return

    # Move to GPU
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
             model.cuda()
             model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = model.cuda(args.gpu if args.gpu is not None else 0)
    
    # Prepare Sample Config
    sample_config = {
        'mlp_ratio': args.mlp_ratio,
        'd_state': args.d_state,
        'expand': args.ssd_expand,
        'depth': args.depth
    }

    # Evaluate
    # If using DDP, model_module inside evaluate will unwrap it automatically using utils.unwrap_model
    metrics = evaluate(model, dataloader, evaluator, logger if logger else get_root_logger(), sample_config, args.amp)
    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info(f"Evaluation Results: {metrics}")


def main():
    args = parse_args()
    
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    cfg = Config.fromfile(args.config)
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        main_worker(0, ngpus_per_node, args, cfg)

if __name__ == '__main__':
    main()
