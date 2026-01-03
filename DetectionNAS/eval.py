import os
import sys
import gc
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from torch.amp import autocast

# 你工程里的工具
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils import convert_arg_line_to_args, get_root_logger, unwrap_model, str2list
from search_space import MambaSearchSpace


def parse_args():
    parser = argparse.ArgumentParser(
        description="MambaDetection eval script (supernet / subnet validation)",
        fromfile_prefix_chars='@'
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('config', help='eval config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path to load')
    parser.add_argument('--devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1"')
    parser.add_argument('--log_directory', type=str, default='./work_dirs/eval', help='log dir')

    # AMP
    parser.add_argument('--amp', action='store_true', help='enable AMP')

    # supernet sampling
    parser.add_argument('--sample_subnet', action='store_true',
                        help='if set, sample a random subnet for each batch (supernet eval)')
    parser.add_argument('--sample_config_json', type=str, default='',
                        help='(optional) fixed subnet config in JSON string. If set, overrides --sample_subnet.')
    parser.add_argument('--mlp_ratio', type=str2list, default=[4.0])
    parser.add_argument('--d_state', type=str2list, default=[64])
    parser.add_argument('--ssd_expand', type=str2list, default=[2])
    parser.add_argument('--open_depth', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # DDP
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='spawn N processes per node, N = num visible gpus')

    if len(sys.argv) == 2:
        # 支持：python eval.py DetectionNAS/configs/xxx.txt  (txt 里写参数)
        args = parser.parse_args(['@' + sys.argv[1]])
    else:
        args = parser.parse_args()
    return args


def is_main_process(args, ngpus_per_node):
    if not args.multiprocessing_distributed:
        return True
    return (args.rank % ngpus_per_node) == 0


@torch.no_grad()
def run_eval(args, model, dataloader_eval, evaluator, logger, ss=None, fixed_sample_config=None):
    model.eval()

    for _, data_batch in enumerate(tqdm(dataloader_eval, disable=not (not dist.is_initialized() or dist.get_rank() == 0))):
        model_module = unwrap_model(model)

        # 1) 固定子网配置（优先级最高）
        if fixed_sample_config is not None:
            model_module.backbone.backbone.set_sample_config(sample_config=fixed_sample_config)

        # 2) 随机采样子网（supernet eval）
        elif args.sample_subnet:
            assert ss is not None, "ss (search space) is required when --sample_subnet is enabled"
            sample_config = ss.sample(n_samples=1)[0]
            model_module.backbone.backbone.set_sample_config(sample_config=sample_config)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            preds = model_module.val_step(data_batch)

        evaluator.process(data_samples=preds, data_batch=data_batch)

    metrics = evaluator.evaluate(len(dataloader_eval.dataset))

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info(metrics)

    return metrics


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu

    # DDP init
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    # logger (after dist init)
    os.makedirs(args.log_directory, exist_ok=True)
    logger = get_root_logger(args.log_directory)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    main_proc = is_main_process(args, ngpus_per_node)
    if main_proc:
        logger.info(f"args: {args}")

    # register mmdet
    init_default_scope('mmdet')
    register_all_modules(init_default_scope=True)

    # build model
    model = MODELS.build(cfg.model)
    model.init_weights()

    # load ckpt
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
    if main_proc:
        logger.info(f"Loaded checkpoint: {args.ckpt}")

    # move to gpu / ddp
    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False
        )

    # build dataloader
    dataloader_eval = Runner.build_dataloader(cfg.val_dataloader, seed=args.seed)

    # evaluator
    evaluator = Evaluator(metrics=cfg.val_evaluator)
    dataset_meta = getattr(dataloader_eval.dataset, 'metainfo', None)
    assert dataset_meta is not None and 'classes' in dataset_meta, \
        'dataset.metainfo 为空或缺少 classes，请在数据集里补全 METAINFO/metainfo。'
    evaluator.dataset_meta = dataset_meta

    # subnet sampling (optional)
    ss = None
    fixed_sample_config = None

    if args.sample_config_json:
        fixed_sample_config = json.loads(args.sample_config_json)
        if main_proc:
            logger.info("Using fixed sample_config_json for evaluation.")
    elif args.sample_subnet:
        ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)
        if main_proc:
            logger.info("Using random subnet sampling for each batch (supernet eval).")
    else:
        if main_proc:
            logger.info("No subnet sampling enabled: evaluate the current (default) model config.")

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # run eval
    if dist.is_initialized():
        dist.barrier()
    run_eval(args, model, dataloader_eval, evaluator, logger, ss=ss, fixed_sample_config=fixed_sample_config)
    if dist.is_initialized():
        dist.barrier()


def main():
    args = parse_args()

    # set visible gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    cfg = Config.fromfile(args.config)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # 单卡/单进程
        if args.gpu is None:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args, cfg)


if __name__ == "__main__":
    main()
