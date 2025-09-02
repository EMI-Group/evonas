import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import random
import os, sys, time
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # to set in config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
from tqdm import tqdm
from mmengine import Config
from dataloaders.dataloader import NewDataLoader


from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, get_root_logger, unwrap_model, str2list, infer, compute_metrics
from networks.model import MambaDepth, make_divisible


'''eval the final selected model'''

### (OFA-init under supernet model) 
# python MambaDepthNAS/eval.py configs/eval/01_nyu_base.txt
### (IN-init under subnet model)
# 

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, SuperNet, VSSD_final, tiny07', default='SuperNet')
    parser.add_argument('--pretrained',                type=str,   help='path of pretrained encoder', default=None)
    parser.add_argument('--devices',                    type=str, default='0', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

    # arch config
    parser.add_argument('--mlp_ratio',                 type=str2list, default=[4.0,4.0,4.0,4.0])
    parser.add_argument('--d_state',                   type=str2list, default=[64,64,64,-1])
    parser.add_argument('--ssd_expand',                type=str2list, default=[4,4,4,-1])
    parser.add_argument('--depth',                     type=str2list, default=[2,4,8,4])
    parser.add_argument('--width_multiplier',          type=float, help='factor to scale the number of channels in each layer (applies only to final model)', default=1.0)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints',default=None)

    # Preprocessing
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    # Multi-gpu training
    parser.add_argument('--num_threads_val',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--persistent_workers',        action='store_true',   help='if set the data loader will not shut down the worker processes after a dataset has been consumed once')
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
    # Online eval
    parser.add_argument('--batch_size_val',            type=int,   help='validation dataloader batch size', default=1)
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')
    # experimental
    parser.add_argument('--dynamic_tanh',              action='store_true', help='if set, will use dynamic tanh for normalization')
    

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args


@torch.no_grad()
def online_eval(args, model, dataloader_eval, gpu, ngpus, post_process=False, logger=None, ss=None):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        image    = eval_sample_batched['image'].cuda(gpu, non_blocking=True)
        gt_depth = eval_sample_batched['depth'].cuda(gpu, non_blocking=True)
        if not eval_sample_batched['has_valid_depth']:
            # print('Invalid depth. continue.')
            continue

        pred_depth = model(image)
        if post_process:
            image_flipped = flip_lr(image)
            pred_depth_flipped = model(image_flipped)
            pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
        # update eval measures
        measures = compute_metrics(gt_depth, pred_depth, args)
        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        # group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        logger.info('Computing errors for {} eval samples, post_process: {}'.format(int(cnt), post_process))
        logger.info("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        logger.info(", ".join(["{:7.4f}".format(eval_measures_cpu[i]) for i in range(9)]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # get logger after init DDP
    logger = get_root_logger(args.log_directory)
    
    if args.gpu is not None:
        logger.info("== Use GPU: {} for training".format(args.gpu))

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info('args: '+str(args))

    # MambaDepth model
    final_config = {
        'mlp_ratio': args.mlp_ratio,
        'd_state': args.d_state + [-1],
        'ssd_expand': args.ssd_expand + [-1],
        'depth': [sum(lst) for lst in args.depth],
    } if args.encoder == 'VSSD_final' else None

    model = MambaDepth(args, version=args.encoder, max_depth=args.max_depth, selected_config=final_config, pretrained=args.pretrained)
    model.train()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info('model: '+str(model))


    # print model params    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("== Total number of parameters: {}".format(num_params))

        num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        logger.info("== Total number of learning parameters: {}".format(num_params_update))

        num_params_backbone = sum([np.prod(p.shape) for name, p in model.named_parameters() if name.startswith("backbone")])
        logger.info("== Total number of backbone parameters: {}".format(num_params_backbone))


    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            assert False,'developing'
    else:
        raise ValueError("Distributed training is not enabled. Please set --distributed flag.")

    # load whole weight from supernet
    if args.ckpt_path:
        key = 'model'
        _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
        logger.info("Successfully load whole ckpt {} from {}".format(args.ckpt_path, key))
        incompatibleKeys = model.load_state_dict(_ckpt[key], strict=False)
        logger.info("== missing_keys: {}".format(incompatibleKeys))
        # key2 = 'distill_module'
        # if key2 in _ckpt and args.f_distill:  # Note!
        #     assert False,'Not use pretrained distill_module'
        #     dis_modules_s4.load_state_dict(_ckpt[key2])
        #     logger.info("Successfully load distill_module ckpt {} from {}".format(args.ckpt_path, key2))
        del _ckpt
    
    model_module = unwrap_model(model)
    if args.encoder == 'SuperNet':
        selected_config = {
            'mlp_ratio': args.mlp_ratio,
            'd_state': args.d_state,
            'expand': args.ssd_expand,
            'depth': args.depth,
        }
        model_module.backbone.set_sample_config(sample_config=selected_config)

    cudnn.benchmark = True
    dataloader_eval = NewDataLoader(args, 'online_eval')

    model.eval()
    with torch.no_grad():
        online_eval(args, model, dataloader_eval, gpu, ngpus_per_node, post_process=True, logger=logger)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
