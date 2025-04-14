import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import random
import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
# from pymoo.factory import get_algorithm, get_crossover, get_mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.pntx import TwoPointCrossover


os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, get_root_logger, sample_mamba_subnet, unwrap_model, str2bool, str2list
from networks.model import MambaDepth

# (choose SpatialMamba) 
# export PYTHONPATH=$PYTHONPATH:/data/code_yzh/Spatial-Mamba-main/kernels/dwconv2d
# export PYTHONPATH=$PYTHONPATH:/data/code_yzh/Spatial-Mamba-main/kernels/selective_scan

# python MambaDepthNAS/search.py configs/search_kitti.txt

parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Search space
parser.add_argument('--mlp_ratio',                 type=str2list, default=[4.0])
parser.add_argument('--d_state',                   type=str2list, default=[64])
parser.add_argument('--ssd_expand',                type=str2list, default=[2])

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Training
parser.add_argument('--resume',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
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
parser.add_argument('--dynamic_tanh', type=str2bool, default=False)
# evolution
parser.add_argument('--population_size',           type=int,    default=40)
parser.add_argument('--n_iter',                    type=int,    default=60)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

space = {
    'mlp_ratio': [0.5, 1.0, 2.0, 3.0, 3.5, 4.0],
    'd_state':  [16, 32, 48, 64],
    'expand':   [0.5, 1, 2, 3, 4],
}

class NasCodec:
    def __init__(self, space):
        self.space = space
        self.num_mlp = 4
        self.num_d = 3
        self.num_expand = 3

    def encode(self, config):
        return (
            [self.space['mlp_ratio'].index(v) for v in config['mlp_ratio']] +
            [self.space['d_state'].index(v) for v in config['d_state']] +
            [self.space['expand'].index(v) for v in config['expand']]
        )

    def decode(self, indices):
        i = 0
        mlp_idx = indices[i:i+self.num_mlp]; i += self.num_mlp
        d_idx   = indices[i:i+self.num_d];   i += self.num_d
        ex_idx  = indices[i:i+self.num_expand]

        return {
            'mlp_ratio': [self.space['mlp_ratio'][j] for j in mlp_idx],
            'd_state':   [self.space['d_state'][j] for j in d_idx],
            'expand':    [self.space['expand'][j] for j in ex_idx],
        }
    
def online_eval(model, dataloader_eval, sample_config, gpu, ngpus, post_process=False, logger=None):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            # random sample subnet
            # sample_config = sample_mamba_subnet(args)
            model_module = unwrap_model(model)
            model_module.backbone.set_sample_config(sample_config=sample_config)
            # print(sample_config)

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    
    if not args.multiprocessing_distributed or gpu == 0:
        logger.info('Computing errors for {} eval samples, post_process: {}'.format(int(cnt), post_process))
        logger.info("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        logger.info(", ".join(["{:7.4f}".format(eval_measures_cpu[i]) for i in range(9)]))
        
    return eval_measures_cpu[6], 1000  # d1, params


def make_eval_func(model, dataloader_eval, gpu, ngpus, post_process=False, logger=None):
    def eval_fn(config):
        return online_eval(
            model=model,
            dataloader_eval=dataloader_eval,
            sample_config=config,
            gpu=gpu,
            ngpus=ngpus,
            post_process=post_process,
            logger=logger
        )
    return eval_fn

class IntegerFromFloatMutation(PolynomialMutation):
    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, eta=eta, at_least_once=at_least_once, **kwargs)

    def _do(self, problem, X, params=None, **kwargs):
        Xp = super()._do(problem, X, params=params, **kwargs)
        Xp_int = np.rint(Xp).astype(int)
        return Xp_int


class NasProblem(Problem):
    def __init__(self, eval_func, min_op=True):
        self.eval_func = eval_func
        self.min_op = min_op
        self.codec = NasCodec(space)
        super().__init__(n_var=10, n_obj=2, n_constr=0,  # 变量数，目标数，约束数
                         type_var=np.int32)
        
        self.xl = np.zeros(self.n_var)
        self.xu = np.array(
                        [len(space['mlp_ratio'])-1] * 4 +
                        [len(space['d_state'])-1] * 3 +
                        [len(space['expand'])-1] * 3,
                        dtype=np.int32
                    )

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        # 假设 predictor 的 predict 方法接受一个二维数组，每行一个架构编码，返回对应的性能指标
        f_err = []
        params_list = []

        for _x in x:
            print('_x:', _x)
            sample_config = self.codec.decode(_x)
            print('sample_config:',sample_config)
            performance, params = self.eval_func(sample_config)
            print(f"Evaluating params for {_x}: {params}M, {performance}")

            if self.min_op:
                f_err.append(performance)
            else:
                f_err.append(1.0 - performance)
            params_list.append(params)

        if self.runner.rank == 0:
            f_err_rounded = [round(err, 4) for err in f_err]  # 将 f_err 保留小数点后 4 位
            params_rounded = [round(par, 4) for par in params_list]  # 将 params 保留小数点后 4 位
            self.runner.logger.info(f"1 - mIoU = {f_err_rounded}")
            self.runner.logger.info(f"params = {params_rounded}")

        for i, (_x, err, par) in enumerate(zip(x, f_err, params_list)):
            f[i, 0] = err
            f[i, 1] = par

        out["F"] = f

        
def main_worker(gpu, ngpus_per_node, args):
    ### logger
    logger = get_root_logger(args.log_directory)
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    ### MambaDepth model
    model = MambaDepth(args, version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain)
    if args.dynamic_tanh:  ### 替换归一化层
        from networks.dynamic_tanh import convert_ln_to_dyt
        model = convert_ln_to_dyt(model)
        logger.info("==> Using dynamic_tanh!")

    model.train()
    ### model params
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("== Total number of parameters: {}".format(num_params))

        num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        logger.info("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    '''show model'''
    with open(os.path.join(args.log_directory, 'max_model.log'),'w') as f:
        f.write(str(model))
    # assert False,'print model'

    if args.distributed:
        logger.info("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        logger.info("== Model Initialized")

    ### load weight
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            logger.info("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])

            logger.info("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            logger.info("== No checkpoint found at '{}'".format(args.checkpoint_path))

        del checkpoint

    cudnn.benchmark = True

    ### get dataset
    dataloader_eval = NewDataLoader(args, 'online_eval')

    ### evolution
    model.eval()
    eval_func = make_eval_func(model=model, dataloader_eval=dataloader_eval, gpu=gpu, ngpus=ngpus_per_node, post_process=True, logger=logger)
    codec = NasCodec(space)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info(f"==> Starting evolution...")

    problem = NasProblem(eval_func=eval_func, min_op=False)
    method = NSGA2(
        pop_size=args.population_size,  # initialize with current nd archs
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=IntegerFromFloatMutation(eta=1.0),
        eliminate_duplicates=True)

    res = minimize(
        problem, method, termination=('n_gen', args.n_iter), save_history=True, verbose=True, seed=1274394)
    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        ### history
        for gen in res.history:
            pop_X = gen.pop.get("X")
            pop_F = gen.pop.get("F")
            logger.info(f"Generation: {gen.n_gen}")
            logger.info(f"Solutions:\n{pop_X}")
            logger.info(f"Objective Values:\n{pop_F}")

        ### result
        optimal_solutions = res.X.tolist()
        optimal_objective_values = res.F

        logger.info("Optimal Code:")
        logger.info(optimal_solutions)

        logger.info("Optimal Objective Values:")
        logger.info(optimal_objective_values)

        logger.info("Optimal Config:")
        logger.info(codec.decode(optimal_solutions))

        logger.info(f"==> Finished evolution!")
    

def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

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
