import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # to set in config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import json, subprocess, tempfile
import argparse
import numpy as np
from tqdm import tqdm
from dataloaders.dataloader import NewDataLoader
from ptflops import get_model_complexity_info
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
# from pymoo.factory import get_algorithm, get_crossover, get_mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation, mut_pm
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.crossover import Crossover
from pymoo.core.variable import get

from functools import partial
from pymoo.core.callback import Callback
import pandas as pd

from utils import post_process_depth, flip_lr, compute_errors, eval_metrics, \
                       convert_arg_line_to_args, get_root_logger, unwrap_model, str2list, is_main_process, compute_metrics
from networks.model import MambaDepth
from search_space import MambaSearchSpace

'''
search the best network on object dataset, e.g. KITTI, NYU

Note: latency tests included; avoid running other GPU tasks.
'''

### develop
# python MambaDepthNAS/search.py configs/search/search_kitti.txt 

### final
# python MambaDepthNAS/search.py configs/search/search_kitti.txt 
# python MambaDepthNAS/search.py configs/search/search_nyu.txt 

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepthSearch')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='SuperNet')
    parser.add_argument('--devices',                   type=str, default='0,1', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False)
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

    # Search space
    parser.add_argument('--mlp_ratio',                 type=str2list, default=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0])
    parser.add_argument('--d_state',                   type=str2list, default=[16, 32, 48, 64])
    parser.add_argument('--ssd_expand',                type=str2list, default=[0.5, 1, 2, 3, 4])
    parser.add_argument('--open_depth',                action='store_true', help='if set, will open depth sampling, otherwise use fixed depth for all stages. Note! Not for search!')
    parser.add_argument('--min_ones',                  type=int,    help='minimum number of active layers in each stage during sampling', default=1)

    # evolution
    parser.add_argument('--population_size',           type=int,    help='number of individuals in the population', default=50)
    parser.add_argument('--n_iter',                    type=int,    help='number of generations to run', default=50)
    parser.add_argument('--cross_p',                   type=float,  help='probability of crossover', default=0.95)
    parser.add_argument('--mut_p',                     type=float,  help='probability of mutation', default=0.1)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints', required=True)

    # Preprocessing
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    # Multi-gpu training
    parser.add_argument('--num_threads_val',               type=int,   help='number of threads to use for data loading', default=1)
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
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=True)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=True)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=True)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    # experimental

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args



def test_lat_mac(args, sample_config_list, gpu, ngpus):
    ### latency
    multi_config  = []
    for sample_config in sample_config_list:
        fps_config = {
            "mlp_ratio": sample_config["mlp_ratio"],
            "d_state": sample_config["d_state"] + [-1],
            "ssd_expand": sample_config["expand"] + [-1],
            "depth": [int(sum(dl)) for dl in sample_config["depth"]],
        }
        multi_config.append(fps_config)
    
    fd, config_path = tempfile.mkstemp(dir="tmp", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(multi_config , f)
    try:
        result = subprocess.run(
            [
                "python", "MambaDepthNAS/sub_test_latency.py",
                "--config_file", config_path,
                "--dataset", args.dataset
            ],
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": args.devices.split(',')[gpu]}
        )
    except subprocess.CalledProcessError as e:
        print("Child process failed!")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
            # print(f"[INFO] Deleted temp config file: {config_path}")

    metrics_list = json.loads(result.stdout)
    other_tensor = torch.tensor(
        [[m["latency"], m["macs"], m["params"]] for m in metrics_list],
        device=gpu
    )

    if args.multiprocessing_distributed:
        dist.all_reduce(tensor=other_tensor, op=dist.ReduceOp.SUM)

    other_tensor /= ngpus

    latency_list = other_tensor[:, 0].cpu().tolist()
    macs_list    = other_tensor[:, 1].cpu().tolist()
    # params_list = other_tensor[:, 2].cpu().tolist()

    # latency, macs_g, params_m = (other_tensor / ngpus).cpu().numpy()

    return latency_list, macs_list


@torch.no_grad()
def online_eval(args, model, dataloader_eval, gpu, ngpus, post_process=False, logger=None, sample_config=None):
    ### performance
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        image    = eval_sample_batched['image'].cuda(gpu, non_blocking=True)
        gt_depth = eval_sample_batched['depth'].cuda(gpu, non_blocking=True)
        # if not eval_sample_batched['has_valid_depth']:
        #     # print('Invalid depth. continue.')
        #     continue

        model_module = unwrap_model(model)
        model_module.backbone.set_sample_config(sample_config=sample_config)

        pred_depth = model(image)
        if post_process:
            image_flipped = flip_lr(image)
            pred_depth_flipped = model(image_flipped)
            pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

        # update eval measures
        assert gt_depth.shape[0] != 1,'batch size == 1'
        for gt_d, pr_d in zip(gt_depth, pred_depth):
            measures = compute_metrics(gt_d, pr_d, args)
            eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
            eval_measures[9] += 1
        # break  # TODO

    if args.multiprocessing_distributed:
        # group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM)

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    abs_rel = eval_measures_cpu[1]

    return abs_rel


def has_empty_stage(depth_list):
    """
    Check if any stage in depth_list is all zeros.

    Args:
        depth_list (list of np.ndarray): e.g. [array([1, 0]), array([0, 0, 0])]

    Returns:
        bool: True if any stage is all zeros, else False
    """
    return any(np.sum(stage) == 0 for stage in depth_list)


def check_safe_correct(code, depth):
    """
    Check and correct depth encoding in-place.

    For each stage in the depth encoding:
    - If all bits are 0, randomly flip one to 1.
    - Returns True if already valid (no all-zero stages), 
      or False if any correction was applied.
    """
    is_safe = True
    depth_len = sum(depth)
    offset = len(code) - depth_len
    id = 0
    for d in depth:
        segment = code[offset + id : offset + id + d]
        if np.sum(segment) == 0:
            rand_idx = np.random.randint(d)
            code[offset + id + rand_idx] = 1
            is_safe = False
        id += d
    return is_safe


class MixedCrossover(Crossover):
    '''
    A mixed crossover:
    1) two-point crossover on the front
    2) uniform crossover on the back
    '''
    def __init__(self, depth, prob=0.9, **kwargs):
        super().__init__(2, 2, prob=prob,**kwargs)
        self.depth = depth
        self.front_crossover = TwoPointCrossover()
        self.back_crossover = UniformCrossover()
    
    def _do(self, _, X, **kwargs):
        # print(f'cross input X: {X}')
        depth_len = sum(self.depth)
        front_len = X.shape[2] - depth_len

        X_f = X[:, :, :front_len].copy()
        X_b = X[:, :, -depth_len:].copy()

        Xp_front = self.front_crossover._do(None, X_f, **kwargs)
        Xp_back = self.back_crossover._do(None, X_b, **kwargs)

        Xp = np.concatenate([Xp_front, Xp_back], axis=-1)
        # print(f'cross Xp: {Xp}')

        # check any stage is all zeros and correct
        for i in range(Xp.shape[0]):
            for j in range(Xp.shape[1]):
                check_safe_correct(Xp[i, j], self.depth)

        return Xp


class MixedIntegerFromFloatMutation(PolynomialMutation):
    '''
    A mixed mutation:
    1) polynomial mutation on the front
    2) 0-1 flip mutation on the back
    '''
    def __init__(self, depth, prob=0.1, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, eta=eta, at_least_once=at_least_once, **kwargs)
        self.depth = depth

    def _do(self, problem, X, params=None, **kwargs):
        # print(f'mut input X: {X}')
        depth_len = sum(self.depth)
        front_len = X.shape[1] - depth_len

        X_f = X[:, :front_len].copy()
        X_b = X[:, -depth_len:].copy()
        
        # int code for mlp_ratio, d_state, ssd_expand
        X_f = X_f.astype(float)

        eta = get(self.eta, size=len(X_f))
        prob_var = self.prob_var if self.prob_var is not None else min(0.5, 1 / front_len)
        prob_var = get(prob_var, size=len(X_f))

        Xp_f = mut_pm(X_f, problem.xl[:front_len], problem.xu[:front_len], eta, prob_var, at_least_once=self.at_least_once)
        Xp_f_int = np.rint(Xp_f).astype(int)

        # 0-1 code for depth
        Xp_b_int = np.rint(X_b).astype(int)
        flip_prob = 1.0 / depth_len
        mask = np.random.rand(*Xp_b_int.shape) < flip_prob
        Xp_b_int[mask] = 1 - Xp_b_int[mask]

        Xp_combined = np.hstack([Xp_f_int, Xp_b_int])
        # print(f'mut Xp_combined: {Xp_combined}')

        # check any stage is all zeros and correct
        for i in range(len(Xp_combined)):
            check_safe_correct(Xp_combined[i], self.depth)
        
        # print(f'mut safe Xp_combined: {Xp_combined}')
        return Xp_combined


class SafeIntegerRandomSampling(IntegerRandomSampling):
    def __init__(self, depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        samples = []
        attempts = 0
        while len(samples) < n_samples:
            attempts += 1
            sample = np.array([np.random.randint(xl[k], xu[k] + 1) for k in range(n)])
            if check_safe_correct(sample, self.depth):
                samples.append(sample)
            if attempts > n_samples * 100:
                raise RuntimeError("Too many attempts, maybe impossible to satisfy safe condition.")
        # print(f'sample: {np.vstack(samples)}')
        return np.vstack(samples)


class NasProblem(Problem):
    def __init__(self, eval_func, latency_func, search_sapce, logger=None):
        self.eval_func = eval_func
        self.latency_func = latency_func
        self.ss = search_sapce
        self.logger = logger
        self.generation_id = 1
        super().__init__(n_var=28, n_obj=3, n_constr=0,  # 变量数，目标数，约束数
                         type_var=np.int32)
        
        self.xl = np.zeros(self.n_var)  # TODO close_depth
        self.xu = np.array(
                        [len(self.ss.mlp_ratio) - 1] * self.ss.num_stages +
                        [len(self.ss.d_state) - 1] * (self.ss.num_stages - 1) +
                        [len(self.ss.ssd_expand) - 1] * (self.ss.num_stages - 1) +
                        [1] * sum(self.ss.depth) ,
                        dtype=np.int32
                    )
        # print(f'xl: {self.xl}')
        # print(f'xu: {self.xu}')

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        # 假设 predictor 的 predict 方法接受一个二维数组，每行一个架构编码，返回对应的性能指标
        abs_rel_list = []
        latency_list = []
        macs_list = []
        sample_config_list = []

        for _x in x:
            # print('_x:', _x)
            sample_config = self.ss.decode(_x)
            sample_config_list.append(sample_config)

            # print('_x after decode:',sample_config)
            abs_rel = self.eval_func(sample_config=sample_config)
            abs_rel_list.append(abs_rel)

        latency_list, macs_list = self.latency_func(sample_config_list=sample_config_list)

        if is_main_process():
            abs_rounded = [round(float(abs), 4) for abs in abs_rel_list]
            lat_rounded = [round(float(lat), 4) for lat in latency_list]
            mac_rounded = [round(float(mac), 4) for mac in macs_list]
            self.logger.info(f"pop_abs_rel = {abs_rounded}")
            self.logger.info(f"pop_latency = {lat_rounded}")
            self.logger.info(f"pop_mac_g = {mac_rounded}")

        for i, (_x, abs, lat, mac) in enumerate(zip(x, abs_rel_list, latency_list, macs_list)):
            f[i, 0] = abs
            f[i, 1] = lat
            f[i, 2] = mac

        out["F"] = f

        if is_main_process():
            gen = self.generation_id
            self.generation_id += 1

            abs_rel_np = np.array(abs_rel_list, dtype=np.float32)
            latency_np = np.array(latency_list, dtype=np.float32)
            macs_np    = np.array(macs_list, dtype=np.float32)

            self.logger.info(f"[Gen {gen}] (New_Pop) abs_rel: mean={abs_rel_np.mean():.4f}, min={abs_rel_np.min():.4f}, max={abs_rel_np.max():.4f}")
            self.logger.info(f"[Gen {gen}] (New_Pop) latency: mean={latency_np.mean():.4f}, min={latency_np.min():.4f}, max={latency_np.max():.4f}")
            self.logger.info(f"[Gen {gen}] (New_Pop) macs_g: mean={macs_np.mean():.4f}, min={macs_np.min():.4f}, max={macs_np.max():.4f}")


class EvolutionLogger(Callback):
    def __init__(self, ss, logger=None, auto_save_path=None):
        super().__init__()
        self.ss = ss
        self.logger = logger
        self.auto_save_path = auto_save_path
        self.data["gen"] = []
        self.data["pop"] = []  # 所有种群 目标值 F 值
        self.data["pop_x"] = []  # 所有种群 编码 X 值

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")

        self.data["gen"].append(algorithm.n_gen)
        self.data.setdefault("pop", []).append(F.tolist())
        self.data.setdefault("pop_x", []).append(X.tolist())

        if self.logger:
            metrics = ["abs_rel", "latency", "macs_g"]
            for i, name in enumerate(metrics):
                vals = F[:, i]
                self.logger.info(
                    f"[Gen {algorithm.n_gen}] (Updated_Pop) {name}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}"
                )
        if self.auto_save_path:
            self.save(self.auto_save_path)

    def save(self, save_path):
        pop_rows = []
        for gen, (F, X) in enumerate(zip(self.data["pop"], self.data["pop_x"])):
            for f, x in zip(F, X):
                config = self.ss.decode(x)
                row = {
                    "gen": gen + 1,
                    "abs_rel": round(f[0], 4),
                    "latency": round(f[1], 4),
                    "macs": round(f[2], 4),
                    "config": str(config).replace("\n", "")
                }
                pop_rows.append(row)
        df_pop = pd.DataFrame(pop_rows)
        df_pop.to_csv(f"{save_path}/pop.csv", index=False)


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

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info('args: '+str(args))

    # MambaDepth model
    model = MambaDepth(args, version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)

    model.train()

    # print model params    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("== Total number of parameters: {}".format(num_params))

        num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        logger.info("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        raise ValueError("Distributed training is not enabled. Please set --distributed flag.")

    # loading supernet weight
    if args.ckpt_path:
        if os.path.isfile(args.ckpt_path):
            key = 'model'
            _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
            logger.info("Loading checkpoint {} from {} (global_step {})".format(args.ckpt_path, key, _ckpt['global_step']))
            # new_state_dict = expand_depth(_ckpt[key], self.state_dict())  # Expand depth of the pretrained weight
            incompatibleKeys = model.load_state_dict(_ckpt[key], strict=False)
            logger.info("== missing_keys: {}".format(incompatibleKeys))
            del _ckpt
        else:
            raise RuntimeError(f"Checkpoint path does not exist: {args.ckpt_path}")

    cudnn.benchmark = True
    dataloader_eval = NewDataLoader(args, 'online_eval')

    ### seach space
    ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)

    ### evolution
    model.eval()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info(f"==> Starting evolution...")
    
    eval_func = partial(online_eval, args=args, model=model, dataloader_eval=dataloader_eval, gpu=gpu, ngpus=ngpus_per_node, post_process=True, logger=logger)
    latency_func = partial(test_lat_mac, args=args, gpu=gpu, ngpus=ngpus_per_node)
    problem = NasProblem(eval_func=eval_func, latency_func=latency_func, search_sapce=ss, logger=logger)
    logger_cb = EvolutionLogger(ss=ss, logger=logger, auto_save_path=args.log_directory)  # to solve GPU Memory error
    method = NSGA2(
        pop_size=args.population_size,  # initialize with current nd archs
        sampling=SafeIntegerRandomSampling(depth=ss.depth),
        crossover=MixedCrossover(depth=ss.depth, prob=args.cross_p),
        mutation=MixedIntegerFromFloatMutation(depth=ss.depth, prob=args.mut_p, eta=1.0),
        eliminate_duplicates=True)

    res = minimize(
        problem, method, termination=('n_gen', args.n_iter), save_history=False, callback=logger_cb, verbose=False, seed=1274395)
    
    ### show results
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        
        # save all pop
        logger_cb.save(save_path=args.log_directory)

        # show best pop
        optimal_solutions = res.X.tolist()
        optimal_objective_values = res.F.tolist()

        logger.info("==="*20)
        logger.info("Optimal Code:")
        logger.info(optimal_solutions)

        logger.info("Optimal Objective Values:")
        for id, val in enumerate(optimal_objective_values):
            logger.info(f"id:{id}, abs_rel={val[0]:.4f}, latency={val[1]:.1f}ms, MACs={val[2]:.1f}G")

        logger.info("Optimal Config:")
        for id, conf in enumerate(optimal_solutions):
            logger.info(f"id:{id}, {ss.decode(conf)}")

        logger.info(f"==> Finished evolution!")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

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

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
