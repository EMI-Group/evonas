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
from mmengine.runner import Runner
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
from pymoo.util.misc import crossover_mask

from functools import partial
from pymoo.core.callback import Callback
import pandas as pd
from typing import Dict, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

from utils import  convert_arg_line_to_args, get_root_logger, unwrap_model, str2list, is_main_process

from search_space import MambaSearchSpace
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.runner.amp import autocast as mautocast
from mmengine.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)
from SegmentNAS.networks import model
'''
search the best network on object dataset, e.g. COCO

Note: latency tests included; avoid running other GPU tasks.
'''


### start search
# python SegmentNAS/search.py SegmentNAS/configs/upernet/01_search/search_cityscapes.txt

### clear
# echo quit | nvidia-cuda-mps-control

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--config',                type=str,   help='model config file path', required=True)
    parser.add_argument('--model_name',                type=str,   help='model name', default='DetectionSearch')
    parser.add_argument('--devices',                   type=str, default='0,1', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

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
    parser.add_argument('--mut_eta',                   type=float,  help='eta of mutation', default=1.0)
    parser.add_argument('--p_bit',                     type=float,  help='probability of depth bit cross', default=0.5)
    

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints', required=True)


    # Multi-gpu searching
    parser.add_argument('--model_batch',               type=int,   help='number of models per process to eval', default=3)
    parser.add_argument('--concurrency',               type=int,   help='number of process per GPU to use for eval', default=4)
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args


def _spawn_one_gpu(idx_chunk, cfg_chunk, gpu_id):
    """
    写临时文件 → 调 sub_test_latency.py → 返回 [(arch_idx, latency, macs), ...]
    """
    fd, path = tempfile.mkstemp(dir="tmp", suffix=f"_GPU{gpu_id}.json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg_chunk, f)

    try:
        res = subprocess.run(
            ["python", "SegmentNAS/sub_test_latency.py",
             "--config_file", path,
             "--base_config", "SegmentNAS/configs/upernet/upernet_nas_subnet_cityscapes.py"],
            capture_output=True, text=True, check=True, timeout=500,
            env={**os.environ,
                 "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1",
                 "CUDA_VISIBLE_DEVICES": gpu_id}
        )
        metrics = json.loads(res.stdout)
        # concate idx
        return [(i, m["latency"], m["macs"]) for i, m in zip(idx_chunk, metrics)]
    
    except subprocess.CalledProcessError as e:
        print("Child process failed!")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lat_mac_mutil(sample_config_list, gpu_arg="0,1,2,3"):

    if isinstance(gpu_arg, str):  # → ['0','1','2','3']
        gpus = [s.strip() for s in gpu_arg.split(',') if s.strip() != '']
    elif isinstance(gpu_arg, (list, tuple)):
        gpus = [str(x) for x in gpu_arg]
    else:
        raise ValueError("gpu_list should be str or list/tuple")
    
    G = len(gpus) 

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    pre_cfgs = []
    for raw in sample_config_list:
        pre_cfgs.append({
            "mlp_ratio":  raw["mlp_ratio"],
            "d_state":    raw["d_state"] + [-1],
            "ssd_expand": raw["expand"]  + [-1],
            "depth": [int(sum(d)) for d in raw["depth"]],
        })
    
    idx_chunks, cfg_chunks = [[] for _ in gpus], [[] for _ in gpus]
    for idx, (cfg, gpu_id) in enumerate(zip(pre_cfgs, cycle(range(G)))):
        idx_chunks[gpu_id].append(idx)
        cfg_chunks[gpu_id].append(cfg)

    latency_out = [None] * len(sample_config_list)
    macs_out    = [None] * len(sample_config_list)

    with ThreadPoolExecutor(max_workers=G) as pool:
        futs = []
        for gid, (idx_chunk, cfg_chunk) in enumerate(zip(idx_chunks, cfg_chunks)):
            if not idx_chunk:   
                continue
            fut = pool.submit(_spawn_one_gpu,
                              idx_chunk, cfg_chunk, gpus[gid])
            futs.append(fut)

        for fut in as_completed(futs):
            for arch_idx, lat, mac in fut.result():
                latency_out[arch_idx] = lat
                macs_out[arch_idx]    = mac

    return latency_out, macs_out


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


class PUniformCrossover(Crossover):

    def __init__(self, p_bit=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.p_bit = p_bit

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < self.p_bit
        _X = crossover_mask(X, M)
        return _X
    

class MixedCrossover(Crossover):
    '''
    A mixed crossover:
    1) two-point crossover on the front
    2) uniform crossover on the back
    '''
    def __init__(self, depth, prob=0.9, p_bit=0.5, **kwargs):
        super().__init__(2, 2, prob=prob,**kwargs)
        self.depth = depth
        self.front_crossover = TwoPointCrossover()
        self.back_crossover = PUniformCrossover(p_bit=p_bit)
        # self.back_crossover = UniformCrossover()
    
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


class JointModel(nn.Module):
    """Wrap a list of sub‑models so *one* forward pass yields a list of outputs.

    • 训练阶段:  forward 合并 → backward 逐模型，梯度互不干扰。
    • 推理阶段:  一次 forward 拿全部输出，吞掉 Python/kernel 开销。
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return [m(**x, mode='predict') for m in self.models]


def worker_main_eval_func(task_q, result_q, args, cfg):
    """ evaluates sub‑networks in parallel """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Please check your GPU or CUDA installation.")
    device = torch.device("cuda")
    dataloader_eval = Runner.build_dataloader(cfg.val_dataloader, seed=42)
    
    dataset_meta = getattr(dataloader_eval.dataset, 'metainfo', None)
    assert dataset_meta is not None and 'classes' in dataset_meta, \
        'dataset.metainfo 为空或缺少 classes，请在数据集里补全 METAINFO/metainfo。'


    if args.ckpt_path:
        if os.path.isfile(args.ckpt_path):
            key = 'model'
            _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
            state_dict = {k.replace('module.', '', 1): v for k, v in _ckpt[key].items()}
            print("Loading checkpoint {} from {} (global_step {})".format(args.ckpt_path, key, _ckpt['global_step']))
        else:
            raise RuntimeError(f"Checkpoint path does not exist: {args.ckpt_path}")
    
    models: List[nn.Module] = []

    while True:
        task: Tuple[int, List[Dict]] = task_q.get()
        if task is None:
            break  # poison‑pill → terminate
        start_idx, tokens = task
        n = len(tokens)

        evaluators = [Evaluator(metrics=cfg.val_evaluator) for _ in range(n)]
        for evaluator in evaluators:
            evaluator.dataset_meta = dataset_meta
            
        if not models:
            for _ in range(n):
                m = MODELS.build(cfg.model).to(device)
                incompatibleKeys = m.load_state_dict(state_dict, strict=False)
                print("== missing_keys: {}".format(incompatibleKeys))
                models.append(m)

        for gene, m in zip(tokens, models):
            model_module = unwrap_model(m)
            model_module.backbone.backbone.set_sample_config(sample_config=gene)

        joint_model = JointModel(models).to(device)
        joint_model.eval()

        with torch.no_grad():
            for _, eval_sample_batched in enumerate(tqdm(dataloader_eval)):
                batched = unwrap_model(models[0]).data_preprocessor(eval_sample_batched, False)
                with mautocast(enabled=False):
                    preds = joint_model(batched)

                # update eval measures
                for i in range(n):
                    evaluators[i].process(data_samples=preds[i], data_batch=eval_sample_batched)

            mIoU_list = []
            for i in range(n):
                metrics_dict = evaluators[i].evaluate(len(dataloader_eval.dataset))
                mIoU_list.append(metrics_dict['mIoU'])
        
        result_q.put((start_idx, mIoU_list))
        torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()


class NasProblem(Problem):
    def __init__(self, latency_func, search_sapce,
                  gpus: int | str,
                  concurrency: int,
                  batch: int,
                  logger=None,
                  w_args=None,
                  m_cfg=None):
        self.worker_eval_func = worker_main_eval_func
        self.latency_func = latency_func
        self.ss = search_sapce
        self.batch = batch
        self.logger = logger
        self.generation_id = 1
        super().__init__(n_var=28, n_obj=3, n_constr=0,  # 变量数，目标数，约束数
                         type_var=np.int32)
        
        self.xl = np.zeros(self.n_var)
        self.xu = np.array(
                [len(self.ss.mlp_ratio) - 1, len(self.ss.d_state) - 1, len(self.ss.ssd_expand) - 1] * (self.ss.num_stages - 1) +
                [len(self.ss.mlp_ratio) - 1] + 
                [1] * sum(self.ss.depth) ,
                dtype=np.int32
                )
        
        # Worker
        gpus_list = [str(gpus)] if isinstance(gpus, int) else [str(g) for g in
                                                               (gpus if isinstance(gpus, list) else gpus.split(',')) if
                                                               g]
        gpus_list = gpus_list or ['']
        num_workers = len(gpus_list) * concurrency
        self.task_q, self.result_q = mp.Queue(), mp.Queue()

        self.workers = []
        for i in range(num_workers):
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list[i % len(gpus_list)]
            p = mp.Process(target=self.worker_eval_func, args=(self.task_q, self.result_q, w_args, m_cfg))
            p.start()
            self.workers.append(p)

    def exit_worker(self):
        for _ in self.workers:
            try:
                self.task_q.put_nowait(None)
            except Exception:
                pass  # queue is closed

        # 关闭队列，回收线程
        try:
            self.task_q.close();  self.task_q.join_thread()
            self.result_q.close(); self.result_q.join_thread()
        except (AttributeError, ValueError):
            pass

        # terminate if wait for long time
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate(); p.join()

        self.workers.clear()

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        # 假设 predictor 的 predict 方法接受一个二维数组，每行一个架构编码，返回对应的性能指标
        sample_config_list = [self.ss.decode(_x) for _x in x]
        eval_s_time = time.time()
        n = x.shape[0]
        for start in range(0, n, self.batch):
            self.task_q.put((start, sample_config_list[start: start + self.batch]))

        mIoU_list = [None] * n
        num_batches = (n + self.batch - 1) // self.batch
        for _ in range(num_batches):
            start_idx, objs_batch = self.result_q.get()
            mIoU_list[start_idx: start_idx + len(objs_batch)] = objs_batch

        eval_cost_time = time.time() - eval_s_time
        lat_s_time = time.time()

        latency_list, macs_list = self.latency_func(sample_config_list=sample_config_list)
        lat_cost_time = time.time() - lat_s_time

        self.logger.info(f"pop_mIoU = {[round(float(miou), 4) for miou in mIoU_list]}")
        self.logger.info(f"pop_latency = {[round(float(lat), 4) for lat in latency_list]}")
        self.logger.info(f"pop_mac_g = {[round(float(mac), 4) for mac in macs_list]}")

        gen = self.generation_id
        self.generation_id += 1

        self.logger.info(f'[Gen {gen}] eval_cost_time: {eval_cost_time:.2f}s')
        self.logger.info(f'[Gen {gen}] lat_cost_time: {lat_cost_time:.2f}s')

        for i, (_x, miou, lat, mac) in enumerate(zip(x, mIoU_list, latency_list, macs_list)):
            f[i, 0] = 1.0 - miou  # mIoU is to be maximized
            f[i, 1] = lat
            f[i, 2] = mac

        out["F"] = f
        
        miou_np = np.array(mIoU_list, dtype=np.float32)
        latency_np = np.array(latency_list, dtype=np.float32)
        macs_np    = np.array(macs_list, dtype=np.float32)

        self.logger.info(f"[Gen {gen}] (New_Pop) box_mIoU: mean={miou_np.mean():.4f}, min={miou_np.min():.4f}, max={miou_np.max():.4f}")
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
            metrics = ["mIoU", "latency", "macs_g"]
            for i, name in enumerate(metrics):
                vals = F[:, i]
                if name == "mIoU":
                    vals = 1.0 - vals  # convert back to mIoU
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
                    "mIoU": round(1.0 - f[0], 4),
                    "latency": round(f[1], 4),
                    "macs": round(f[2], 4),
                    "config": str(config).replace("\n", "")
                }
                pop_rows.append(row)
        df_pop = pd.DataFrame(pop_rows)
        df_pop.to_csv(f"{save_path}/pop.csv", index=False)



def start_mps():
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
    subprocess.run(["pkill", "-9", "nvidia-cuda-mps-control"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        print("[MPS] Started successfully", flush=True)
    except Exception as e:
        print(f"[MPS] Failed: {e}", flush=True)


def main():
    start_mps()
    args = parse_args()
    cfg = Config.fromfile(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    args.distributed = False

    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    # torch.cuda.empty_cache()
    logger = get_root_logger(args.log_directory)

    # cudnn.benchmark = True

    ### seach space
    ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)

    ### evolution
    logger.info(f"==> Starting evolution...")
    
    latency_func = partial(test_lat_mac_mutil, gpu_arg=args.devices)
    problem = NasProblem(
        latency_func=latency_func,
        search_sapce=ss, 
        gpus=args.devices, 
        concurrency=args.concurrency, 
        batch=args.model_batch, 
        logger=logger,
        w_args=args,
        m_cfg=cfg
    )
    logger_cb = EvolutionLogger(
        ss=ss, 
        logger=logger, 
        auto_save_path=args.log_directory
    )  # to solve GPU Memory error
    method = NSGA2(
        pop_size=args.population_size,  # initialize with current nd archs
        sampling=SafeIntegerRandomSampling(depth=ss.depth),
        crossover=MixedCrossover(depth=ss.depth, prob=args.cross_p, p_bit=args.p_bit),
        mutation=MixedIntegerFromFloatMutation(depth=ss.depth, prob=args.mut_p, eta=args.mut_eta),
        eliminate_duplicates=True
    )

    res = minimize(
        problem, 
        method, 
        termination=('n_gen', args.n_iter), 
        save_history=False, 
        callback=logger_cb, 
        verbose=False, 
        seed=1274395
    )
    
    problem.exit_worker()

    ### show results
    logger_cb.save(save_path=args.log_directory)

    optimal_solutions = res.X.tolist()
    optimal_objective_values = res.F.tolist()

    logger.info("==="*20)
    logger.info("Optimal Code:")
    logger.info(optimal_solutions)

    logger.info("Optimal Objective Values:")
    for id, val in enumerate(optimal_objective_values):
        logger.info(f"id:{id}, mIoU={1.0 - val[0]:.4f}, latency={val[1]:.1f}ms, MACs={val[2]:.1f}G")

    logger.info("Optimal Config:")
    for id, conf in enumerate(optimal_solutions):
        logger.info(f"id:{id}, {ss.decode(conf)}")

    logger.info(f"==> Finished evolution!")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
