import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import random
import os, sys, time, gc
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # to set in config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
from tqdm import tqdm
from mmengine import Config
from timm.scheduler.cosine_lr import CosineLRScheduler

from tensorboardX import SummaryWriter

from utils import barrier, block_print, enable_print, build_optimizer, \
                       convert_arg_line_to_args, get_root_logger, unwrap_model, str2list
from DetectionNAS.networks import model
from search_space import MambaSearchSpace
from torch.amp import autocast, GradScaler

from supernet_pool import ArchitecturePool
from utils import build_iter_lambda_scheduler
from mmdet.registry import MODELS, METRICS  
from mmengine.evaluator import Evaluator
from torch.utils.data import DataLoader
from contextlib import nullcontext

from mmengine.registry import init_default_scope
init_default_scope('mmdet')
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.amp import autocast as mautocast
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

'''fine-tune supernet on object dataset, e.g. COCO'''

### develop
# python DetectionNAS/train.py DetectionNAS/configs/coco_nas_kd.txt 

### final
# sh scripts/detection/supernet_steptrain.sh

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDetection PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('config',              help='train config file path')
    parser.add_argument('teacher_config', nargs='?', default=None, help='teacher config file path (optional)')
    parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='SuperNet')
    parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
    parser.add_argument('--devices',                    type=str, default='0,1', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

    # Search space
    parser.add_argument('--mlp_ratio',                 type=str2list, default=[4.0])
    parser.add_argument('--d_state',                   type=str2list, default=[64])
    parser.add_argument('--ssd_expand',                type=str2list, default=[2])
    parser.add_argument('--open_depth',                action='store_true', help='if set, will open depth sampling, otherwise use fixed depth for all stages')
    parser.add_argument('--min_ones',                  type=int,    help='minimum number of active layers in each stage during sampling', default=1)
    parser.add_argument('--width_multiplier',          type=float, help='factor to scale the number of channels in each layer (applies only to final model)', default=1.0)

    # Knowledge distillation
    parser.add_argument('--kd_ratio',                  type=float,   help='the ratio of knowledge distillation', default=0)
    parser.add_argument('--f_distill',                 action='store_true',   help='if set, will use mid feature distillation loss')
    parser.add_argument('--alpha_1',                   type=float, help='weight coefficient for spatial loss (spat_loss)', default=0.08)
    parser.add_argument('--alpha_2',                   type=float, help='weight coefficient for frequency loss (freq_loss)', default=0.06)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints',default=None)

    # Training
    parser.add_argument('--optimizer',                 type=str,   help='optimizer to use', default='adamw')
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--weight_decay',              type=float, help='weight decay for optimizer', default=0.05)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--dynamic_batch_size',        type=int,   help='the number of dynamic batch size', default=1)
    parser.add_argument('--max_norm',                  type=float, help='maximum norm for gradient clipping, (grad clip is not used if -1 or 0)', default=-1)
    parser.add_argument('--amp',                                   help='enable mixed precision (AMP)', action='store_true')

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
    # Online eval
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')
    # experimental
    parser.add_argument('--arch_pool', action='store_true', help='if set, will train arch pool of subnet during training')
    parser.add_argument('--accum_steps', type=int, default=1,
                    help='gradient accumulation steps across different batches')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args


@torch.no_grad()
def online_eval(args, model, dataloader_eval, evaluator=None, logger=None, ss=None):
    model.eval()
    # 显存有限时，防止OOM
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval)):
        # random sample subnet
        sample_config = ss.sample(n_samples=1)[0]
        model_module = unwrap_model(model)
        model_module.backbone.backbone.set_sample_config(sample_config=sample_config)  # TODO
        with mautocast(enabled=args.amp):
            preds = model_module.val_step(eval_sample_batched)
        evaluator.process(data_samples=preds, data_batch=eval_sample_batched)

    metrics_dict = evaluator.evaluate(len(dataloader_eval.dataset))
    logger.info(metrics_dict)

    # eval 完再清一次，避免回到 train 还残留
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_dict


def main_worker(gpu, ngpus_per_node, args, cfg):
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

    # model = MambaDepth(args, version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain, remap=args.remap)
    model = MODELS.build(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.init_weights()
    model.train()
    
    # load_checkpoint(model, './checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth', map_location='cpu')

    ### teacher model
    if args.kd_ratio > 0 or args.f_distill:
        t_cfg = Config.fromfile(args.teacher_config)
        teacher_model = MODELS.build(t_cfg.model)
        load_checkpoint(teacher_model, './checkpoints/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220510_201004-3d24f5a4.pth', map_location='cpu', strict=True)
        logger.info("Successed loading weights for teacher model")
    
    from distillation.fmdv2 import FreqMaskingDistillLossv2
    dis_modules_s4 = FreqMaskingDistillLossv2(
        alpha=[args.alpha_1, args.alpha_2],
        student_dims=512, # student feature dimension
        teacher_dims=768,  # teacher feature dimension
        pos_dims=768,  # same to teacher_dims
        self_query=True,
        softmax_scale=[5.,5.],
        num_heads=16
    ) if args.f_distill else None

    # print model params    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info("== Total number of parameters: {}".format(num_params))

        num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
        logger.info("== Total number of learning parameters: {}".format(num_params_update))

        def count_params(m):
            if m is None:
                return 0, 0
            total = sum(p.numel() for p in m.parameters()) / 1e6
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
            return total, trainable

        parts = {
            "backbone": getattr(model, "backbone", None),
            "neck": getattr(model, "neck", None),
            "rpn_head": getattr(model, "rpn_head", None),
            "roi_head": getattr(model, "roi_head", None),
        }

        used = set()
        for name, module in parts.items():
            total, trainable = count_params(module)
            logger.info(f"== {name:<8}: total {total:.2f}M, trainable {trainable:.2f}M")
            if module is not None:
                used |= set(module.parameters())

        others = [p for p in model.parameters() if p not in used]
        total_others = sum(p.numel() for p in others) / 1e6
        trainable_others = sum(p.numel() for p in others if p.requires_grad) / 1e6
        logger.info(f"== {'others':<8}: total {total_others:.2f}M, trainable {trainable_others:.2f}M")

        total_all = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"== {'model':<8}: total {total_all:.2f}M, trainable {trainable_all:.2f}M")

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False)
            if args.kd_ratio > 0 or args.f_distill:
                teacher_model.cuda(args.gpu)
                teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], broadcast_buffers=False)
            if args.f_distill: 
                dis_modules_s4.init_weights()
                dis_modules_s4.cuda(args.gpu)
                dis_modules_s4 = torch.nn.parallel.DistributedDataParallel(dis_modules_s4, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False)
        else:
            assert False, 'developing'
    else:
        raise ValueError("Distributed training is not enabled. Please set --distributed flag.")


    if args.ckpt_path:
        key = 'model'
        _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
        logger.info("Successfully load whole ckpt {} from {}".format(args.ckpt_path, key))
        incompatibleKeys = model.load_state_dict(_ckpt[key], strict=False)
        logger.info("== missing_keys: {}".format(incompatibleKeys))
        del _ckpt

    if args.kd_ratio > 0 or args.f_distill:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    ################## build dataset #######################
    dataloader = Runner.build_dataloader(cfg.train_dataloader, seed=42)
    dataloader_eval = Runner.build_dataloader(cfg.val_dataloader, seed=42)

    evaluator = Evaluator(metrics=cfg.val_evaluator)
    dataset_meta = getattr(dataloader_eval.dataset, 'metainfo', None)
    assert dataset_meta is not None and 'classes' in dataset_meta, \
        'dataset.metainfo 为空或缺少 classes，请在数据集里补全 METAINFO/metainfo。'
    evaluator.dataset_meta = dataset_meta

    model_module = unwrap_model(model)

    for m in model_module.modules():
        if isinstance(m, nn.ReLU):  # TODO
            m.inplace = False
            
    ################## build optimizer ######################
    optimizer = build_optimizer(args, model_module, logger, dis_modules_s4)
        
    scaler = GradScaler(enabled=args.amp)
    # cudnn.benchmark = True

    ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)

    # metrics_dict = online_eval(args, teacher_model, dataloader_eval, evaluator=evaluator, logger=logger, ss=ss)
    # assert False,'test'

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    global_step = -1
    key_metrics = ['coco/bbox_mAP', 'coco/segm_mAP']
    best_vals = {k: float('-inf') for k in key_metrics}
    best_steps = {k: -1 for k in key_metrics}

    start_time = time.time()
    duration = 0

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = 0  # need equal to global_step // steps_per_epoch if resume

    if args.num_epochs==36:
        milestones = (24, 33)
    elif args.num_epochs==12:
        milestones = (8, 11)
    elif args.num_epochs==8:
        milestones = (5, 7)
    else:
        milestones = (max(1, int(args.num_epochs*0.67)),)
    scheduler = build_iter_lambda_scheduler(
        optimizer,
        warmup_iters=1000,
        warmup_ratio=0.001,
        milestones=milestones,
        iters_per_epoch=steps_per_epoch,
        gamma=0.1
    )
    # from torch.optim.lr_scheduler import MultiStepLR
    # scheduler = MultiStepLR(
    #     optimizer,
    #     milestones=[8, 11],
    #     gamma=0.1
    # )

    # with open('./detection_model_nas.log', 'w') as f:
    #     f.write(str(model))
    # assert False,'debug'

    features = {}
    def hook_fn_S(module, input, output):
        features['feat_S_s4'] = output[3]
    def hook_fn_T(module, input, output):
        features['feat_T_s4'] = output[3]

    if args.arch_pool:
        arch_pool = ArchitecturePool(pool_size=3)

    
    # 辅助函数：将重复的蒸馏 Loss 计算提取出来
    def calc_distill_loss_and_return_components(args, loss, feat_T_s4, features, global_step, num_total_steps):
        spat_loss = torch.tensor(0.0).to(loss.device)
        freq_loss = torch.tensor(0.0).to(loss.device)
        kd_loss = torch.tensor(0.0).to(loss.device)

        # 特征蒸馏
        if args.f_distill and feat_T_s4 is not None:
            feat_S_s4 = features.pop('feat_S_s4')
            spat_loss, freq_loss = dis_modules_s4(feat_S_s4, feat_T_s4)
            
            warmup = min(1.0, global_step / (0.05 * num_total_steps))
            w_ratio = warmup * (0.2 + 0.8 * (1 - global_step / num_total_steps))
            spat_loss = w_ratio * spat_loss
            freq_loss = w_ratio * freq_loss
        
        # 总 Loss 计算
        if args.kd_ratio > 0 or args.f_distill:
            total_loss = (1 - args.kd_ratio) * loss + args.kd_ratio * kd_loss + spat_loss + freq_loss
        else:
            total_loss = loss
            
        return total_loss, spat_loss, freq_loss

    ################### train loop (Sandwich Rule Modified) ########################
    model.train()
    # 定义三明治法则的采样策略：Max + Min + 2个Random
    n_random_subnets = 2
    total_subnets_per_step = 1 + n_random_subnets
    max_config = {'mlp_ratio': [4.0,4.0,4.0,4.0], 'd_state': [64,64,64,64], 'expand': [4,4,4,4], 'depth': [[1,1],[1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1]]}

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)
        
        random.seed(epoch)
        np.random.seed(epoch)

        optimizer.zero_grad()

        for step, sample_batched in enumerate(dataloader):
            before_op_time = time.time()
            global_step += 1

            log_avg_loss = 0.0
            log_spat_loss = 0.0
            log_freq_loss = 0.0

            feat_T_s4 = None
            if args.kd_ratio > 0 or args.f_distill:
                with torch.no_grad():
                    teacher_model.eval()
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                        model_module = unwrap_model(teacher_model)
                        handle_T = model_module.backbone.register_forward_hook(hook_fn_T)
                        # tearcher_preds = model_module.val_step(sample_batched)
                        batched = model_module.data_preprocessor(sample_batched, False)
                        model_module.extract_feat(batched["inputs"])  # 输出的是FPN之后的结果
                    if args.f_distill:
                        feat_T_s4 = features.pop('feat_T_s4')
                        handle_T.remove()

            my_context = model.no_sync() if args.distributed else nullcontext()

            # Sandwich Rule Step 1: 训练 Max Subnet (The Top Bun)
            with my_context:
                with autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    model_module = unwrap_model(model)
                    model_module.backbone.backbone.set_sample_config(sample_config=max_config)
                    batched = model_module.data_preprocessor(sample_batched, True)

                    handle_S = None
                    if args.f_distill:
                        handle_S = model_module.backbone.register_forward_hook(hook_fn_S)

                    losses = model(**batched, mode='loss')
                    loss_max, _ = model_module.parse_losses(losses)
                    loss_max, s_loss, f_loss = calc_distill_loss_and_return_components(
                        args, loss_max, feat_T_s4, features, global_step, num_total_steps
                    )
                    if handle_S: handle_S.remove()
                
                # 记录 Log
                log_avg_loss += loss_max.item()
                log_spat_loss += s_loss.item()
                log_freq_loss += f_loss.item()

                loss_max = loss_max / total_subnets_per_step
                if args.amp:
                    scaler.scale(loss_max).backward()
                else:    
                    loss_max.backward()

            # Sandwich Rule Step 3: 训练 Random Subnets (The Meat)
            for i in range(n_random_subnets):
                # 只有最后一个子网 Backward 时才同步梯度 (DDP)
                is_last_subnet = (i == n_random_subnets - 1)
                my_context = nullcontext() if (is_last_subnet or not args.distributed) else model.no_sync()

                with my_context:
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                        sample_config = ss.sample(n_samples=1)[0]
                        model_module.backbone.backbone.set_sample_config(sample_config=sample_config)

                        if args.f_distill:
                            handle_S = model_module.backbone.register_forward_hook(hook_fn_S)

                        losses = model(**batched, mode='loss')
                        loss_rand, _ = model_module.parse_losses(losses)
                        loss_rand, s_loss, f_loss = calc_distill_loss_and_return_components(
                            args, loss_rand, feat_T_s4, features, global_step, num_total_steps
                        )
                        if handle_S: handle_S.remove()
                    
                    # 记录 Log
                    log_avg_loss += loss_rand.item()
                    log_spat_loss += s_loss.item()
                    log_freq_loss += f_loss.item()

                    loss_rand = loss_rand / total_subnets_per_step
                    if args.amp:
                        scaler.scale(loss_rand).backward()
                    else:
                        loss_rand.backward()

            if args.max_norm > 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                # print(f'grad norm: {grad_norm:.2f}')

            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            current_lr_kd = optimizer.param_groups[1]['lr'] if args.f_distill else 0

            # 计算 Average
            log_avg_loss /= total_subnets_per_step
            log_spat_loss /= total_subnets_per_step
            log_freq_loss /= total_subnets_per_step

            # show log and save result
            if global_step % 100 == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    logger.info(
                        '[epoch][s/gs]: [{}][{}/{}], lr: {:.8f}, '
                        'Avg_Loss: {:.4f}, Spat_Loss: {:.4f}, Freq_Loss: {:.4f}'.format(
                        epoch, step, global_step, current_lr, 
                        log_avg_loss, log_spat_loss, log_freq_loss
                    ))
                    
                    if torch.isnan(torch.tensor(log_avg_loss)).any():
                        logger.info('NaN in loss occurred. Aborting training.')
                        return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = cfg.train_dataloader.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                logger.info(print_string.format(args.gpu, examples_per_sec, log_avg_loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('train_loss', log_avg_loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('learning_rate_kd', current_lr_kd, global_step)
                    writer.add_scalar('var_average', var_sum.item()/var_cnt, global_step)


        if args.do_online_eval:
            barrier()
            time.sleep(0.1)
            metrics_dict = online_eval(args, model, dataloader_eval, evaluator=evaluator, logger=logger, ss=ss)
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                for k, v in metrics_dict.items():
                    eval_summary_writer.add_scalar(k, float(v), int(global_step))
                eval_summary_writer.flush()

                if metrics_dict is not None:
                    for metric_name in ['coco/bbox_mAP', 'coco/segm_mAP']:
                        curr = float(metrics_dict[metric_name])
                        is_best = False
                        if curr >= best_vals[metric_name]:
                            old_best = best_vals[metric_name]
                            old_step = best_steps[metric_name]

                            best_vals[metric_name] = curr
                            best_steps[metric_name] = int(global_step)
                            is_best = True
                        
                        if is_best:
                            # clean old best（if exists）
                            old_name = f"/model-{old_step}-best_{metric_name.replace('/', '_')}_{old_best:.5f}"
                            old_path = os.path.join(args.log_directory, args.model_name + old_name)
                            if os.path.exists(old_path):
                                os.remove(old_path)

                            # save new best weight
                            save_name = f"/model-{global_step}-best_{metric_name.replace('/', '_')}_{curr:.5f}"
                            save_path = os.path.join(args.log_directory, args.model_name + save_name)
                            print(f'New best for {metric_name}: {curr:.5f}. Saving {save_name}')
                            checkpoint = {
                                'global_step': global_step,
                                'model': model.state_dict(),
                                # 'optimizer': optimizer.state_dict(),  # if need
                                'best_vals': best_vals,
                                'best_steps': best_steps,
                                'best_metric_name': metric_name,
                            }
                            torch.save(checkpoint, save_path)

                        if is_best and metric_name == 'coco/bbox_mAP':
                            link_map = {
                                'coco/bbox_mAP': 'best_bbox_mAP.pth',
                                'coco/segm_mAP': 'best_segm_mAP.pth'
                            }
                            link_name = link_map[metric_name]
                            symlink_path = os.path.join(args.log_directory, link_name)
                            # 目标是相对路径，便于移动目录
                            rel_target = os.path.relpath(save_path, start=os.path.dirname(symlink_path))
                            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                                os.remove(symlink_path)
                            os.symlink(rel_target, symlink_path)

            model.train()
            block_print()
            enable_print()

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices

    command = 'mkdir -p ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1][1:] + ' ' + args_out_path
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        command = 'cp DetectionNAS/train.py ' + aux_out_path
        os.system(command)
        # command = 'mkdir -p ' + networks_savepath + ' && cp DetectionNAS/networks/*.py ' + networks_savepath
        # os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        main_worker(args.gpu, ngpus_per_node, args, cfg)


if __name__ == '__main__':
    main()
