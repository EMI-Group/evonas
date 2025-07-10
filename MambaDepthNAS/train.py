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
from timm.scheduler.cosine_lr import CosineLRScheduler

from tensorboardX import SummaryWriter

from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, get_root_logger, unwrap_model, str2list, infer, compute_metrics
from networks.model import MambaDepth
from search_space import MambaSearchSpace

'''fine-tune supernet on object dataset, e.g. KITTI, NYU'''

### develop
# python MambaDepthNAS/train.py configs/fine_tune_kitti_0_maxnet_debug.txt

### final
# sh whole_run.sh
# or
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_0_maxnet.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_1_state_1.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_2_state_2.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_3_mlp_1.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_4_mlp_2.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_5_ssd_1.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_6_ssd_2.txt
# python MambaDepthNAS/train.py configs/prog_shrink/supernet_train_kitti_7_depth.txt

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--teacher_config",            type=str,   required=True)
    parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)
    parser.add_argument('--devices',                    type=str, default='0,1', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

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
    parser.add_argument('--open_depth',                action='store_true', help='if set, will open depth sampling, otherwise use fixed depth for all stages')
    parser.add_argument('--min_ones',                  type=int,    help='minimum number of active layers in each stage during sampling', default=1)

    # Knowledge distillation
    parser.add_argument('--kd_ratio',                  type=float,   help='the ratio of knowledge distillation', default=0)
    parser.add_argument('--f_distill',                 action='store_true',   help='if set, will use mid feature distillation loss')
    parser.add_argument('--alpha_1',                   type=float, help='weight coefficient for spatial loss (spat_loss)', default=0.08)
    parser.add_argument('--alpha_2',                   type=float, help='weight coefficient for frequency loss (freq_loss)', default=0.06)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints',default=None)

    # Training
    parser.add_argument('--min_lr',                    type=float, help='', default=5e-6)
    parser.add_argument('--warmup_lr',                 type=float, help='', default=5e-7)
    parser.add_argument('--warmup_epochs',             type=int, help='', default=3)
    parser.add_argument('--weight_decay',              type=float, help='', default=0.05)
    parser.add_argument('--dynamic_batch_size',        type=int,   help='the number of dynamic batch size', default=1)
    parser.add_argument('--resume',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--batch_size',                type=int,   help='this is the global batch size for all gpus', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--max_norm',                  type=float, help='maximum norm for gradient clipping, (grad clip is not used if -1 or 0)', default=-1)
    parser.add_argument('--optimizer',                 type=str,   help='optimizer to use, adam or adamw', default='adamw')

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
        # random sample subnet
        sample_config = ss.sample(n_samples=1)[0]
        model_module = unwrap_model(model)
        model_module.backbone.set_sample_config(sample_config=sample_config)
        # print(sample_config)

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
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

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
    model = MambaDepth(args, version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain)
    if args.dynamic_tanh:  ### 替换归一化层（会破坏预训练权重，暂不使用）
        assert False, 'Dynamic tanh is not good for this yet!'
        from networks.dynamic_tanh import convert_ln_to_dyt
        model = convert_ln_to_dyt(model)
        logger.info("==> Using dynamic_tanh!")

    model.train()

    ### teacher model
    if args.kd_ratio > 0:
        from networks.depthAny.builder import build_model
        t_config = Config.fromfile(args.teacher_config)
        teacher_model = build_model(t_config)  # include loading pretrained weight

        if args.f_distill:
            from distillation.fmdv2 import FreqMaskingDistillLossv2
            dis_modules_s4 = FreqMaskingDistillLossv2(
                alpha=[args.alpha_1, args.alpha_2],
                student_dims=512,
                teacher_dims=1024,
                query_hw=(14,19),  # shape of tearcher feature 
                pos_hw=(int(args.input_height/32), int(args.input_width/32)),  # shape of student feature 
                pos_dims=1024,  # teacher feature dimension
                self_query=True,
                softmax_scale=[5.,5.],
                num_heads=16
            )

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
            args.batch_size = int(args.batch_size / ngpus_per_node)  # note that args.batch_size is the total batch size
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            if args.kd_ratio > 0:
                teacher_model.cuda(args.gpu)
                teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], broadcast_buffers=False)

                if args.f_distill: 
                    dis_modules_s4.cuda(args.gpu)
                    dis_modules_s4 = torch.nn.parallel.DistributedDataParallel(dis_modules_s4, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            if args.kd_ratio > 0:
                teacher_model.cuda()
                teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, broadcast_buffers=False)
    else:
        raise ValueError("Distributed training is not enabled. Please set --distributed flag.")

    if args.ckpt_path:
        key = 'model'
        _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
        logger.info("Successfully load whole ckpt {} from {}".format(args.ckpt_path, key))
        # new_state_dict = expand_depth(_ckpt[key], self.state_dict())  # Expand depth of the pretrained weight
        incompatibleKeys = model.load_state_dict(_ckpt[key], strict=False)
        logger.info("== missing_keys: {}".format(incompatibleKeys))
        del _ckpt

    if args.kd_ratio > 0:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    '''show model'''
    # with open(os.path.join(args.log_directory, 'max_model.log'),'w') as f:
    #     f.write(str(model))
    # assert False,'print model'

    '''show param'''
    # 计算FLOPs 和 Params
    '''https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file'''
    # print('*'*20, ' calflops ', '*'*20)
    # from calflops import calculate_flops
    # # from torchvision import models
    # import sys
    # with open('./FLOPs_Params.log', 'w') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    #     # model = models.resnet50()
    #     input_shape = (8, 3, 352, 1120)
    #     flops, macs, params = calculate_flops(model=model, 
    #                                         input_shape=input_shape,
    #                                         output_as_string=True,
    #                                         output_precision=2)
    #     sys.stdout = original_stdout
    #     print("net FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))        
    #     #Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M 
    # assert False, 'over'

    if args.distributed:
        logger.info("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        logger.info("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    if args.optimizer == 'adamw':
        backbone_params = list(model.module.backbone.parameters())
        other_params = [p for n, p in model.module.named_parameters() if not n.startswith("backbone.")]

        param_groups = [
            {'params': backbone_params, 'lr': args.learning_rate},
            {'params': other_params,    'lr': args.learning_rate},
        ]
        if args.f_distill:
            param_groups.append(
                {'params': dis_modules_s4.parameters(), 'lr': args.learning_rate}
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)  # TODO 可学习VSSD的权重衰减策略
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate)   
    else:
        raise ValueError("Unsupported optimizer: {}".format(args.optimizer))
    

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    if args.optimizer == 'adamw':
        scheduler = CosineLRScheduler(optimizer, num_total_steps, t_mul=1., lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=warmup_steps, cycle_limit=1, t_in_epochs=False, warmup_prefix=True)
    elif args.optimizer == 'adam':
        end_learning_rate = 0.1 * args.learning_rate

    ss = MambaSearchSpace(args.mlp_ratio, args.d_state, args.ssd_expand, open_depth=args.open_depth)

    # training
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
         
        random.seed(epoch)
        np.random.seed(epoch)
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image    = sample_batched['image'].cuda(args.gpu, non_blocking=True)
            depth_gt = sample_batched['depth'].cuda(args.gpu, non_blocking=True)

            if args.kd_ratio > 0:
                with torch.no_grad():
                    teacher_model.eval()
                    depth_tea, mid_feats_T = infer(teacher_model, image, dataset=args.dataset, focal=None)

            for _ in range(args.dynamic_batch_size):
                # random sample subnet
                sample_config = ss.sample(n_samples=1)[0]
                model_module = unwrap_model(model)
                model_module.backbone.set_sample_config(sample_config=sample_config)
                # print(sample_config)  # {'mlp_ratio': [2.0, 4.0, 4.0, 4.0], 'd_state': [32, 32, 64], 'expand': [2, 4, 2], 'depth': [[0, 0, 1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]}

                depth_est, mid_feats_S = model(image, mid_features=True)
                loss = silog_criterion.forward(depth_est, depth_gt)

                ### teacher model
                spat_loss = torch.tensor(0.0).cuda(args.gpu)
                freq_loss = torch.tensor(0.0).cuda(args.gpu)
                kd_loss = torch.tensor(0.0).cuda(args.gpu)
                if args.f_distill:
                    feat_T_s4 = mid_feats_T[3]
                    feat_S_s4 = mid_feats_S[3]

                    spat_loss, freq_loss = dis_modules_s4(feat_S_s4, feat_T_s4)
                    # print('spat_loss', spat_loss, 'freq_loss', freq_loss)
                    # spat_loss tTensor(43.6711, device='cuda:0', grad_fn=<AliasBackward0>) freq_loss tTensor(31.9551, device='cuda:0', grad_fn=<AliasBackward0>)
                    # assert False,'debug'

                if args.kd_ratio > 0:
                    kd_loss = silog_criterion.forward(depth_est, depth_tea, interpolate=True)
                    # print('loss', (1 - args.kd_ratio) * loss, 'kd_loss', args.kd_ratio * kd_loss, 'spat_loss', spat_loss, 'freq_loss', freq_loss)
                    loss = (1 - args.kd_ratio) * loss + args.kd_ratio * kd_loss + spat_loss + freq_loss
                
                
                loss = loss / args.dynamic_batch_size
                loss.backward()

            if args.optimizer == 'adamw':
                scheduler.step_update(epoch * steps_per_epoch + step)
                current_lr = optimizer.param_groups[1]['lr']  # for decoder
            elif args.optimizer == 'adam':
                for param_group in optimizer.param_groups:
                    current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                    param_group['lr'] = current_lr

            if args.max_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)  # TODO max_norm need to set
                # print(f'grad norm: {grad_norm:.2f}')
            optimizer.step()

            # show log and save result
            if global_step % 100 == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    logger.info('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.10f}, total_loss: {:.6f}, kd_loss: {:.6f}, spat_loss: {:.6f}, freq_loss: {:.6f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss.item(), args.kd_ratio * kd_loss.item(), spat_loss.item(), freq_loss.item()))
                    if np.isnan(loss.cpu().item()):
                        logger.info('NaN in loss occurred. Aborting training.')
                        return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                logger.info(print_string.format(args.gpu, examples_per_sec, loss.item(), var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss.item(), global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var_average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    for i in range(num_log_images):
                        writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :].data), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est[i, :, :, :].data), global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(args, model, dataloader_eval, gpu, ngpus_per_node, post_process=True, logger=logger, ss=ss)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            if eval_metrics[i] not in ['abs_rel', 'd1']:
                                continue  # 减少冗余文件
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                        if is_best and eval_metrics[i] == 'abs_rel':
                            symlink_path = os.path.join(args.log_directory, 'abs_rel_best_weight.pth')
                            target_path = os.path.join(args.log_directory, args.model_name + model_save_name)
                            relative_target_path = os.path.relpath(target_path, start=os.path.dirname(symlink_path))
                            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                                os.remove(symlink_path)  # rm the old link
                            os.symlink(relative_target_path, symlink_path)

                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            global_step += 1

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    args = parse_args()
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
        command = 'cp MambaDepthNAS/train.py ' + aux_out_path
        os.system(command)
        # command = 'mkdir -p ' + networks_savepath + ' && cp MambaDepthNAS/networks/*.py ' + networks_savepath
        # os.system(command)

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
