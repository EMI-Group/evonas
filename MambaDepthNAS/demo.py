import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"  # to set in config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
from PIL import Image 
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, get_root_logger, unwrap_model, str2list, infer, compute_metrics
from networks.model import MambaDepth


'''predict the single picture'''
# python MambaDepthNAS/demo.py configs/debug.txt 

def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, SuperNet, VSSD_final, tiny07', default='SuperNet')
    parser.add_argument('--devices',                    type=str, default='0', help='CUDA_VISIBLE_DEVICES value, e.g., "0" or "0,1"')

    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--image_path',                type=str,   help='path to the image for inference', required=True)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

    # arch config
    parser.add_argument('--mlp_ratio',                 type=str2list, default=[4.0,4.0,4.0,4.0])
    parser.add_argument('--d_state',                   type=str2list, default=[64,64,64,-1])
    parser.add_argument('--ssd_expand',                type=str2list, default=[4,4,4,-1])
    parser.add_argument('--depth',                     type=str2list, default=[2,4,8,4])
    parser.add_argument('--width_multiplier',          type=float, help='factor to scale the number of channels in each layer (applies only to final model)', default=1.0)

    # Log and save
    parser.add_argument('--ckpt_path',                 type=str,   help='path to load checkpoints',default=None)


    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args


def inference(args, model, post_process=False):
    
    image = np.asarray(Image.open(args.image_path), dtype=np.float32) / 255.0
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    with torch.no_grad():
        image = torch.autograd.Variable(image.unsqueeze(0).cuda())
       
        pred_depth = model(image)
        if post_process:
            image_flipped = flip_lr(image)
            pred_depths_flipped = model(image_flipped)
            pred_depth = post_process_depth(pred_depth, pred_depths_flipped)

        # if args.dataset == 'nyu':
        #     pred_depth = pred_depth[:, :, 10:470, 10:630]  # 裁剪掉nyu的图片边框

        pred_depth = pred_depth.cpu().numpy().squeeze()

        save_name = f"{args.dataset}_{os.path.splitext(os.path.basename(args.image_path))[0]}_{args.model_name}_depth.png"
        if args.dataset == 'kitti':
            plt.imsave(save_name, np.log10(pred_depth), cmap='magma')  # , vmin=0, vmax=2
        else:
            plt.imsave(save_name, pred_depth, cmap='jet')

            # ## 打印原始深度值，不颜色映射
            # depth_image = Image.fromarray(np.uint8(pred_depth))  # 如果需要转换为其他类型请根据需要修改
            # depth_image.save(save_name)
            
    print(f'== Finished inferenced and saved in {save_name}')



def main_worker(args):


    # MambaDepth model
    final_config = {
        'mlp_ratio': args.mlp_ratio,
        'd_state': args.d_state + [-1],
        'ssd_expand': args.ssd_expand + [-1],
        'depth': [sum(lst) for lst in args.depth],
    } if args.encoder == 'VSSD_final' else None

    model = MambaDepth(args, version=args.encoder, max_depth=args.max_depth, selected_config=final_config)
    model.cuda()

    # load whole weight from supernet
    if args.ckpt_path:
        key = 'model'
        _ckpt = torch.load(open(args.ckpt_path, "rb"), map_location=torch.device("cpu"))
        print("Successfully load whole ckpt {} from {}".format(args.ckpt_path, key))
        import re
        new_sd = {re.sub(r"^module\.", "", k): v for k, v in _ckpt[key].items()}

        incompatibleKeys = model.load_state_dict(new_sd, strict=False)
        print("== missing_keys: {}".format(incompatibleKeys))
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

    model.eval()
    with torch.no_grad():
        inference(args, model, post_process=True)



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    main_worker(args)


if __name__ == '__main__':
    main()
